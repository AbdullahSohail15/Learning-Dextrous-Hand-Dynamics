"""Whole-horizon Shadow Hand dynamics models (project_iteration_v2/shadow_hand_flow_matching_architecture.md).

Inputs (physical units, normalized inside the model with statistics stored in the checkpoint):
  past_states (B, 4, 48)   q deg + qdot deg/s for x_{t-3..t}
  past_actions (B, 3, 20)  applied targets a_{t-3..t-1}, deg
  future_actions (B, 12, 20) a_{t..t+11}, deg
Output: future states (B, 12, 48); internally the model works on the normalized delta y = (x_{t+h} - x_t - mean_h) / std_h.

Kinds:
  'flow_tcn'  TCN history encoder + future-action MLP + flow-time embedding + FiLM residual MLP flow head (main model)
  'flow_mlp'  same with a flattened-history MLP encoder
  'det_mlp'   flattened-history encoder, deterministic residual MLP head on the delta trajectory (MSE)
Physics-prior kinds (experiment 09; shadow_dynamics.physics_prior.NominalHand, its parameters stored in config['prior']):
  'physics'       the nominal prior alone, no learned parameters
  'res_det'       det_mlp on the residual R = Y_true - Y_physics, with the nominal trajectory as an extra input
  'res_flow_tcn'  flow_tcn on the residual, same extra input; prediction = Y_physics + R
"""
from __future__ import annotations
import math
import torch
from torch import nn

H, HIST, PAST_A, SD, AD = 12, 4, 3, 48, 20


class TCNEncoder(nn.Module):
    """Per history step [x_k, a_{k-1}, flag]; a_{t-4} is outside the window, so the first step gets zeros and flag 0."""

    def __init__(self, width, out, sd=SD, ad=AD):
        super().__init__()
        self.ad = ad; self.proj = nn.Linear(sd + ad + 1, width)
        self.conv = nn.Sequential(nn.Conv1d(width, width, 2), nn.SiLU(), nn.Conv1d(width, width, 2), nn.SiLU())
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(width * (HIST - 2), out))

    def forward(self, s, a):
        B = s.shape[0]
        prev = torch.cat([torch.zeros(B, 1, self.ad, device=s.device, dtype=s.dtype), a], 1)
        flag = torch.cat([torch.zeros(B, 1, 1, device=s.device, dtype=s.dtype), torch.ones(B, HIST - 1, 1, device=s.device, dtype=s.dtype)], 1)
        z = self.proj(torch.cat([s, prev, flag], -1)).transpose(1, 2)
        return self.out(self.conv(z))


class MLPEncoder(nn.Module):
    def __init__(self, width, out, sd=SD, ad=AD):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(HIST * sd + PAST_A * ad, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, out))

    def forward(self, s, a):
        return self.net(torch.cat([s.flatten(1), a.flatten(1)], -1))


class FiLMBlock(nn.Module):
    def __init__(self, width, ctx):
        super().__init__()
        self.norm = nn.LayerNorm(width); self.film = nn.Linear(ctx, 2 * width)
        self.net = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, h, c):
        scale, shift = self.film(c).chunk(2, -1)
        return h + self.net(self.norm(h) * (1 + scale) + shift)


class HandDynamics(nn.Module):
    def __init__(self, kind="flow_tcn", hist_width=256, cond_dim=256, width=512, depth=6, time_dim=64, stats=None, prior=None, state_dim=SD, loss_channels=None, contact_loss_weight=None, action_dim=AD):
        super().__init__()
        assert kind in ("flow_tcn", "flow_mlp", "det_mlp", "physics", "res_det", "res_flow_tcn")
        self.kind = kind; self.flow = kind in ("flow_tcn", "flow_mlp", "res_flow_tcn"); self.residual = kind.startswith("res_"); self.uses_prior = self.residual or kind == "physics"
        self.config = dict(kind=kind, hist_width=hist_width, cond_dim=cond_dim, width=width, depth=depth, time_dim=time_dim, prior=prior, state_dim=state_dim, loss_channels=loss_channels, contact_loss_weight=contact_loss_weight)
        if action_dim != AD: self.config["action_dim"] = action_dim   # experiment 46: Allegro (16 actions); Shadow configs stay unchanged
        self.ad = ad = action_dim
        self.loss_channels = loss_channels   # experiment 34: train only the first k output channels (later channels are inputs only)
        w = torch.ones(state_dim)
        if contact_loss_weight is not None: w[63:] = contact_loss_weight   # experiment 34 follow-up: down-weight the contact channels (>= 63) in the loss
        self.register_buffer("loss_w", w if loss_channels is None else w[:loss_channels], persistent=False)
        self.sd = sd = state_dim   # 48 = hand only; 63 = hand + cube (experiment 12): channels 48.. use persistence as their prior
        assert prior is not None or not self.uses_prior, "physics kinds need the prior config"
        self._prior = {}
        self.hist = TCNEncoder(hist_width, cond_dim, sd, ad) if kind in ("flow_tcn", "res_flow_tcn") else MLPEncoder(hist_width, cond_dim, sd, ad)
        self.act = nn.Sequential(nn.Linear(H * ad, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        ctx_in = 2 * cond_dim
        if self.residual:   # physics encoder over the normalized nominal delta trajectory
            self.phys = nn.Sequential(nn.Linear(H * sd, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)); ctx_in += cond_dim
        if self.flow:
            self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(1000), time_dim // 2)))
            self.time = nn.Sequential(nn.Linear(time_dim, 128), nn.SiLU(), nn.Linear(128, 128)); ctx_in += 128
            self.inp = nn.Linear(H * sd, width)
        else:
            self.inp = nn.Linear(ctx_in, width)
        self.ctx = nn.Sequential(nn.Linear(ctx_in, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([FiLMBlock(width, width) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, H * sd))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)
        for k in ("state_mean", "state_std", "action_mean", "action_std"):
            self.register_buffer(k, torch.zeros(sd if k.startswith("state") else ad))
        self.register_buffer("delta_mean", torch.zeros(H, sd)); self.register_buffer("delta_std", torch.ones(H, sd))
        if self.residual: self.register_buffer("res_mean", torch.zeros(H, sd)); self.register_buffer("res_std", torch.ones(H, sd))   # only residual kinds (old checkpoints load strictly)
        if stats is not None: self.set_stats(stats)

    def set_stats(self, st):
        for k in ("state_mean", "state_std", "action_mean", "action_std", "delta_mean", "delta_std") + (("res_mean", "res_std") if self.residual else ()):
            getattr(self, k).copy_(torch.as_tensor(st[k], dtype=torch.float32))

    def physics(self, s, a, fa):
        """Nominal trajectory (B, 12, 48) in deg, deg/s, or None for black-box kinds. Uses x_t and the future actions only.
        Experiment 28: config['prior'] = {'source': 'model', 'path': ckpt} uses a frozen learned model's prediction as the prior (all channels);
        {'source': 'sim'} = nominal-simulator futures, which the caller must pass as `phys` (sample(..., phys=...))."""
        if not self.uses_prior: return None
        src = self.config["prior"].get("source") if isinstance(self.config["prior"], dict) else None
        if src == "model":
            key = ("model", str(s.device))
            if key not in self._prior: self._prior[key] = load_model(self.config["prior"]["path"], s.device)[0]
            return self._prior[key].sample(s, a, fa)[0]
        if src == "sim": raise ValueError("simulator prior: pass phys= (nominal-simulator futures) to sample()")
        from .physics_prior import NominalHand
        key = str(s.device)
        if key not in self._prior:   # experiment 36: a prior config with an 'identified' block is the parametric (grey-box) hand model
            if self.config["prior"].get("hand") == "allegro":   # experiment 46: Allegro equation (shadow_dynamics.allegro_hand_model)
                from .allegro_hand_model import AllegroHand
                self._prior[key] = AllegroHand(self.config["prior"], s.device).eval()
            elif "identified" in self.config["prior"]:
                from .parametric_hand import ParametricHand
                self._prior[key] = ParametricHand(self.config["prior"], s.device).eval()
            else: self._prior[key] = NominalHand(self.config["prior"], s.device)
        hsd = 2 * len(self.config["prior"]["physics"]["joint_names"])   # hand state channels (Shadow 48, Allegro 32)
        hand = self._prior[key].rollout(s[..., :hsd], fa, a)
        if self.sd == hsd: return hand
        return torch.cat([hand, s[:, -1:, hsd:].expand(-1, H, -1)], -1)   # object channels: persistence prior

    def condition(self, s, a, fa, phys=None):
        parts = []
        if self.residual: parts.append(self.phys((((phys - s[:, -1:, :]) - self.delta_mean) / self.delta_std).flatten(1)))
        s = (s - self.state_mean) / self.state_std; a = (a - self.action_mean) / self.action_std; fa = (fa - self.action_mean) / self.action_std
        return torch.cat([self.hist(s, a), self.act(fa.flatten(1))] + parts, -1)

    def head(self, h, c):
        for b in self.blocks: h = b(h, c)
        return self.out(h).view(-1, H, self.sd)

    def velocity(self, y, tau, cond):
        ph = tau[:, None] * self.freqs[None] * 2 * math.pi
        c = self.ctx(torch.cat([cond, self.time(torch.cat([ph.sin(), ph.cos()], -1))], -1))
        return self.head(self.inp(y.flatten(1)), c)

    def loss(self, s, a, fa, y1, phys=None):
        """y1: normalized delta target (B, 12, 48); for residual kinds the normalized residual (pass the same phys)."""
        cond = self.condition(s, a, fa, phys)
        if not self.flow:
            c = self.ctx(cond); return self._wmse((self.head(self.inp(cond), c) - y1)[..., :self.loss_channels])
        y0 = torch.randn_like(y1); tau = torch.rand(y1.shape[0], device=y1.device)
        yt = (1 - tau[:, None, None]) * y0 + tau[:, None, None] * y1
        return self._wmse((self.velocity(yt, tau, cond) - (y1 - y0))[..., :self.loss_channels])

    def _wmse(self, e):   # channel-weighted mean square; equals .mean() when all weights are 1
        return (e.pow(2) * self.loss_w).sum(-1).mean() / self.loss_w.sum()

    def normalize_target(self, s, future, phys=None):
        if self.residual: return (future - phys - self.res_mean) / self.res_std
        return (future - s[:, -1:, :] - self.delta_mean) / self.delta_std

    @torch.no_grad()
    def sample(self, s, a, fa, K=1, steps=4, generator=None, phys=None):
        """Physical future states, (K, B, 12, 48). Deterministic model returns K identical copies. phys: an external prior trajectory (experiment 28)."""
        if phys is None: phys = self.physics(s, a, fa) if self.uses_prior else None
        if self.kind == "physics": return phys[None].expand(K, -1, -1, -1)
        cond = self.condition(s, a, fa, phys); B = s.shape[0]
        if not self.flow:
            c = self.ctx(cond); y = self.head(self.inp(cond), c)[None].expand(K, -1, -1, -1)
        else:
            cond = cond.repeat(K, 1)
            y = torch.randn(K * B, H, self.sd, device=s.device, generator=generator); dt = 1.0 / steps
            for i in range(steps):
                tau = torch.full((K * B,), i * dt, device=s.device); y = y + dt * self.velocity(y, tau, cond)
            y = y.view(K, B, H, self.sd)
        if self.residual: return phys[None] + self.res_mean + y * self.res_std
        return s[None, :, -1:, :] + self.delta_mean + y * self.delta_std


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = HandDynamics(**ck["config"]).to(device); m.load_state_dict(ck["model"]); m.eval()
    return m, ck


class EnsembleDynamics(torch.nn.Module):
    """Experiment 25: a deterministic ensemble as a sampled forward model. sample() returns one future per member, (M, B, 12, D),
    whatever K is; `flow` is True so that the planner averages over members and the LCB uses their spread."""

    def __init__(self, members):
        super().__init__(); self.members = torch.nn.ModuleList(members); self.flow = True; self.kind = "ensemble:" + members[0].kind
        assert all(not m.flow for m in members)

    @torch.no_grad()
    def sample(self, s, a, fa, K=1, steps=4, generator=None):
        return torch.cat([m.sample(s, a, fa, K=1) for m in self.members], 0)


def load_forward(spec, device="cpu"):
    """'path' -> HandDynamics; 'path1+path2+...' -> EnsembleDynamics (experiment 25)."""
    paths = spec.split("+")
    if len(paths) == 1: return load_model(paths[0], device)
    ms = [load_model(p_, device) for p_ in paths]; return EnsembleDynamics([m for m, _ in ms]).eval(), ms[0][1]
