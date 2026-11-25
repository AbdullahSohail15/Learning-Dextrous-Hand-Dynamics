"""Parametric (grey-box) Allegro hand model with identifiable physical parameters (experiment 46). The Allegro analogue of
parametric_hand.ParametricHand: same interface (rollout, residual_load, values / load_values, to_prior_config, param_spec), no tendons.

Every joint j (all 16 actuated; hand gravity off), implicit Euler at 120 Hz (6 substeps per 20 Hz action):

    M_j(q) qdd_j = u_j + s_j g_j(q) - Fc_j tanh(qd_j / 0.02) + tau_ff_j [+ tau_c_j]
    u_j          = clip(K_j (a_j - q_j) - D_j qd_j, -u_max_j, u_max_j)          PhysX implicit position drive with an effort limit
    g_j(q)       = gravity torque about joint j's axis: sum over the bodies b below j of m_b ((p_b - o_j) x g) . axis_j (export masses / COMs,
                   g in the root frame = physics['gravity_root']; zero when hand gravity is off); recomputed at every control step, s_j a learned scale
    M_j(q)       = composite subtree inertia about joint j's axis at the window start (FK + parallel-axis, physics_prior.NominalHand) + armature_j
    |qd_j|       <= v_max_j                                                       PhysX joint velocity limit (360 deg/s on the stock asset)

Unsaturated substeps use the linear implicit update; where the implicit drive would exceed u_max, the drive is the clipped value (a known
torque, semi-implicit). Joint limits clamp q and zero qd. tau_c is either the quasi-static load observer of the export (cfg['disturbance'],
"exportobs": tau_c = -u(q, qd, a_{t-1}) + friction, held over the horizon) or the residual observer of experiment 37a (cfg['residual_observer']).

Parameters (raw; `values()` gives physical units): tau_j = D_j / K_j [s], K_j [N m/rad], armature_j [kg m^2], Fc_j [N m], gravity scale s_j, u_max_j [N m],
v_max_j [rad/s] (u_max and v_max fixed at the export unless freed). Units at the interface: deg, deg/s in and out; radians inside.
"""
from __future__ import annotations
import copy, json, math
from pathlib import Path
import numpy as np
import torch
from .physics_prior import NominalHand
V_EPS = 0.02   # rad/s, Coulomb smoothing


class AllegroHand(torch.nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg = copy.deepcopy(cfg); base_cfg = copy.deepcopy(cfg); base_cfg.pop("identified", None)
        self.base = NominalHand(base_cfg, device); b = self.base; dev = torch.device(device)
        self.names = b.joint_names; n = self.n = len(self.names); assert b.act == list(range(n)) and not b.pas
        self.act = b.act; self.act_names = list(self.names)
        self.register_buffer("act_mask", torch.ones(n, dtype=torch.bool, device=dev))
        self.register_buffer("lo", b.lo.clone()); self.register_buffer("hi", b.hi.clone()); self.substeps, self.h = b.substeps, b.h
        P = cfg["physics"]; t = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
        self.register_buffer("K_nom", b.K.clone()); self.register_buffer("tau_nom", b.D / b.K)
        self.log_tau = torch.nn.Parameter(torch.log(b.D / b.K)); self.log_K = torch.nn.Parameter(torch.log(b.K.clone()))
        # cfg["af_param"] == "lin" (experiment 47): armature = 1e-3 softplus_20(log_arm) kg m^2 and Fc = 0.01 softplus_20(raw_fc) N m, initialised
        # at the export's armature (floor 1e-5) and joint friction, so Adam steps move them on a physical scale. Default (exps 46-46c): log armature
        # from 1e-6 and Fc from 3e-6 N m, which the fit never moves (gradients ~1e-7).
        self.af_lin = cfg.get("af_param") == "lin"
        if self.af_lin:
            self.log_arm = torch.nn.Parameter(self._inv_sp(b.arm.clamp_min(1e-5) / 1e-3))
            self.raw_fc = torch.nn.Parameter(self._inv_sp(t(P.get("friction", [0.0] * n)).clamp_min(1e-4) / 0.01))
        else:
            self.log_arm = torch.nn.Parameter(torch.log(b.arm.clamp_min(1e-6))); self.raw_fc = torch.nn.Parameter(torch.full((n,), -8.0, device=dev))
        self.log_umax = torch.nn.Parameter(torch.log(t(P["max_force"]))); self.log_vmax = torch.nn.Parameter(torch.log(t(P["max_velocity"])))
        self.log_gscale = torch.nn.Parameter(torch.zeros(n, device=dev))
        self.register_buffer("g_root", t(P.get("gravity_root", [0.0, 0.0, 0.0]))); self.gravity = bool(self.g_root.abs().sum() > 0)
        self.observer = bool(cfg.get("disturbance", False))
        if "identified" in cfg: self.load_values(cfg["identified"]["values"])

    # ---- physical values
    def K(self): return self.log_K.exp()
    def D(self): return self.log_tau.exp() * self.log_K.exp()
    @staticmethod
    def _inv_sp(y, beta=20.0): return y + torch.log(-torch.expm1(-beta * y)) / beta   # stable inverse of softplus_beta
    def arm(self): return 1e-3 * torch.nn.functional.softplus(self.log_arm, beta=20.0) if self.af_lin else self.log_arm.exp()
    def Fc(self): return 0.01 * torch.nn.functional.softplus(self.raw_fc, beta=20.0) if self.af_lin else torch.nn.functional.softplus(self.raw_fc) * 0.01
    def umax(self): return self.log_umax.exp()
    def vmax(self): return self.log_vmax.exp()
    def gscale(self): return self.log_gscale.exp()

    @torch.no_grad()
    def gravity_torque(self, q):
        """(N, 16) N m at pose q (rad), export masses, scale 1."""
        b = self.base; Tb, Tj = b._fk(q); R = Tb[..., :3, :3]; p = (R @ b.com[None, :, :, None])[..., 0] + Tb[..., :3, 3]   # COMs (N, B, 3)
        axis = Tj[..., :3, 0]; o = Tj[..., :3, 3]; F = b.mass[None, :, None] * self.g_root                               # (N, J, 3), (1, B, 3)
        tq = torch.cross(p[:, None] - o[:, :, None], F[:, None].expand(len(q), axis.shape[1], -1, -1), dim=-1)            # (N, J, B, 3)
        return ((tq * axis[:, :, None]).sum(-1) * b.sub[None]).sum(-1)

    def values(self):
        A = lambda x: x.detach().cpu().numpy().tolist()
        return dict(joints=list(self.names), tau_s=A(self.log_tau.exp()), K_Nm_rad=A(self.K()), D_Nms_rad=A(self.D()), armature_kgm2=A(self.arm()),
                    Fc_Nm=A(self.Fc()), gravity_scale=A(self.gscale()), umax_Nm=A(self.umax()), vmax_rad_s=A(self.vmax()))

    @torch.no_grad()
    def load_values(self, v):
        idx = [self.names.index(j) for j in v["joints"]]; t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.log_K.device)
        self.log_tau[idx] = t(v["tau_s"]).log(); self.log_K[idx] = t(v["K_Nm_rad"]).log(); self.log_arm[idx] = t(v["armature_kgm2"]).clamp_min(1e-9).log()
        fc = t(v["Fc_Nm"]).clamp_min(1e-7) / 0.01; self.raw_fc[idx] = fc + torch.log(-torch.expm1(-fc))   # inverse softplus
        if self.af_lin: self.log_arm[idx] = self._inv_sp(t(v["armature_kgm2"]).clamp_min(1e-7) / 1e-3); self.raw_fc[idx] = self._inv_sp(t(v["Fc_Nm"]).clamp_min(1e-6) / 0.01)
        if "umax_Nm" in v: self.log_umax[idx] = t(v["umax_Nm"]).log()
        if "vmax_rad_s" in v: self.log_vmax[idx] = t(v["vmax_rad_s"]).log()
        if "gravity_scale" in v: self.log_gscale[idx] = t(v["gravity_scale"]).clamp_min(1e-6).log()

    def to_prior_config(self, extra=None):
        cfg = copy.deepcopy(self.cfg); cfg["identified"] = dict(values=self.values(), **(extra or {})); cfg["disturbance"] = self.observer; return cfg

    def subtree_inertia(self, q):
        return self.base.effective_inertia(q) - self.base.arm[None]

    # ---- dynamics
    def rollout(self, s, fa, pa=None, tau_ff=None, grad=False, tau_ff_direct=None):
        """s (N, >=1, 32) deg/deg/s (last state used), fa (N, H, 16) future applied targets deg, pa (N, >=1, 16) past applied targets (observers),
        tau_ff / tau_ff_direct (N, H, 16) N m known torques or None -> (N, H, 32) deg, deg/s."""
        ctx = torch.enable_grad() if grad else torch.no_grad(); n = self.n
        with ctx:
            x = s[:, -1].float(); q = torch.deg2rad(x[:, :n]); v = torch.deg2rad(x[:, n:]); A = torch.deg2rad(fa.float())
            with torch.no_grad(): Msub = self.subtree_inertia(q)
            M = Msub + self.arm()[None]; K, D, Fc, um, vm = self.K()[None], self.D()[None], self.Fc()[None], self.umax()[None], self.vmax()[None]; h = self.h
            fric = lambda v: -Fc * torch.tanh(v / V_EPS)
            tau_c = torch.zeros_like(q)
            ro = self.cfg.get("residual_observer")
            if ro:
                assert pa is not None and s.shape[1] >= 2, "the residual observer needs past states and actions"
                tau_c = self.residual_load(s, pa, float(ro.get("gain", 1.0)), float(ro.get("max_nm", 0.3)))
            grav = (lambda q: self.gravity_torque(q.detach()) * self.gscale()[None]) if self.gravity else (lambda q: torch.zeros_like(q))
            if self.observer:
                assert pa is not None, "the observer needs the last applied target"
                u0 = (K * (torch.deg2rad(pa[:, -1].float()) - q) - D * v).clamp(-um, um); tau_c = -u0 - fric(v) - grav(q)
            out = []
            for k in range(A.shape[1]):
                tgt = A[:, k]; ff = torch.zeros_like(q); tg = grav(q)
                if tau_ff is not None: ff = tau_ff[:, k].float()
                elif tau_ff_direct is not None: ff = tau_ff_direct[:, k].float()
                for _ in range(self.substeps):
                    tau_o = fric(v) + ff + tau_c + tg
                    vl = (M * v + h * (K * (tgt - q) + tau_o)) / (M + h * D + h * h * K)        # linear implicit step
                    ul = K * (tgt - q - h * vl) - D * vl                                         # its drive torque at the end of the step
                    us = ul.clamp(-um, um); vs = v + h * (us + tau_o) / M                       # saturated: clipped drive as a known torque
                    vn = torch.where(ul.abs() > um, vs, vl).clamp(-vm, vm)
                    qn = q + h * vn
                    at = (qn < self.lo) | (qn > self.hi); q = torch.minimum(torch.maximum(qn, self.lo), self.hi); v = torch.where(at, torch.zeros_like(vn), vn)
                out.append(torch.cat([torch.rad2deg(q), torch.rad2deg(v)], -1))
            return torch.stack(out, 1)

    @torch.no_grad()
    def residual_load(self, s, pa, gain=1.0, max_nm=0.3, eps=1e-2):
        """(N, 16) N m constant torque per joint that best explains the model's one-step angle errors over the past window (as ParametricHand)."""
        n = self.n; W = min(s.shape[1] - 1, pa.shape[1]); N = s.shape[0]
        x0 = s[:, -W - 1:-1].reshape(N * W, 1, -1); x1 = torch.deg2rad(s[:, -W:, :n].reshape(N * W, n).float()); A = pa[:, -W:].reshape(N * W, 1, n)
        cfg_ro = self.cfg.pop("residual_observer", None); obs = self.observer; self.observer = False
        try:
            base = torch.deg2rad(self.rollout(x0, A)[:, 0, :n])
            pert = torch.deg2rad(self.rollout(x0, A, tau_ff_direct=torch.zeros(N * W, 1, n, device=s.device) + eps)[:, 0, :n])
        finally:
            self.observer = obs
            if cfg_ro is not None: self.cfg["residual_observer"] = cfg_ro
        g = ((pert - base) / eps).reshape(N, W, n); e = (x1 - base).reshape(N, W, n)
        d = gain * (g * e).sum(1) / (g.square().sum(1) + 1e-12)
        return d.clamp(-max_nm, max_nm).float()

    def param_spec(self, with_lam=False):
        spec = []
        for i, j in enumerate(self.names):
            spec += [(f"log_tau[{j}]", self.log_tau, i), (f"log_K[{j}]", self.log_K, i), (f"log_arm[{j}]", self.log_arm, i), (f"raw_fc[{j}]", self.raw_fc, i)]
            if self.gravity: spec.append((f"log_gscale[{j}]", self.log_gscale, i))
        return spec


def allegro_prior_config(export_dir, disturbance=False):
    """Prior config of the export (no identified block): physics parameters + kinematic tree, tagged hand='allegro'."""
    from .physics_prior import prior_config
    d = Path(export_dir); c = prior_config(d / "physics_parameters.json", d / "model.json", {}, disturbance=disturbance)
    keep = ("max_velocity", "friction", "gravity_root"); P = json.loads((d / "physics_parameters.json").read_text())
    c["physics"].update({k: P[k] for k in keep}); c["hand"] = "allegro"; return c


def load_allegro(path_or_cfg, device="cpu"):
    cfg = json.loads(Path(path_or_cfg).read_text()) if isinstance(path_or_cfg, (str, Path)) else path_or_cfg
    return AllegroHand(cfg, device).eval()
