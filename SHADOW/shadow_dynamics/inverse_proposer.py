"""Inverse-dynamics action proposers (experiment 30): a rectified-flow or deterministic model of the command that brings the system from its
recent history to a desired outcome, written as a prior plus a learned residual (the structure of the rectified-flow inverse-dynamics diagram
the user shared: u = u_phys + residual, trained with flow matching and a consistency loss through a frozen forward model).

Inputs: past states s (B, 4, D), past applied targets a (B, 3, 20), desired outcome d (B, Dd), prior command u_phys (B, 20).
  gestures, variant T: d = 15 fingertip coordinates (mm), u_phys = the current command (hold); output g = held 20-D target (deg)
  gestures, variant J: d = 24 joint angles (deg),       u_phys = the desired active joints (direct command)
  cube:               d = desired cube rotation over 0.6 s (rotation vector, deg, world frame), u_phys = P's action (x units, [-1, 1])
The network works on the normalized residual y = (u - u_phys) / res_std. Flow kind: linear path y_tau = (1 - tau) z + tau y, velocity target y - z,
sampled with Euler steps from z ~ N(0, I). Det kind: one pass, MSE.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .flow_model import FiLMBlock, HIST, PAST_A, AD, H


class InverseProposer(nn.Module):
    def __init__(self, kind, sd, dd, task, width=512, depth=4, stats=None):
        super().__init__()
        assert kind in ("flow", "det"); self.kind, self.flow = kind, kind == "flow"
        self.config = dict(kind=kind, sd=sd, dd=dd, task=task, width=width, depth=depth)
        self.enc = nn.Sequential(nn.Linear(HIST * sd + PAST_A * AD + dd + AD, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        if self.flow:
            self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(1000), 32)))
            self.time = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128)); self.inp = nn.Linear(AD, width); ctx = width + 128
        else:
            self.inp = nn.Linear(width, width); ctx = width
        self.ctx = nn.Sequential(nn.Linear(ctx, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([FiLMBlock(width, width) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, AD)); nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)
        for k, n in (("s_mean", sd), ("s_std", sd), ("a_mean", AD), ("a_std", AD), ("d_mean", dd), ("d_std", dd), ("u_mean", AD), ("u_std", AD), ("res_std", AD)):
            self.register_buffer(k, torch.zeros(n) if k.endswith("mean") else torch.ones(n))
        if stats is not None:
            for k in ("s_mean", "s_std", "a_mean", "a_std", "d_mean", "d_std", "u_mean", "u_std", "res_std"): getattr(self, k).copy_(torch.as_tensor(stats[k], dtype=torch.float32))

    def condition(self, s, a, d, u_phys):
        z = torch.cat([((s - self.s_mean) / self.s_std).flatten(1), ((a - self.a_mean) / self.a_std).flatten(1), (d - self.d_mean) / self.d_std, (u_phys - self.u_mean) / self.u_std], -1)
        return self.enc(z)

    def head(self, h, c):
        for b in self.blocks: h = b(h, c)
        return self.out(h)

    def velocity(self, y, tau, cond):
        ph = tau[:, None] * self.freqs[None] * 2 * math.pi
        return self.head(self.inp(y), self.ctx(torch.cat([cond, self.time(torch.cat([ph.sin(), ph.cos()], -1))], -1)))

    def loss(self, s, a, d, u_phys, u):
        """Flow-matching or MSE loss on the normalized residual of the label command u."""
        y1 = (u - u_phys) / self.res_std; cond = self.condition(s, a, d, u_phys)
        if not self.flow: return (self.head(self.inp(cond), self.ctx(cond)) - y1).square().mean()
        z = torch.randn_like(y1); tau = torch.rand(len(y1), device=y1.device)
        return (self.velocity((1 - tau[:, None]) * z + tau[:, None] * y1, tau, cond) - (y1 - z)).square().mean()

    def sample(self, s, a, d, u_phys, K=1, steps=4, generator=None, grad=False):
        """(K, B, 20) commands. With grad=True (consistency loss) gradients flow through the Euler steps."""
        with torch.set_grad_enabled(grad):
            cond = self.condition(s, a, d, u_phys); B = len(s)
            if not self.flow:
                y = self.head(self.inp(cond), self.ctx(cond))[None].expand(K, -1, -1)
            else:
                c = cond.repeat(K, 1); y = torch.randn(K * B, AD, device=s.device, generator=generator); dt = 1.0 / steps
                for i in range(steps): y = y + dt * self.velocity(y, torch.full((K * B,), i * dt, device=s.device), c)
                y = y.view(K, B, AD)
            return u_phys[None] + y * self.res_std


def forward_with_grad(F, s, a, fa):
    """Deterministic forward model (det_mlp / res_det / physics kinds of flow_model.HandDynamics) with gradients w.r.t. fa. Returns (B, 12, D)."""
    assert not F.flow
    phys = F.physics(s, a, fa) if F.uses_prior else None
    if F.kind == "physics": return phys
    cond = F.condition(s, a, fa, phys); y = F.head(F.inp(cond), F.ctx(cond))
    if F.residual: return phys + F.res_mean + y * F.res_std
    return s[:, -1:, :] + F.delta_mean + y * F.delta_std


def load_proposer(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = InverseProposer(**ck["config"]).to(device); m.load_state_dict(ck["model"]); m.eval(); return m, ck
