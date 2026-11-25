"""Experiment 50: discrete-time control barrier function (CBF) safety filter against self-collision.

Barrier per link pair p: h_p(q) = clearance_p(q) + tol (m; HandCSpace.clearance: > 0 apart). `tol` lets a task keep the contact its own
goal needs (curled-finger and pinch goals touch). Condition over n control steps with the command held: h(q_{k+n}) >= (1 - gamma)^n h(q_k).
q_{k+n}(u) is predicted by the identified model and linearised in u (finite differences); h is linearised in q (finite differences through
the forward kinematics). Pairs farther than `d_act` are ignored. The filtered command is the closest one (least squares) satisfying the
linearised conditions (dual projected gradient), then projected on the joint box and the slew limit.
"""
from __future__ import annotations
import torch


class CBFFilter:
    def __init__(self, cspace, step_fn, low, high, slew=4.0, n=2, gamma=0.3, d_act=0.010, eps_u=0.1, eps_q=0.05, iters=300):
        self.cs = cspace; self.f = step_fn; self.low, self.high = low, high; self.slew = slew; self.n = n; self.g = gamma
        self.d_act = d_act; self.eps_u = eps_u; self.eps_q = eps_q; self.iters = iters; self.stats = dict(steps=0, active=0, modified=0, dnorm=0.0)

    def _predict(self, x, u):
        y = x
        for _ in range(self.n): y = self.f(y, u)
        return y[:, :24]

    @torch.no_grad()
    def filter(self, x, up, u_nom, tol):
        """x (E, 48) measured state, up (E, 20) last command, u_nom (E, 20), tol (E,) m -> (E, 20) filtered command."""
        E = x.shape[0]; dev = x.device
        h0 = self.cs.clearance(x[:, :24]) + tol[:, None]                                               # (E, P)
        U = torch.cat([u_nom[:, None], u_nom[:, None] + self.eps_u * torch.eye(20, device=dev)[None]], 1)   # (E, 21, 20)
        Q = self._predict(x[:, None].expand(E, 21, 48).reshape(-1, 48), U.reshape(-1, 20)).view(E, 21, 24)
        q1 = Q[:, 0]; S = ((Q[:, 1:] - q1[:, None]) / self.eps_u).transpose(1, 2)                      # (E, 24, 20) dq/du
        qs = torch.cat([q1[:, None], q1[:, None] + self.eps_q * torch.eye(24, device=dev)[None]], 1)     # (E, 25, 24)
        C = self.cs.clearance(qs.reshape(-1, 24)).view(E, 25, -1) + tol[:, None, None]
        h1 = C[:, 0]; Gq = ((C[:, 1:] - h1[:, None]) / self.eps_q).transpose(1, 2)                      # (E, P, 24) dh/dq
        G = (Gq @ S) * 1000; b = ((1 - self.g) ** self.n * h0 - h1) * 1000                            # mm; need G d >= b
        active = (h0 < self.d_act) | (h1 < self.d_act); G = G * active[..., None]; b = torch.where(active, b, torch.full_like(b, -1e9))
        need = (b > 0).any(1)
        d = torch.zeros(E, 20, device=dev)
        if need.any():
            Gn, bn = G[need], b[need].clamp_min(-1e3); lam = torch.zeros_like(bn)
            L = (Gn @ Gn.transpose(1, 2)).diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-9)              # step 1 / trace >= 1 / lambda_max
            for _ in range(self.iters):
                dd = torch.einsum("epj,ep->ej", Gn, lam); lam = (lam + (bn - torch.einsum("epj,ej->ep", Gn, dd)) / L[:, None]).clamp_min(0)
            d[need] = torch.einsum("epj,ep->ej", Gn, lam)
        u = (up + (u_nom + d - up).clamp(-self.slew, self.slew)).clamp(self.low, self.high)
        s = self.stats; s["steps"] += E; s["active"] += int(active.any(1).sum()); s["modified"] += int(need.sum()); s["dnorm"] += float(d.norm(dim=1).sum())
        return u
