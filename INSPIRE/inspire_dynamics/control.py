"""Experiments 03-04: controllers on the identified servo-linkage model (6 servos), batched over E tasks, torch.

State x = (p, v) in deg, deg/s (E, 12); command u = 6 servo targets (deg). f(x, u) = one 20 Hz step of servo_model with fitted theta.
  CEM      held-command MPC: candidates = 4 knots over the 0.6 s horizon (linearly interpolated), Gaussian around the mean, 3 iterations,
           cost = mean squared servo error to the reference over the horizon (+2x terminal) + smoothness; re-solved every 3 steps.
  ILQR     iLQR (Li & Todorov 2004) as receding-horizon MPC with TVLQR feedback between re-solves; same cost; Jacobians of f by batched
           forward differences (1 + 12 + 6 = 19 model calls per time step, vs 69 on the Shadow hand); box on u by projection.
  CBF      discrete-time barrier filter against self-collision: h_p = clearance_p + tol >= 0 (27 link pairs, InspireKin), condition
           h(q_{k+n}) >= (1 - gamma)^n h(q_k) with the command held n steps, q_{k+n}(u) by the model, linearised; closest command (dual
           projected gradient), then the box. Fallback: if the filtered command, re-predicted, still breaks the condition
           (and the barrier is negative), the command becomes "hold the measured servo positions" (the servos are non-backdrivable, so the
           hand stops where it is); the caller then replans globally (RRT) from the measured pose.
  RRT      SHADOW's BatchRRTConnect + PathFollower on the 6-D servo space (deg), edges checked by InspireKin (cspace wrapper below).
"""
from __future__ import annotations

import math
import numpy as np
import torch

from . import servo_model as sm

D2R = math.pi / 180


class ModelStep:
    """f(x, u) in degrees for one fitted theta (1, 6, 7)."""

    def __init__(self, theta, lo, hi):
        self.th = theta; self.lo, self.hi = lo, hi

    def __call__(self, x, u):
        p, v = sm.step(self.th, (x[:, :6] * D2R)[None], (x[:, 6:] * D2R)[None], (u * D2R)[None], self.lo, self.hi)
        return torch.cat([p[0], v[0]], -1) / D2R


def _ref(ref, H):
    return ref[:, None].expand(-1, H, -1) if ref.dim() == 2 else ref


class CEM:
    def __init__(self, f, low, high, H=12, n=64, iters=3, elite=8, std=15.0, smooth=1e-3, seed=0):
        self.f, self.low, self.high, self.H, self.n, self.iters, self.elite, self.std, self.smooth = f, low, high, H, n, iters, elite, std, smooth
        self.g = torch.Generator(device=low.device).manual_seed(seed); self.mean = None

    def _expand(self, knots):   # (..., 4, 6) -> (..., H, 6), knot k at step 3k+2, linear in between, held from the start
        t = torch.arange(self.H, device=knots.device, dtype=torch.float32); pos = torch.tensor([0., 3., 6., 9.], device=knots.device) + 2
        w = torch.clamp(1 - (t[:, None] - pos[None]).abs() / 3, 0); w[:2, 0] = 1; w[pos[-1].long():, -1] = 1; w = w / w.sum(1, keepdim=True)
        return torch.einsum("tk,...kd->...td", w, knots)

    @torch.no_grad()
    def plan(self, x0, up, ref):
        E = x0.shape[0]; R = _ref(ref, self.H)
        mean = (self.mean if self.mean is not None and self.mean.shape[0] == E else R[:, [2, 5, 8, 11]].clone())
        std = torch.full_like(mean, self.std); bestJ = torch.full((E,), float("inf"), device=x0.device); bestK = mean.clone()
        w = torch.full((self.H,), 1.0 / self.H, device=x0.device); w[-1] += 2.0
        Rr = R[:, None].expand(E, self.n, self.H, 6).reshape(-1, self.H, 6)
        for it in range(self.iters):
            K = mean[:, None] + std[:, None] * torch.randn(E, self.n, 4, 6, generator=self.g, device=x0.device)
            K[:, 0] = mean; K[:, 1] = R[:, [2, 5, 8, 11]]; K[:, 2] = bestK
            K = torch.maximum(torch.minimum(K, self.high), self.low); U = self._expand(K)                # (E, n, H, 6)
            x = x0[:, None].expand(E, self.n, 12).reshape(-1, 12); Uf = U.reshape(-1, self.H, 6); J = 0
            for t in range(self.H): x = self.f(x, Uf[:, t]); J = J + w[t] * (x[:, :6] - Rr[:, t]).square().mean(-1)
            dU = torch.diff(torch.cat([up[:, None, None].expand(E, self.n, 1, 6), U], 2), dim=2)
            J = J.view(E, self.n) + self.smooth * dU.square().sum((2, 3))
            el = J.topk(self.elite, dim=1, largest=False).indices
            KE = torch.gather(K, 1, el[..., None, None].expand(-1, -1, 4, 6)); mean = KE.mean(1); std = KE.std(1).clamp_min(0.5)
            jb = J.gather(1, el[:, :1])[:, 0]; imp = jb < bestJ; bestJ = torch.where(imp, jb, bestJ); bestK = torch.where(imp[:, None, None], KE[:, 0], bestK)
        self.mean = torch.cat([bestK[:, 1:], bestK[:, -1:]], 1)                                              # shift for the next solve
        return self._expand(bestK)                                                                           # best sample found (E, H, 6)


class ILQR:
    def __init__(self, f, low, high, H=12, smooth=1e-3, iters_first=8, iters=3, eps=(0.05, 0.5, 0.05)):
        self.f, self.low, self.high, self.H, self.smooth = f, low, high, H, smooth; self.it0, self.it = iters_first, iters; self.eps = eps
        self.U = None; self.mu = None; self.nx, self.nu = 12, 6

    def _forward(self, x0, Ubar, Xbar=None, K=None, k=None, a=0.0):
        X = [x0]; U = []
        for t in range(self.H):
            u = Ubar[:, t]
            if K is not None: u = u + a * k[:, t] + torch.einsum("eij,ej->ei", K[:, t], X[-1] - Xbar[:, t])
            u = torch.maximum(torch.minimum(u, self.high), self.low); X.append(self.f(X[-1], u)); U.append(u)
        return torch.stack(X, 1), torch.stack(U, 1)

    def _w(self, dev):
        w = torch.full((self.H,), 1.0 / self.H, device=dev); w[-1] += 2.0; return w

    def _cost(self, X, U, up0, R):
        e = (X[:, 1:, :6] - R).square().mean(-1); v = torch.diff(torch.cat([up0[:, None], U], 1), dim=1)
        J = (e * self._w(X.device)).sum(1) + self.smooth * v.square().sum((1, 2)); return torch.where(torch.isfinite(J), J, torch.full_like(J, float("inf")))

    def _jac(self, X, U):
        E = X.shape[0]; nx, nu = self.nx, self.nu; eq, ev, eu = self.eps; dev = X.device
        dx = torch.cat([torch.full((6,), eq, device=dev), torch.full((6,), ev, device=dev)]); n = 1 + nx + nu
        x = X[:, :self.H]; u = U                                                                              # (E, H, .)
        xs = x[:, :, None].expand(E, self.H, n, nx).clone(); us = u[:, :, None].expand(E, self.H, n, nu).clone()
        xs[:, :, 1:1 + nx] += torch.diag(dx); us[:, :, 1 + nx:] += torch.eye(nu, device=dev) * eu
        y = self.f(xs.reshape(-1, nx), us.reshape(-1, nu)).view(E, self.H, n, nx)
        fx = ((y[:, :, 1:1 + nx] - y[:, :, :1]) / dx[:, None]).transpose(2, 3); fu = ((y[:, :, 1 + nx:] - y[:, :, :1]) / eu).transpose(2, 3)
        return torch.nan_to_num(fx), torch.nan_to_num(fu)

    def _backward(self, X, U, up0, R, fx, fu, mu):
        E = X.shape[0]; dev = X.device; w = self._w(dev); nx, nu = self.nx, self.nu; I = torch.eye(nu, device=dev)
        Vx = torch.zeros(E, nx, device=dev); Vxx = torch.zeros(E, nx, nx, device=dev)
        K = torch.zeros(E, self.H, nu, nx, device=dev); k = torch.zeros(E, self.H, nu, device=dev)
        for t in reversed(range(self.H)):
            # stage cost of the state reached by u_t: x_{t+1}; folded into V at t+1 (w[t])
            Vx = Vx.clone(); Vxx = Vxx.clone()
            Vx[:, :6] += 2 * w[t] / 6 * (X[:, t + 1, :6] - R[:, t]); Vxx[:, range(6), range(6)] += 2 * w[t] / 6
            A, B = fx[:, t], fu[:, t]; du = U[:, t] - (U[:, t - 1] if t > 0 else up0)
            Qx = torch.einsum("eji,ej->ei", A, Vx); Qu = 2 * self.smooth * du + torch.einsum("eji,ej->ei", B, Vx)
            Qxx = A.transpose(1, 2) @ Vxx @ A; Quu = 2 * self.smooth * I + B.transpose(1, 2) @ Vxx @ B; Qux = B.transpose(1, 2) @ Vxx @ A
            Quu = 0.5 * (Quu + Quu.transpose(1, 2))
            L, info = torch.linalg.cholesky_ex(Quu + mu[:, None, None] * I)                       # Levenberg-Marquardt regularised
            rhs = torch.cat([Qu[..., None], Qux], -1); sol = torch.cholesky_solve(rhs, L)
            sol = torch.where((info == 0)[:, None, None], sol, torch.zeros_like(sol))              # not positive definite: no step
            kt = torch.nan_to_num(-sol[..., 0]); Kt = torch.nan_to_num(-sol[..., 1:]); k[:, t] = kt; K[:, t] = Kt
            Vx = Qx + torch.einsum("eji,ej->ei", Kt, Qu) + torch.einsum("eji,ej->ei", Qux, kt) + torch.einsum("eji,ejk,ek->ei", Kt, Quu, kt)
            Vxx = Qxx + Kt.transpose(1, 2) @ Quu @ Kt + Kt.transpose(1, 2) @ Qux + Qux.transpose(1, 2) @ Kt; Vxx = 0.5 * (Vxx + Vxx.transpose(1, 2))
        return K, k

    @torch.no_grad()
    def plan(self, x0, up0, ref, shift=0):
        E = x0.shape[0]; dev = x0.device; R = _ref(ref, self.H)
        X, U = self._forward(x0, torch.maximum(torch.minimum(R, self.high), self.low)); J = self._cost(X, U, up0, R)   # guess: command the reference
        if self.U is None or self.U.shape[0] != E: n_it = self.it0; self.mu = torch.full((E,), 1e-2, device=dev)
        else:
            n_it = self.it; Uw = torch.cat([self.U[:, shift:], self.U[:, -1:].expand(E, shift, 6)], 1) if shift else self.U.clone()
            Xw, Uw = self._forward(x0, Uw); Jw = self._cost(Xw, Uw, up0, R); b = Jw < J
            X = torch.where(b[:, None, None], Xw, X); U = torch.where(b[:, None, None], Uw, U); J = torch.where(b, Jw, J)
        for _ in range(n_it):
            fx, fu = self._jac(X, U); K, k = self._backward(X, U, up0, R, fx, fu, self.mu)
            al = torch.tensor([1.0, 0.5, 0.25, 0.1], device=dev); na = len(al)                    # line search, all step sizes in one batch
            rep = lambda z: z.repeat_interleave(na, 0)
            Xa, Ua = self._forward(rep(x0), rep(U), rep(X), rep(K), rep(k) * al.repeat(E)[:, None, None], 1.0)
            Ja = self._cost(Xa, Ua, rep(up0), rep(R)).view(E, na); ja, ib = Ja.min(1); sel = torch.arange(E, device=dev) * na + ib
            imp = ja < J; best = (torch.where(imp, ja, J), torch.where(imp[:, None, None], Xa[sel], X), torch.where(imp[:, None, None], Ua[sel], U)); self.mu = torch.where(imp, (self.mu / 3).clamp_min(1e-4), (self.mu * 10).clamp_max(1e4)); J, X, U = best
        fx, fu = self._jac(X, U); K, k = self._backward(X, U, up0, R, fx, fu, self.mu)
        self.U, self.X, self.K, self.J = U, X, K, J
        return X, U, K

    def feedback(self, j, x):
        """TVLQR: command for step j of the current plan, corrected by the measured state."""
        u = self.U[:, j] + torch.einsum("eij,ej->ei", self.K[:, j], x - self.X[:, j])
        u = torch.maximum(torch.minimum(u, self.high), self.low); return torch.where(torch.isfinite(u).all(1, keepdim=True), u, self.U[:, j])


class CBF:
    def __init__(self, kin, f, low, high, n=2, gamma=0.3, d_act=0.010, eps_u=0.1, eps_q=0.05, iters=300):
        self.kin, self.f, self.low, self.high, self.n, self.g, self.d_act, self.eps_u, self.eps_q, self.iters = kin, f, low, high, n, gamma, d_act, eps_u, eps_q, iters
        self.stats = dict(steps=0, active=0, modified=0, dnorm=0.0, fallback=0); self.last_fallback = None

    def clearance(self, qdeg):
        return -self.kin.depth(qdeg * D2R)

    @torch.no_grad()
    def filter(self, x, u_nom, tol):
        E = x.shape[0]; dev = x.device; nu = 6
        h0 = self.clearance(x[:, :6]) + tol[:, None]
        U = torch.cat([u_nom[:, None], u_nom[:, None] + self.eps_u * torch.eye(nu, device=dev)[None]], 1)
        y = x[:, None].expand(E, nu + 1, 12).reshape(-1, 12); Uf = U.reshape(-1, nu)
        for _ in range(self.n): y = self.f(y, Uf)
        Q = y[:, :6].view(E, nu + 1, 6); q1 = Q[:, 0]; S = ((Q[:, 1:] - q1[:, None]) / self.eps_u).transpose(1, 2)        # (E, 6, 6) dq/du
        qs = torch.cat([q1[:, None], q1[:, None] + self.eps_q * torch.eye(6, device=dev)[None]], 1)
        C = self.clearance(qs.reshape(-1, 6)).view(E, 7, -1) + tol[:, None, None]
        h1 = C[:, 0]; Gq = ((C[:, 1:] - h1[:, None]) / self.eps_q).transpose(1, 2)
        G = (Gq @ S) * 1000; b = ((1 - self.g) ** self.n * h0 - h1) * 1000
        active = (h0 < self.d_act) | (h1 < self.d_act); G = G * active[..., None]; b = torch.where(active, b, torch.full_like(b, -1e9))
        need = (b > 0).any(1); d = torch.zeros(E, nu, device=dev)
        if need.any():
            Gn, bn = G[need], b[need].clamp_min(-1e3); lam = torch.zeros_like(bn)
            L = (Gn @ Gn.transpose(1, 2)).diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-9)
            for _ in range(self.iters):
                dd = torch.einsum("epj,ep->ej", Gn, lam); lam = (lam + (bn - torch.einsum("epj,ej->ep", Gn, dd)) / L[:, None]).clamp_min(0)
            d[need] = torch.einsum("epj,ep->ej", Gn, lam)
        u = torch.maximum(torch.minimum(u_nom + d, self.high), self.low)
        # verification (Lecture 16 fallback trigger): re-predict with the filtered command; if an active pair still breaks the barrier
        # condition by > 0.2 mm (the linearised correction was not enough, or the box clipped it), hold the current servo positions.
        y = x
        for _ in range(self.n): y = self.f(y, u)
        hu = self.clearance(y[:, :6]) + tol[:, None]
        bad = (active & (hu < (1 - self.g) ** self.n * h0 - 2e-4) & (hu < 0)).any(1)
        u = torch.where(bad[:, None], x[:, :6], u); self.last_fallback = bad
        s = self.stats; s["steps"] += E; s["active"] += int(active.any(1).sum()); s["modified"] += int(need.sum()); s["dnorm"] += float(d.norm(dim=1).sum()); s["fallback"] += int(bad.sum())
        return u


class CSpaceDeg:
    """SHADOW BatchRRTConnect's configuration-space interface on the Inspire servo space (deg)."""

    def __init__(self, kin, threshold_m=0.0):
        self.kin = kin; self.threshold = threshold_m

    def penetration(self, a_deg):
        return self.kin.penetration(np.asarray(a_deg, np.float32) * D2R)


def make_rrt(kin, low_deg, high_deg, seed=0, step=10.0, res=2.0):
    from shadow_dynamics.rrt_connect import BatchRRTConnect

    class RRT6(BatchRRTConnect):
        def _points(self, segs):
            if not segs: return np.zeros((0, 6)), np.zeros(0, int), np.zeros(0)
            return super()._points(segs)
    return RRT6(CSpaceDeg(kin), low_deg, high_deg, step=step, res=res, exempt=2.0, seed=seed)
