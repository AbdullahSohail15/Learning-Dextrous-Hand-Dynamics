"""Experiment 50: batched iLQR (Li & Todorov 2004) on the identified hand model, run as receding-horizon MPC with TVLQR feedback.

Dynamics: x_{t+1} = f(x_t, u_t), x = (q, qd) in deg / deg/s (48), u = 20 joint targets (deg), f = one 20 Hz step of the model.
Augmented state z = (x, u_prev) (68), control v = u - u_prev (20) with |v| <= slew (the 80 deg/s limit) and u inside the joint box,
both enforced by projection in the forward pass. Cost (the gesture CEM's): sum_{t=1..H} w_t ||q_t - r||^2 / 24 + 1e-3 sum ||v_t||^2,
w_t = 1/H, plus 2 at t = H. Jacobians of f by batched forward differences. Levenberg-Marquardt regularisation and a per-task line search.
"""
from __future__ import annotations
import torch


def _ref_traj(ref, H):
    """ref (E, 24) fixed target or (E, H, 24) time-varying reference (experiment 51) -> (E, H, 24)."""
    return ref[:, None].expand(-1, H, -1) if ref.dim() == 2 else ref


class ILQR:
    def __init__(self, step_fn, low, high, act, slew=4.0, H=12, smooth=1e-3, iters_first=8, iters=3, eps_q=0.05, eps_v=0.5, eps_u=0.05, fb_scale=1.0):
        self.f = step_fn; self.act = act; self.low, self.high = low, high; self.slew = slew; self.H = H; self.smooth = smooth; self.fb_scale = fb_scale
        self.it0, self.it = iters_first, iters; self.eps = (eps_q, eps_v, eps_u); self.U = None; self.mu = None

    # ---------- dynamics and cost
    def _forward(self, x0, up0, Ubar, Xbar, K, k, alpha):
        """Roll out with feedback; Xbar (E, H+1, 48), Ubar (E, H, 20) nominal; returns X, U."""
        X = [x0]; U = []; up = up0
        for t in range(self.H):
            dz = torch.cat([X[-1] - Xbar[:, t], up - (Ubar[:, t - 1] if t > 0 else up0)], -1) if K is not None else None
            v = Ubar[:, t] - (Ubar[:, t - 1] if t > 0 else up0)
            if K is not None: v = v + alpha * k[:, t] + torch.einsum("eij,ej->ei", K[:, t], dz)
            v = v.clamp(-self.slew, self.slew); u = (up + v).clamp(self.low, self.high)
            X.append(self.f(X[-1], u)); U.append(u); up = u
        return torch.stack(X, 1), torch.stack(U, 1)

    def _cost(self, X, U, up0, ref):
        w = torch.full((self.H,), 1.0 / self.H, device=X.device); w[-1] += 2.0
        e = (X[:, 1:, :24] - _ref_traj(ref, self.H)).square().mean(-1)                                         # (E, H)
        v = torch.diff(torch.cat([up0[:, None], U], 1), dim=1)
        J = (e * w).sum(1) + self.smooth * v.square().sum((1, 2))
        return torch.where(torch.isfinite(J), J, torch.full_like(J, float("inf")))   # a diverged model rollout is never selected

    def _jacobians(self, X, U):
        """Forward differences: fx (E, H, 48, 48), fu (E, H, 48, 20) at the nominal (X[:, t], U[:, t])."""
        E = X.shape[0]; eq, ev, eu = self.eps; dev = X.device
        dx = torch.cat([torch.full((24,), eq, device=dev), torch.full((24,), ev, device=dev)]); fx, fu = [], []
        for t in range(self.H):
            x, u = X[:, t], U[:, t]
            xs = torch.cat([x[:, None].expand(E, 49, 48).clone(), x[:, None].expand(E, 20, 48)], 1)
            us = torch.cat([u[:, None].expand(E, 49, 20), u[:, None].expand(E, 20, 20).clone()], 1)
            xs[:, 1:49] += torch.diag(dx)[None]; us[:, 49:] += torch.eye(20, device=dev)[None] * eu
            y = self.f(xs.reshape(-1, 48), us.reshape(-1, 20)).view(E, 69, 48)
            fx.append(((y[:, 1:49] - y[:, :1]) / dx[None, :, None]).transpose(1, 2)); fu.append(((y[:, 49:] - y[:, :1]) / eu).transpose(1, 2))
        return torch.nan_to_num(torch.stack(fx, 1), 0.0, 0.0, 0.0), torch.nan_to_num(torch.stack(fu, 1), 0.0, 0.0, 0.0)

    def _backward(self, X, U, up0, ref, fx, fu, mu):
        E = X.shape[0]; dev = X.device; nz = 68; R = _ref_traj(ref, self.H)
        w = torch.full((self.H + 1,), 1.0 / self.H, device=dev); w[-1] += 2.0
        Vx = torch.zeros(E, nz, device=dev); Vxx = torch.zeros(E, nz, nz, device=dev)
        Vx[:, :24] = 2 * w[-1] / 24 * (X[:, -1, :24] - R[:, -1]); Vxx[:, range(24), range(24)] = 2 * w[-1] / 24
        K = torch.zeros(E, self.H, 20, nz, device=dev); k = torch.zeros(E, self.H, 20, device=dev); ok = torch.ones(E, dtype=torch.bool, device=dev)
        I20 = torch.eye(20, device=dev)
        for t in reversed(range(self.H)):
            A = torch.zeros(E, nz, nz, device=dev); A[:, :48, :48] = fx[:, t]; A[:, :48, 48:] = fu[:, t]; A[:, 48:, 48:] = I20
            B = torch.cat([fu[:, t], I20.expand(E, 20, 20)], 1)                                          # (E, 68, 20)
            v = U[:, t] - (U[:, t - 1] if t > 0 else up0)
            lz = torch.zeros(E, nz, device=dev); lzz = torch.zeros(E, nz, nz, device=dev)
            if t > 0:
                lz[:, :24] = 2 * w[t - 1] / 24 * (X[:, t, :24] - R[:, t - 1]); lzz[:, range(24), range(24)] = 2 * w[t - 1] / 24
            Qz = lz + torch.einsum("eji,ej->ei", A, Vx); Qv = 2 * self.smooth * v + torch.einsum("eji,ej->ei", B, Vx)
            VA = Vxx @ A; VB = Vxx @ B
            Qzz = lzz + A.transpose(1, 2) @ VA; Qvv = 2 * self.smooth * I20 + B.transpose(1, 2) @ VB; Qvz = B.transpose(1, 2) @ VA
            Qvv = 0.5 * (Qvv + Qvv.transpose(1, 2)); ev, V = torch.linalg.eigh(Qvv); ok &= ev.amin(1) > 0
            Qinv = V @ torch.diag_embed(1.0 / (ev.clamp_min(0) + mu[:, None])) @ V.transpose(1, 2)   # eigenvalue-clamped, regularised inverse
            kt = -(Qinv @ Qv[..., None])[..., 0]; Kt = -Qinv @ Qvz
            kt = torch.nan_to_num(kt, 0.0, 0.0, 0.0); Kt = torch.nan_to_num(Kt, 0.0, 0.0, 0.0); k[:, t] = kt; K[:, t] = Kt
            Vx = Qz + torch.einsum("eji,ej->ei", Kt, Qv) + torch.einsum("eji,ej->ei", Qvz, kt) + torch.einsum("eji,ejk,ek->ei", Kt, Qvv, kt)
            Vxx = Qzz + Kt.transpose(1, 2) @ Qvv @ Kt + Kt.transpose(1, 2) @ Qvz + Qvz.transpose(1, 2) @ Kt; Vxx = 0.5 * (Vxx + Vxx.transpose(1, 2))
        return K, k, ok

    # ---------- MPC interface
    @torch.no_grad()
    def plan(self, x0, up0, ref, shift=0, U_init=None):
        """x0 (E, 48) measured state, up0 (E, 20) last applied command, ref (E, 24) or (E, H, 24). Returns nominal X (E, H+1, 48), U (E, H, 20), K."""
        E = x0.shape[0]; dev = x0.device
        Us = []; p = up0.clone()   # heuristic initial guess: slew straight toward the reference's commanded joints
        R = _ref_traj(ref, self.H)
        for t in range(self.H): p = (p + (R[:, t][:, self.act] - p).clamp(-self.slew, self.slew)).clamp(self.low, self.high); Us.append(p)
        X, U = self._forward(x0, up0, torch.stack(Us, 1), None, None, None, 0.0); J = self._cost(X, U, up0, ref)
        if self.U is None or self.U.shape[0] != E:
            n_it = self.it0; self.mu = torch.full((E,), 1e-2, device=dev)
        else:   # warm start: the previous plan shifted, where it is cheaper than the fresh guess
            Uw = torch.cat([self.U[:, shift:], self.U[:, -1:].expand(E, shift, 20)], 1) if shift else self.U.clone(); n_it = self.it
            Xw, Uw = self._forward(x0, up0, Uw, None, None, None, 0.0); Jw = self._cost(Xw, Uw, up0, ref); b = Jw < J
            X = torch.where(b[:, None, None], Xw, X); U = torch.where(b[:, None, None], Uw, U); J = torch.where(b, Jw, J)
        if U_init is not None:   # experiment 51: a supplied command sequence (the cube planner's choice) competes as the initial guess
            Xi, Ui = self._forward(x0, up0, U_init, None, None, None, 0.0); Ji = self._cost(Xi, Ui, up0, ref); b = Ji < J
            X = torch.where(b[:, None, None], Xi, X); U = torch.where(b[:, None, None], Ui, U); J = torch.where(b, Ji, J)
        K = None
        for _ in range(n_it):
            fx, fu = self._jacobians(X, U); K, k, ok = self._backward(X, U, up0, ref, fx, fu, self.mu)
            best = (J.clone(), X, U)
            for a in (1.0, 0.5, 0.25, 0.1):
                Xa, Ua = self._forward(x0, up0, U, X, K, k, a); Ja = self._cost(Xa, Ua, up0, ref); imp = Ja < best[0]
                best = (torch.where(imp, Ja, best[0]), torch.where(imp[:, None, None], Xa, best[1]), torch.where(imp[:, None, None], Ua, best[2]))
            imp = best[0] < J; self.mu = torch.where(imp, (self.mu / 3).clamp_min(1e-4), (self.mu * 10).clamp_max(1e4))
            J, X, U = best
        fx, fu = self._jacobians(X, U); K, k, ok = self._backward(X, U, up0, ref, fx, fu, self.mu)   # feedback gains about the final nominal
        self.U = U; self.X = X; self.K = K; self.J = J
        return X, U, K

    @torch.no_grad()
    def gains(self, x0, up0, U, Xref):
        """Experiment 54: TVLQR only. Feedback gains about a supplied nominal (no optimisation): U (E, H, 20) the cube planner's commands,
        Xref (E, H, 48) the predicted states they lead to (the soft hybrid's contact-aware prediction). feedback() then tracks Xref."""
        E = x0.shape[0]; X = torch.cat([x0[:, None], Xref], 1)
        if self.mu is None or self.mu.shape[0] != E: self.mu = torch.full((E,), 1e-2, device=x0.device)
        fx, fu = self._jacobians(X, U); K, k, ok = self._backward(X, U, up0, Xref[..., :24], fx, fu, self.mu)
        self.U = U; self.X = X; self.K = K
        return K

    def feedback(self, j, x, up, up0_plan):
        """Command for step j of the current plan with TVLQR feedback on the measured state x and the actual previous command up."""
        dz = torch.cat([x - self.X[:, j], up - (self.U[:, j - 1] if j > 0 else up0_plan)], -1)
        v = self.U[:, j] - (self.U[:, j - 1] if j > 0 else up0_plan) + self.fb_scale * torch.einsum("eij,ej->ei", self.K[:, j], dz)   # fb_scale 0 = open-loop plan between replans (exp 51)
        u = (up + v.clamp(-self.slew, self.slew)).clamp(self.low, self.high)
        fb = torch.isfinite(u).all(1)                      # non-finite feedback (diverged nominal): fall back to the open-loop plan, then to holding
        u = torch.where(fb[:, None], u, self.U[:, j]); u = torch.where(torch.isfinite(u).all(1)[:, None], u, up)
        self.n_fallback = getattr(self, "n_fallback", 0) + int((~fb).sum()); return u
