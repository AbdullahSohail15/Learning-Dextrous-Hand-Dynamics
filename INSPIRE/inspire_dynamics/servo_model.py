"""Experiment 02: the servo-linkage model of the Inspire hand (6 generalised coordinates), batched over M models, differentiable.

State per servo i: p_i (rad, the angle of the joint it drives), v_i = dp_i/dt. The 12 joint angles follow q = c(p) (linkage, torch_kin).
One 120 Hz substep h (implicit position loop, like the PhysX drive it stands for), per servo, independent of the others:
    v* = (m v + h (K (a - p) - g(p))) / (m + h D + h^2 K)                        implicit spring-damper toward the target a
    tau = m (v* - v) / h + g(p);  if |tau| > u_max:  v* = v + h (sign(tau) u_max - g(p)) / m      force threshold (torque cap)
    v' = clip(v*, -v_max, v_max);  p' = clip(p + h v', lo, hi)                       servo speed limit, joint range
    g(p) = g1 cos p + g2 sin p      gravity seen at the servo (Y(p) pi, linear in pi; units 1e-3 N m)
m = J_c^T M J_c (diagonal) + armature: link inertia seen through the linkage + reflected rotor inertia, held constant (gate 0 shows it
varies < 10% over the range). Six substeps per 20 Hz command. Parameters theta (M, 6, 7) = log K, log D, log m, log v_max, log u_max, g1, g2.
"""
from __future__ import annotations

import math
import torch

SUB, DT = 6, 1 / 20
H = DT / SUB
NAMES = ["logK", "logD", "logm", "logvmax", "logumax", "g1", "g2"]


def pack(K, D, m, vmax, umax, g1=None, g2=None):
    """(M, 6) physical values -> theta (M, 6, 7)."""
    z = torch.zeros_like(K)
    return torch.stack([K.log(), D.log(), m.log(), vmax.log(), umax.log(), z if g1 is None else g1, z if g2 is None else g2], -1)


def unpack(th):
    return dict(K=th[..., 0].exp(), D=th[..., 1].exp(), m=th[..., 2].exp(), vmax=th[..., 3].exp(), umax=th[..., 4].exp(),
                g1=th[..., 5] * 1e-3, g2=th[..., 6] * 1e-3)


def substep(th, p, v, a, lo, hi):
    """th (M, 6, 7); p, v, a (M, B, 6). Returns p', v'."""
    P = {k: x[:, None] for k, x in unpack(th).items()}
    g = P["g1"] * torch.cos(p) + P["g2"] * torch.sin(p)
    vs = (P["m"] * v + H * (P["K"] * (a - p) - g)) / (P["m"] + H * P["D"] + H * H * P["K"])
    tau = P["m"] * (vs - v) / H + g
    vc = v + H * (torch.sign(tau) * P["umax"] - g) / P["m"]
    vs = torch.where(tau.abs() > P["umax"], vc, vs)
    vs = torch.maximum(torch.minimum(vs, P["vmax"]), -P["vmax"])
    pn = p + H * vs; pc = torch.maximum(torch.minimum(pn, hi), lo)
    return pc, torch.where(pc != pn, torch.zeros_like(vs), vs)


def step(th, p, v, a, lo, hi):
    """One 20 Hz control step (6 substeps) with the target a held."""
    a = torch.maximum(torch.minimum(a, hi), lo)
    for _ in range(SUB): p, v = substep(th, p, v, a, lo, hi)
    return p, v


def rollout(th, p0, v0, A, lo, hi):
    """p0, v0 (M, B, 6); A (M, B, T, 6) targets -> P (M, B, T, 6) positions after each step, and the final v."""
    P = []; p, v = p0, v0
    for t in range(A.shape[2]): p, v = step(th, p, v, A[:, :, t], lo, hi); P.append(p)
    return torch.stack(P, 2), v


def windows(p, a, valid, W=12, stride=1):
    """One trajectory p (T, 6) measured at 20 Hz, a (T, 6) target applied over step t -> t+1, valid (T,) bool (no contact).
    Returns start index list s such that p[s-1..s+W] are all valid (v0 by backward difference needs p[s-1])."""
    T = p.shape[0]; ok = valid.clone()
    s = [k for k in range(1, T - W, stride) if bool(ok[k - 1:k + W + 1].all())]
    return torch.tensor(s, dtype=torch.long)


def gather(p, a, s, W=12):
    """-> p0 (B, 6), v0 (B, 6), A (B, W, 6), Y (B, W, 6) for window starts s."""
    idx = s[:, None] + torch.arange(W)[None]
    return p[s], (p[s] - p[s - 1]) / DT, a[idx], p[idx + 1]


def fit(th0, data, lo, hi, iters=400, lr=0.03, batch=1024, delta_deg=0.5, fix=(), seed=0, prior=None, prior_sigma=None, prior_weight=0.0):
    """Fit M models jointly (each on its own data). data: list of M tuples (p0, v0, A, Y). Huber loss on positions (deg), 0.6 s rollouts.
    fix: names of parameters held at th0. prior (M, 6, 7), prior_sigma (7,): MAP fit, adds prior_weight * mean(((th - prior)/sigma)^2)
    (log-space for K, D, m, v_max, u_max) so parameters the data cannot see stay at the prior instead of drifting. Returns fitted theta (M, 6, 7) and the final loss per model."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    th = th0.clone().requires_grad_(True); opt = torch.optim.Adam([th], lr=lr)
    mask = torch.ones(7, device=th0.device)
    for n in fix: mask[NAMES.index(n)] = 0
    nB = [d[0].shape[0] for d in data]; B = min(batch, max(nB))
    for it in range(iters):
        sel = [torch.randint(n, (B,), generator=g) for n in nB]
        p0 = torch.stack([d[0][s] for d, s in zip(data, sel)]); v0 = torch.stack([d[1][s] for d, s in zip(data, sel)])
        A = torch.stack([d[2][s] for d, s in zip(data, sel)]); Y = torch.stack([d[3][s] for d, s in zip(data, sel)])
        P, _ = rollout(th, p0, v0, A, lo, hi)
        e = (P - Y) * (180 / math.pi)
        loss_m = torch.nn.functional.huber_loss(e, torch.zeros_like(e), delta=delta_deg, reduction="none").mean((1, 2, 3))
        tot = loss_m.sum()
        if prior is not None and prior_weight > 0: tot = tot + prior_weight * (((th - prior) / prior_sigma) ** 2).mean((1, 2)).sum()
        opt.zero_grad(); tot.backward(); th.grad *= mask; opt.step()
        with torch.no_grad(): th[..., 2].clamp_(math.log(1e-6), math.log(1e-2)); th[..., 0].clamp_(math.log(0.3), math.log(300))
    return th.detach(), loss_m.detach()


@torch.no_grad()
def window_error(th, data, lo, hi, chunk=4096):
    """Open-loop 0.6 s window error (deg): per model the median over windows of the mean |error| over the window and servos,
    and the per-servo median of the final-step error. data: list of M tuples (p0, v0, A, Y) (same length not required)."""
    med, per = [], []
    for m, d in enumerate(data):
        E = []
        for i in range(0, d[0].shape[0], chunk):
            P, _ = rollout(th[m:m + 1], d[0][None, i:i + chunk], d[1][None, i:i + chunk], d[2][None, i:i + chunk], lo, hi)
            E.append(((P[0] - d[3][i:i + chunk]).abs() * 180 / math.pi))
        E = torch.cat(E); med.append(float(E.mean((1, 2)).median())); per.append(E[:, -1].median(0).values.tolist())
    return med, per


def info_gain(th, p0, v0, A, lo, hi, F0, stride=3, W=12):
    """D-optimal score of candidate target programs for ONE model. th (1, 6, 7); p0, v0 (C, 6) start state; A (C, T, 6) candidates.
    Predicts each candidate open loop, takes 0.6 s windows every `stride` steps from the predicted states (as the fit sees data), and
    returns sum_i log det(F0_i + J_i^T J_i) per candidate (C,), J_i = d(window positions of servo i, deg)/d(theta_i). F0 (6, 7, 7)."""
    C, T, _ = A.shape
    with torch.no_grad():
        P, _ = rollout(th, p0[None], v0[None], A[None], lo, hi); P = P[0]                         # (C, T, 6)
        Pf = torch.cat([p0[:, None], P], 1)
    starts = torch.arange(1, T - W, stride)
    ps = Pf[:, starts]; vs = (Pf[:, starts] - Pf[:, starts - 1]) / DT                              # (C, S, 6)
    idx = starts[:, None] + torch.arange(W)[None]; Aw = A[:, idx]                                  # (C, S, W, 6)
    S = len(starts)

    def f(thf):
        Pw, _ = rollout(thf.view(1, 6, 7), ps.reshape(1, C * S, 6), vs.reshape(1, C * S, 6), Aw.reshape(1, C * S, W, 6), lo, hi)
        return Pw[0] * (180 / math.pi)                                                             # (C*S, W, 6)
    J = torch.func.jacfwd(f)(th.reshape(-1)).view(C, S * W, 6, 6, 7)                                # d out[..., i] / d theta[j, k]
    J = torch.stack([J[:, :, i, i] for i in range(6)], 1)                                           # (C, 6, S*W, 7) own-servo block
    F = F0[None] + J.transpose(-1, -2) @ J
    return torch.linalg.slogdet(F + 1e-6 * torch.eye(7, device=F.device))[1].sum(1), F


def window_info(th, p0, v0, A, lo, hi):
    """Fisher information (6, 7, 7) of measured windows (B, ...) for one model (used to accumulate F0 from data)."""
    B, W = A.shape[:2]

    def f(thf):
        Pw, _ = rollout(thf.view(1, 6, 7), p0[None], v0[None], A[None], lo, hi)
        return Pw[0] * (180 / math.pi)
    F = torch.zeros(6, 7, 7, device=A.device)
    for i in range(0, B, 512):
        sl = slice(i, i + 512)
        def g(thf, sl=sl):
            Pw, _ = rollout(thf.view(1, 6, 7), p0[None, sl], v0[None, sl], A[None, sl], lo, hi)
            return Pw[0] * (180 / math.pi)
        J = torch.func.jacfwd(g)(th.reshape(-1)).view(-1, W, 6, 6, 7)
        J = torch.stack([J[:, :, i_, i_].reshape(-1, 7) for i_ in range(6)], 0)                     # (6, n, 7)
        F += J.transpose(-1, -2) @ J
    return F
