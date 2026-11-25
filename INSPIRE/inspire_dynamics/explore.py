"""Experiment 02: self-exploration programs in the 6-D servo space.

A chunk = 3 s (60 steps at 20 Hz) of piecewise-constant servo targets. Per servo, holds of 0.15-0.8 s; each new level is either a large
jump (uniform over the range: excites the speed limit and torque cap) or a small step of 1-10 deg (excites the position loop K, D, m).
The mix differs per candidate so the designer has a choice. Three ways to build a program:
  R   random chunks, no screening (open loop);
  D0  chunks chosen once, up front, by D-optimal score under the spec model N0, safety-screened under N0 (open loop);
  CL  closed loop: each round designs its chunks with the model fitted to all data so far (D-optimal + safety screen with that model),
      runs them on the hand, refits, and designs the next round from the new fit.
Safety screen: the predicted servo trajectory, mapped through the linkage, must keep every checked link pair >= `margin` apart.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from . import servo_model as sm

CHUNK = 60


def gen_chunks(rng: np.random.Generator, lo, hi, n, start=None, T=CHUNK):
    """n candidate chunks (n, T, 6) as numpy (rad), continuing from `start` (6,) target levels."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float); out = np.zeros((n, T, 6))
    for c in range(n):
        p_big = rng.uniform(0.15, 0.85); hold_max = rng.integers(6, 17)
        for i in range(6):
            lev = rng.uniform(lo[i], hi[i]) if start is None else float(start[i]); t = 0
            while t < T:
                d = int(rng.integers(3, hold_max + 1))
                if rng.random() < p_big: lev = rng.uniform(lo[i], hi[i])
                else: lev = float(np.clip(lev + rng.choice([-1, 1]) * np.radians(rng.uniform(1, 10)), lo[i], hi[i]))
                out[c, t:t + d, i] = lev; t += d
    return out


@torch.no_grad()
def screen(kin, th, p0, v0, A, lo, hi, margin=0.002):
    """Predicted min clearance (m) over each candidate (C,) under model th (1, 6, 7); True where >= margin."""
    P, _ = sm.rollout(th, p0[None], v0[None], A[None], lo, hi); P = P[0]
    d = kin.depth(P.reshape(-1, 6)).amax(1).view(P.shape[0], -1).amax(1)
    return -d >= margin, -d


def design_round(kin, th, p0, v0, F0, n_chunks, rng, lo, hi, n_cand=48, margin=0.002):
    """Greedy D-optimal chunks for one hand from state (p0, v0) (6,), accumulated information F0 (6, 7, 7). Each chosen chunk is
    appended to the program; its predicted information is added to F0 and its predicted end state starts the next choice.
    Returns targets (n_chunks*CHUNK, 6) numpy, stats."""
    dev = th.device; lot, hit = lo, hi; prog = []; scores = []; rejected = 0
    start = p0.cpu().numpy(); p, v = p0.clone(), v0.clone(); F = F0.clone()
    for k in range(n_chunks):
        A = torch.tensor(gen_chunks(rng, lot.cpu().numpy(), hit.cpu().numpy(), n_cand, start), dtype=torch.float32, device=dev)
        ok, clr = screen(kin, th, p.expand(n_cand, 6), v.expand(n_cand, 6), A, lo, hi, margin)
        rejected += int((~ok).sum())
        if not ok.any(): ok = clr >= clr.max()                                    # nothing safe: the safest candidate
        s, Fc = sm.info_gain(th, p.expand(n_cand, 6), v.expand(n_cand, 6), A, lo, hi, F)
        s = torch.where(ok, s, torch.full_like(s, -1e30)); b = int(s.argmax())
        prog.append(A[b]); F = Fc[b]; scores.append(float(s[b])); start = A[b, -1].cpu().numpy()
        P, vn = sm.rollout(th, p[None, None], v[None, None], A[b][None, None], lo, hi); p, v = P[0, 0, -1], vn[0, 0]
    return torch.cat(prog).cpu().numpy(), dict(score=scores, rejected=rejected, candidates=n_cand * n_chunks)
