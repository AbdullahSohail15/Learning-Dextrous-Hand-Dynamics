"""Experiment 50: the Shadow Hand's configuration space for planning.

Configuration = the 20 commanded joint angles (deg, action order); the 4 passive fingertip joints follow J0 = clip(1.143 J1, 0, 90) deg (the
rule of the gesture benchmark). Obstacles = self-collision of the link meshes (k-DOP hulls of HandGeometry: points of one link inside the
other's hull). `penetration` gives the largest depth (m, >= 0) per configuration; `clearance` gives, per link pair, the signed k-DOP depth
with its sign flipped (m; > 0 apart, < 0 penetrating), differentiable through the forward kinematics, for the safety filter.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from shadow_dynamics.hand_geometry import HandGeometry, TorchFK

J0_GAIN = 1.143


class HandCSpace:
    def __init__(self, model_dir="artifacts/viz_hand_model", device="cuda:0", threshold_m=1e-3):
        self.dev = device; self.fk = TorchFK(model_dir, device); self.geo = HandGeometry.from_viz_model(model_dir); self.threshold = threshold_m
        names = json.loads((Path(model_dir) / "model.json").read_text())["joint_names"]; self.names = names
        self.act = [i for i, n in enumerate(names) if not any(n == f"robot0_{f}J0" for f in ("FF", "MF", "RF", "LF"))]
        self.pas = [names.index(f"robot0_{f}J0") for f in ("FF", "MF", "RF", "LF")]; self.mid = [names.index(f"robot0_{f}J1") for f in ("FF", "MF", "RF", "LF")]
        self.lidx = [list(self.fk.body_names).index(k) for k in self.geo.links]; self.pairs = list(self.geo.pairs)

    def full(self, a):
        """a (..., 20) deg torch -> q (..., 24) deg."""
        q = torch.zeros(a.shape[:-1] + (24,), device=a.device, dtype=a.dtype); q[..., self.act] = a
        q[..., self.pas] = (J0_GAIN * q[..., self.mid]).clamp(0, 90); return q

    def _signed_depth(self, q24, rows=256):
        """Row-chunked _signed_depth_rows (exp 51: 3000-4096-row calls ran out of memory on a shared GPU; values are identical)."""
        return torch.cat([self._signed_depth_rows(q24[i:i + rows]) for i in range(0, len(q24), rows)], 0) if len(q24) > rows else self._signed_depth_rows(q24)

    def _signed_depth_rows(self, q24, chunk=64):
        """q24 (N, 24) deg -> (N, n_pairs) signed depth in m (> 0 penetrating), max over both directions."""
        T = self.fk(q24)[:, self.lidx]; R, p = T[..., :3, :3], T[..., :3, 3]
        g = self.geo._tensors(q24.device); K = len(self.pairs); out = []
        for s in range(0, len(g["src"]), chunk):
            src, dst = g["src"][s:s + chunk], g["dst"][s:s + chunk]
            P = torch.einsum("nkij,kpj->nkpi", R[:, src], g["points"][src]) + p[:, src][:, :, None]
            L = torch.einsum("nkji,nkpj->nkpi", R[:, dst], P - p[:, dst][:, :, None])
            depth = (g["offsets"][dst][None, :, None, :] - torch.einsum("nkpi,kdi->nkpd", L, g["normals"][dst])).amin(-1)
            out.append(depth.amax(-1))
        d = torch.cat(out, 1); return torch.maximum(d[:, :K], d[:, K:])

    def clearance(self, q24):
        """(N, n_pairs) m, differentiable; > 0 means apart (a lower bound proxy from the most separating hull face)."""
        return -self._signed_depth(q24)

    @torch.no_grad()
    def penetration(self, a, batch=4096):
        """a (..., 20) deg (torch or numpy) -> (...) max penetration (m, >= 0) as numpy."""
        t = torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a, dtype=torch.float32, device=self.dev); flat = t.reshape(-1, 20); out = []
        for i in range(0, len(flat), batch): out.append(self._signed_depth(self.full(flat[i:i + batch])).clamp_min(0).amax(1))
        r = torch.cat(out) if out else torch.zeros(0, device=self.dev); return r.reshape(t.shape[:-1]).cpu().numpy()

    def valid(self, a):
        return self.penetration(a) <= self.threshold
