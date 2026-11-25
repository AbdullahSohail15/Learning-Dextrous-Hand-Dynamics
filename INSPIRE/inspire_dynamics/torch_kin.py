"""Batched torch forward kinematics and self-collision geometry for the Inspire hand, on the 6-D servo space.

Configuration a (..., 6) = servo angles (rad, ACTUATED order). The 12 joint angles follow the sim linkage law q_link = k * q_servo.
Collision: each link's collision hull is approximated by a polytope of at most 74 planes (its 48 largest hull facets + the 26 k-DOP
directions; a superset of the hull, so depths are conservative) and at most 160 surface points (hull vertices, farthest-point subsampled).
Depth of pair (A, B) = max over points of A inside B (and of B inside A) of the distance to the nearest plane; > 0 means penetrating.
Used by the configuration space (RRT edge checks), the CBF filter and the exploration safety screen (experiments 02-04).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .urdf_kin import URDFKin, ROOT

ACT = json.load(open(ROOT / "assets/inspire_hand/inspire_hand_right_meta.json"))["actuated"]
THUMB = ["thumb_proximal", "thumb_intermediate", "thumb_distal"]
FINGERS = {f: [f"{f}_proximal", f"{f}_intermediate"] for f in ("index", "middle", "ring", "pinky")}
PAIRS = ([(t, l) for t in THUMB for l in FINGERS["index"] + FINGERS["middle"] + FINGERS["ring"]] +
         [("index_intermediate", "middle_intermediate"), ("middle_intermediate", "ring_intermediate"), ("ring_intermediate", "pinky_intermediate")] +
         [(t, "hand_base_link") for t in ("thumb_intermediate", "thumb_distal")] +
         [(f"{f}_intermediate", "hand_base_link") for f in FINGERS])
TIPS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
CACHE = ROOT / "assets/inspire_hand/collision_polytopes.npz"


def _kdop26():
    d = [np.array(v, float) for v in np.ndindex(3, 3, 3)]; d = [v - 1 for v in d if (v - 1).any()]
    return np.stack([v / np.linalg.norm(v) for v in d])


def build_polytopes(kin: URDFKin, n_planes=48, n_pts=160, seed=0):
    """link -> (normals (P, 3), offsets (P,), points (K, 3)) in the link frame; merged over a link's hulls for points, planes per hull."""
    import trimesh
    rng = np.random.default_rng(seed); out = {}
    for link, hulls in kin.collision_hulls().items():
        H = trimesh.util.concatenate(hulls).convex_hull if len(hulls) > 1 else hulls[0]
        areas = np.array([H.area_faces[f].sum() for f in H.facets]) if len(H.facets) else np.zeros(0)
        normals = [H.facets_normal[i] for i in np.argsort(-areas)[:n_planes]] if len(areas) else []
        N = np.concatenate([np.array(normals).reshape(-1, 3), _kdop26()]); off = (H.vertices @ N.T).max(0)
        V = H.vertices; idx = [int(rng.integers(len(V)))]
        dmin = np.linalg.norm(V - V[idx[0]], axis=1)
        while len(idx) < min(n_pts, len(V)): i = int(dmin.argmax()); idx.append(i); dmin = np.minimum(dmin, np.linalg.norm(V - V[i], axis=1))
        out[link] = (N, off, V[idx])
    return out


class InspireKin:
    """Torch FK + collision depth. a: (..., 6) servo rad."""

    def __init__(self, device="cuda:0", pairs=PAIRS):
        self.dev = device; k = URDFKin(); self.kin = k
        self.links = [k.root] + [j["child"] for j in k.order]; li = {n: i for i, n in enumerate(self.links)}
        self.parent = torch.tensor([li[j["parent"]] for j in k.order]); self.child = [li[j["child"]] for j in k.order]
        self.T0 = torch.tensor(np.stack([j["T"] for j in k.order]), dtype=torch.float32, device=device)
        ax = np.array([np.asarray(j["axis"]) / np.linalg.norm(j["axis"]) for j in k.order]); self.axis = torch.tensor(ax, dtype=torch.float32, device=device)
        self.rev = torch.tensor([j["type"] == "revolute" for j in k.order], device=device)
        # joint angle of every joint in `order` = gain * a[src]  (fixed joints: gain 0)
        src, gain = [], []
        for j in k.order:
            if j["type"] != "revolute": src.append(0); gain.append(0.0)
            elif j["mimic"] is None: src.append(ACT.index(j["name"])); gain.append(1.0)
            else: src.append(ACT.index(j["mimic"][0])); gain.append(j["mimic"][1])
        self.src = torch.tensor(src, device=device); self.gain = torch.tensor(gain, dtype=torch.float32, device=device)
        lim = {j["name"]: (j["lo"], j["hi"]) for j in k.joints}
        self.lo = torch.tensor([lim[n][0] for n in ACT], device=device); self.hi = torch.tensor([lim[n][1] for n in ACT], device=device)
        if CACHE.exists(): z = np.load(CACHE, allow_pickle=True); poly = z["poly"].item()
        else: poly = build_polytopes(k); np.savez(CACHE, poly=np.array(poly, dtype=object))
        self.pairs = [(a, b) for a, b in pairs if a in poly and b in poly]
        self.li = li; self.poly = {n: tuple(torch.tensor(x, dtype=torch.float32, device=device) for x in v) for n, v in poly.items()}
        self.tip_idx = [li[t] for t in TIPS]

    def fk(self, a):
        """a (N, 6) -> (N, L, 4, 4) link transforms in the hand base frame."""
        N = a.shape[0]; q = a[:, self.src] * self.gain                                            # (N, J)
        x, y, z = self.axis.T; c, s = torch.cos(q), torch.sin(q); C = 1 - c
        R = torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s,
                         z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1).view(N, -1, 3, 3)
        R = torch.where(self.rev[None, :, None, None], R, torch.eye(3, device=a.device).expand_as(R))
        X = [None] * len(self.links); X[0] = torch.eye(4, device=a.device).expand(N, 4, 4)
        for j in range(len(self.child)):
            T = X[self.parent[j]] @ self.T0[j]; M = torch.zeros(N, 4, 4, device=a.device); M[:, :3, :3] = R[:, j]; M[:, 3, 3] = 1
            X[self.child[j]] = T @ M
        return torch.stack(X, 1)

    def tips(self, a):
        """(N, 6) -> (N, 5, 3) fingertip positions (m): thumb, index, middle, ring, pinky."""
        return self.fk(a)[:, self.tip_idx, :3, 3]

    def _depth_dir(self, X, A, B):
        NB, oB, _ = self.poly[B]; _, _, PA = self.poly[A]
        TA, TB = X[:, self.li[A]], X[:, self.li[B]]
        P = PA @ TA[:, :3, :3].transpose(1, 2) + TA[:, None, :3, 3]                               # world points of A
        L = (P - TB[:, None, :3, 3]) @ TB[:, :3, :3]                                              # in B's frame
        return (oB[None, None] - L @ NB.T).amin(-1).amax(-1)                                      # max over points of min plane slack

    def depth(self, a, rows=2048):
        """a (N, 6) -> (N, n_pairs) signed depth (m, > 0 penetrating; < 0 a clearance proxy). Differentiable."""
        outs = []
        for i in range(0, a.shape[0], rows):
            X = self.fk(a[i:i + rows])
            outs.append(torch.stack([torch.maximum(self._depth_dir(X, A, B), self._depth_dir(X, B, A)) for A, B in self.pairs], 1))
        return torch.cat(outs, 0)

    @torch.no_grad()
    def penetration(self, a):
        """a (..., 6) torch/numpy -> (...) max penetration (m, >= 0) as numpy."""
        t = torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a, dtype=torch.float32, device=self.dev)
        flat = t.reshape(-1, 6)
        if flat.shape[0] == 0: return np.zeros(t.shape[:-1])
        return self.depth(flat).clamp_min(0).amax(1).reshape(t.shape[:-1]).cpu().numpy()
