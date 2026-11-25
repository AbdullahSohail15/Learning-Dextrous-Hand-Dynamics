"""Visual-geometry interpenetration of Shadow Hand links (torch, GPU or CPU).

The PhysX collision shapes of the stock hand are capsules and boxes that are thinner than the visual links, so fingers
can visibly pass through each other while the simulator reports almost no overlap. This module measures overlap of the
*visual* links: every link is approximated by a k-DOP (its visual mesh bounded along 64 fixed directions, a slightly
conservative convex hull), and the penetration of link A into link B is the deepest of A's surface samples inside B's
k-DOP. Pairs that are kinematically adjacent (parent/child, or joined through a link without visual mesh) are skipped.

Inputs are body poses in any common frame: world-frame positions + wxyz quaternions from the simulator, or 4x4
transforms from forward kinematics (shadow_viz.kinematics). Geometry comes from artifacts/viz_hand_model
(scripts/export_shadow_viz_model.py); `save`/`load` keep a self-contained copy next to each dataset.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

EXCLUDED_LINKS = ("robot0_forearm",)   # the forearm only meets the wrist, which is adjacent


def fibonacci_directions(n=64):
    i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def surface_samples(vertices, faces, n, seed=0):
    """Area-weighted surface samples plus all hull-extreme vertices (deterministic)."""
    rng = np.random.default_rng(seed); tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pick = rng.choice(len(faces), n, p=area / area.sum()); u, v = rng.uniform(size=(2, n)); flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]; t = tri[pick]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


class HandGeometry:
    def __init__(self, links, parents, normals, offsets, points, pairs, palm_mode="hull"):
        self.links = list(links); self.parents = dict(parents); self.normals = normals; self.offsets = offsets
        self.points = points; self.pairs = [tuple(p) for p in pairs]; self.palm_mode = palm_mode
        self._dev = {}

    @classmethod
    def from_viz_model(cls, model_dir, n_dirs=64, n_points=128, exclude=EXCLUDED_LINKS, skip_pairs=()):
        d = Path(model_dir); meta = json.loads((d / "model.json").read_text()); meshes = np.load(d / "meshes.npz")
        links = [k for k in meta["mesh_links"] if k not in exclude]
        parents = {}
        for j in meta["joints"]:
            if j["body0"] is not None: parents[j["body1"].rsplit("/", 1)[-1]] = j["body0"].rsplit("/", 1)[-1]
        dirs = fibonacci_directions(n_dirs); normals, offsets, points = [], [], []
        for k in links:
            v, f = meshes[f"{k}/vertices"].astype(np.float64), meshes[f"{k}/faces"]
            normals.append(dirs); offsets.append((v @ dirs.T).max(0)); points.append(surface_samples(v, f, n_points))
        def ancestors(n):
            out = []
            while n in parents: n = parents[n]; out.append(n)
            return out
        meshed = set(meta["mesh_links"])
        def adjacent(a, b):   # within two joints, or connected through bodies without a visual mesh
            for x, y in ((a, b), (b, a)):
                for depth, anc in enumerate(ancestors(x)):
                    if anc == y: return True
                    if anc in meshed and depth >= 1: break
            return False
        skip = {tuple(sorted(p)) for p in skip_pairs}
        pairs = [(a, b) for i, a in enumerate(links) for b in links[i + 1:] if not adjacent(a, b) and tuple(sorted((a, b))) not in skip]
        return cls(links, parents, np.stack(normals), np.stack(offsets), np.stack(points), pairs)

    def save(self, path):
        np.savez_compressed(path, links=np.array(self.links), normals=self.normals, offsets=self.offsets, points=self.points,
                            pairs=np.array(self.pairs), parents=json.dumps(self.parents))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["links"].tolist(), json.loads(str(z["parents"])), z["normals"], z["offsets"], z["points"], z["pairs"].tolist())

    def _tensors(self, device):
        key = str(device)
        if key not in self._dev:
            li = {k: i for i, k in enumerate(self.links)}
            src = [li[a] for a, b in self.pairs] + [li[b] for a, b in self.pairs]   # directed: points of src inside dst
            dst = [li[b] for a, b in self.pairs] + [li[a] for a, b in self.pairs]
            t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
            self._dev[key] = dict(src=t(src, torch.long), dst=t(dst, torch.long), normals=t(self.normals), offsets=t(self.offsets), points=t(self.points))
        return self._dev[key]

    def penetration(self, R, p, chunk=64):
        """R (N, L, 3, 3), p (N, L, 3) poses of self.links (same order) -> per-pair depth (N, n_pairs) in metres (>= 0)."""
        g = self._tensors(R.device); K = len(self.pairs); out = []
        for s in range(0, len(g["src"]), chunk):
            src, dst = g["src"][s:s + chunk], g["dst"][s:s + chunk]
            Rs, ps, Rd, pd = R[:, src], p[:, src], R[:, dst], p[:, dst]
            P = torch.einsum("nkij,kpj->nkpi", Rs, g["points"][src]) + ps[:, :, None]          # world
            L = torch.einsum("nkji,nkpj->nkpi", Rd, P - pd[:, :, None])                        # in dst frame
            depth = (g["offsets"][dst][None, :, None, :] - torch.einsum("nkpi,kdi->nkpd", L, g["normals"][dst])).amin(-1)
            out.append(depth.amax(-1))
        d = torch.cat(out, 1).clamp_min(0)
        return torch.maximum(d[:, :K], d[:, K:])

    def penetration_from_world(self, body_names, pos, quat_wxyz):
        idx = [list(body_names).index(k) for k in self.links]
        return self.penetration(quat_to_matrix(quat_wxyz[:, idx]), pos[:, idx])

    def penetration_from_transforms(self, body_names, T):
        """T (N, B, 4, 4) torch (e.g. forward kinematics)."""
        idx = [list(body_names).index(k) for k in self.links]
        return self.penetration(T[:, idx, :3, :3], T[:, idx, :3, 3])


class TorchFK:
    """Batched forward kinematics (same tree and conventions as shadow_viz.kinematics.HandModel, verified there against Isaac)."""

    def __init__(self, model_dir, device="cpu"):
        meta = json.loads((Path(model_dir) / "model.json").read_text()); self.body_names = meta["body_names"]; self.joint_names = meta["joint_names"]
        bidx = {b: i for i, b in enumerate(self.body_names)}; jidx = {j: i for i, j in enumerate(self.joint_names)}
        def frame(pos, quat):
            T = np.eye(4); T[:3, :3] = quat_to_matrix(torch.tensor(quat, dtype=torch.float64)).numpy(); T[:3, 3] = pos; return T
        chain = []
        for j in meta["joints"]:
            if j["body0"] is None: continue
            A = frame(j["local_pos0"], j["local_rot0"]); B = np.linalg.inv(frame(j["local_pos1"], j["local_rot1"]))
            chain.append((bidx[j["body0"]], bidx[j["body1"]], A, B, jidx.get(j["name"], -1) if j["type"] == "PhysicsRevoluteJoint" else -1))
        order, done = [], {bidx[meta["root_body"]]}
        while len(order) < len(chain):
            for c in chain:
                if c[0] in done and c[1] not in done: order.append(c); done.add(c[1])
        self.chain = [(p, c, torch.tensor(A, dtype=torch.float32, device=device), torch.tensor(B, dtype=torch.float32, device=device), ji) for p, c, A, B, ji in order]
        self.root = bidx[meta["root_body"]]; self.device = device

    def __call__(self, q_deg):
        """q_deg (N, 24) torch -> (N, B, 4, 4) relative to the root body."""
        q = torch.deg2rad(q_deg.float()); N = len(q)
        T = [None] * len(self.body_names); T[self.root] = torch.eye(4, device=q.device).expand(N, 4, 4)
        for p, c, A, B, ji in self.chain:
            M = T[p] @ A
            if ji >= 0:
                a = q[:, ji]; R = torch.zeros(N, 4, 4, device=q.device); R[:, 0, 0] = 1; R[:, 3, 3] = 1
                R[:, 1, 1] = torch.cos(a); R[:, 1, 2] = -torch.sin(a); R[:, 2, 1] = torch.sin(a); R[:, 2, 2] = torch.cos(a)
                M = M @ R
            T[c] = M @ B
        return torch.stack(T, 1)


def quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))
