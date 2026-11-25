"""The allowed prior of experiment 49: the Shadow Hand's kinematics only (joint tree, joint frames and axes, joint limits, link meshes for the
self-collision check). Built from `artifacts/viz_hand_model/model.json`, which holds no mass, centre of mass, inertia, armature, drive gain,
friction or tendon property; `load_kinematics` refuses any file that does.

Joint convention (as in physics_prior.NominalHand): child body frame T_c = T_parent @ A_j @ Rx(q_j) @ inv(B_j); the joint axis is the x axis
of T_parent @ A_j, its origin that frame's origin. Units: m, rad (the json limits are deg).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

FORBIDDEN = ("mass", "inertia", "com", "armature", "stiffness", "damping", "friction", "max_force", "tendon")


def _quat(q):   # wxyz -> 3x3
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_kinematics(path="artifacts/viz_hand_model/model.json"):
    """Plain-data kinematics; asserts that no dynamic quantity is present at any level of the file."""
    m = json.loads(Path(path).read_text())
    def keys(x):
        if isinstance(x, dict):
            for k, v in x.items(): yield k; yield from keys(v)
        elif isinstance(x, list):
            for v in x: yield from keys(v)
    leaked = sorted({k for k in keys(m) if any(f in k.lower() for f in FORBIDDEN)})
    assert not leaked, f"kinematics file carries dynamic fields: {leaked}"
    return dict(joint_names=m["joint_names"], body_names=m["body_names"], root_body=m["root_body"], joints=m["joints"],
                lower_deg=m["joint_lower_deg"], upper_deg=m["joint_upper_deg"], active=m["active_indices"], passive=m["passive_indices"], source=str(path))


class Kinematics(torch.nn.Module):
    """Batched forward kinematics: q (N, nj) rad -> body frames (N, B, 4, 4), joint axes and origins (N, nj, 3), in the root frame."""

    def __init__(self, kin, device="cpu"):
        super().__init__()
        self.joint_names, self.body_names = kin["joint_names"], kin["body_names"]; nb, nj = len(self.body_names), len(self.joint_names)
        bidx = {b: i for i, b in enumerate(self.body_names)}; jidx = {j: i for i, j in enumerate(self.joint_names)}
        def frame(pos, quat):
            T = np.eye(4); T[:3, :3] = _quat(quat); T[:3, 3] = pos; return T
        chain = []
        for j in kin["joints"]:
            if j["body0"] is None: continue
            A = frame(j["local_pos0"], j["local_rot0"]); B = np.linalg.inv(frame(j["local_pos1"], j["local_rot1"]))
            chain.append((bidx[j["body0"]], bidx[j["body1"]], A, B, jidx.get(j["name"], -1) if j["type"] == "PhysicsRevoluteJoint" else -1))
        order, done = [], {bidx[kin["root_body"]]}
        while len(order) < len(chain):
            for c in chain:
                if c[0] in done and c[1] not in done: order.append(c); done.add(c[1])
        self.order = [(pa, c, ji) for pa, c, _, _, ji in order]; self.root = bidx[kin["root_body"]]
        self.register_buffer("A", torch.tensor(np.stack([o[2] for o in order]), dtype=torch.float32, device=device))
        self.register_buffer("B", torch.tensor(np.stack([o[3] for o in order]), dtype=torch.float32, device=device))
        children = {i: [] for i in range(nb)}
        for pa, c, _ in self.order: children[pa].append(c)
        def subtree(b): return [b] + [d for c in children[b] for d in subtree(c)]
        sub = torch.zeros(nj, nb, device=device)   # sub[j, b] = 1 if body b moves with joint j
        for pa, c, ji in self.order:
            if ji >= 0: sub[ji, subtree(c)] = 1
        self.register_buffer("sub", sub)
        self.register_buffer("lo", torch.deg2rad(torch.tensor(kin["lower_deg"], dtype=torch.float32, device=device)))
        self.register_buffer("hi", torch.deg2rad(torch.tensor(kin["upper_deg"], dtype=torch.float32, device=device)))
        self.active, self.passive = list(kin["active"]), list(kin["passive"])

    def forward(self, q):
        N = len(q); dev = q.device; T = [None] * len(self.body_names); ax = [None] * len(self.joint_names); org = [None] * len(self.joint_names)
        T[self.root] = torch.eye(4, device=dev).expand(N, 4, 4)
        for k, (pa, c, ji) in enumerate(self.order):
            M = T[pa] @ self.A[k]
            if ji >= 0:
                ax[ji] = M[:, :3, 0]; org[ji] = M[:, :3, 3]; a = q[:, ji]; R = torch.zeros(N, 4, 4, device=dev); R[:, 0, 0] = 1; R[:, 3, 3] = 1
                R[:, 1, 1] = torch.cos(a); R[:, 1, 2] = -torch.sin(a); R[:, 2, 1] = torch.sin(a); R[:, 2, 2] = torch.cos(a); M = M @ R
            T[c] = M @ self.B[k]
        return torch.stack(T, 1), torch.stack(ax, 1), torch.stack(org, 1)
