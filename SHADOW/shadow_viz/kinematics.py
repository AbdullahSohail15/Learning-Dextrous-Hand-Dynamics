"""Forward kinematics of the stock Isaac Lab Shadow Hand from the exported USD joint tree (scripts/export_shadow_viz_model.py).

Poses are relative to the root body (robot0_hand_mount). Joint angles are in degrees, in the simulator's runtime joint order.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

FINGERS = ("ff", "mf", "rf", "lf", "th")


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def frame(pos, quat):
    T = np.eye(4); T[:3, :3] = quat_matrix(quat); T[:3, 3] = pos; return T


class HandModel:
    def __init__(self, model_dir):
        d = Path(model_dir); self.meta = json.loads((d / "model.json").read_text())
        self.joint_names = self.meta["joint_names"]; self.body_names = self.meta["body_names"]; self.root = self.meta["root_body"]
        bidx = {b: i for i, b in enumerate(self.body_names)}; jidx = {j: i for i, j in enumerate(self.joint_names)}
        self.chain = []   # (parent body, child body, A = J0, B = inv(J1), runtime joint index or -1 for fixed)
        for j in self.meta["joints"]:
            if j["body0"] is None: continue   # root joint to the world
            assert j["type"] == "PhysicsFixedJoint" or j.get("axis") == "X", j
            A = frame(j["local_pos0"], j["local_rot0"]); B = np.linalg.inv(frame(j["local_pos1"], j["local_rot1"]))
            self.chain.append((bidx[j["body0"]], bidx[j["body1"]], A, B, jidx.get(j["name"], -1) if j["type"] == "PhysicsRevoluteJoint" else -1))
        order, done = [], {bidx[self.root]}   # topological order from the root
        while len(order) < len(self.chain):
            for c in self.chain:
                if c[0] in done and c[1] not in done: order.append(c); done.add(c[1])
        self.chain = order
        self.joint_child = {c[4]: c[1] for c in self.chain if c[4] >= 0}   # runtime joint -> child body (its origin lies on the joint axis)
        meshes = np.load(d / "meshes.npz"); self.meshes = {k: (meshes[f"{k}/vertices"], meshes[f"{k}/faces"]) for k in self.meta["mesh_links"]}
        self.tips = {}
        for f in FINGERS:   # fingertip: centre of the distal mesh's last 1.5 mm along the link axis
            v = self.meshes[f"robot0_{f}distal"][0]; top = v[v[:, 2] > v[:, 2].max() - 0.0015]
            self.tips[f"robot0_{f}distal"] = np.array([top[:, 0].mean(), top[:, 1].mean(), v[:, 2].max()])

    def fk(self, q_deg):
        """q_deg (..., 24) -> body transforms (..., B, 4, 4) relative to the root body."""
        q = np.deg2rad(np.asarray(q_deg, np.float64)); lead = q.shape[:-1]
        T = np.zeros(lead + (len(self.body_names), 4, 4)); T[..., bidx_root(self), :, :] = np.eye(4)
        for p, c, A, B, ji in self.chain:
            if ji < 0: R = np.eye(4)
            else:
                a = q[..., ji]; R = np.zeros(lead + (4, 4)); R[..., 0, 0] = 1; R[..., 3, 3] = 1
                R[..., 1, 1] = np.cos(a); R[..., 1, 2] = -np.sin(a); R[..., 2, 1] = np.sin(a); R[..., 2, 2] = np.cos(a)
            T[..., c, :, :] = T[..., p, :, :] @ A @ R @ B
        return T

    def keypoints(self, q_deg):
        """Joint origins (one per revolute joint, runtime order) and the five fingertips: (..., 24 + 5, 3) and their names."""
        T = self.fk(q_deg); bodies = [self.joint_child[j] for j in range(len(self.joint_names))]
        pts = [T[..., b, :3, 3] for b in bodies]
        for name, off in self.tips.items():
            b = self.body_names.index(name); pts.append(np.einsum("...ij,j->...i", T[..., b, :3, :3], off) + T[..., b, :3, 3])
        return np.stack(pts, -2), list(self.joint_names) + [n.replace("distal", "_tip") for n in self.tips]


def bidx_root(m): return m.body_names.index(m.root)


def finger_of(name):
    n = name.replace("robot0_", "").lower()
    for f in FINGERS:
        if n.startswith(f): return f
    return "wr"
