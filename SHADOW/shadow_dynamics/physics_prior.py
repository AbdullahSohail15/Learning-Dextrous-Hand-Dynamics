"""Nominal per-finger physics prior for the Shadow Hand (spec: project_iteration_v2/shadow_hand_physics_prior_flow_matching_experimental_setup.md, §5, §8, §15-16, §33).

Built from the exported articulation (`physics_parameters.json` of a schema-3 dataset) and the hand's kinematic tree
(`artifacts/viz_hand_model/model.json`), plus one identified term (the J1 -> J0 tendon coupling):

  actuated joint j:  M_j(q) qdd = K_j (a_j - q_j) - D_j qd_j        (PhysX implicit position drive, Isaac Lab ImplicitActuator)
                     implicit Euler at the physics rate (120 Hz, 6 substeps per 20 Hz action); joint limits clamp q and zero qd
                     M_j = armature_j + composite inertia of the subtree below j about the joint axis, at the window's current pose
  load torque (disturbance observer, cfg['disturbance']): tau_j = D_j qd_j - K_j (a_{t-1,j} - q_j) from the last observed state and
                     the last applied target (quasi-static), held constant over the horizon and added to the drive. It stands for
                     the unmodelled tendon, coupling and contact loads that hold the weak finger drives (K = 1 N m/rad) off target.
  passive J0 (FF/MF/RF/LF):  q0 = clip(c_f q1 + b_f, limits), qd0 = c_f qd1 inside the limits (fixed-tendon coupling, fitted on train)

Each joint uses only its own state and command (local prior); joints are coupled only through M_j(q) and the tendon.
Units: inputs/outputs deg and deg/s (model-facing), radians internally. The prior uses x_t and the future actions only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

PHYSICS_HZ, CONTROL_HZ = 120, 20
FINGERS = ("FF", "MF", "RF", "LF")


def _quat(q):   # wxyz -> 3x3
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def prior_config(physics_json, viz_model_json, coupling, disturbance=True):
    """Everything the prior needs, as plain data (stored in checkpoints)."""
    p = json.loads(Path(physics_json).read_text()); m = json.loads(Path(viz_model_json).read_text())
    assert p["joint_names"] == m["joint_names"] and p["body_names"] == m["body_names"]
    keep = ("joint_names", "body_names", "mass", "inertia", "com", "stiffness", "damping", "limits", "armature", "max_force", "actuated_indices", "passive_indices")
    return dict(physics={k: p[k] for k in keep}, tree=dict(joints=m["joints"], root_body=m["root_body"]), coupling=coupling, disturbance=bool(disturbance),
                physics_hz=PHYSICS_HZ, control_hz=CONTROL_HZ, source=dict(physics=str(physics_json), viz_model=str(viz_model_json)))


class NominalHand:
    def __init__(self, cfg, device="cpu"):
        P, tree = cfg["physics"], cfg["tree"]; self.cfg = cfg; dev = torch.device(device); self.device = dev
        self.joint_names = P["joint_names"]; self.body_names = P["body_names"]; nb = len(self.body_names)
        bidx = {b: i for i, b in enumerate(self.body_names)}; jidx = {j: i for i, j in enumerate(self.joint_names)}
        def frame(pos, quat):
            T = np.eye(4); T[:3, :3] = _quat(quat); T[:3, 3] = pos; return T
        chain = []
        for j in tree["joints"]:
            if j["body0"] is None: continue
            A = frame(j["local_pos0"], j["local_rot0"]); B = np.linalg.inv(frame(j["local_pos1"], j["local_rot1"]))
            chain.append((bidx[j["body0"]], bidx[j["body1"]], A, B, jidx.get(j["name"], -1) if j["type"] == "PhysicsRevoluteJoint" else -1))
        order, done = [], {bidx[tree["root_body"]]}
        while len(order) < len(chain):
            for c in chain:
                if c[0] in done and c[1] not in done: order.append(c); done.add(c[1])
        self.chain = [(pa, c, torch.tensor(A, dtype=torch.float32, device=dev), torch.tensor(B, dtype=torch.float32, device=dev), ji) for pa, c, A, B, ji in order]
        self.root = bidx[tree["root_body"]]
        children = {i: [] for i in range(nb)}
        for pa, c, *_ in self.chain: children[pa].append(c)
        def subtree(b): return [b] + [d for c in children[b] for d in subtree(c)]
        self.joint_child = {ji: c for _, c, _, _, ji in self.chain if ji >= 0}
        self.sub = torch.zeros(len(self.joint_names), nb, device=dev)   # bodies moved by joint j
        for j, c in self.joint_child.items(): self.sub[j, subtree(c)] = 1
        # body masses, inertia tensors (body frame; the exported COM frames are identity rotations), COM offsets
        self.mass = torch.tensor(P["mass"], dtype=torch.float32, device=dev); self.I = torch.tensor(np.array(P["inertia"]).reshape(nb, 3, 3), dtype=torch.float32, device=dev)
        self.com = torch.tensor(np.array(P["com"])[:, :3], dtype=torch.float32, device=dev)
        cq = np.array(P["com"])[:, 3:]   # experiment 46: PhysX COM frame quaternion (xyzw); the inertia is diagonal in that frame (Allegro). Shadow: identity
        if cq.shape[1] == 4 and not np.allclose(cq, [0, 0, 0, 1], atol=1e-6):
            Rc = torch.tensor(np.stack([_quat([w, x, y, z]) for x, y, z, w in cq]), dtype=torch.float32, device=dev); self.I = Rc @ self.I @ Rc.transpose(-1, -2)
        f = lambda k: torch.tensor(P[k], dtype=torch.float32, device=dev)
        self.K, self.D, self.arm = f("stiffness"), f("damping"), f("armature")
        lim = torch.tensor(P["limits"], dtype=torch.float32, device=dev); self.lo, self.hi = lim[:, 0], lim[:, 1]
        self.act = list(P["actuated_indices"]); self.pas = list(P["passive_indices"])
        if not self.pas:   # experiment 46: Allegro, every joint actuated, no tendon coupling
            self.mid = self.dist = []; self.c = self.b = torch.zeros(0, device=dev)
        else:
          self.mid = [self.joint_names.index(f"robot0_{f}J1") for f in FINGERS]; self.dist = [self.joint_names.index(f"robot0_{f}J0") for f in FINGERS]
          assert sorted(self.dist) == sorted(self.pas)
          cpl = cfg["coupling"]; self.c = torch.tensor([cpl[f]["gain"] for f in FINGERS], dtype=torch.float32, device=dev); self.b = torch.tensor([np.deg2rad(cpl[f]["offset_deg"]) for f in FINGERS], dtype=torch.float32, device=dev)
        self.substeps = cfg["physics_hz"] // cfg["control_hz"]; self.h = 1.0 / cfg["physics_hz"]

    def _fk(self, q):   # q rad (N, 24) -> body transforms (N, B, 4, 4), joint frames (N, 24, 4, 4)
        N = len(q); T = [None] * len(self.body_names); J = [None] * len(self.joint_names)
        T[self.root] = torch.eye(4, device=q.device).expand(N, 4, 4)
        for pa, c, A, B, ji in self.chain:
            M = T[pa] @ A
            if ji >= 0:
                J[ji] = M; a = q[:, ji]; R = torch.zeros(N, 4, 4, device=q.device); R[:, 0, 0] = 1; R[:, 3, 3] = 1
                R[:, 1, 1] = torch.cos(a); R[:, 1, 2] = -torch.sin(a); R[:, 2, 1] = torch.sin(a); R[:, 2, 2] = torch.cos(a); M = M @ R
            T[c] = M @ B
        return torch.stack(T, 1), torch.stack(J, 1)

    def effective_inertia(self, q):
        """(N, 24) kg m^2: armature + subtree inertia about each joint axis (parallel-axis theorem), at pose q (rad)."""
        Tb, Tj = self._fk(q); R = Tb[..., :3, :3]; p = (R @ self.com[None, :, :, None])[..., 0] + Tb[..., :3, 3]   # COM in root frame (N, B, 3)
        Iw = R @ self.I[None] @ R.transpose(-1, -2)                                                                  # (N, B, 3, 3)
        axis = Tj[..., :3, 0]; o = Tj[..., :3, 3]                                                                     # (N, 24, 3)
        rot = torch.einsum("nji,nbik,njk->njb", axis, Iw, axis)
        r = p[:, None] - o[:, :, None]; perp = torch.cross(r, axis[:, :, None].expand_as(r), dim=-1)
        I = ((rot + self.mass[None, None] * perp.square().sum(-1)) * self.sub[None]).sum(-1)
        return I + self.arm[None]

    @torch.no_grad()
    def rollout(self, s, fa, pa=None):
        """s (N, >=1, 48) past states (deg, deg/s; only the last is used), fa (N, H, 20) future applied targets, pa (N, >=1, 20) past
        applied targets (only a_{t-1} is used, for the load estimate) -> (N, H, 48) deg, deg/s."""
        x = s[:, -1].float(); n = len(self.joint_names); q = torch.deg2rad(x[:, :n]); v = torch.deg2rad(x[:, n:]); a_all = torch.deg2rad(fa.float())
        M = self.effective_inertia(q); K, D, h = self.K, self.D, self.h; act = self.act
        tau = torch.zeros_like(q)
        if self.cfg.get("disturbance", False):
            assert pa is not None, "the load estimate needs the last applied target"
            tau[:, act] = D[act] * v[:, act] - K[act] * (torch.deg2rad(pa[:, -1].float()) - q[:, act])
        out = []
        for k in range(a_all.shape[1]):
            tgt = q.clone(); tgt[:, act] = a_all[:, k]
            for _ in range(self.substeps):   # implicit Euler of M v' = K (a - q) - D v on the actuated joints
                vn = (M * v + h * (K * (tgt - q) + tau)) / (M + h * D + h * h * K)
                qn = q + h * vn
                at_lim = (qn < self.lo) | (qn > self.hi); qn = qn.clamp(self.lo, self.hi); vn = torch.where(at_lim, torch.zeros_like(vn), vn)
                q = torch.where(self._act_mask(q), qn, q); v = torch.where(self._act_mask(q), vn, v)
            if self.dist:
                q1, v1 = q[:, self.mid], v[:, self.mid]   # passive distal joints follow the tendon coupling
                q0 = self.c * q1 + self.b; inside = (q0 > self.lo[self.dist]) & (q0 < self.hi[self.dist])
                q[:, self.dist] = q0.clamp(self.lo[self.dist], self.hi[self.dist]); v[:, self.dist] = torch.where(inside, self.c * v1, torch.zeros_like(v1))
            out.append(torch.cat([torch.rad2deg(q), torch.rad2deg(v)], -1))
        return torch.stack(out, 1)

    def _act_mask(self, q):
        if not hasattr(self, "_am"):
            m = torch.zeros(len(self.joint_names), dtype=torch.bool, device=q.device); m[self.act] = True; self._am = m
        return self._am


def fit_coupling(episodes, joint_names, validity_ok=(0,)):
    """Least-squares q0 = c q1 + b per finger on no-contact states with both joints strictly inside their limits (deg)."""
    out = {}
    for f in FINGERS:
        i1, i0 = joint_names.index(f"robot0_{f}J1"), joint_names.index(f"robot0_{f}J0")
        q1 = np.concatenate([e["q"][np.isin(e["validity"], validity_ok), i1] for e in episodes]); q0 = np.concatenate([e["q"][np.isin(e["validity"], validity_ok), i0] for e in episodes])
        m = (q0 > 1) & (q0 < 89) & (q1 > 1) & (q1 < 89)
        A = np.stack([q1[m], np.ones(m.sum())], 1); (c, b), *_ = np.linalg.lstsq(A, q0[m], rcond=None)
        resid = q0[m] - (c * q1[m] + b)
        out[f] = dict(gain=float(c), offset_deg=float(b), rmse_deg=float(np.sqrt((resid ** 2).mean())), states=int(m.sum()))
    return out
