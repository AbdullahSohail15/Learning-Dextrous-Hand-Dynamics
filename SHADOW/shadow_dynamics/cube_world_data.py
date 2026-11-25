"""Hand + cube windows for the world model of experiment 12 (same interface as flow_data.StructuredDataset).

State (63): q (24, deg), qdot (24, deg/s), cube position (3, mm, relative to the in-hand point), cube orientation as the
first two columns of its rotation matrix (6, the continuous "6D" representation), cube linear velocity (3, mm/s),
cube angular velocity (3, deg/s). Shards come from scripts/collect_shadow_cube.py. Episodes end at the first drop; the
drop state itself is kept, so the model sees cubes leaving the hand.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

HISTORY, PAST_ACTIONS, HORIZON = 4, 3, 12
STATE_DIM = 63
POS, ROT, LIN, ANG = slice(48, 51), slice(51, 57), slice(57, 60), slice(60, 63)


def quat_to_6d(q):
    """(..., 4) wxyz -> (..., 6): first two columns of the rotation matrix (numpy or torch)."""
    w, x, y, z = (q[..., i] for i in range(4)); lib = torch if isinstance(q, torch.Tensor) else np
    c0 = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
    c1 = [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
    return lib.stack(c0 + c1, -1)


def sixd_to_matrix(r):
    """(..., 6) torch -> (..., 3, 3) rotation (Gram-Schmidt)."""
    a, b = r[..., :3], r[..., 3:]
    c0 = torch.nn.functional.normalize(a, dim=-1); b = b - (c0 * b).sum(-1, keepdim=True) * c0; c1 = torch.nn.functional.normalize(b, dim=-1)
    return torch.stack([c0, c1, torch.cross(c0, c1, dim=-1)], -1)


def quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def rotation_distance(Ra, Rb):
    """Angle (rad) between rotation matrices (..., 3, 3), the same quantity as Isaac's rotation_distance."""
    c = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.arccos(c.clamp(-1 + 1e-7, 1 - 1e-7))


CONTACT_FEATURES = ("cube_force", "cube_contact", "self_force")   # experiment 34, 26 links each


def contact_channels(z, i, L):
    """(L + 1, 78): log(1 + max |F| hand->cube) per link, cube contact flag per link, log(1 + self-contact force) per link."""
    return np.concatenate([np.log1p(z["link_cube_force_max"][i, :L + 1].astype(np.float32)), z["link_cube_contact"][i, :L + 1].astype(np.float32),
                           np.log1p(z["link_self_force"][i, :L + 1].astype(np.float32))], 1)


class CubeDataset:
    def __init__(self, directories, contact=False, mask_invalid=True):
        """contact: append the 78 experiment-34 contact channels to the 63-dim state (state = 141; needs --contact-sensors all data).
        mask_invalid: windows touching a hand-validity class >= 2 (self-penetration / numerical) are not used (experiment 34 decision)."""
        self.episodes, self.meta = [], []; manifests = []; self.contact = contact
        for d in directories:
            d = Path(d); man = json.loads((d / "manifest.json").read_text()); manifests.append(man)
            for sh in man["shards"]:
                with np.load(d / sh["file"]) as f: z = {k: f[k] for k in f.files}   # decompress each array once (indexing an NpzFile re-reads it)
                for i in range(len(z["length"])):
                    L = int(z["length"][i]); c = z["cube"][i, :L + 1]
                    val = z["validity"][i, :L + 1].astype(np.int8) if (mask_invalid and "validity" in z) else np.zeros(L + 1, np.int8)
                    self.episodes.append(dict(q=z["q"][i, :L + 1], qdot=z["qdot"][i, :L + 1], action_applied=z["action"][i, :L], validity=val,
                                              cube=np.concatenate([c[:, :3], quat_to_6d(c[:, 3:7]), c[:, 7:10], c[:, 10:13]], 1).astype(np.float32), quat=c[:, 3:7]))
                    if contact:
                        assert "link_cube_contact" in z, f"{d}: no contact channels (collect with --contact-sensors all)"
                        self.episodes[-1]["contact"] = contact_channels(z, i, L)
                    ds_ = int(z["drop_step"][i]) if "drop_step" in z else (L if bool(z["dropped"][i]) else -1)   # older shards stop at the drop
                    self.meta.append(dict(seed=int(z["seed"][i]), split=str(z["split"][i]), dropped=bool(z["dropped"][i]), drop_step=ds_, source=f"{d}:{sh['file']}:{i}", behaviour=man["arguments"]["behaviour"]))
        m = manifests[0]
        for x in manifests: assert x["joint_names"] == m["joint_names"] and x["control_hz"] == 20
        self.mapping = dict(joint_names=m["joint_names"], action_names=m["action_names"], active_indices=m["active_indices"], passive_indices=m["passive_indices"])
        self.split = np.array([x["split"] for x in self.meta]); self.seed = np.array([x["seed"] for x in self.meta])
        self.speed = np.full(len(self.meta), 80.0); self.family = np.array([x["behaviour"] for x in self.meta]); self.state_dim = STATE_DIM + (78 if contact else 0)
        assert len(set(self.seed.tolist())) == len(self.seed), "duplicate seeds across shards"

    def windows(self, split, **_):
        out = []
        for e, ep in enumerate(self.episodes):
            if self.split[e] != split: continue
            T = len(ep["action_applied"])
            if T < HORIZON + HISTORY - 1: continue
            t = np.arange(HISTORY - 1, T - HORIZON + 1)
            bad = ep["validity"] >= 2
            if bad.any():   # window states t-3 .. t+12 must all be valid
                cb = np.concatenate([[0], np.cumsum(bad)]); t = t[cb[t + HORIZON + 1] - cb[t - HISTORY + 1] == 0]
            out.append(np.stack([np.full(len(t), e), t], 1))
        return np.concatenate(out) if out else np.zeros((0, 2), int)

    def state(self, e):
        ep = self.episodes[e]; return np.concatenate([ep["q"], ep["qdot"], ep["cube"]] + ([ep["contact"]] if self.contact else []), 1)

    def gather(self, W):
        if not hasattr(self, "_X"): self._X = [self.state(e) for e in range(len(self.episodes))]
        hs, ha, fa, y = [], [], [], []
        for e, t in W:
            x, u = self._X[e], self.episodes[e]["action_applied"]
            hs.append(x[t - HISTORY + 1:t + 1]); ha.append(u[t - PAST_ACTIONS:t]); fa.append(u[t:t + HORIZON]); y.append(x[t + 1:t + HORIZON + 1])
        hs, ha, fa, y = map(np.stack, (hs, ha, fa, y))
        return dict(past_states=hs, past_actions=ha, future_actions=fa, future_states=y, future_delta=y - hs[:, -1:, :])

    def normalization(self, split="train", max_windows=400_000, seed=0, eps=1e-3):
        assert split == "train"
        W = self.windows("train"); rng = np.random.default_rng(seed)
        if len(W) > max_windows: W = W[rng.choice(len(W), max_windows, replace=False)]
        b = self.gather(W); s = b["past_states"].reshape(-1, self.state_dim); a = np.concatenate([b["past_actions"].reshape(-1, 20), b["future_actions"].reshape(-1, 20)]); d = b["future_delta"]
        return dict(state_mean=s.mean(0), state_std=np.maximum(s.std(0), eps), action_mean=a.mean(0), action_std=np.maximum(a.std(0), eps),
                    delta_mean=d.mean(0), delta_std=np.maximum(d.std(0), eps), n_windows=len(W),
                    units=dict(state="q deg (24) + qdot deg/s (24) + cube pos mm (3) + rot 6D (6) + linvel mm/s (3) + angvel deg/s (3)", action="deg"))


# ---- experiment 24: partial observability of the cube (occlusion + noise); the hand joints stay clean.
# What the controller sees of the cube at a step: if visible, the true reading plus Gaussian noise (position sigma_p mm, orientation a
# world-frame rotation with axis-angle ~ N(0, sigma_r), velocities the noise of a finite difference: sqrt(2) * sigma * 20 Hz); if
# occluded, the last reading it saw (pose and velocities held). Occlusion is a two-state Markov chain per environment:
# visible -> occluded with probability `enter`, occluded -> occluded with probability `stay` (stale fraction enter / (enter + 1 - stay),
# mean run 1 / (1 - stay) steps). The same definition is used for training windows (corrupt_history) and in Isaac (CubeOcclusionObserver).
OCC_BURN_IN = 16


def axis_angle_matrix(v):
    """(..., 3) rotation vectors (rad) -> (..., 3, 3) (Rodrigues)."""
    th = v.norm(dim=-1, keepdim=True).clamp_min(1e-9); k = v / th; th = th[..., None]
    K = torch.zeros(v.shape[:-1] + (3, 3), device=v.device, dtype=v.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0], K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 2], k[..., 1], k[..., 2], -k[..., 0], -k[..., 1], k[..., 0]
    return torch.eye(3, device=v.device, dtype=v.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def _cube_noise(shape, cfg, gen, device):
    """Noise draws for `shape` cube readings: position (3), rotation vector (3, rad), linear (3) and angular (3) velocity."""
    n = lambda sd: torch.randn(shape + (3,), generator=gen, device=device) * sd; k = 2 ** 0.5 * 20
    return n(cfg["pos_mm"]), n(np.deg2rad(cfg["rot_deg"])), n(cfg["pos_mm"] * k), n(cfg["rot_deg"] * k)


def corrupt_history(X, e, t, cfg, gen):
    """Observed 4-step histories for windows (e, t): X (E, T+1, 63) true states, e / t (B, 1) episode and anchor indices.
    Returns s (B, 4, 63) with clean hand channels and corrupted cube channels, and the stale age (B, 4) in steps (0 = fresh)."""
    B, dev = e.shape[0], X.device; occ = torch.zeros(B, dtype=torch.bool, device=dev); age = torch.zeros(B, dtype=torch.long, device=dev); ages = []
    for k in range(OCC_BURN_IN + HISTORY):   # chain starts visible OCC_BURN_IN steps before the window (near-stationary by the window)
        if k:
            u = torch.rand(B, generator=gen, device=dev); occ = torch.where(occ, u < cfg["stay"], u < cfg["enter"]); age = torch.where(occ, age + 1, torch.zeros_like(age))
        if k >= OCC_BURN_IN: ages.append(age.clone())
    age = torch.stack(ages, 1)                                                                   # (B, 4)
    tt = t + torch.arange(-HISTORY + 1, 1, device=dev)[None]; src = (tt - age).clamp_min(0)
    s = X[e, tt].clone(); cube = X[e, src, 48:].clone()                                           # (B, 4, 15) held readings
    dp, dr, dl, da = _cube_noise((B, HISTORY), cfg, gen, dev)
    for j in range(1, HISTORY):   # a held reading keeps the noise it was seen with
        same = (src[:, j] == src[:, j - 1])[:, None]
        dp[:, j], dr[:, j], dl[:, j], da[:, j] = (torch.where(same, x[:, j - 1], x[:, j]) for x in (dp, dr, dl, da))
    R = axis_angle_matrix(dr) @ torch.stack([cube[..., 3:6], cube[..., 6:9]], -1)                # world-frame noise on the two stored columns
    s[..., 48:51] = cube[..., :3] + dp; s[..., 51:54], s[..., 54:57] = R[..., 0], R[..., 1]
    s[..., 57:60] = cube[..., 9:12] + dl; s[..., 60:63] = cube[..., 12:15] + da
    return s, age


class CubeOcclusionObserver:
    """Per-step version of corrupt_history for Isaac: cube13 (N, 13) = pos mm, quat wxyz, linvel mm/s, angvel deg/s -> observed cube13 and age (N,)."""

    def __init__(self, n, cfg, gen, device):
        self.cfg, self.g, self.dev = cfg, gen, device; self.occ = torch.zeros(n, dtype=torch.bool, device=device)
        self.age = torch.zeros(n, dtype=torch.long, device=device); self.last = None

    def __call__(self, c):
        if self.last is not None:
            u = torch.rand(len(c), generator=self.g, device=self.dev); self.occ = torch.where(self.occ, u < self.cfg["stay"], u < self.cfg["enter"])
            self.age = torch.where(self.occ, self.age + 1, torch.zeros_like(self.age))
        dp, dr, dl, da = _cube_noise((len(c),), self.cfg, self.g, self.dev)
        R = axis_angle_matrix(dr) @ quat_to_matrix(c[:, 3:7]); o = c.clone()
        o[:, :3] += dp; o[:, 3:7] = matrix_to_quat(R); o[:, 7:10] += dl; o[:, 10:13] += da
        if self.last is not None: o = torch.where(self.occ[:, None], self.last, o)
        self.last = o.clone(); return o, self.age.clone()


def matrix_to_quat(R):
    """(N, 3, 3) -> (N, 4) wxyz (w >= 0), via the largest-diagonal branch."""
    m = R; t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.stack([1 + t, 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], -1)
    i = q.argmax(-1); out = torch.zeros(len(R), 4, device=R.device, dtype=R.dtype)
    s = lambda a_, b_: m[:, a_[0], a_[1]] - m[:, b_[0], b_[1]]; p = lambda a_, b_: m[:, a_[0], a_[1]] + m[:, b_[0], b_[1]]
    cand = [torch.stack([q[:, 0], s((2, 1), (1, 2)), s((0, 2), (2, 0)), s((1, 0), (0, 1))], -1),
            torch.stack([s((2, 1), (1, 2)), q[:, 1], p((0, 1), (1, 0)), p((0, 2), (2, 0))], -1),
            torch.stack([s((0, 2), (2, 0)), p((0, 1), (1, 0)), q[:, 2], p((1, 2), (2, 1))], -1),
            torch.stack([s((1, 0), (0, 1)), p((0, 2), (2, 0)), p((1, 2), (2, 1)), q[:, 3]], -1)]
    for k_ in range(4): out = torch.where((i == k_)[:, None], cand[k_], out)
    out = torch.nn.functional.normalize(out, dim=-1); return out * torch.where(out[:, :1] < 0, -1.0, 1.0)


def occlusion_cfg(pos_mm, rot_deg, enter, stay):
    return None if enter <= 0 and pos_mm <= 0 and rot_deg <= 0 else dict(pos_mm=float(pos_mm), rot_deg=float(rot_deg), enter=float(enter), stay=float(stay))
