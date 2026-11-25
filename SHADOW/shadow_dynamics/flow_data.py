"""Training windows for the whole-horizon flow-matching hand model (schema 3.0.0-structured datasets).

Model I/O (project_iteration_v2/shadow_hand_flow_matching_architecture.md):
  past states     x_{t-3..t}      (4, 48)   [q (24) deg, qdot (24) deg/s]
  past actions    a_{t-3..t-1}    (3, 20)   applied targets, deg
  future actions  a_{t..t+11}     (12, 20)  applied targets, deg
  target          x_{t+1..t+12}   (12, 48)  and its delta to x_t (the recommended flow variable)
Action a_k is applied between states k and k+1. A window is valid when every state t-3..t+12 is VALID_NO_CONTACT or
VALID_CONTACT. Normalization statistics come from the train split only (per channel; deltas per horizon step and channel).

    ds = StructuredDataset(["artifacts/shadow_structured_v3/train_speeds"])
    stats = ds.normalization("train")                  # dict of numpy arrays, save with the checkpoint
    W = ds.windows("train")                            # (n, 2) int: episode index, anchor t
    batch = ds.gather(W[:256])                         # dict of numpy arrays in physical units
"""
from __future__ import annotations
import json
from pathlib import Path
import h5py
import numpy as np

SCHEMA_PREFIX = "3."
HISTORY, PAST_ACTIONS, HORIZON = 4, 3, 12
EXPECTED_UNITS = {"position_unit": "deg", "velocity_unit": "deg/s", "action_unit": "deg", "control_hz": 20}


class StructuredDataset:
    def __init__(self, directories, fields=("q", "qdot", "action_applied", "validity")):
        self.episodes, self.meta, self.manifests = [], [], []
        for d in directories:
            d = Path(d); shards = sorted(d.glob("shard_*.h5")); assert shards, f"no shards in {d}"
            for p in shards:
                with h5py.File(p, "r") as f:
                    assert f.attrs.get("complete", False), p
                    assert str(f.attrs["schema_version"]).startswith(SCHEMA_PREFIX), p
                    for k, v in EXPECTED_UNITS.items(): assert f.attrs[k] == v, (p, k, f.attrs[k])
                    man = json.loads(f.attrs["manifest_json"]); self.manifests.append(man)
                    for name in f:
                        g = f[name]; m = json.loads(g.attrs["metadata_json"]); m["source"] = f"{p}:{name}"
                        self.episodes.append({k: g[k][:] for k in fields}); self.meta.append(m)
        self.mapping = self.manifests[0]["mapping"]
        for man in self.manifests: assert man["mapping"]["joint_names"] == self.mapping["joint_names"] and man["control_hz"] == 20
        self.split = np.array([m["split"] for m in self.meta]); self.seed = np.array([m["seed"] for m in self.meta])
        self.speed = np.array([m["speed_deg_s"] for m in self.meta]); self.family = np.array([m["motion_family"] for m in self.meta])
        seeds = {}
        for s, sp in zip(self.seed, self.split): assert seeds.setdefault(s, sp) == sp, f"seed {s} in two splits"
        assert not np.isin(self.speed[self.split == "train"], [15, 25, 50, 70]).any(), "held-out speed in the training split"

    def windows(self, split, families=None, speeds=None):
        out = []
        for e, (ep, sp) in enumerate(zip(self.episodes, self.split)):
            if sp != split or (families is not None and self.family[e] not in families) or (speeds is not None and self.speed[e] not in speeds): continue
            T = len(ep["action_applied"]); ok = ep["validity"] <= 1
            if T < HORIZON + HISTORY - 1: continue
            t = np.arange(HISTORY - 1, T - HORIZON + 1)
            bad = np.r_[0, np.cumsum(~ok)]   # windows need states t-3 .. t+12 valid
            valid = (bad[t + HORIZON + 1] - bad[t - HISTORY + 1]) == 0
            out.append(np.stack([np.full(valid.sum(), e), t[valid]], 1))
        return np.concatenate(out) if out else np.zeros((0, 2), int)

    def state(self, e):
        ep = self.episodes[e]; return np.concatenate([ep["q"], ep["qdot"]], 1)

    def gather(self, W):
        X = [self.state(e) for e in range(len(self.episodes))] if not hasattr(self, "_X") else self._X; self._X = X
        hs, ha, fa, y = [], [], [], []
        for e, t in W:
            x, u = X[e], self.episodes[e]["action_applied"]
            hs.append(x[t - HISTORY + 1:t + 1]); ha.append(u[t - PAST_ACTIONS:t]); fa.append(u[t:t + HORIZON]); y.append(x[t + 1:t + HORIZON + 1])
        hs, ha, fa, y = map(np.stack, (hs, ha, fa, y))
        return dict(past_states=hs, past_actions=ha, future_actions=fa, future_states=y, future_delta=y - hs[:, -1:, :])

    def normalization(self, split="train", max_windows=400_000, seed=0, eps=1e-3):
        assert split == "train", "normalization statistics come from the train split only"
        W = self.windows("train"); rng = np.random.default_rng(seed)
        if len(W) > max_windows: W = W[rng.choice(len(W), max_windows, replace=False)]
        b = self.gather(W); s = b["past_states"].reshape(-1, b["past_states"].shape[-1]); na = b["past_actions"].shape[-1]; a = np.concatenate([b["past_actions"].reshape(-1, na), b["future_actions"].reshape(-1, na)])
        d = b["future_delta"]
        stats = dict(state_mean=s.mean(0), state_std=np.maximum(s.std(0), eps), action_mean=a.mean(0), action_std=np.maximum(a.std(0), eps),
                     delta_mean=d.mean(0), delta_std=np.maximum(d.std(0), eps), n_windows=len(W),
                     units=dict(state="q deg (24) + qdot deg/s (24)", action="deg", delta="same as state, per horizon step (12, 48)"))
        return stats


def save_stats(stats, path):
    Path(path).write_text(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()}, indent=1))
