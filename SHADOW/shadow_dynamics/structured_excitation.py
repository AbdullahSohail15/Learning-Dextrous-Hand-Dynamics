"""Structured, physically valid excitation programs for the Shadow Hand (schema 3.0.0-structured).

Programs are requested active-joint targets (T, 20) in DEGREES at the control rate (20 Hz), in runtime actuator order.
The collector applies the per-episode slew limit (= speed regime, deg/s) and the joint limits; the result is the applied
action. Families follow project_iteration_v2/shadow_hand_flow_matching_data_generation.md section 18:

  isolated finger : isolated_multisine, isolated_chirp, isolated_limit_sweep, isolated_limit_triangle, isolated_random_walk
  finger pairs    : pair_multisine, pair_random_walk, pair_synchronized, pair_opposing
  coordinated     : pose_open_to_fist, pose_fist_to_open, pose_victory, pose_thumb_index_pinch, pose_thumb_middle_pinch,
                    pose_tripod, pose_thumb_opposition, pose_finger_wave
  local           : local_pose_perturbation
  random          : constraint_aware_random

Smooth motion only: keyframes are joined by minimum-jerk segments whose peak speed is at most the regime speed, and
sinusoids/chirps are frequency-capped by the same speed. Every program is checked before simulation (`ProgramChecker`):
its slew-limited target path is put through forward kinematics (passive J0 = clip(1.13 J1)) and the visual-link
penetration test; overlapping programs are redrawn. The simulator's own per-step labels remain the authority.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SCHEMA = "3.0.0-structured"
CONTROL_HZ = 20
TRAIN_SPEEDS_DEG = [1, 5, 10, 20, 30, 45, 60, 80]   # 45-80 added in experiment 09
OOD_SPEEDS_DEG = [15, 25, 50, 70]   # 50, 70 added in experiment 09
SPEED_SHARES = {1: 0.10, 5: 0.20, 10: 0.25, 20: 0.25, 30: 0.20}
FINGER_CHANNELS = {"thumb": ["THJ4", "THJ3", "THJ2", "THJ1", "THJ0"], "index": ["FFJ3", "FFJ2", "FFJ1"], "middle": ["MFJ3", "MFJ2", "MFJ1"],
                   "ring": ["RFJ3", "RFJ2", "RFJ1"], "little": ["LFJ4", "LFJ3", "LFJ2", "LFJ1"]}
WRIST = ["WRJ1", "WRJ0"]
FINGERS = list(FINGER_CHANNELS)
PAIRS = [("thumb", "index"), ("thumb", "middle"), ("thumb", "ring"), ("thumb", "little"), ("index", "middle"), ("middle", "ring"),
         ("ring", "little"), ("index", "ring"), ("index", "little"), ("middle", "little")]
GROUPS = {
    "isolated": ["isolated_multisine", "isolated_chirp", "isolated_limit_sweep", "isolated_limit_triangle", "isolated_random_walk"],
    "pair": ["pair_multisine", "pair_random_walk", "pair_synchronized", "pair_opposing"],
    "coordinated": ["pose_open_to_fist", "pose_fist_to_open", "pose_victory", "pose_thumb_index_pinch", "pose_thumb_middle_pinch",
                    "pose_tripod", "pose_thumb_opposition", "pose_finger_wave"],
    "local": ["local_pose_perturbation"],
    "random": ["constraint_aware_random"],
    "lateral": ["lateral_limit_sweep"],   # top-up only (share 0 in the main mix): spread / thumb-rotation / wrist limits
}
GROUP_SHARES = {"isolated": 0.30, "pair": 0.20, "coordinated": 0.25, "local": 0.15, "random": 0.10, "lateral": 0.0}
LATERAL = ["FFJ3", "MFJ3", "RFJ3", "LFJ3", "LFJ4", "THJ4", "THJ2", "THJ1", "WRJ1", "WRJ0"]
FAMILIES = [f for g in GROUPS.values() for f in g]
SHADOW_LAYOUT = dict(finger_channels=FINGER_CHANNELS, wrist=WRIST, pairs=PAIRS, lateral=LATERAL, wave_fingers=["index", "middle", "ring", "little"],
                     wave_skip={"FFJ3", "MFJ3", "RFJ3", "LFJ4", "LFJ3"}, closed_skip={"FFJ3", "MFJ3", "RFJ3", "LFJ4", "THJ4", "THJ3"},
                     thumb_pair_pose={"index": "thumb_index_pinch", "middle": "thumb_middle_pinch", "ring": "thumb_ring_pinch", "little": "thumb_opposition"})
_AL = lambda f: [f"{f}_joint_{i}" for i in range(4)]
ALLEGRO_LAYOUT = dict(finger_channels={"thumb": _AL("thumb"), "index": _AL("index"), "middle": _AL("middle"), "ring": _AL("ring")}, wrist=[],   # experiment 46
                      pairs=[("thumb", "index"), ("thumb", "middle"), ("thumb", "ring"), ("index", "middle"), ("middle", "ring"), ("index", "ring")],
                      lateral=["index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0", "thumb_joint_1"], wave_fingers=["index", "middle", "ring"],
                      wave_skip={"index_joint_0", "middle_joint_0", "ring_joint_0"}, closed_skip={"index_joint_0", "middle_joint_0", "ring_joint_0"},
                      thumb_pair_pose={"index": "thumb_index_pinch", "middle": "thumb_middle_pinch", "ring": "thumb_ring_pinch"})
FAMILY_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}
MINJERK_PEAK = 1.875   # peak / mean speed of a minimum-jerk segment
AMPLITUDE_FRACTIONS = [0.05, 0.15, 0.4, 1.0]   # of the half range


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0); return s ** 3 * (10 - 15 * s + 6 * s ** 2)


class Spec:
    """Joint names, limits (deg) and the reference pose library, in runtime actuator order."""

    def __init__(self, action_names, low_deg, high_deg, poses_deg, layout=None):
        self.L = layout or SHADOW_LAYOUT   # experiment 46: hand layout (finger channels, wrist, pairs, lateral, skip sets)
        self.names = [n.replace("robot0_", "") for n in action_names]; self.index = {n: i for i, n in enumerate(self.names)}; self.n = len(self.names)
        self.low = np.asarray(low_deg, np.float64); self.high = np.asarray(high_deg, np.float64)
        self.range = self.high - self.low; self.mid = (self.low + self.high) / 2
        self.poses = {k: np.clip(np.asarray(v, np.float64), self.low, self.high) for k, v in poses_deg.items()}
        th = [self.index[n] for n in self.L["finger_channels"]["thumb"]]
        for k in ("fist", "victory"):   # via poses: fingers closed, thumb still at the open pose
            if k in self.poses: q = self.poses[k].copy(); q[th] = self.poses["open"][th]; self.poses[f"{k}_thumb_out"] = q
        self.finger = {f: [self.index[n] for n in ch] for f, ch in self.L["finger_channels"].items()}; self.wrist = [self.index[n] for n in self.L["wrist"]]
        self.fingers = list(self.L["finger_channels"])

    def ch(self, fingers):
        return [c for f in fingers for c in self.finger[f]]


def keyframe_path(start, keys, T, speed, rng, hz=CONTROL_HZ, slow=(1.0, 1.4)):
    """Minimum-jerk path through keyframes [(target (20,), dwell_s)]; segment time from the largest joint change at <= speed.

    Returns (T, 20). The schedule repeats the keyframes cyclically until T is covered.
    """
    out, t, cur, k = [], 0, np.asarray(start, np.float64), 0
    out.append(cur[None])
    while sum(len(o) for o in out) < T + 1:
        target, dwell = keys[k % len(keys)]; k += 1
        dur = MINJERK_PEAK * np.abs(target - cur).max() / speed * rng.uniform(*slow)
        n = max(1, int(np.ceil(dur * hz)))
        s = min_jerk(np.arange(1, n + 1) / n)[:, None]; out.append(cur + s * (target - cur)); cur = np.asarray(target, np.float64)
        if dwell > 0: out.append(np.repeat(cur[None], int(round(dwell * hz)), 0))
        if k > 10000: break
    return np.concatenate(out)[:T]


def sines(rng, T, center, amp, speed, n_max=3, hz=CONTROL_HZ):
    """Sum of 1..n_max sinusoids with peak amplitude amp; the fastest component's peak speed is about `speed`."""
    t = np.arange(T) / hz; n = int(rng.integers(1, n_max + 1)); fmax = max(0.002, speed / (2 * np.pi * max(amp, 1e-3)))
    freqs = np.exp(rng.uniform(np.log(fmax / 8), np.log(fmax), n)); phases = rng.uniform(0, 2 * np.pi, n); w = rng.uniform(0.3, 1.0, n)
    sig = sum(wi * np.sin(2 * np.pi * f * t + p) for wi, f, p in zip(w, freqs, phases)); sig = sig / max(np.abs(sig).max(), 1e-9)
    return center + amp * sig, dict(freq_hz=freqs.round(4).tolist())


def chirp_signal(rng, T, center, amp, speed, hz=CONTROL_HZ):
    t = np.arange(T) / hz; seconds = T / hz; f1 = max(0.002, speed / (2 * np.pi * max(amp, 1e-3))); f0 = f1 / rng.uniform(5, 20)
    if rng.uniform() < 0.5: f0, f1 = f1, f0   # up- or down-chirp
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t ** 2)
    return center + amp * np.sin(phase + rng.uniform(0, 2 * np.pi)), dict(f0=round(f0, 4), f1=round(f1, 4))


class Generator:
    def __init__(self, spec: Spec, seconds=20.0, hz=CONTROL_HZ, hold_mode="library"):
        """hold_mode: 'library' holds idle fingers at neutral/open/grasp poses (main set); 'balanced' holds them at a random
        mixed posture (mostly bent), used by the balance top-up to reduce the share of straight fingers."""
        self.s = spec; self.T = int(round(seconds * hz)); self.hz = hz; self.hold_mode = hold_mode

    # ---------- helpers ----------
    def amplitude(self, rng, c, speed, frac=None):
        """Amplitude (deg) for channel c: a fraction of the half range, limited to what the speed can traverse in the episode."""
        frac = frac if frac is not None else AMPLITUDE_FRACTIONS[int(rng.integers(len(AMPLITUDE_FRACTIONS)))]
        reach = speed * self.T / self.hz / 2   # half the distance the regime can travel in one episode
        return float(min(frac * self.s.range[c] / 2, max(1.0, reach))), frac

    def hold_pose(self, rng):
        """A safe hold for fingers that do not move: neutral or a library pose, plus a little noise (or a random mixed pose)."""
        if self.hold_mode == "balanced" and rng.uniform() < 0.85: return self.random_valid_pose(rng), "random_mixed"
        name = ["neutral", "neutral", "open", "cylindrical_grasp", "spherical_grasp"][int(rng.integers(5))]
        return np.clip(self.s.poses[name] + rng.normal(0, 2.0, self.s.n), self.s.low, self.s.high), name

    def channel_signal(self, rng, kind, c, speed, held):
        s, T = self.s, self.T
        a, frac = self.amplitude(rng, c, speed)
        center = rng.uniform(s.low[c] + a, s.high[c] - a) if s.range[c] > 2 * a else s.mid[c]
        if kind == "multisine":
            sig, p = sines(rng, T, center, a, speed)
        elif kind == "chirp":
            sig, p = chirp_signal(rng, T, center, a, speed)
        elif kind in ("limit_sweep", "limit_triangle"):
            lo, hi = s.low[c] + 0.02 * s.range[c], s.high[c] - 0.02 * s.range[c]
            dwell = float(rng.choice([0.0, 0.25, 0.5, 1.0])); seq = [lo, hi] if rng.uniform() < 0.5 else [hi, lo]
            keys = [(np.full(s.n, v), dwell) for v in seq]
            x0 = held[c] if kind == "limit_sweep" else rng.uniform(lo, hi)
            sig = keyframe_path(np.full(s.n, x0), keys, T, speed, rng, slow=(1.0, 1.0) if kind == "limit_triangle" else (1.0, 1.3))[:, 0]
            p = dict(dwell_s=dwell)
        elif kind == "random_walk":
            step = a; v = held[c]; keys = []
            for _ in range(200):
                v = float(np.clip(v + rng.normal(0, step), s.low[c], s.high[c])); keys.append((np.full(s.n, v), float(rng.choice([0.0, 0.1, 0.25, 0.5, 1.0]))))
            sig = keyframe_path(np.full(s.n, held[c]), keys, T, speed, rng)[:, 0]; p = dict(step_deg=round(step, 2))
        else:
            raise ValueError(kind)
        return np.clip(sig, s.low[c], s.high[c]), dict(kind=kind, amp_deg=round(a, 2), amp_frac=frac, center_deg=round(float(center), 2), **p)

    def maybe_wrist(self, rng, out, speed, prob):
        if self.s.wrist and rng.uniform() < prob:
            for c in self.s.wrist: out[:, c], _ = self.channel_signal(rng, "multisine", c, speed, out[0])
            return True
        return False

    def pose_cycle(self, rng, poses, speed, dwell_choices=(0.0, 0.25, 0.5, 1.0), jitter=3.0):
        """Cycle through library poses with min-jerk, starting at the first pose. When the first transition is longer than an
        episode (slow regimes), the episode starts at a random point of that transition, so episodes cover different segments."""
        keys = [(np.clip(self.s.poses[p] + rng.normal(0, jitter, self.s.n), self.s.low, self.s.high), float(rng.choice(dwell_choices))) for p in poses * 40]
        start = keys[0][0]; path = keyframe_path(start, keys[1:], self.T * 40 + 1, speed, rng)
        first = MINJERK_PEAK * np.abs(keys[1][0] - start).max() / speed * self.hz
        offset = int(rng.integers(0, int(first - self.T) + 1)) if first > self.T else int(rng.integers(0, max(1, int(keys[0][1] * self.hz) + 1)))
        return path[offset:offset + self.T]

    # ---------- families ----------
    def generate(self, rng, family, speed):
        """Returns (requested (T,20) deg float32, info dict)."""
        s, T = self.s, self.T; info = dict(family=family, group=FAMILY_GROUP[family], speed_deg_s=float(speed))
        if family.startswith("isolated_"):
            finger = s.fingers[int(rng.integers(len(s.fingers)))]; held, hold_name = self.hold_pose(rng); out = np.repeat(held[None], T, 0)
            kind = family.replace("isolated_", ""); chans = {}
            for c in s.finger[finger]: out[:, c], chans[s.names[c]] = self.channel_signal(rng, kind, c, speed, held)
            info.update(active_fingers=[finger], hold_pose=hold_name, channels=chans); info["wrist_moving"] = self.maybe_wrist(rng, out, speed, 0.25)
        elif family.startswith("pair_"):
            pair = list(s.L["pairs"][int(rng.integers(len(s.L["pairs"])))]); held, hold_name = self.hold_pose(rng); out = np.repeat(held[None], T, 0)
            kind = family.replace("pair_", ""); info.update(active_fingers=pair, hold_pose=hold_name)
            if kind in ("multisine", "random_walk"):
                chans = {}
                for c in s.ch(pair): out[:, c], chans[s.names[c]] = self.channel_signal(rng, kind, c, speed, held)
                info["channels"] = chans
            else:   # one shared phase s(t) in [0, 1] drives both fingers between an open and a closed (contact-near) pose
                closed = self.pair_closed_pose(pair); depth = rng.uniform(0.6, 1.0)
                travel = np.abs(closed - s.poses["open"])[s.ch(pair)].max() * depth
                f = max(0.002, speed / (np.pi * max(travel, 1e-3))) * rng.uniform(0.4, 1.0)
                tt = np.arange(T) / self.hz; phase = rng.uniform(0, 2 * np.pi)
                base_sig = 0.5 - 0.5 * np.cos(2 * np.pi * f * tt + phase)
                ramp = depth * base_sig
                for i, finger in enumerate(pair):
                    sig = ramp if (kind == "synchronized" or i == 0) else depth * (1 - base_sig)
                    for c in s.finger[finger]: out[:, c] = s.poses["open"][c] + sig * (closed[c] - s.poses["open"][c])
                info.update(depth=round(depth, 3), freq_hz=round(f, 4))
            info["wrist_moving"] = self.maybe_wrist(rng, out, speed, 0.25)
        elif family.startswith("pose_"):
            seq = {"pose_open_to_fist": ["open", "fist_thumb_out", "fist", "fist_thumb_out"], "pose_fist_to_open": ["fist", "fist_thumb_out", "open", "fist_thumb_out"],
                   "pose_victory": ["open", "victory_thumb_out", "victory", "victory_thumb_out"],
                   "pose_thumb_index_pinch": ["open", "thumb_index_pinch"], "pose_thumb_middle_pinch": ["open", "thumb_middle_pinch"],
                   "pose_tripod": ["neutral", "tripod"], "pose_thumb_opposition": ["open", "thumb_opposition"]}.get(family)
            if seq:
                out = self.pose_cycle(rng, seq, speed)   # the thumb leaves before the fingers open and wraps after they close
                info.update(poses=seq, active_fingers=s.fingers)
            else:   # finger wave: index..little flex and extend in sequence with a lag; thumb holds
                held, hold_name = self.hold_pose(rng); out = np.repeat(held[None], T, 0); amp = rng.uniform(0.4, 1.0)
                lag = rng.uniform(0.15, 0.5); seg = 1.0
                for k, finger in enumerate(s.L["wave_fingers"]):
                    for c in s.finger[finger]:
                        name = s.names[c]
                        if name in s.L["wave_skip"]: continue
                        top = s.low[c] + amp * s.range[c]; travel = top - s.low[c]
                        seg = MINJERK_PEAK * travel / speed * rng.uniform(1.0, 1.2)
                        period = 2 * seg + 4 * lag + rng.uniform(0, 1.0)
                        tt = (np.arange(T) / self.hz + rng.uniform(0, period) * 0 - k * lag) % period
                        up = min_jerk(tt / seg); down = 1 - min_jerk((tt - seg - lag) / seg)
                        out[:, c] = s.low[c] + travel * np.where(tt < seg + lag, up, down)
                info.update(active_fingers=list(s.L["wave_fingers"]), hold_pose=hold_name, amp_frac=round(amp, 3), lag_s=round(lag, 3))
            info["wrist_moving"] = self.maybe_wrist(rng, out, speed, 0.3)
        elif family == "local_pose_perturbation":
            ref = ["neutral", "open", "fist", "victory", "thumb_index_pinch", "thumb_middle_pinch", "tripod", "thumb_opposition",
                   "cylindrical_grasp", "spherical_grasp", "random"][int(rng.integers(11))]
            q_ref = self.random_valid_pose(rng) if ref == "random" else s.poses[ref]
            start = np.clip(q_ref + rng.uniform(-5, 5, s.n), s.low, s.high); out = np.repeat(start[None], T, 0)
            sub = ["walk", "sine", "chirp", "targets"][int(rng.integers(4))]; box = rng.uniform(3.0, 8.0)
            lo, hi = np.maximum(q_ref - box, s.low), np.minimum(q_ref + box, s.high)
            if sub in ("sine", "chirp"):
                for c in range(s.n):
                    a = (hi[c] - lo[c]) / 2 * rng.uniform(0.3, 1.0); ctr = (hi[c] + lo[c]) / 2
                    out[:, c] = (sines if sub == "sine" else chirp_signal)(rng, T, ctr, a, speed)[0]
                out[0] = out[0]   # start where the signal starts
            else:
                keys = [(rng.uniform(lo, hi), float(rng.choice([0.0, 0.1, 0.25, 0.5] if sub == "walk" else [0.25, 0.5, 1.0]))) for _ in range(400)]
                out = keyframe_path(start, keys, T, speed, rng, slow=(1.0, 1.2))
            info.update(reference_pose=ref, sub_program=sub, box_deg=round(box, 2), active_fingers=s.fingers)
            info["reference_deg"] = np.round(q_ref, 2).tolist()
        elif family == "lateral_limit_sweep":   # 1-2 spread / thumb-rotation / wrist channels sweep to both limits, the rest hold
            held, hold_name = self.hold_pose(rng); out = np.repeat(held[None], T, 0); chans = {}
            names = list(rng.choice(s.L["lateral"], int(rng.integers(1, 3)), replace=False)); kind = ["limit_sweep", "limit_triangle"][int(rng.integers(2))]
            for n in names:
                c = s.index[n]; out[:, c], chans[n] = self.channel_signal(rng, kind, c, speed, held)
            fingers = sorted({f for f, ch in s.L["finger_channels"].items() for n in names if n in ch})
            info.update(active_fingers=fingers or ["wrist"], hold_pose=hold_name, channels=chans)
        elif family == "constraint_aware_random":
            sub = ["walk", "sine", "chirp", "group"][int(rng.integers(4))]
            if sub == "walk":
                keys = [(self.random_valid_pose(rng), float(rng.choice([0.0, 0.25, 0.5, 1.0]))) for _ in range(60)]
                out = keyframe_path(keys[-1][0], keys, T, speed, rng)
            elif sub in ("sine", "chirp"):
                ctr = self.random_valid_pose(rng); out = np.repeat(ctr[None], T, 0); fr = AMPLITUDE_FRACTIONS[int(rng.integers(3))]
                for c in range(s.n):
                    a = min(fr * s.range[c] / 2, ctr[c] - s.low[c], s.high[c] - ctr[c], max(1.0, speed * T / self.hz / 2))
                    if a > 0.2: out[:, c] = (sines if sub == "sine" else chirp_signal)(rng, T, ctr[c], a, speed)[0]
            else:   # a random group of 2-4 fingers follows one shared phase between two valid poses
                fingers = list(rng.choice(s.fingers, int(rng.integers(2, len(s.fingers))), replace=False)); a0, a1 = self.random_valid_pose(rng), self.random_valid_pose(rng)
                out = np.repeat(a0[None], T, 0); travel = np.abs(a1 - a0)[s.ch(fingers)].max()
                f = max(0.002, speed / (np.pi * max(travel, 1e-3))) * rng.uniform(0.4, 1.0); ph = 0.5 - 0.5 * np.cos(2 * np.pi * f * np.arange(T) / self.hz + rng.uniform(0, 2 * np.pi))
                for c in s.ch(fingers): out[:, c] = a0[c] + ph * (a1[c] - a0[c])
                info["group_fingers"] = fingers
            info.update(sub_program=sub, active_fingers=s.fingers); info["wrist_moving"] = self.maybe_wrist(rng, out, speed, 0.4)
        else:
            raise ValueError(family)
        info.setdefault("active_fingers", s.fingers)
        return np.clip(out, s.low, s.high).astype(np.float32), info

    def pair_closed_pose(self, pair):
        s = self.s
        if "thumb" in pair:
            other = [f for f in pair if f != "thumb"][0]
            return s.poses[s.L["thumb_pair_pose"][other]].copy()
        q = s.poses["open"].copy(); f = s.poses["fist"]
        for finger in pair:
            for c in s.finger[finger]:
                if s.names[c] not in s.L["closed_skip"]: q[c] = f[c] * 0.8
        return q

    def random_valid_pose(self, rng):
        """Candidate from mixing two library poses plus noise; validity is enforced by the program checker."""
        names = [k for k in self.s.poses if not k.endswith("_thumb_out")]; a, b = rng.choice(names, 2, replace=False); w = rng.uniform()
        q = w * self.s.poses[a] + (1 - w) * self.s.poses[b] + rng.normal(0, 8.0, self.s.n) * self.s.range / self.s.range.max()
        return np.clip(q, self.s.low, self.s.high)


def apply_slew(requested, previous, slew_deg_s, low, high, hz=CONTROL_HZ):
    """Applied target (deg): previous moved toward requested by at most slew/hz per step, within limits."""
    step = slew_deg_s / hz
    return np.clip(previous + np.clip(requested - previous, -step, step), low, high)


def applied_path(requested, start, slew_deg_s, low, high, hz=CONTROL_HZ):
    out = np.empty_like(requested, dtype=np.float64); prev = np.asarray(start, np.float64)
    for t in range(len(requested)):
        prev = apply_slew(requested[t], prev, slew_deg_s, low, high, hz); out[t] = prev
    return out


def passive_from_active(a_deg, joint_names, action_names):
    """Full 24-joint estimate from 20 active targets: J0 = clip(1.13 * J1, 0, 90) deg (fit to simulator data, r >= 0.995)."""
    J = [j.replace("robot0_", "") for j in joint_names]; A = [n.replace("robot0_", "") for n in action_names]
    q = np.zeros(a_deg.shape[:-1] + (24,)); q[..., [J.index(n) for n in A]] = a_deg
    for f in ("FF", "MF", "RF", "LF"): q[..., J.index(f + "J0")] = np.clip(1.13 * q[..., J.index(f + "J1")], 0, 90)
    return q


class ProgramChecker:
    """FK + visual-link penetration on the slew-limited target path, on GPU."""

    def __init__(self, geometry, fk, joint_names, action_names, threshold_m, device, every=4):
        import torch
        self.torch = torch; self.g = geometry; self.fk = fk; self.J = joint_names; self.A = action_names
        self.threshold = threshold_m; self.device = device; self.every = every

    def max_penetration(self, targets_deg):
        """targets_deg (..., 20) -> (...) max penetration in metres."""
        torch = self.torch; flat = targets_deg.reshape(-1, 20)
        q = torch.as_tensor(passive_from_active(flat, self.J, self.A), dtype=torch.float32, device=self.device)
        out = []
        for i in range(0, len(q), 1024):
            T = self.fk(q[i:i + 1024]); out.append(self.g.penetration_from_transforms(self.fk.body_names, T).amax(1))
        return (torch.cat(out).cpu().numpy() if out else np.zeros(0)).reshape(targets_deg.shape[:-1])

    def repair(self, programs, grid=17, window_s=0.5, hz=CONTROL_HZ):
        """Scale each program toward its first frame where it would overlap: r(t) = r0 + lam(t) (r(t) - r0).

        lam(t) is the largest grid value in [0, 1] whose pose passes the threshold, then min-filtered and averaged over
        `window_s` (the average of a min-filter with the same window never exceeds the pointwise valid value, so the
        smoothing keeps validity on the grid). Programs keep their timing and only lose amplitude where contact is near.
        programs (N, T, 20) deg -> (repaired, lam (N, T))."""
        N, T, _ = programs.shape; lams = np.linspace(0, 1, grid); r0 = programs[:, :1]
        lam = np.ones((N, T)); bad = self.max_penetration(programs) > self.threshold          # full amplitude first
        if bad.any():
            n_i, t_i = np.nonzero(bad); d = (programs[n_i, t_i] - r0[n_i, 0])
            cand = r0[n_i, 0][:, None] + lams[None, :, None] * d[:, None]                     # (M, G, 20)
            ok = self.max_penetration(cand) <= self.threshold; ok[:, 0] = True                  # lam = 0: start pose (checked separately)
            first_bad = np.where((~ok).any(-1), (~ok).argmax(-1), grid)
            lam[n_i, t_i] = lams[np.maximum(first_bad - 1, 0)]
        w = max(1, int(round(window_s * hz))); pad = w // 2
        lp = np.pad(lam, ((0, 0), (pad, pad)), mode="edge")
        mn = np.lib.stride_tricks.sliding_window_view(lp, w, axis=1).min(-1)[:, :T]
        mp = np.pad(mn, ((0, 0), (pad, pad)), mode="edge")
        lam_s = np.minimum(np.lib.stride_tricks.sliding_window_view(mp, w, axis=1).mean(-1)[:, :T], lam)
        return (r0 + lam_s[..., None] * (programs - r0)).astype(np.float32), lam_s

    def check_paths(self, paths):
        """paths (N, T, 20) applied targets -> (N,) worst penetration over every `every`-th step and the last step."""
        idx = np.unique(np.r_[np.arange(0, paths.shape[1], self.every), paths.shape[1] - 1])
        return self.max_penetration(paths[:, idx]).max(1)


def schedule(n_episodes_per_speed, speeds, rng, mode="shares"):
    """List of (speed, family) with family counts per speed following the group shares (or uniform for the pilot)."""
    out = []
    for sp in speeds:
        n = n_episodes_per_speed[sp] if isinstance(n_episodes_per_speed, dict) else n_episodes_per_speed
        if mode == "uniform":
            fams = [f for f in FAMILIES for _ in range(n // len(FAMILIES))]
        else:
            fams = []
            for g, share in GROUP_SHARES.items():
                k = int(round(n * share)); fs = GROUPS[g]; fams += [fs[i % len(fs)] for i in range(k)]
            fams = fams[:n] + [FAMILIES[i % len(FAMILIES)] for i in range(n - len(fams))]
        out += [(sp, f) for f in fams]
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def split_for_seed(seed):
    return "validation" if seed % 10 == 8 else "test" if seed % 10 == 9 else "train"
