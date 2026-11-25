"""Hand-only excitation programs, episode labels, recorder, and verifier (schema 2.0.0-excitation).

Programs return requested active-joint targets (T, 20) in radians at 60 Hz. The collector applies a
per-episode slew limit and the joint limits; both are recorded and checked by `verify`.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import h5py
import numpy as np

SCHEMA = "2.1.0-excitation"
CONTROL_HZ = 60   # set by set_control_hz(); every program uses the module value at call time


def set_control_hz(hz):
    global CONTROL_HZ; CONTROL_HZ = int(hz)


def episode_seconds(speed_deg, minimum=10.0, widest_range_deg=120.0, traversals=2.5):
    """Long enough for the widest joint to traverse its range 2.5 times at this speed."""
    return float(max(minimum, traversals * widest_range_deg / speed_deg))
UNITS = {"FF": ["FFJ3", "FFJ2", "FFJ1"], "MF": ["MFJ3", "MFJ2", "MFJ1"], "RF": ["RFJ3", "RFJ2", "RFJ1"],
         "LF": ["LFJ4", "LFJ3", "LFJ2", "LFJ1"], "TH": ["THJ4", "THJ3", "THJ2", "THJ1", "THJ0"], "WR": ["WRJ1", "WRJ0"]}
FLEXION = ["FFJ2", "FFJ1", "MFJ2", "MFJ1", "RFJ2", "RFJ1", "LFJ2", "LFJ1"]
ABDUCTION = ["FFJ3", "MFJ3", "RFJ3", "LFJ3", "LFJ4"]
AMPLITUDE_FRACTIONS = [0.02, 0.05, 0.15, 0.4, 1.0]     # of the half range
SPEEDS_DEG = [1, 3, 6, 10, 15, 20, 25, 30]              # deg/s ramp speed and slew limit (user-specified grid)
SPEEDS = [float(np.deg2rad(v)) for v in SPEEDS_DEG]       # rad/s
DWELLS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]               # s
FAMILIES = ["limit_sweep", "limit_triangle", "multisine", "chirp", "steps", "random_walk",
            "unit_isolated", "unit_pair", "curl", "spread", "thumb_opposition", "finger_wave", "wrist", "mixed"]


class Limits:
    def __init__(self, action_names, low, high):
        self.names = [n.replace("robot0_", "") for n in action_names]
        self.low = np.asarray(low, np.float64); self.high = np.asarray(high, np.float64)
        self.range = self.high - self.low; self.mid = (self.high + self.low) / 2
        self.index = {n: i for i, n in enumerate(self.names)}

    def channels(self, names):
        return [self.index[n] for n in names]


def choose(rng, values):
    return values[int(rng.integers(len(values)))]


def posture(rng, lim, interior=0.0):
    """Random hold posture; `interior` shrinks toward the middle of the range."""
    return lim.low + lim.range * rng.uniform(interior, 1 - interior, size=len(lim.low))


def interpolate(keys, T):
    """Piecewise-linear keyframes [(t_s, value)] -> (T,) at 60 Hz, holding the last value."""
    t = np.arange(T) / CONTROL_HZ
    times = np.array([k[0] for k in keys]); values = np.array([k[1] for k in keys])
    return np.interp(t, times, values)


def ramp_keys(start_t, start_v, targets, speed, dwell):
    """Keyframes ramping through targets at `speed` rad/s, holding `dwell` s at each."""
    keys = [(start_t, start_v)]; t, v = start_t, start_v
    for target in targets:
        t += abs(target - v) / speed; keys.append((t, target)); v = target
        t += dwell; keys.append((t, v))
    return keys, t


def limit_sweep(rng, lim, T, group_size, speed, dwell, held):
    """Groups of channels visit low and high limits in turn; others hold a posture."""
    seconds = T / CONTROL_HZ; out = np.repeat(held[None], T, 0).copy()
    channel_keys = {c: [(0.0, held[c])] for c in range(20)}; ends = {c: 0.0 for c in range(20)}
    t0 = 0.0
    while t0 < seconds:
        order = rng.permutation(20)
        for g in range(0, 20, group_size):
            group = order[g:g + group_size]; t_end = t0
            for c in group:
                first = choose(rng, ["low", "high"])
                seq = [lim.low[c], lim.high[c]] if first == "low" else [lim.high[c], lim.low[c]]
                seq.append(rng.uniform(lim.low[c], lim.high[c]))
                keys, t_c = ramp_keys(t0, channel_keys[c][-1][1], seq, speed, dwell)
                channel_keys[c] += keys[1:]; ends[c] = t_c; t_end = max(t_end, t_c)
            t0 = t_end
            if t0 >= seconds: break
    for c in range(20): out[:, c] = interpolate(channel_keys[c], T)
    return out


def limit_triangle(rng, lim, T, speed, dwell, subset=None):
    """Every (or a subset of) channel(s) runs low<->high triangles with holds, random phase."""
    seconds = T / CONTROL_HZ; out = np.zeros((T, 20)); subset = range(20) if subset is None else subset
    held = posture(rng, lim)
    for c in range(20):
        if c not in subset: out[:, c] = held[c]; continue
        period = 2 * lim.range[c] / speed + 2 * dwell; phase = rng.uniform(0, period)
        keys = [(-phase, lim.low[c])]; t = -phase
        while t < seconds:
            t += lim.range[c] / speed; keys.append((t, lim.high[c])); t += dwell; keys.append((t, lim.high[c]))
            t += lim.range[c] / speed; keys.append((t, lim.low[c])); t += dwell; keys.append((t, lim.low[c]))
        out[:, c] = interpolate(keys, T)
    return out


def multisine(rng, lim, T, amp_frac, speed, subset=None, held=None):
    """1-3 sinusoids per channel; frequency bounded so the slew limit is respected."""
    t = np.arange(T) / CONTROL_HZ; out = np.zeros((T, 20)); subset = range(20) if subset is None else subset
    held = posture(rng, lim) if held is None else held; params = {}
    for c in range(20):
        if c not in subset: out[:, c] = held[c]; continue
        a = amp_frac * lim.range[c] / 2 * rng.uniform(0.6, 1.0)
        center = rng.uniform(lim.low[c] + a, lim.high[c] - a)
        n = int(rng.integers(1, 4)); fmax = min(5.0, speed / (2 * np.pi * max(a, 1e-3)))
        freqs = np.exp(rng.uniform(np.log(0.1), np.log(max(0.11, fmax)), n)); phases = rng.uniform(0, 2 * np.pi, n)
        weights = rng.uniform(0.3, 1.0, n); signal = sum(w * np.sin(2 * np.pi * f * t + p) for w, f, p in zip(weights, freqs, phases))
        signal = signal / max(np.abs(signal).max(), 1e-6) * a
        out[:, c] = center + signal; params[lim.names[c]] = dict(center=float(center), amplitude=float(a), freq_hz=freqs.round(3).tolist())
    return out, params


def chirp(rng, lim, T, amp_frac, speed, subset=None, held=None):
    t = np.arange(T) / CONTROL_HZ; seconds = T / CONTROL_HZ; out = np.zeros((T, 20)); subset = range(20) if subset is None else subset
    held = posture(rng, lim) if held is None else held; params = {}
    for c in range(20):
        if c not in subset: out[:, c] = held[c]; continue
        a = amp_frac * lim.range[c] / 2; center = rng.uniform(lim.low[c] + a, lim.high[c] - a)
        f0 = min(0.1, speed / (2 * np.pi * max(a, 1e-3)) / 2); f1 = min(6.0, max(2 * f0, speed / (2 * np.pi * max(a, 1e-3))))
        phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t ** 2)
        out[:, c] = center + a * np.sin(phase + rng.uniform(0, 2 * np.pi))
        params[lim.names[c]] = dict(center=float(center), amplitude=float(a), f0=f0, f1=float(f1))
    return out, params


def steps(rng, lim, T, dwell, amp_frac=None, subset=None, held=None):
    """Piecewise-constant targets: uniform in the limits, or a random walk with step scale amp_frac."""
    out = np.zeros((T, 20)); subset = range(20) if subset is None else subset; held = posture(rng, lim) if held is None else held
    for c in range(20):
        if c not in subset: out[:, c] = held[c]; continue
        t = 0; v = rng.uniform(lim.low[c], lim.high[c])
        while t < T:
            n = max(1, int(round(rng.uniform(0.5, 1.5) * dwell * CONTROL_HZ)))
            if amp_frac is None: v = rng.uniform(lim.low[c], lim.high[c])
            else: v = np.clip(v + rng.normal(0, amp_frac * lim.range[c] / 2), lim.low[c], lim.high[c])
            out[t:t + n, c] = v; t += n
    return out


def coordinated(rng, lim, T, kind, speed, dwell, amp_frac):
    t = np.arange(T) / CONTROL_HZ; held = posture(rng, lim, interior=0.1); out = np.repeat(held[None], T, 0).copy()
    def sweep(c, low, high, phase, freq):
        return (low + high) / 2 + (high - low) / 2 * np.sin(2 * np.pi * freq * t + phase)
    if kind == "curl":
        freq = min(2.0, speed / (np.pi * 1.571)) * rng.uniform(0.3, 1.0); phase = rng.uniform(0, 2 * np.pi)
        for name in FLEXION:
            c = lim.index[name]; out[:, c] = sweep(c, lim.low[c], lim.low[c] + amp_frac * lim.range[c], phase + rng.uniform(-0.3, 0.3), freq)
        for name in ABDUCTION:
            c = lim.index[name]; out[:, c] = held[c] if rng.uniform() < 0.5 else sweep(c, lim.low[c], lim.high[c], rng.uniform(0, 2 * np.pi), freq * 0.5)
    elif kind == "spread":
        freq = min(2.0, speed / (np.pi * 0.8)) * rng.uniform(0.3, 1.0)
        for name in ABDUCTION:
            c = lim.index[name]; out[:, c] = sweep(c, lim.low[c], lim.high[c], rng.uniform(0, 2 * np.pi), freq)
        for name in FLEXION:
            c = lim.index[name]; out[:, c] = rng.uniform(lim.low[c], lim.low[c] + 0.5 * lim.range[c])
    elif kind == "thumb_opposition":
        freq = min(1.5, speed / (np.pi * 2.1)) * rng.uniform(0.3, 1.0)
        for name, f in [("THJ4", 1.0), ("THJ3", 1.0), ("THJ2", 2.0), ("THJ1", 1.5), ("THJ0", 1.5)]:
            c = lim.index[name]; out[:, c] = sweep(c, lim.low[c], lim.high[c], rng.uniform(0, 2 * np.pi), freq * f)
        for name in FLEXION:
            c = lim.index[name]; out[:, c] = rng.uniform(lim.low[c] + 0.2 * lim.range[c], lim.low[c] + 0.7 * lim.range[c])
    elif kind == "finger_wave":
        lag = rng.uniform(0.1, 0.5); period = max(1.0, 2 * 1.571 / speed) * rng.uniform(1.0, 2.0)
        for k, finger in enumerate(["FF", "MF", "RF", "LF"]):
            for name in [f"{finger}J2", f"{finger}J1"]:
                c = lim.index[name]; keys = [(0.0, lim.low[c])]; tt = k * lag
                while tt < T / CONTROL_HZ:
                    keys += [(tt, lim.low[c]), (tt + period / 3, lim.high[c] * amp_frac), (tt + period / 3 + dwell, lim.high[c] * amp_frac), (tt + 2 * period / 3 + dwell, lim.low[c])]
                    tt += period + 4 * lag
                out[:, c] = interpolate(sorted(keys), T)
    elif kind == "wrist":
        for name in ["WRJ1", "WRJ0"]:
            c = lim.index[name]; freq = min(2.0, speed / (np.pi * lim.range[c])) * rng.uniform(0.2, 1.0)
            out[:, c] = sweep(c, lim.low[c], lim.high[c], rng.uniform(0, 2 * np.pi), freq)
        if rng.uniform() < 0.5:
            sig, _ = multisine(rng, lim, T, amp_frac, speed, subset=[lim.index[n] for n in FLEXION], held=held)
            for name in FLEXION: out[:, lim.index[name]] = sig[:, lim.index[name]]
    return out


def mixed(rng, lim, T, speed=None):
    out = np.zeros((T, 20)); choice = {}
    for c in range(20):
        kind = choose(rng, ["hold", "sine", "steps", "walk", "triangle", "chirp"]); amp = choose(rng, AMPLITUDE_FRACTIONS); speed = choose(rng, SPEEDS) if speed is None else speed; dwell = choose(rng, DWELLS)
        if kind == "hold": out[:, c] = rng.uniform(lim.low[c], lim.high[c])
        elif kind == "sine": out[:, c] = multisine(rng, lim, T, amp, speed, subset=[c])[0][:, c]
        elif kind == "steps": out[:, c] = steps(rng, lim, T, dwell, subset=[c])[:, c]
        elif kind == "walk": out[:, c] = steps(rng, lim, T, dwell, amp_frac=amp, subset=[c])[:, c]
        elif kind == "triangle": out[:, c] = limit_triangle(rng, lim, T, speed, dwell, subset=[c])[:, c]
        else: out[:, c] = chirp(rng, lim, T, amp, speed, subset=[c])[0][:, c]
        choice[lim.names[c]] = dict(kind=kind, amp_frac=amp, speed=speed, dwell=dwell)
    return out, choice


def generate(rng, lim, T, family=None, speed=None):
    """One episode program. Returns (requested (T,20) float32, slew_limit rad/s, parameter dict)."""
    family = family or choose(rng, FAMILIES)
    amp = choose(rng, AMPLITUDE_FRACTIONS); speed = choose(rng, SPEEDS) if speed is None else float(speed); dwell = choose(rng, DWELLS)
    params = dict(family=family, amp_frac=amp, speed=speed, speed_deg_s=float(np.rad2deg(speed)), dwell=dwell)
    if family == "limit_sweep":
        group = choose(rng, [1, 2, 4, 5, 20]); held = posture(rng, lim); out = limit_sweep(rng, lim, T, group, speed, dwell, held); params.update(group_size=group)
    elif family == "limit_triangle": out = limit_triangle(rng, lim, T, speed, dwell)
    elif family == "multisine": out, p = multisine(rng, lim, T, amp, speed); params.update(channels=p)
    elif family == "chirp": out, p = chirp(rng, lim, T, amp, speed); params.update(channels=p)
    elif family == "steps": out = steps(rng, lim, T, dwell)
    elif family == "random_walk": out = steps(rng, lim, T, dwell, amp_frac=amp)
    elif family in ("unit_isolated", "unit_pair"):
        units = list(UNITS); chosen = [choose(rng, units)] if family == "unit_isolated" else list(rng.choice(units, 2, replace=False))
        subset = [lim.index[n] for u in chosen for n in UNITS[u]]; sub = choose(rng, ["triangle", "multisine", "chirp", "steps", "walk"])
        held = posture(rng, lim)
        if sub == "triangle": out = limit_triangle(rng, lim, T, speed, dwell, subset=subset); out[:, [c for c in range(20) if c not in subset]] = held[[c for c in range(20) if c not in subset]]
        elif sub == "multisine": out, _ = multisine(rng, lim, T, amp, speed, subset=subset, held=held)
        elif sub == "chirp": out, _ = chirp(rng, lim, T, amp, speed, subset=subset, held=held)
        elif sub == "steps": out = steps(rng, lim, T, dwell, subset=subset, held=held)
        else: out = steps(rng, lim, T, dwell, amp_frac=amp, subset=subset, held=held)
        params.update(units=chosen, sub_program=sub)
    elif family in ("curl", "spread", "thumb_opposition", "finger_wave", "wrist"): out = coordinated(rng, lim, T, family, speed, dwell, amp)
    elif family == "mixed": out, p = mixed(rng, lim, T, speed); params.update(channels=p)
    else: raise ValueError(family)
    out = np.clip(out, lim.low, lim.high)
    slew = speed
    params["slew_limit_rad_s"] = float(slew)
    return out.astype(np.float32), float(slew), params


def apply_slew(requested, previous, slew, lim):
    step = slew / CONTROL_HZ
    return np.clip(previous + np.clip(requested - previous, -step, step), lim.low, lim.high)


def labels(states, actions, mapping, limit_tol=0.03):
    """Achieved-coverage labels: which joints actually reached their limits, plus validity."""
    q, v = states["joint_pos"], states["joint_vel"]; limits = np.asarray(mapping["joint_limits"]); rng_ = limits[:, 1] - limits[:, 0]
    active, passive = mapping["active_indices"], mapping["passive_indices"]
    finite = all(np.isfinite(x).all() for x in states.values()) and np.isfinite(actions).all()
    excess = np.maximum(limits[:, 0] - q, q - limits[:, 1]).max(-1)
    force = np.linalg.norm(states["fingertip_net_force"], axis=-1).max(-1)          # includes finger-finger / finger-palm self-contact
    object_force = np.linalg.norm(states["fingertip_object_force"], axis=-1).max(-1)  # cube contact only; must be zero
    constraint = (excess <= 0.08) & (np.abs(v).max(-1) <= 40) & (force <= 40)
    qmin, qmax = q.min(0), q.max(0)
    low_hit = qmin <= limits[:, 0] + limit_tol; high_hit = qmax >= limits[:, 1] - limit_tol
    coverage = (qmax - qmin) / rng_
    result = dict(physics_valid=bool(finite and excess.max() < 0.5 and np.abs(v).max() < 200), constraint_ok=bool(constraint.all()),
        contact_free=bool(object_force.max() == 0.0), self_contact=bool(force.max() > 0.0),
        max_self_contact_force_N=float(force.max()), self_contact_fraction=float((force > 0).mean()), task_success=None,
        coverage_fraction=coverage.round(4).tolist(), low_limit_reached=low_hit.tolist(), high_limit_reached=high_hit.tolist(),
        active_channels_both_limits=int(np.sum(low_hit[active] & high_hit[active])), passive_joints_both_limits=int(np.sum(low_hit[passive] & high_hit[passive])),
        mean_active_coverage=float(coverage[active].mean()), mean_passive_coverage=float(coverage[passive].mean()),
        joint_velocity_rms=np.sqrt((v ** 2).mean(0)).round(4).tolist(), max_abs_velocity_rad_s=float(np.abs(v).max()),
        max_limit_excess_rad=float(excess.max()), max_fingertip_net_force_N=float(force.max()),
        excitation_success=bool(finite and constraint.all() and coverage[active].mean() >= 0.1),
        constraint_ok_per_state=constraint,
        criteria=dict(limit_tolerance_rad=limit_tol, velocity_limit_rad_s=40, fingertip_force_limit_N=40, physics_valid_velocity_rad_s=200,
                      contact_free="no fingertip-object (cube) force; finger self-contact is allowed and labeled", excitation_rule="mean active coverage >= 0.1 of range"))
    return result


class ExcitationRecorder:
    def __init__(self, output, manifest):
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=False)
        self.partial = self.output / "trajectories.h5.partial"; self.file = h5py.File(self.partial, "w")
        self.file.attrs["schema_version"] = SCHEMA; self.file.attrs["manifest_json"] = json.dumps(manifest)
        self.manifest = manifest; self.index = []
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def episode(self, states, actions, requested, physics, metadata):
        actions, requested = np.asarray(actions, np.float32), np.asarray(requested, np.float32); n = len(actions)
        assert all(len(x) == n + 1 for x in states.values()) and requested.shape == actions.shape
        episode_id = f"episode_{len(self.index):06d}"; group = self.file.create_group(episode_id)
        group.attrs["metadata_json"] = json.dumps(metadata)
        for namespace, values in (("state", states), ("action", {"target_rad": actions, "requested_rad": requested})) + ((("physics", physics),) if physics else ()):
            sub = group.create_group(namespace)
            for key, value in values.items(): sub.create_dataset(key, data=np.asarray(value), compression="gzip", compression_opts=1, shuffle=True)
        lab = labels(states, actions, self.manifest["mapping"]); mask = lab.pop("constraint_ok_per_state")
        group.create_dataset("constraint_ok_per_state", data=mask); group.attrs["labels_json"] = json.dumps(lab)
        entry = {"episode_id": episode_id, "steps": n, **{k: v for k, v in metadata.items() if k != "program"}, "family": metadata["program"]["family"],
                 **{k: lab[k] for k in ["physics_valid", "constraint_ok", "contact_free", "self_contact", "max_self_contact_force_N", "excitation_success", "active_channels_both_limits", "passive_joints_both_limits", "mean_active_coverage", "mean_passive_coverage", "max_abs_velocity_rad_s", "max_limit_excess_rad"]}}
        self.index.append(entry); self.file.flush()
        with (self.output / "episodes.jsonl").open("a") as stream: stream.write(json.dumps(entry) + "\n")
        return entry

    def close(self):
        self.file.attrs["complete"] = True; self.file.flush(); self.file.close(); os.replace(self.partial, self.output / "trajectories.h5")
        views = {"all_valid": [e["episode_id"] for e in self.index if e["physics_valid"]], "constraint_ok": [e["episode_id"] for e in self.index if e["constraint_ok"]],
                 "excitation_success": [e["episode_id"] for e in self.index if e["excitation_success"]], "task_success": []}
        (self.output / "views.json").write_text(json.dumps(views, indent=2))


def split_for_seed(seed):
    return "validation" if seed % 10 == 8 else "test" if seed % 10 == 9 else "train"


def verify(path):
    """Independent check of timing, limits, per-episode slew, requested/applied alignment, labels, contact, splits."""
    counts = {"episodes": 0, "transitions": 0, "valid": 0, "constraint_ok": 0, "contact_free": 0, "self_contact": 0}; seed_splits = {}; families = {}
    with h5py.File(path, "r") as f:
        assert f.attrs.get("complete", False), "Incomplete file"; assert f.attrs["schema_version"] == SCHEMA
        manifest = json.loads(f.attrs["manifest_json"]); mapping = manifest["mapping"]; set_control_hz(manifest["control_hz"]); dec = manifest["physics_hz"] // CONTROL_HZ
        active = mapping["active_indices"]; limits = np.asarray(mapping["joint_limits"])[active]
        for name, ep in f.items():
            meta = json.loads(ep.attrs["metadata_json"]); lab = json.loads(ep.attrs["labels_json"])
            state = {k: v[:] for k, v in ep["state"].items()}; u = ep["action/target_rad"][:]; r = ep["action/requested_rad"][:]; n = len(u)
            assert all(len(v) == n + 1 for v in state.values()), name
            assert np.allclose(np.diff(state["time"]), 1 / CONTROL_HZ, atol=1e-8), name
            if "physics" in ep:
                assert all(len(v) == n * dec + 1 for v in ep["physics"].values()), name
                assert np.allclose(ep["physics/joint_pos"][::dec], state["joint_pos"]), name
                assert np.array_equal(ep["physics/action_index"][:], np.r_[-1, np.repeat(np.arange(n), dec)]), name
            assert ((u >= limits[:, 0] - 1e-6) & (u <= limits[:, 1] + 1e-6)).all(), "Command limit violation"
            slew = meta["program"]["slew_limit_rad_s"]; assert np.abs(np.diff(u, axis=0)).max() <= slew / CONTROL_HZ + 1e-5, "Command slew violation"
            prev = np.vstack((np.asarray(meta["initial_target_rad"]), u[:-1])); step = slew / CONTROL_HZ
            expected = np.clip(prev + np.clip(r - prev, -step, step), limits[:, 0], limits[:, 1])
            assert np.allclose(expected, u, atol=1e-5), "Requested/applied action alignment mismatch"
            recomputed = labels(state, u, mapping)
            for field in ("physics_valid", "constraint_ok", "contact_free", "self_contact", "excitation_success", "active_channels_both_limits"):
                assert lab[field] == recomputed[field], (name, field)
            assert np.array_equal(ep["constraint_ok_per_state"][:], recomputed["constraint_ok_per_state"]), name
            if not meta["object_present"]: assert lab["contact_free"], "Hand-only dataset must have no cube contact"
            if meta["seed"] in seed_splits: assert seed_splits[meta["seed"]] == meta["split"], "Seed leakage"
            seed_splits[meta["seed"]] = meta["split"]; families[meta["program"]["family"]] = families.get(meta["program"]["family"], 0) + 1
            counts["episodes"] += 1; counts["transitions"] += n; counts["valid"] += lab["physics_valid"]; counts["constraint_ok"] += lab["constraint_ok"]; counts["contact_free"] += lab["contact_free"]; counts["self_contact"] += lab["self_contact"]
    return {"passed": True, **counts, "families": families, "seed_splits": seed_splits}
