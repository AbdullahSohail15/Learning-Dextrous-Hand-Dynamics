"""Experiment 01: which non-adjacent link pairs of the dex-urdf collision geometry overlap, at the open pose and over the joint range.
Usage: .venv_viz python scripts/setup/self_collision_pairs.py  (writes artifacts/exp01/self_collision_pairs.json)"""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inspire_dynamics.urdf_kin import URDFKin, hull_penetration
K = URDFKin(); H = K.collision_hulls(); A = K.actuated
adj = {frozenset((j["parent"], j["child"])) for j in K.joints}
pairs = [p for p in itertools.combinations(sorted(H), 2) if frozenset(p) not in adj]
rng = np.random.default_rng(0); lim = {j["name"]: (j["lo"], j["hi"]) for j in K.joints}
poses = [dict.fromkeys(A, 0.0)] + [{n: rng.uniform(*lim[n]) for n in A} for _ in range(400)]
res = {f"{a}|{b}": [] for a, b in pairs}
for q in poses:
    X = K.fk(q)
    for a, b in pairs:
        d = 0.0
        for ha in H[a]:
            ha_w = ha.copy(); ha_w.apply_transform(X[a])
            for hb in H[b]:
                hb_w = hb.copy(); hb_w.apply_transform(X[b]); d = max(d, hull_penetration(ha_w, hb_w))
        res[f"{a}|{b}"].append(d)
out = {k: dict(open_mm=1e3 * v[0], frac_random=float(np.mean(np.array(v[1:]) > 1e-4)), max_mm=1e3 * max(v)) for k, v in res.items() if max(v) > 1e-4}
print(json.dumps(out, indent=1)); Path("artifacts/exp01").mkdir(parents=True, exist_ok=True)
json.dump(dict(n_random=len(poses) - 1, overlapping=out), open("artifacts/exp01/self_collision_pairs.json", "w"), indent=1)
