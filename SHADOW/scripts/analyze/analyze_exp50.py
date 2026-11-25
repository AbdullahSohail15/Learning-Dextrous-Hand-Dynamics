"""Experiment 50 tables: per controller over the task seeds (success all 15 gestures and excluding thumb_middle_pinch, tip success,
contact and invalid-penetration fractions, planning time), per-gesture success, and paired exact McNemar tests for the hypotheses."""
import json, sys, collections
from math import comb
from pathlib import Path
import numpy as np

X = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/exp50"); EXCL = "thumb_middle_pinch"
runs = collections.defaultdict(lambda: collections.defaultdict(list)); names = []
for d in sorted(X.glob("gestures_s*")):
    f = d / "results.json"
    if not f.exists(): continue
    j = json.loads(f.read_text()); nm = [t["target_name"].split(":")[1] for t in j["tasks"]]
    for k, r in j["runs"].items():
        k = k.split(":")[0]; pt = r["per_task"]
        for key in ("final_joint_rmse", "final_tip_mm", "contact_frac", "invalid_frac"):
            runs[k][key] += pt.get(key, [np.nan] * len(nm))
        runs[k]["gesture"] += nm; runs[k]["seed"] += [d.name] * len(nm); runs[k]["plan_ms"].append(r["plan_ms"])
        for key in ("global_plan_s", "rrt_needed", "rrt_replans", "ilqr_fallback", "nonfinite_cmd"):
            if key in r: runs[k][key].append(r[key])
R = {k: {kk: np.array(v) for kk, v in d.items()} for k, d in runs.items()}
order = [k for k in ("direct", "cem", "rrt_direct", "rrt_cem", "ilqr", "rrt_ilqr", "cem_cbf", "rrt_ilqr_cbf") if k in R]
out = dict(controllers={}, gestures={}, tests={})
print(f"{'controller':14s} {'n':>4s} {'succ all':>8s} {'succ excl':>9s} {'tip<5mm':>8s} {'joint':>6s} {'contact':>8s} {'invalid':>8s} {'plan ms':>8s}")
for k in order:
    r = R[k]; e = r["final_joint_rmse"]; keep = r["gesture"] != EXCL
    row = dict(n=int(len(e)), seeds=int(len(set(r["seed"]))), success_all=float((e < 2).mean() * 100), success_excl=float((e[keep] < 2).mean() * 100),
               tip_success=float((r["final_tip_mm"] < 5).mean() * 100), joint_mean=float(e.mean()), contact=float(np.nanmean(r["contact_frac"]) * 100),
               invalid=float(np.nanmean(r["invalid_frac"]) * 100), plan_ms=float(np.mean(r["plan_ms"])),
               per_seed_success=[float((e[r["seed"] == s] < 2).mean() * 100) for s in sorted(set(r["seed"]))])
    for key in ("global_plan_s", "rrt_needed", "rrt_replans", "ilqr_fallback", "nonfinite_cmd"):
        if key in r: row[key] = r[key].tolist()
    out["controllers"][k] = row
    print(f"{k:14s} {row['n']:4d} {row['success_all']:7.1f}% {row['success_excl']:8.1f}% {row['tip_success']:7.1f}% {row['joint_mean']:5.2f}° {row['contact']:7.1f}% {row['invalid']:7.2f}% {row['plan_ms']:8.0f}")
G = sorted(set(R[order[0]]["gesture"]))
print("\nper gesture success (%):\n" + f"{'gesture':20s}" + "".join(f"{k[:12]:>13s}" for k in order))
for g in G:
    out["gestures"][g] = {k: float((R[k]["final_joint_rmse"][R[k]["gesture"] == g] < 2).mean() * 100) for k in order}
    print(f"{g:20s}" + "".join(f"{out['gestures'][g][k]:13.0f}" for k in order))
def mcnemar(a, b):
    x, y = R[a]["final_joint_rmse"] < 2, R[b]["final_joint_rmse"] < 2; n10, n01 = int((x & ~y).sum()), int((~x & y).sum()); n = n10 + n01
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n) if n else 1.0; return n10, n01, p
print("\npaired tests (a only, b only, exact McNemar p); contact difference a - b (pp, paired mean):")
for a, b in (("rrt_cem", "cem"), ("rrt_ilqr", "ilqr"), ("rrt_direct", "direct"), ("ilqr", "cem"), ("rrt_ilqr", "cem"), ("cem_cbf", "cem"), ("rrt_ilqr_cbf", "rrt_ilqr")):
    if a in R and b in R and len(R[a]["final_joint_rmse"]) == len(R[b]["final_joint_rmse"]):
        n10, n01, p = mcnemar(a, b); dc = float(np.nanmean(R[a]["contact_frac"] - R[b]["contact_frac"]) * 100); di = float(np.nanmean(R[a]["invalid_frac"] - R[b]["invalid_frac"]) * 100)
        out["tests"][f"{a}_vs_{b}"] = dict(a_only=n10, b_only=n01, p=p, contact_diff_pp=dc, invalid_diff_pp=di)
        print(f"  {a:13s} vs {b:9s}: {n10:3d} vs {n01:3d}  p = {p:.2g}   contact {dc:+.1f} pp   invalid {di:+.2f} pp")
(X / "summary.json").write_text(json.dumps(out, indent=1))
