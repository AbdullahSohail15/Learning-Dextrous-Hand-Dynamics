"""Experiment 03 summary: per controller over seeds (success, tips, contact, deep penetration, time to 2 deg, latency, RRT/CBF/fallback
counts), per-gesture success, paired exact McNemar tests, and the failing tasks. Usage: python3 analyze_exp03.py artifacts/exp03/bench"""
import json, sys
from math import comb
from pathlib import Path
import numpy as np

d = json.load(open(Path(sys.argv[1]) / "results.json")); R = d["runs"]; runs = sorted({k.split("|")[0] for k in R}, key=lambda n: list(R).index(f"{n}|{min(k.split('|')[1] for k in R)}"))
seeds = sorted({k.split("|")[1] for k in R})


def cat(name, key): return np.concatenate([np.array(R[f"{name}|{s}"]["per_task"][key]) for s in seeds if f"{name}|{s}" in R])


def mcnemar(a, b):
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum()); n = n10 + n01
    if n == 0: return n10, n01, 1.0
    k = min(n10, n01); return n10, n01, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


print(f"{'run':12s} {'n':>4s} {'succ':>6s} {'tips':>6s} {'rmse':>6s} {'contact%':>8s} {'deep%':>6s} {'t2deg':>6s} {'plan ms':>8s} {'cbf ms':>7s} {'rrt s':>6s} replans fallback")
S = {}
for n in runs:
    ok = cat(n, "final_rmse") < 2.0; S[n] = ok; ks = [f"{n}|{s}" for s in seeds if f"{n}|{s}" in R]
    t = cat(n, "time_to_tol"); t = t[t >= 0]
    print(f"{n:12s} {len(ok):4d} {100 * ok.mean():6.1f} {100 * (cat(n, 'final_tip_mm') < 5).mean():6.1f} {cat(n, 'final_rmse').mean():6.2f} {100 * cat(n, 'contact_frac').mean():8.2f} "
          f"{100 * cat(n, 'deep_frac').mean() if 'deep_frac' in R[ks[0]]['per_task'] else float('nan'):6.2f} {np.median(t):6.2f} "
          f"{np.median([R[k]['plan_ms_median'] for k in ks]):8.1f} {np.median([R[k]['cbf_ms_median'] for k in ks]):7.1f} {np.mean([R[k]['rrt_s'] for k in ks]):6.2f} "
          f"{sum(R[k]['rrt_replans'] for k in ks):7d} {sum(R[k].get('fallback_replans', 0) for k in ks):8d}")
meta = sum([R[f"{runs[0]}|{s}"]["meta"] for s in seeds], [])
goals = np.array([m[0] for m in meta]); starts = np.array([m[1] for m in meta])
print("\nper goal success (%):", " ".join(runs))
for g in dict.fromkeys(goals):
    m = goals == g; print(f"{g:12s}", " ".join(f"{100 * S[n][m].mean():5.0f}" for n in runs))
print("\npaired (A only, B only, p):")
for a, b in [("ilqr", "cem"), ("rrt_direct", "direct"), ("rrt_ilqr", "ilqr"), ("rrt_cem", "cem"), ("cem_cbf", "cem"), ("full", "rrt_ilqr"), ("full", "direct"), ("full", "full_N0")]:
    if a in S and b in S: print(f"  {a} vs {b}: {mcnemar(S[a], S[b])}")
print("\nfailures (run: goal <- start, final rmse):")
for n in runs:
    f = np.where(~S[n])[0]
    if len(f): print(f"  {n}: " + "; ".join(f"{goals[i]}<-{starts[i]} {cat(n, 'final_rmse')[i]:.1f}" for i in f[:8]))
if "latency_ms" in d: print("\nlatency ms:", json.dumps({k: round(v, 1) for k, v in d["latency_ms"].items()}))
