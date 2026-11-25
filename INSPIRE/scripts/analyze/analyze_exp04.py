"""Experiment 04 summary: per arm over seeds (success, final success, drops, time to goal, plan ms), per goal size, paired McNemar vs
soft_CL and vs oracle. Usage: python3 analyze_exp04.py artifacts/exp04/cube"""
import json, sys
from math import comb
from pathlib import Path
import numpy as np

d = json.load(open(Path(sys.argv[1]) / "results.json")); R = d["runs"]
arms = list(dict.fromkeys(k.split("|")[0] for k in R)); seeds = sorted({k.split("|")[1] for k in R})


def cat(a, key): return np.concatenate([np.array(R[f"{a}|{s}"]["per_task"][key]) for s in seeds if f"{a}|{s}" in R])


def mcnemar(a, b):
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum()); n = n10 + n01
    if n == 0: return n10, n01, 1.0
    return n10, n01, min(1.0, 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n)


print(f"{'arm':14s} {'n':>4s} {'succ%':>6s} {'final%':>7s} {'drop%':>6s} {'t_reach':>8s} {'plan ms':>8s}  seeds")
S = {}
for a in arms:
    ok = cat(a, "reached") > 0.5; S[a] = ok; t = cat(a, "t_reach"); t = t[t >= 0]; ks = [f"{a}|{s}" for s in seeds if f"{a}|{s}" in R]
    print(f"{a:14s} {len(ok):4d} {100 * ok.mean():6.1f} {100 * (cat(a, 'final_ok') > 0.5).mean():7.1f} {100 * (cat(a, 'dropped') > 0.5).mean():6.1f} "
          f"{np.median(t) if len(t) else float('nan'):8.2f} {np.median([R[k]['plan_ms_median'] for k in ks]):8.0f}  " + " ".join(f"{100 * R[k]['success']:.0f}" for k in ks)
          + (f"  cbf {R[ks[0]]['cbf']}" if "cbf" in R[ks[0]] else ""))
g = np.abs(cat(arms[0], "goal_deg")).round()
print("\nper |goal| success %:", arms)
for v in sorted(set(g)): print(f"{v:5.0f}", " ".join(f"{100 * S[a][g == v].mean():6.0f}" for a in arms))
print("\npaired:")
for b in ("param_CL", "oracle"):
    for a in arms:
        if a != b and b in S and len(S[a]) == len(S[b]): print(f"  {a} vs {b}: {mcnemar(S[a], S[b])}")
