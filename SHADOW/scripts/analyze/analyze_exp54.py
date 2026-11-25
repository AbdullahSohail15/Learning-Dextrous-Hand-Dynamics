"""Experiment 54: cube-aware stack on the soft hybrid. Each arm vs the soft CEM baseline (exact McNemar), plus exp-53 kinematic references.

Usage: .venv_viz/bin/python scripts/analyze/analyze_exp54.py [seed]      (default 790000; writes artifacts/exp54/analysis_s<seed>.json)
"""
import json, sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; seed = sys.argv[1] if len(sys.argv) > 1 else "790000"
BASE = "cem:P:hyb_soft_tg0:safe"


def load(p):
    R = json.load(open(p))["runs"]; out = {}
    for a, r in R.items():
        pc, s = r["per_case"], r["summary"]; v = np.array(pc["valid_start"], bool); t = np.array(pc["time_to_success"], float)
        out[a] = dict(ok=~np.isnan(t) & v, ok10=~np.isnan(t) & (t <= 10 + 1e-6) & v, v=v, s=s)
    return out


def mcnemar(a, b):
    x, y = int((a & ~b).sum()), int((~a & b).sum()); n = x + y
    return x, y, (min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0)


A = load(ROOT / f"artifacts/exp54/cube_s{seed}/results.json")
ref = ROOT / f"artifacts/exp53/cube_s{seed}/results.json"
if ref.exists(): A.update({f"exp53 {k}": v for k, v in load(ref).items()})
res = {}
print(f"{'arm':48s} {'@20s':>6s} {'@10s':>6s} {'vs soft CEM (only arm/only base, p)':>36s} {'drop':>5s} {'>1mm':>5s} {'>3mm':>5s} {'rrtp':>5s} {'min':>5s}")
for a, r in A.items():
    s = r["s"]; b = A[BASE]["ok"]; x, y, p = mcnemar(r["ok"], b) if a != BASE else (0, 0, 1.0)
    st = s.get("stack") or {}
    res[a] = dict(success=float(r["ok"][r["v"]].mean()), success_10s=float(r["ok10"][r["v"]].mean()), only_arm=x, only_base=y, p=p,
                  drop=s.get("drop_rate"), pen_1mm=s.get("self_pen_1mm_frac"), pen_3mm=s.get("self_pen_3mm_frac"), rrtp_chosen=st.get("rrtp_chosen"),
                  rrt_needed=st.get("rrt_needed"), decisions=st.get("decisions"), minutes=s.get("seconds", 0) / 60, soft_track_deg=s.get("soft_track_deg"))
    q = res[a]
    print(f"{a:48s} {100 * q['success']:6.1f} {100 * q['success_10s']:6.1f} {f'{x}/{y}, p {p:.3g}':>36s} {q['drop'] or 0:5.2f} {q['pen_1mm'] or 0:5.2f} {q['pen_3mm'] or 0:5.3f} {str(q['rrtp_chosen'] or '-'):>5s} {q['minutes']:5.0f}")
json.dump(res, open(ROOT / f"artifacts/exp54/analysis_s{seed}.json", "w"), indent=1)
