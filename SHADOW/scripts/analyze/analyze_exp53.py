"""Experiment 53: both controller stacks with a 20 s cap. Success vs time from the same runs, paired tests, consistency vs exp 50/51.

Usage: .venv_viz/bin/python scripts/analyze/analyze_exp53.py      (writes artifacts/exp53/analysis.json, prints tables)
"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; X = ROOT / "artifacts/exp53"
CUBE = {"CEM": "cem:P:hyb_tg0:safe", "STACK": "stk:P:hyb_tg0:safe:rrt+ilqr+cbf"}
GEST = {"CEM": "cem", "STACK": "rrt_ilqr_cbf"}
HZ, LAST, JTOL = 20, 10, 2.0      # gesture success: mean joint RMSE over the last 0.5 s (10 steps) < 2 deg


def mcnemar(a, b):
    x, y = int((a & ~b).sum()), int((~a & b).sum()); n = x + y
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return dict(only_stack=x, only_cem=y, p=p)


def cube():
    out, per = {}, {k: dict(ok={}, t=[], drop=[], pen1=[], pen3=[]) for k in CUBE}
    times = [5, 10, 15, 20]
    for d in sorted(X.glob("cube_s*")):
        if not (d / "results.json").exists() or "_partial_" in d.name: continue
        R = json.load(open(d / "results.json"))["runs"]
        if not all(a in R for a in CUBE.values()): continue
        for k, a in CUBE.items():
            pc, s = R[a]["per_case"], R[a]["summary"]
            v = np.array(pc["valid_start"], bool); t = np.array(pc["time_to_success"], float)[v]
            for T in times: per[k]["ok"].setdefault(T, []).append(~np.isnan(t) & (t <= T + 1e-6))
            per[k]["t"].append(t); per[k]["drop"].append(np.array(pc["dropped"], bool)[v])
            per[k]["pen1"].append(s.get("self_pen_1mm_frac", np.nan)); per[k]["pen3"].append(s.get("self_pen_3mm_frac", np.nan))
        out.setdefault("seeds", []).append(d.name)
    if "seeds" not in out: return None
    for k in CUBE:
        P = per[k]; t = np.concatenate(P["t"])
        out[k] = dict(n=int(t.size), success_at={T: float(np.concatenate(P["ok"][T]).mean()) for T in times},
                      per_seed_20s=[float(x.mean()) for x in P["ok"][20]], drop=float(np.concatenate(P["drop"]).mean()),
                      median_time_to_success=float(np.nanmedian(t)) if np.isfinite(t).any() else None,
                      pen_1mm=float(np.nanmean(P["pen1"])), pen_3mm=float(np.nanmean(P["pen3"])))
    out["paired"] = {T: mcnemar(np.concatenate(per["STACK"]["ok"][T]), np.concatenate(per["CEM"]["ok"][T])) for T in times}
    out["gain_10_to_20"] = {k: mcnemar(np.concatenate(per[k]["ok"][20]), np.concatenate(per[k]["ok"][10])) for k in CUBE}
    ref = {}   # exp 51 at 10 s, same seeds (approximately paired: settled starts differ slightly between processes)
    for s, files in {"790000": ["eval_s790000_merged", "eval_s790000_b", "eval_s790000"], "791000": ["eval_s791000"]}.items():
        for k, a in CUBE.items():
            for f in files:
                p = ROOT / "artifacts/exp51" / f / "results.json"
                if p.exists() and a in (R := json.load(open(p))["runs"]): ref.setdefault(k, {})[s] = R[a]["summary"]["success_rate"]; break
    out["exp51_10s_reference"] = ref
    return out


def gestures():
    out, ok = {}, {k: {} for k in GEST}; times = [6, 10, 15, 20]; pen = {k: [] for k in GEST}
    for d in sorted(X.glob("gestures_s*")):
        if not (d / "traces.npz").exists() or "_partial_" in d.name: continue
        Z = np.load(d / "traces.npz"); R = json.load(open(d / "results.json"))["runs"]
        if not all(f"{c}/joint_rmse" in Z.files for c in GEST.values()): continue
        for k, c in GEST.items():
            e = Z[f"{c}/joint_rmse"]
            for T in times:
                i = T * HZ; ok[k].setdefault(T, []).append(e[:, i - LAST + 1:i + 1].mean(1) < JTOL)
            pen[k].append(R[c]["invalid_frac"])
        out.setdefault("seeds", []).append(d.name)
    if "seeds" not in out: return None
    for k in GEST:
        out[k] = dict(n=int(np.concatenate(ok[k][20]).size), success_at={T: float(np.concatenate(ok[k][T]).mean()) for T in times},
                      per_seed_20s=[float(x.mean()) for x in ok[k][20]], invalid_frac=float(np.mean(pen[k])))
    out["paired"] = {T: mcnemar(np.concatenate(ok["STACK"][T]), np.concatenate(ok["CEM"][T])) for T in times}
    out["gain_6_to_20"] = {k: mcnemar(np.concatenate(ok[k][20]), np.concatenate(ok[k][6])) for k in GEST}
    ref = {}
    for d in sorted((ROOT / "artifacts/exp50").glob("gestures_s*")):
        R = json.load(open(d / "results.json"))["runs"]
        for k, c in GEST.items(): ref.setdefault(k, {})[d.name] = R[c]["success_joint"]
    out["exp50_6s_reference"] = ref
    return out


res = dict(cube=cube(), gestures=gestures())
json.dump(res, open(X / "analysis.json", "w"), indent=1)
for task, tag in [("cube", "s"), ("gestures", "s")]:
    r = res[task]
    if not r: print(f"{task}: no finished seeds"); continue
    T = list(r["CEM"]["success_at"])
    print(f"\n{task} ({', '.join(r['seeds'])}; n = {r['CEM']['n']} per arm)")
    print(f"{'arm':6s} " + " ".join(f"{'@' + str(t) + tag:>7s}" for t in T))
    for k in ["CEM", "STACK"]: print(f"{k:6s} " + " ".join(f"{100 * r[k]['success_at'][t]:7.1f}" for t in T))
    print("stack vs CEM p: " + "  ".join(f"@{t}{tag} {r['paired'][t]['p']:.3g}" for t in T))
    print({k: v for k, v in r.items() if k not in ("CEM", "STACK", "paired", "seeds")})
