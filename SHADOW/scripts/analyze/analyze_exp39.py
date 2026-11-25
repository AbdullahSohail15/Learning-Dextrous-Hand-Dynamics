"""Experiment 39 analysis: cube success per arm (pooled over seeds, Wilson CIs, McNemar pairs) and the hybrid-vs-oracle ranking diagnostic
from the mpc:<P>:oracle arm (every hybrid hand model and learned model scored the oracle's candidates): per-decision Spearman correlation,
top-1 agreement, oracle-score regret of each model's pick, and, for the oracle's chosen candidate, the hybrid's predicted hand / cube error."""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics.cube_world_data import sixd_to_matrix, rotation_distance
p = argparse.ArgumentParser(); p.add_argument("--runs", required=True); p.add_argument("--output", required=True); p.add_argument("--pairs", nargs="*", default=[])
p.add_argument("--oracle-arm", default="mpc:P:oracle:safe")
a = p.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=False); rep = dict(task={}, pairs={}, ranking={}, prediction={})
def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d; return [c - h, c + h]
def mcn(x, y):
    b, c = int((x & ~y).sum()), int((~x & y).sum()); n = b + c; return b, c, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n))
dirs = [Path(d) for d in a.runs.split(",")]; R = [json.loads((d / "results.json").read_text()) for d in dirs]
oc = lambda arm, k="success": np.concatenate([np.array(r["runs"][arm]["per_case"][k], bool) & np.array(r["runs"][arm]["per_case"]["valid_start"], bool) for r in R])
for arm in R[0]["runs"]:
    s, dr = oc(arm), oc(arm, "dropped"); ci = wilson(int(s.sum()), len(s))
    secs = sum(r["runs"][arm]["summary"]["seconds"] for r in R)
    rep["task"][arm] = dict(success=float(s.mean()), ci95=ci, drop=float(dr.mean()), n=len(s), seconds=secs,
                            per_seed=[r["runs"][arm]["summary"]["success_rate"] for r in R])
    print(f"TASK {arm:28s} success {s.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  drop {dr.mean():.3f}  seeds " + " ".join(f"{x:.3f}" for x in rep["task"][arm]["per_seed"]) + f"  {secs:.0f}s", flush=True)
for pr in a.pairs:
    A_, B_ = pr.split(",")
    if A_ in R[0]["runs"] and B_ in R[0]["runs"]:
        for k in ("success", "dropped"):
            b, c, pv = mcn(oc(A_, k), oc(B_, k)); rep["pairs"][f"{A_} vs {B_} [{k}]"] = dict(only_A=b, only_B=c, diff_pts=100 * (b - c) / len(oc(A_)), p=pv)
            print(f"PAIR {A_} vs {B_} [{k}] only A {b} only B {c} diff {100 * (b - c) / len(oc(A_)):+.1f} pts p {pv:.4g}", flush=True)

def spearman(x, y):   # rows of (n, K), average ranks for ties
    def rk(v): return np.apply_along_axis(lambda r: __import__("scipy.stats", fromlist=["rankdata"]).rankdata(r), 1, v)
    rx, ry = rk(x), rk(y); rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1)); return np.where(den > 0, (rx * ry).sum(1) / np.maximum(den, 1e-12), np.nan)
tag = a.oracle_arm.replace(":", "__"); Z = [np.load(d / f"traces_{tag}.npz") for d in dirs if (d / f"traces_{tag}.npz").exists()]
if Z:
    labs = [k[5:] for k in Z[0].files if k.startswith("rank_") and k not in ("rank_alive", "rank_oracle")]
    for lab in labs:
        rho, top1, regret = [], [], []
        for z in Z:
            O, M, al = z["rank_oracle"], z[f"rank_{lab}"], z["rank_alive"] & z["valid"][None]   # (D, E, K), (D, E)
            o, m = O[al], M[al]; ok = np.isfinite(o).all(1) & np.isfinite(m).all(1); o, m = o[ok], m[ok]
            rho.append(spearman(o, m)); pick = m.argmax(1); top1.append(pick == o.argmax(1))
            spread = o.max(1) - np.median(o, 1); regret.append((o.max(1) - o[np.arange(len(o)), pick]) / np.maximum(spread, 1e-9))
        rho, top1, regret = np.concatenate(rho), np.concatenate(top1), np.concatenate(regret)
        rep["ranking"][lab] = dict(decisions=len(rho), spearman_median=float(np.nanmedian(rho)), spearman_mean=float(np.nanmean(rho)), top1=float(top1.mean()),
                                   regret_median=float(np.median(regret)), chance_top1=1 / Z[0]["rank_oracle"].shape[-1])
        print(f"RANK {lab:14s} decisions {len(rho)}  Spearman median {np.nanmedian(rho):.3f} mean {np.nanmean(rho):.3f}  top-1 {top1.mean():.3f} (chance {1 / Z[0]['rank_oracle'].shape[-1]:.3f})  "
              f"regret median {np.median(regret):.3f} (oracle best - pick, / (best - median))", flush=True)
    for k in [k for k in Z[0].files if k.startswith("hyb_chosen_pred_")]:
        lab = k[len("hyb_chosen_pred_"):]; jq, cp, cr, cr_last = [], [], [], []
        for z in Z:
            al = z["dec_alive"] & z["valid"][None]; Yo, Yh = z["oracle_chosen_pred"][al], z[k][al]   # (n, 12, 63)
            jq.append(np.sqrt(((Yh[..., :24] - Yo[..., :24]) ** 2).mean(-1)))
            cp.append(np.linalg.norm(Yh[..., 48:51] - Yo[..., 48:51], axis=-1))
            Ro, Rh = sixd_to_matrix(torch.as_tensor(Yo[..., 51:57]).float()), sixd_to_matrix(torch.as_tensor(Yh[..., 51:57]).float())
            cr.append(np.degrees(rotation_distance(Rh.reshape(-1, 3, 3), Ro.reshape(-1, 3, 3)).numpy()).reshape(Yo.shape[:2]))
        jq, cp, cr = np.concatenate(jq), np.concatenate(cp), np.concatenate(cr)
        rep["prediction"][lab] = dict(n=len(jq), joint_rmse_deg_h12_mean=float(jq[:, -1].mean()), joint_rmse_deg_h6_mean=float(jq[:, 5].mean()),
                                      cube_pos_mm_h12_median=float(np.median(cp[:, -1])), cube_rot_deg_h12_median=float(np.median(cr[:, -1])),
                                      cube_rot_deg_h12_mean=float(cr[:, -1].mean()), cube_rot_deg_h6_median=float(np.median(cr[:, 5])), cube_pos_mm_h12_p90=float(np.percentile(cp[:, -1], 90)))
        v = rep["prediction"][lab]
        print(f"PRED hyb_{lab:10s} n {len(jq)}  hand joint RMSE vs oracle h6 {v['joint_rmse_deg_h6_mean']:.2f} h12 {v['joint_rmse_deg_h12_mean']:.2f} deg  "
              f"cube rot h6 {v['cube_rot_deg_h6_median']:.2f} h12 {v['cube_rot_deg_h12_median']:.2f} deg (median)  cube pos h12 {v['cube_pos_mm_h12_median']:.1f} mm (p90 {v['cube_pos_mm_h12_p90']:.1f})", flush=True)
(out / "analysis.json").write_text(json.dumps(rep, indent=1, default=float)); print("EXP39_ANALYSIS_COMPLETE", flush=True)
