"""Experiment 02 summary table from artifacts/exp02/explore/report.json (no Isaac)."""
import json, sys
import numpy as np
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "artifacts/exp02/explore/report.json"))
print(f"TRUE window err {np.round(r['true_err'], 3)}  N0 {np.round(r['N0_err'], 3)}")
print("contact fraction over the whole exploration:", {a: np.round(v, 4).tolist() for a, v in r["contact_frac_total"].items()})
names = ["yaw", "bend", "idx", "mid", "ring", "pinky"]
for a, rows in r["arms"].items():
    print(f"\n== {a}")
    for row in rows:
        rel = {k: np.array(v) for k, v in row["rel_err"].items()}            # (hands, 6)
        print(f"{row['seconds']:5.0f}s err {np.mean(row['err']):.3f} (hands {np.round(row['err'], 3).tolist()})  "
              f"vmax {100 * np.median(rel['vmax']):.1f}% (yaw {100 * np.median(rel['vmax'][:, 0]):.1f}%)  umax {100 * np.median(rel['umax']):.0f}%  "
              f"K {100 * np.median(rel['K']):.0f}%  D {100 * np.median(rel['D']):.0f}%  m {100 * np.median(rel['m']):.0f}%  contact {np.round(row['contact_frac'], 3).tolist()}")
d = r.get("design", [])
if d: print("\nCL design: rejected", sum(x["rejected"] for x in d), "of", sum(x["candidates"] for x in d), "candidates by the safety screen; seconds", round(sum(x["seconds"] for x in d)))
