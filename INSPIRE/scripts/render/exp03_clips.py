"""Experiment 03: cut one gesture task out of artifacts/exp03/<run>/traces.npz for render_traces.py (goal gesture as a ghost hand).

  python3 scripts/render/exp03_clips.py artifacts/exp03/crossing full 0 28 clips/t28_full.npz      # controller, seed, task
  python3 scripts/render/exp03_clips.py artifacts/exp03/crossing --compare direct full 0            # tasks where 'direct' fails and 'full' succeeds

Output npz: q (T, 12) joint angles (rad), joint_names, dt, ghost_q (12,) = the goal gesture with its linkage joints filled in.
No Isaac needed (numpy only).
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from inspire_dynamics.urdf_kin import URDFKin


def main(argv):
    run = Path(argv[0]); R = json.loads((run / "results.json").read_text())["runs"]
    if argv[1] == "--compare":
        a, b, seed = argv[2], argv[3], argv[4]; tol = 2.0
        ra, rb = R[f"{a}|{seed}"], R[f"{b}|{seed}"]
        ea, eb = np.array(ra["per_task"]["final_rmse"]), np.array(rb["per_task"]["final_rmse"])
        idx = np.where((ea >= tol) & (eb < tol) & (eb >= 0))[0]
        print(f"{len(idx)} tasks where {a} fails and {b} succeeds (final servo RMSE, deg; seed {seed}):")
        for i in idx[np.argsort(-ea[idx])]:
            print(f"  task {i:3d}  goal {rb['meta'][i][0]:12s} start {rb['meta'][i][1]:16s}  {a} {ea[i]:6.2f}  {b} {eb[i]:5.2f}")
        return
    ctrl, seed, task, outp = argv[1], argv[2], int(argv[3]), Path(argv[4])
    z = np.load(run / "traces.npz"); q = z[f"{ctrl}__{seed}"][task]; names = [str(n) for n in z["joint_names"]]
    lib = json.loads((ROOT / "assets/inspire_hand/gesture_library.json").read_text())
    goal = R[f"{ctrl}|{seed}"]["meta"][task][0]
    kin = URDFKin(ROOT / "assets/inspire_hand/inspire_hand_right.urdf")
    full = kin.full_q(dict(zip(lib["servo_order"], lib["gestures"][goal]["a"])))
    dt = float(z["dt"]) if "dt" in z.files else 0.05
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, q=q.astype(np.float32), joint_names=np.array(names), dt=dt, ghost_q=np.array([full[n] for n in names]))
    print(outp, q.shape, "goal", goal, "final servo RMSE", R[f"{ctrl}|{seed}"]["per_task"]["final_rmse"][task])


if __name__ == "__main__":
    main(sys.argv[1:])
