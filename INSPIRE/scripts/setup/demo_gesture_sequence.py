"""Experiment 01 (preview of 03): a sequence of hand-written gestures sent as direct servo commands (no planning), recorded for rendering.
Shows what the stiff servos do in free space and where thumb-finger contact stops them (the failure mode experiment 03 targets).

Usage: ./run_python.sh scripts/setup/demo_gesture_sequence.py --headless --device cuda:0 --output artifacts/exp01/gesture_sequence
"""
import argparse, json
from pathlib import Path
from omni.isaac.lab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--hold", type=float, default=1.5)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app

import numpy as np, torch
from inspire_dynamics.sim import ACTUATED, CONTROL_DT, InspireSim

# fractions of each servo's range: thumb yaw, thumb bend, index, middle, ring, little (draft gestures; experiment 03 settles them in Isaac)
G = dict(open=[0, 0, 0, 0, 0, 0], point=[0.6, 0.6, 0, 0.9, 0.9, 0.9], peace=[0.6, 0.6, 0, 0, 0.9, 0.9], thumbs_up=[0, 0, 0.9, 0.9, 0.9, 0.9],
         rock=[0.6, 0.6, 0, 0.9, 0.9, 0], call=[0, 0, 0.9, 0.9, 0.9, 0], pinch=[1.0, 0.45, 0.42, 0, 0, 0], tripod=[1.0, 0.45, 0.42, 0.40, 0, 0],
         fist=[0.5, 0.9, 0.9, 0.9, 0.9, 0.9])
SEQ = ["open", "point", "peace", "thumbs_up", "rock", "call", "pinch", "tripod", "fist", "pinch", "open"]
out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
S = InspireSim(num_envs=1, device=args.device); lo, hi = S.lo.cpu(), S.hi.cpu(); S.reset()
Q, U, QA, labels = [], [], [], []
for g in SEQ:
    a = lo + torch.tensor(G[g]) * (hi - lo)
    for _ in range(int(round(args.hold / CONTROL_DT))):
        S.advance(a[None]); st = S.state(); Q.append(st["q"][0].cpu().numpy()); QA.append(st["qa"][0].cpu().numpy()); U.append(a.numpy()); labels.append(g)
Q, QA, U = map(np.array, (Q, QA, U)); n = int(round(args.hold / CONTROL_DT))
rep = {f"{i:02d}_{g}": dict(final_servo_err_deg=np.degrees(QA[(i + 1) * n - 1] - U[(i + 1) * n - 1]).round(2).tolist()) for i, g in enumerate(SEQ)}
np.savez(out / "traces.npz", q=Q, qa=QA, target=U, joint_names=np.array(S.names), dt=CONTROL_DT, labels=np.array(labels))
json.dump(dict(sequence=SEQ, gestures=G, servo_order=ACTUATED, hold_s=args.hold, results=rep), open(out / "report.json", "w"), indent=1)
print(json.dumps(rep, indent=1)); print("GESTURE_SEQUENCE_COMPLETE"); app.close()
