"""Experiment 04 feasibility probe: can the Inspire hand, palm up, change the yaw of a cube resting on its palm without dropping it?

Mount: hand rotated -90 deg about y (palm normal +x -> world +z, fingers -> world -x), base at (0, 0, 0.5). Palm surface at z = 0.528,
cube centre at x = --palm-x (default -0.07, the palm centre). Cube: `--size` m, 30 g, friction 1.0, dropped onto the palm centre and settled 1.5 s with the hand open.
Programs (8 s each, hand 0's true servo parameters): random servo chunks at three speeds of change (exp 02 generator), plus scripted
single-finger curls. Reports per env the final cube yaw change, max |yaw| reached while on the palm, drop (cube centre below the palm
surface or > 6 cm off the palm centre), and saves traces for rendering.

Usage: ./run_python.sh scripts/evaluate/exp04_probe.py --headless --device cuda:0 --output artifacts/exp04/probe
"""
import argparse, json, math
from pathlib import Path
from omni.isaac.lab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); ap.add_argument("--size", type=float, default=0.04)
ap.add_argument("--envs", type=int, default=96); ap.add_argument("--palm-x", type=float, default=-0.07); ap.add_argument("--closure", type=float, default=0.8); ap.add_argument("--seconds", type=float, default=8.0)
AppLauncher.add_app_launcher_args(ap); args = ap.parse_args(); app = AppLauncher(args).app

import numpy as np, torch
from inspire_dynamics.sim import InspireSim, true_hand_params, CONTROL_DT
from inspire_dynamics import explore as ex

ROT = (0.70710678, 0.0, -0.70710678, 0.0); PALM_Z = 0.523; PALM_C = (args.palm_x, 0.0)   # palm surface z from the settled cube (0.5428 - 0.02)
out = Path(args.output); out.mkdir(parents=True, exist_ok=False); dev = args.device; N = args.envs
S = InspireSim(num_envs=N, device=dev, cube=True, root_rot=ROT, cube_size=args.size, cube_mass=0.03, cube_pos=(PALM_C[0], 0.0, PALM_Z + args.size / 2 + 0.01))
hand = true_hand_params(0); S.set_servo_params(**{k: v[None].expand(N, 6) for k, v in hand.items()}); lo, hi = S.lo, S.hi


def yaw(q):   # wxyz -> yaw about world z
    w, x, y, z = q.unbind(-1); return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


S.reset()
for _ in range(30): S.advance(torch.zeros(N, 6, device=dev))
c0 = S.cube_state(); y0 = yaw(c0[:, 3:7]); print("settled cube z", c0[:, 2].mean().item(), "x", c0[:, 0].mean().item(), "y", c0[:, 1].mean().item())
rng = np.random.default_rng(0); T = int(args.seconds / CONTROL_DT); A = np.zeros((N, T, 6), np.float32); kind = []
for e in range(N):
    if e < 6:   # scripted: one servo at a time to 60 % and back, twice
        a = np.zeros((T, 6)); i = e; lev = lo[i].item() + 0.6 * (hi[i] - lo[i]).item()
        for k in range(T): a[k, i] = lev if (k // 40) % 2 == 0 else 0.0
        A[e] = a; kind.append(f"servo{i}")
    else:
        chunks = ex.gen_chunks(rng, lo.cpu().numpy(), (lo + (hi - lo) * args.closure).cpu().numpy(), T // ex.CHUNK + 1, np.zeros(6)); A[e] = np.concatenate(list(chunks))[:T]; kind.append("random")
Ys, Cs, Qs = [], [], []
for k in range(T):
    S.advance(torch.tensor(A[:, k], device=dev)); c = S.cube_state(); Cs.append(c.cpu()); Ys.append(yaw(c[:, 3:7]).cpu()); Qs.append(S.state()["q"].cpu())
C = torch.stack(Cs, 1); Y = torch.stack(Ys, 1); dy = torch.atan2(torch.sin(Y - y0.cpu()[:, None]), torch.cos(Y - y0.cpu()[:, None]))
off = ((C[..., 0] - PALM_C[0]) ** 2 + (C[..., 1] - PALM_C[1]) ** 2).sqrt(); dropped_t = (C[..., 2] < PALM_Z) | (off > 0.06)
drop = dropped_t.any(1); first = torch.where(drop, dropped_t.float().argmax(1), torch.full((N,), T))
maxyaw = torch.stack([dy[e, :first[e]].abs().max() if first[e] > 0 else torch.tensor(0.0) for e in range(N)])
rep = dict(size=args.size, settled=dict(z=float(c0[:, 2].mean()), x=float(c0[:, 0].mean()), y=float(c0[:, 1].mean())),
           drop_rate=float(drop.float().mean()), max_abs_yaw_deg_on_palm=dict(median=float(np.degrees(maxyaw.median())), p90=float(np.degrees(maxyaw.quantile(0.9))), max=float(np.degrees(maxyaw.max()))),
           kept_and_turned_over_30deg=float(((~drop) & (dy[:, -1].abs() > math.radians(30))).float().mean()),
           scripted={kind[e]: dict(final_yaw_deg=float(np.degrees(dy[e, -1])), drop=bool(drop[e]), max_yaw_deg=float(np.degrees(maxyaw[e]))) for e in range(6)})
json.dump(rep, open(out / "probe.json", "w"), indent=1)
np.savez_compressed(out / "traces.npz", q=torch.stack(Qs, 1).numpy(), cube=C.numpy(), target=A, joint_names=np.array(S.names), dt=CONTROL_DT)
print(json.dumps(rep, indent=1)); print("PROBE_COMPLETE"); app.close()
