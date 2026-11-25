"""Experiment 01, gate 0 smoke test: import, mimic coupling on the GPU pipeline, servo speed / hold under gravity, joint-space traces for rendering.

Usage: ./run_python.sh scripts/setup/gate0_smoke.py --headless --coupling mimic --output artifacts/exp01/smoke_mimic
Writes <output>/{report.json, traces.npz}.
"""
import argparse
import json
from pathlib import Path

from omni.isaac.lab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--num-envs", type=int, default=2)
p.add_argument("--self-collision", type=int, default=1)
p.add_argument("--gravity", type=int, default=1)
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
app = AppLauncher(args).app

import math
import numpy as np
import torch
from pxr import Usd
from inspire_dynamics.sim import ACTUATED, LINKAGE, CONTROL_DT, InspireSim, ensure_usd

out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
usd = ensure_usd()
stage = Usd.Stage.Open(usd)
apis = sorted({f"{p_.GetName()}:{a}" for p_ in stage.Traverse() for a in p_.GetAppliedSchemas() if "Mimic" in a or "Tendon" in a})
print("USD mimic/tendon schemas:", apis)

S = InspireSim(num_envs=args.num_envs, device=args.device, self_collision=bool(args.self_collision), gravity=bool(args.gravity))
print("joints:", S.names); print("bodies:", S.hand.body_names)
rep = dict(self_collision=args.self_collision, gravity=args.gravity, usd_schemas=apis, joint_names=S.names, body_names=S.hand.body_names, lo=S.lo.tolist(), hi=S.hi.tolist())

# program: open 1 s -> each actuated joint alone to 90% of range and back -> full fist -> hold 2 s -> open. Env 1 = same, unchanged (repeatability).
lo, hi = S.lo.cpu(), S.hi.cpu(); seg = []
def hold(a, s): seg.extend([a.clone()] * int(round(s / CONTROL_DT)))
z = lo.clone(); hold(z, 1.0)
for j in range(6):
    a = z.clone(); a[j] = lo[j] + 0.9 * (hi[j] - lo[j]); hold(a, 1.5); hold(z, 1.5)
fist = lo + 0.9 * (hi - lo); fist[0] = lo[0] + 0.5 * (hi[0] - lo[0]); hold(fist, 3.0); hold(z, 1.5)
U = torch.stack(seg)
S.reset(z[None].repeat(S.N, 1))
Q, QA, TQ, TIP = [], [], [], []
tips = [n for n in S.hand.body_names if n.endswith("_tip")]
for a in U:
    S.advance(a[None].repeat(S.N, 1)); st = S.state()
    Q.append(st["q"][0].cpu()); QA.append(st["qa"][0].cpu()); TQ.append(st["torque"][0].cpu()); TIP.append(S.body_pos(tips)[0].cpu())
Q, QA, TQ, TIP = map(torch.stack, (Q, QA, TQ, TIP)); il = S.il
link_err = (Q[:, il] - S.linkage_of(QA.to(S.device)).cpu()).abs()
rep["linkage_err_deg"] = dict(max=math.degrees(float(link_err.max())), mean=math.degrees(float(link_err.mean())),
                              per_joint_max=dict(zip(LINKAGE, np.degrees(link_err.max(0).values.numpy()).round(3).tolist())))
err = (QA - U.clamp(lo, hi)).abs()
rep["final_tracking_err_deg_in_holds"] = {n: math.degrees(float(err[t - 1, j])) for j, n in enumerate(ACTUATED) for t in [20 + 30 * (2 * j + 1)]}
# time to 90% of a step for each actuated joint (from its step-up segment)
rise = {}
for j, n in enumerate(ACTUATED):
    t0 = 20 + 60 * j; tgt = U[t0, j]; tr = QA[t0:t0 + 30, j]; k = torch.nonzero((tr - lo[j]) >= 0.9 * (tgt - lo[j]))
    rise[n] = float((k[0, 0] + 1) * CONTROL_DT) if len(k) else None
rep["rise_time_90pct_s"] = rise
rep["fist_hold_sag_deg"] = dict(zip(ACTUATED, np.degrees((QA[20 + 360 + 59] - fist.clamp(lo, hi)).numpy()).round(3).tolist()))
rep["repeatability_env1_deg"] = math.degrees(float((S.state()["q"][1] - S.state()["q"][0]).abs().max())) if S.N > 1 else None
rep["tips"] = tips; rep["tip_pos_open"] = TIP[19].numpy().round(4).tolist(); rep["root_pos"] = (S.hand.data.root_pos_w[0] - S.scene.env_origins[0]).tolist()
np.savez(out / "traces.npz", q=Q.numpy(), qa=QA.numpy(), target=U.numpy(), torque=TQ.numpy(), tip=TIP.numpy(), joint_names=np.array(S.names), dt=CONTROL_DT)
json.dump(rep, open(out / "report.json", "w"), indent=1); print(json.dumps(rep, indent=1)); print("GATE0_SMOKE_COMPLETE")
app.close()
