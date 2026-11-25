"""Print the hand-root (FK root) pose and the in-hand point in env-0 coordinates (for re-framing exp-51/52 MPC traces for the cube viewer)."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics.resources import restrict_simulator_gpu
restrict_simulator_gpu()
from omni.isaac.lab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--output", required=True); AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.physics_hz = 120; args.control_hz = 20; args.mode = "cube"; args.num_envs = 1; args.seed = 0
app = AppLauncher(args, multi_gpu=False).app
from shadow_dynamics.env import ResearchShadowEnv, make_cfg
cfg = make_cfg(args); cfg.object_cfg.spawn.rigid_props.kinematic_enabled = False
env = ResearchShadowEnv(cfg); env.configure_research("cube")
o = env.scene.env_origins[0]; b = env.hand.data.body_state_w[0, 0]
from pxr import UsdGeom, Usd
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeLocalBound(env.sim.stage.GetPrimAtPath("/World/envs/env_0/object"))
r = dict(root_pos=(b[:3] - o).tolist(), root_quat_wxyz=b[3:7].tolist(), in_hand=env.in_hand_pos[0].tolist(), cube_size=[float(x) for x in bb.GetRange().GetSize()], joint_names=list(env.hand.joint_names))
Path(args.output).write_text(json.dumps(r, indent=1)); print("PROBE", r["root_pos"], r["root_quat_wxyz"], r["in_hand"], r["cube_size"], flush=True)
app.close()
