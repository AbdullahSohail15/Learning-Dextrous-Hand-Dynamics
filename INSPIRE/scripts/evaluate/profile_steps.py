"""Timing of the Isaac step pieces with a cube, for sizing experiment 04 (CPU-bound)."""
import argparse, time
from omni.isaac.lab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--envs", type=int, default=20); AppLauncher.add_app_launcher_args(ap); args = ap.parse_args(); app = AppLauncher(args).app
import torch
from inspire_dynamics.sim import InspireSim
S = InspireSim(num_envs=args.envs, device=args.device, cube=True, root_rot=(0.70710678, 0.0, -0.70710678, 0.0), spacing=0.6, cube_size=0.04, cube_mass=0.03, cube_pos=(-0.105, 0, 0.55))
S.reset()
def T(fn, n=60):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000
print("envs", args.envs)
print("sim.step only ms", T(lambda: S.sim.step(render=False)))
print("write_data_to_sim ms", T(lambda: S.scene.write_data_to_sim()))
print("scene.update ms", T(lambda: S.scene.update(S.sim.get_physics_dt())))
print("joint_pos read ms", T(lambda: S.hand.data.joint_pos[0, 0].item()))
print("cube_state ms", T(lambda: S.cube_state()))
tgt = S.hand.data.joint_pos_target.clone()
print("set target ms", T(lambda: S.hand.set_joint_position_target(tgt)))
print("full advance (6 substeps) ms", T(lambda: S.advance(torch.zeros(S.N, 6, device=args.device)), 20))
app.close()
