"""Export the stock Isaac Lab Shadow Hand for the web viewer: link meshes, kinematic tree, an FK check, test episodes.

Output (one directory, never overwritten):
  model.json         joint names (runtime order), kinematic tree (USD joints), limits (fingertips are derived from the distal meshes in shadow_viz.kinematics)
  meshes.npz         visual triangle meshes of every link, vertices in that link's frame (m)
  fk_check.npz       random joint configurations and the simulator's body poses (for numpy FK verification)
  episodes.npz       short hand-only excitation episodes at 20 Hz: q (deg), qdot (deg/s), body poses relative to the root
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shadow_dynamics.resources import restrict_simulator_gpu
PHYSICAL_GPU = restrict_simulator_gpu()
from omni.isaac.lab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--fk-samples", type=int, default=32)
parser.add_argument("--families", nargs="+", default=["finger_wave", "thumb_opposition", "curl", "mixed"])
parser.add_argument("--speed-deg", type=float, default=20.0)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--seed", type=int, default=310001)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.mode = "free"; args.physics_hz = 120; args.control_hz = 20
launcher = AppLauncher(args, multi_gpu=False); app = launcher.app
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics
from shadow_dynamics.env import ResearchShadowEnv, make_cfg
from shadow_dynamics.excitation import Limits, generate, apply_slew, set_control_hz

ROBOT = "/World/envs/env_0/Robot"


def mat(m):
    """Gf.Matrix4d (row-vector convention, p' = p M) -> numpy 4x4 column-vector convention."""
    return np.array(m, dtype=np.float64).T


def owner_body(prim, bodies):
    while prim and prim.IsValid():
        if str(prim.GetPath()) in bodies: return str(prim.GetPath())
        prim = prim.GetParent()
    return None


def export_model(env, out):
    stage = env.sim.stage; root = stage.GetPrimAtPath(ROBOT); cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    bodies = {str(p.GetPath()): p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    body_T = {k: mat(cache.GetLocalToWorldTransform(p)) for k, p in bodies.items()}
    for k, T in body_T.items():
        s = np.linalg.norm(T[:3, :3], axis=0); assert np.allclose(s, 1, atol=1e-4), (k, s)   # FK assumes unscaled bodies
    meshes, skipped = {}, []
    for p in prims:
        if not p.IsA(UsdGeom.Mesh): continue
        path = str(p.GetPath()); low = path.lower()
        purpose = UsdGeom.Imageable(p).ComputePurpose()
        if "collision" in low or purpose == UsdGeom.Tokens.guide or UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible:
            skipped.append(path); continue
        body = owner_body(p, bodies)
        if body is None: skipped.append(path); continue
        m = UsdGeom.Mesh(p); pts = np.asarray(m.GetPointsAttr().Get(), np.float64)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get()); idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
        tris, o = [], 0
        for c in counts:
            tris += [(idx[o], idx[o + j], idx[o + j + 1]) for j in range(1, c - 1)]; o += c
        T = np.linalg.inv(body_T[body]) @ mat(cache.GetLocalToWorldTransform(p))
        v = pts @ T[:3, :3].T + T[:3, 3]
        name = body.rsplit("/", 1)[1]; meshes.setdefault(name, []).append((v.astype(np.float32), np.asarray(tris, np.int32), path))
    joints = []
    for p in prims:
        if not p.IsA(UsdPhysics.Joint): continue
        j = UsdPhysics.Joint(p); b0 = j.GetBody0Rel().GetTargets(); b1 = j.GetBody1Rel().GetTargets()
        def q(r): return [r.GetReal(), *r.GetImaginary()]   # wxyz
        rec = dict(name=p.GetName(), type=p.GetTypeName(), body0=str(b0[0]).rsplit("/", 1)[1] if b0 else None, body1=str(b1[0]).rsplit("/", 1)[1] if b1 else None,
                   local_pos0=list(j.GetLocalPos0Attr().Get()), local_rot0=q(j.GetLocalRot0Attr().Get()), local_pos1=list(j.GetLocalPos1Attr().Get()), local_rot1=q(j.GetLocalRot1Attr().Get()))
        if p.IsA(UsdPhysics.RevoluteJoint):
            r = UsdPhysics.RevoluteJoint(p); rec.update(axis=r.GetAxisAttr().Get(), lower_deg=r.GetLowerLimitAttr().Get(), upper_deg=r.GetUpperLimitAttr().Get())
        joints.append(rec)
    body_names = list(env.hand.body_names); joint_names = list(env.hand.joint_names)
    model = dict(source="Isaac Lab 1.2 ShadowHandEnvCfg robot (shadow_hand_instanceable.usd), exported from the live stage",
                 joint_names=joint_names, body_names=body_names, root_body=body_names[0],
                 joint_lower_deg=np.rad2deg(env.hand_dof_lower_limits[0].cpu().numpy()).tolist(), joint_upper_deg=np.rad2deg(env.hand_dof_upper_limits[0].cpu().numpy()).tolist(),
                 active_indices=list(map(int, env.active)), passive_indices=list(map(int, env.passive)), joints=joints,
                 mesh_links=sorted(meshes), mesh_source_paths={k: [m[2] for m in v] for k, v in meshes.items()}, skipped_mesh_paths=skipped,
                 units=dict(length="m", angle="deg (model.json limits); simulator radians converted at this boundary"), quaternion_order="wxyz")
    (out / "model.json").write_text(json.dumps(model, indent=1))
    arrays = {}
    for k, v in meshes.items():
        verts, faces, off = [], [], 0
        for vv, ff, _ in v: verts.append(vv); faces.append(ff + off); off += len(vv)
        arrays[f"{k}/vertices"] = np.concatenate(verts); arrays[f"{k}/faces"] = np.concatenate(faces)
    np.savez_compressed(out / "meshes.npz", **arrays)
    print(f"EXPORT bodies={len(bodies)} mesh_links={len(meshes)} triangles={sum(len(a) for k, a in arrays.items() if k.endswith('faces'))} joints={len(joints)} skipped_meshes={len(skipped)}", flush=True)
    return model


def body_poses(env):
    """Body poses relative to the root body: (B, 4, 4) per env."""
    st = env.hand.data.body_state_w[..., :7].cpu().numpy().astype(np.float64)
    def T(s):
        w, x, y, z = s[3:7]; R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = s[:3]; return M
    out = []
    for e in range(st.shape[0]):
        Ts = np.stack([T(s) for s in st[e]]); out.append(np.linalg.inv(Ts[0]) @ Ts)
    return np.stack(out)


def fk_check(env, out, rng):
    lo, hi = env.hand_dof_lower_limits[0], env.hand_dof_upper_limits[0]; qs, poses = [], []
    for _ in range(args.fk_samples):
        q = lo + (hi - lo) * torch.as_tensor(rng.uniform(0.05, 0.95, len(lo)), dtype=torch.float32, device=env.device)
        q = q.repeat(env.num_envs, 1); env.hand.write_joint_state_to_sim(q, torch.zeros_like(q)); env.hand.set_joint_position_target(q[:, env.active], joint_ids=env.active)
        env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(env.physics_dt)
        qs.append(env.hand.data.joint_pos[0].cpu().numpy().astype(np.float64)); poses.append(body_poses(env)[0])   # FK is checked against the achieved joints
    np.savez_compressed(out / "fk_check.npz", q_rad=np.stack(qs), body_T=np.stack(poses), body_names=np.array(env.hand.body_names))
    print(f"FK_CHECK samples={len(qs)}", flush=True)


def episodes(env, out):
    set_control_hz(args.control_hz); HZ = args.control_hz; N = env.num_envs; T = round(args.seconds * HZ)
    lim = Limits(env.action_names, env.low.cpu().numpy(), env.high.cpu().numpy()); speed = float(np.deg2rad(args.speed_deg))
    rngs = [np.random.default_rng(args.seed + i) for i in range(N)]
    programs = [generate(r, lim, T, args.families[i], speed=speed) for i, r in enumerate(rngs)]
    requested = np.stack([p[0] for p in programs], 1)
    env.reset_physical(); last = env.last_target.cpu().numpy().astype(np.float64); start = requested[0].astype(np.float64)
    for _ in range(2 * HZ):   # settle to the program's first target (not recorded)
        last = np.stack([apply_slew(start[i], last[i], 4.0, lim) for i in range(N)]); env.advance(torch.as_tensor(last, dtype=torch.float32, device=env.device))
    q, qd, poses, acts = [env.hand.data.joint_pos.cpu().numpy()], [env.hand.data.joint_vel.cpu().numpy()], [body_poses(env)], []
    for t in range(T):
        last = np.stack([apply_slew(requested[t, i].astype(np.float64), last[i], programs[i][1], lim) for i in range(N)])
        env.advance(torch.as_tensor(last, dtype=torch.float32, device=env.device))
        acts.append(env.last_target.cpu().numpy()); q.append(env.hand.data.joint_pos.cpu().numpy()); qd.append(env.hand.data.joint_vel.cpu().numpy()); poses.append(body_poses(env))
    q = np.rad2deg(np.stack(q, 1)); qd = np.rad2deg(np.stack(qd, 1)); acts = np.rad2deg(np.stack(acts, 1))
    np.savez_compressed(out / "episodes.npz", q_deg=q.astype(np.float32), qdot_deg_s=qd.astype(np.float32), action_deg=acts.astype(np.float32), body_T=np.stack(poses, 1).astype(np.float32),
                        families=np.array(args.families), seeds=np.arange(args.seed, args.seed + N), control_hz=HZ, speed_deg_s=args.speed_deg,
                        joint_names=np.array(env.hand.joint_names), action_names=np.array(env.action_names), body_names=np.array(env.hand.body_names))
    print(f"EPISODES n={N} steps={T} hz={HZ} speed={args.speed_deg} max|q|={np.abs(q).max():.1f} deg", flush=True)


def main():
    torch.set_num_threads(1); assert sorted(os.sched_getaffinity(0)) == __import__('shadow_dynamics.resources', fromlist=['budget_cpus']).budget_cpus()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    args.num_envs = len(args.families); args.seed_env = 0
    cfg = make_cfg(argparse.Namespace(**{**vars(args), "seed": 0})); cfg.object_cfg.spawn.rigid_props.kinematic_enabled = True
    env = ResearchShadowEnv(cfg)
    try:
        env.configure_research("free")
        export_model(env, out); fk_check(env, out, np.random.default_rng(args.seed)); episodes(env, out)
        print("EXPORT_COMPLETE " + str(out), flush=True)
    finally:
        env.close()


try: main()
except Exception:
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
finally: app.close()
