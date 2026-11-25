"""Experiment 49 self-exploration collector: the Shadow Hand (gravity on) runs motion programs and records what a real hand could record.

Programs: --programs <npz> (commanded (N, T, 20) deg, speed, family, seed; from design_tendon_excitation.py) or --random N from the structured
generator (structured_excitation) at --speeds. Every program is repaired and checked kinematically (FK of the link meshes, penetration
<= --precheck-mm) before it runs; failing programs are dropped. Uses only allowed knowledge: kinematics, joint limits, link meshes.

Recorded at every physics substep (120 Hz): joint angles and velocities (24), the applied target (20), and Isaac's drive torque (20; the
simulated tendon-force sensor, within 3.5% of the implicit drive force, gate 0). Recorded at every control step: the validity class (0 free,
1 self-contact, 2 invalid penetration, 3 numerical). Each episode is truncated at its first invalid state (class >= 2), as the structured
collector does; self-contact states are kept and labelled (fits mask them). Units: deg, deg/s, N m. Writes episodes.npz and manifest.json.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics.resources import restrict_simulator_gpu
PHYSICAL_GPU = restrict_simulator_gpu()
from omni.isaac.lab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", required=True); p.add_argument("--programs", default=None); p.add_argument("--random", type=int, default=0)
p.add_argument("--speeds", nargs="+", type=float, default=[1, 5, 10]); p.add_argument("--seconds", type=float, default=20.0); p.add_argument("--program-seed", type=int, default=4910000)
p.add_argument("--precheck-mm", type=float, default=1.0); p.add_argument("--settle-s", type=float, default=1.0); p.add_argument("--no-gravity", action="store_true")
p.add_argument("--manifest", default="artifacts/shadow_structured_v3/train_speeds/manifest.json", help="action names and joint limits only")
p.add_argument("--poses", default="shadow_dynamics/data/shadow_pose_library.json")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.physics_hz = 120; args.control_hz = 20; args.mode = "free"; args.seed = 0; args.gravity = not args.no_gravity
launcher = AppLauncher(args, multi_gpu=False); app = launcher.app
import numpy as np, torch
from shadow_dynamics import structured_excitation as sx
from shadow_dynamics.hand_geometry import HandGeometry, TorchFK


def programs():
    man = json.loads(Path(args.manifest).read_text())["mapping"]; lib = json.loads(Path(args.poses).read_text()); lim = np.array(man["action_limits_deg"])
    spec = sx.Spec(man["action_names"], lim[:, 0], lim[:, 1], lib["poses_deg"]); T = int(round(args.seconds * 20))
    if args.programs:
        z = np.load(args.programs); P = z["commanded"].astype(np.float32); info = dict(speed=z["speed"].astype(float), family=z["family"].astype(str), seed=z["seed"].astype(int))
    else:
        gen = sx.Generator(spec, args.seconds, 20); rng = np.random.default_rng(args.program_seed); P, sp, fam = [], [], []
        shares = np.array([sx.SPEED_SHARES.get(int(s), 0.2) for s in args.speeds]); shares /= shares.sum()
        for i in range(args.random):
            s = float(rng.choice(args.speeds, p=shares)); f = sx.FAMILIES[int(rng.integers(len(sx.FAMILIES)))]; pg, _ = gen.generate(rng, f, s); P.append(pg.astype(np.float32)); sp.append(s); fam.append(f)
        P = np.stack(P); info = dict(speed=np.array(sp), family=np.array(fam), seed=np.arange(args.program_seed, args.program_seed + len(P)))
    return man, spec, P[:, :T], info


def main():
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False); t0 = time.time()
    man, spec, P, info = programs(); dev = args.device
    geo = HandGeometry.from_viz_model("artifacts/viz_hand_model"); fk = TorchFK("artifacts/viz_hand_model", dev)
    checker = sx.ProgramChecker(geo, fk, man["joint_names"], man["action_names"], args.precheck_mm / 1000, dev); P, _ = checker.repair(P)
    paths = np.stack([sx.applied_path(pg, pg[0], s, spec.low, spec.high) for pg, s in zip(P, info["speed"])])
    ok = (checker.check_paths(paths) <= checker.threshold) & (checker.max_penetration(paths[:, 0]) <= checker.threshold)
    keep = np.where(ok)[0]; assert len(keep), "no program passed the kinematic check"
    paths = paths[keep]; info = {k: v[keep] for k, v in info.items()}; N, T = paths.shape[0], paths.shape[1]
    args.num_envs = N
    from shadow_dynamics.env import ResearchShadowEnv, make_cfg
    from shadow_dynamics.validity import ValidityMonitor
    env = ResearchShadowEnv(make_cfg(args))
    try:
        env.configure_research("free"); h = env.hand; act = env.active
        assert [a.replace("robot0_", "") for a in env.action_names] == [a.replace("robot0_", "") for a in man["action_names"]]
        mon = ValidityMonitor(env, geo, dict(touch_separation_m=0.0, touch_force_n=1e-3, limit_excess_rad=0.5, velocity_rad_s=200.0, hull_invalid_m=0.003, hull_palm_invalid_m=0.008,
                                             physx_penetration_m=0.002, hull_contact_m=0.0005))
        start = torch.zeros(N, 24, device=env.device); start[:, act] = torch.deg2rad(torch.as_tensor(paths[:, 0], dtype=torch.float32, device=env.device))
        env.reset_physical(start)
        for _ in range(int(args.settle_s * 20)): env.advance(start[:, act])
        rec = dict(q=[h.data.joint_pos.clone()], qd=[h.data.joint_vel.clone()], tau=[], target=[])
        def on_sub():
            rec["q"].append(h.data.joint_pos.clone()); rec["qd"].append(h.data.joint_vel.clone()); rec["tau"].append(h.data.applied_torque[:, act].clone())
        valid = [mon.measure()["validity"].clone()]
        for t in range(T):
            env.advance(torch.deg2rad(torch.as_tensor(paths[:, t], dtype=torch.float32, device=env.device)), on_substep=on_sub)
            for _ in range(6): rec["target"].append(env.last_target.clone())
            valid.append(mon.measure()["validity"].clone())
        V = torch.stack(valid, 1).cpu().numpy()                                     # (N, T+1)
        bad = V >= 2; n_ok = np.where(bad.any(1), bad.argmax(1), T + 1)              # control states kept per episode (before the first invalid)
        cat = lambda k: torch.rad2deg(torch.stack(rec[k], 1)).cpu().numpy().astype(np.float32)
        np.savez_compressed(out / "episodes.npz", q_deg=cat("q"), qd_deg=cat("qd"), target_deg=cat("target"), tau=torch.stack(rec["tau"], 1).cpu().numpy().astype(np.float32),
                            validity=V.astype(np.uint8), n_valid_states=n_ok, speed=info["speed"], family=info["family"], seed=info["seed"], commanded_deg=paths.astype(np.float32))
        valid_free_s = float(sum(((V[i, :n_ok[i]] == 0).sum()) for i in range(N)) / 20)
        man_out = dict(schema="exp49-collect-1", gravity=args.gravity, gravity_root=env.gravity_root(), physics_hz=120, control_hz=20, programs_requested=int(len(P)) if args.programs else int(args.random),
                       programs_run=int(N), programs_dropped_precheck=int((~ok).sum()), seconds=args.seconds, speeds=info["speed"].tolist(), families=info["family"].tolist(),
                       truncated=int((n_ok <= T).sum()), valid_minutes=float(n_ok.sum() / 20 / 60), valid_free_minutes=valid_free_s / 60,
                       validity_fraction={n_: float((V == i).mean()) for i, n_ in enumerate(["free", "contact", "invalid_pen", "invalid_num"])},
                       source=args.programs or f"random:{args.program_seed}", joint_names=h.joint_names, action_names=env.action_names, wall_s=time.time() - t0)
        (out / "manifest.json").write_text(json.dumps(man_out, indent=1))
        print("COLLECT_EXP49_COMPLETE", json.dumps({k: man_out[k] for k in ("programs_run", "programs_dropped_precheck", "valid_minutes", "valid_free_minutes", "validity_fraction")}), flush=True)
    finally:
        env.close()


try: main()
except Exception:
    import traceback; traceback.print_exc(); sys.stdout.flush()
finally: app.close()
