"""Experiment 04: turn a cube lying on the palm of the Inspire hand (palm up) to a goal yaw with CEM-MPC; Isaac cube in the planner loop.

Envs: per task one live env + NB branch envs. Every 3 control steps CEM (2 iterations x NB candidates of 4 held-target knots over 12
steps) copies the live state (hand joints, cube pose/velocity) into the task's branches, rolls every candidate out for 0.6 s and scores
the cube yaw error (+2x terminal), a drop penalty and an off-centre penalty; live envs are restored afterwards and execute the best
candidate's first 3 steps. Branch hand per arm (see experiments/04_*.md): oracle (true servo parameters), kin (identified model,
joint state written every substep), soft (identified model prediction as a stiff drive target, u_max = identified). Optional layers on
the executed command: CBF self-collision filter, TVLQR tracking of the chosen candidate's branch-measured states (exp 54 cube-aware).
Output: <output>/results.json, traces.npz.

Usage: ./run_python.sh scripts/evaluate/exp04_cube.py --headless --device cuda:0 --output artifacts/exp04/cube --fits artifacts/exp02/explore/fits.pt
"""
import argparse, json, math, time
from pathlib import Path
from omni.isaac.lab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); ap.add_argument("--fits", required=True)
ap.add_argument("--arms", default="oracle,kin_CL,soft_CL,soft_N0,soft_CL_cbf,soft_CL_stack"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--tasks", type=int, default=30); ap.add_argument("--nb", type=int, default=16); ap.add_argument("--iters", type=int, default=2)
ap.add_argument("--goals-deg", default="20,-20,40,-40"); ap.add_argument("--seconds", type=float, default=10.0); ap.add_argument("--size", type=float, default=0.04)
ap.add_argument("--tol", type=float, default=0.1); ap.add_argument("--palm-x", type=float, default=-0.105); ap.add_argument("--soft-gain", type=float, default=10.0); ap.add_argument("--profile", action="store_true"); ap.add_argument("--horizon", type=int, default=12); ap.add_argument("--gate", default=None)
AppLauncher.add_app_launcher_args(ap); args = ap.parse_args(); app = AppLauncher(args).app

import numpy as np, torch
from inspire_dynamics.sim import InspireSim, true_hand_params, spec_params, CONTROL_DT, SUBSTEPS
from inspire_dynamics.torch_kin import InspireKin
from inspire_dynamics import servo_model as sm, control as C

ROT = (0.70710678, 0.0, -0.70710678, 0.0); PALM_Z = 0.523; PALM_C = (args.palm_x, 0.0)
out = Path(args.output); out.mkdir(parents=True, exist_ok=False); dev = args.device
NT, NB, H, RP = args.tasks, args.nb, args.horizon, 3; G = NB + 1; N = NT * G
S = InspireSim(num_envs=N, device=dev, cube=True, root_rot=ROT, spacing=0.6, cube_size=args.size, cube_mass=0.03, cube_pos=(PALM_C[0], 0.0, PALM_Z + args.size / 2 + 0.01))
kin = InspireKin(dev); lo, hi = S.lo, S.hi; lod, hid = lo / C.D2R, hi / C.D2R
live = torch.arange(NT, device=dev) * G; br = (live[:, None] + 1 + torch.arange(NB, device=dev)[None]).reshape(-1)
F = torch.load(args.fits); TH = dict(CL=F["fits"]["CL"][-1]["theta"][0:1].to(dev), N0=F["th_N0"][None].to(dev))
if "REG" in F["fits"]: TH["REG"] = F["fits"]["REG"][-1]["theta"][0:1].to(dev)   # experiment 04b: MAP refit (fit_regularised.py)
hand = true_hand_params(0); goals = [math.radians(float(g)) for g in args.goals_deg.split(",")]; steps = int(args.seconds / CONTROL_DT)
ia = torch.tensor(S.ia, device=dev); il = torch.tensor(S.il, device=dev)


def yaw(q):
    w, x, y, z = q.unbind(-1); return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a): return torch.atan2(torch.sin(a), torch.cos(a))


def off_palm(c):  # (..., 13) -> distance from palm centre (m), dropped flag
    off = ((c[..., 0] - PALM_C[0]) ** 2 + (c[..., 1] - PALM_C[1]) ** 2).sqrt(); return off, (c[..., 2] < PALM_Z) | (off > 0.06)


def full_q(p):   # servo (n, 6) rad -> 12 joints in Isaac order
    q = torch.zeros(p.shape[0], len(S.names), device=dev); q[:, ia] = p; q[:, il] = S.linkage_of(p); return q


GATE = json.load(open(Path(args.gate) if args.gate else Path(args.fits).parent.parent / "gate/gate.json")); MLINK = torch.tensor(GATE["Mbar_diag_mean"])


def set_branch_hand(mode, th):
    P = {k: v[None].expand(N, 6).clone() for k, v in hand.items()}
    if mode == "param":   # the planner's simulated hand carries the identified parameters (armature = identified m - link inertia)
        u = sm.unpack(th.cpu())
        for k_, v_ in dict(K=u["K"][0], D=u["D"][0], vmax=u["vmax"][0], effort=u["umax"][0], armature=(u["m"][0] - MLINK).clamp_min(1e-6)).items():
            P[k_][br.cpu()] = v_.float()
    if mode in ("soft", "kin"):
        u = sm.unpack(th.cpu())
        for k_, v_ in dict(K=hand["K"] * 0 + 10.0 * args.soft_gain, D=torch.full((6,), 0.07 * args.soft_gain), vmax=torch.full((6,), 20.0), effort=u["umax"][0]).items():
            P[k_][br.cpu()] = v_.float()
    S.set_servo_params(**P)


def knots_to_U(Kn):   # (..., 4, 6) -> (..., H, 6), same interpolation as control.CEM
    t = torch.arange(H, device=dev, dtype=torch.float32); sp = H / 4; pos = (torch.arange(4, device=dev) + 1) * sp - 1
    w = torch.clamp(1 - (t[:, None] - pos[None]).abs() / sp, 0); w[: int(pos[0]) + 1, 0] = 1; w[int(pos[-1]):, -1] = 1; w = w / w.sum(1, keepdim=True)
    return torch.einsum("tk,...kd->...td", w, Kn)


def rollout_branches(mode, th, U, liveS):
    """U (NT, NB, H, 6) rad targets. Copies the live state into branches, runs H steps; returns cube states (NT, NB, H, 13), servo pos (NT, NB, H, 6)."""
    q0, qd0, cube0, p0, v0 = liveS
    rep = lambda z: z.repeat_interleave(NB, 0); tp = time.time()
    S.write_hand_state(rep(q0), rep(qd0), U[:, :, 0].reshape(-1, 6), br); S.write_cube_state(rep(cube0), br)
    if args.profile: torch.cuda.synchronize(); print(f"  write branches {1000 * (time.time() - tp):.1f} ms", flush=True); tp = time.time()
    if mode not in ("oracle", "param"):   # model prediction at substep resolution
        pp, vv = rep(p0)[None], rep(v0)[None]; Pk = []
        Uf = U.reshape(-1, H, 6)
        for t in range(H):
            a = torch.maximum(torch.minimum(Uf[:, t], hi), lo)[None]; sub = []
            for _ in range(SUBSTEPS): pp, vv = sm.substep(th, pp, vv, a, lo, hi); sub.append((pp[0].clone(), vv[0].clone()))
            Pk.append(sub)
    Cs, Ps = [], []
    for t in range(H):
        for s_ in range(SUBSTEPS):
            if mode in ("oracle", "param"): tgt = U[:, :, t].reshape(-1, 6)
            else:
                pp_, vv_ = Pk[t][s_]
                if mode == "kin":
                    qq = full_q(pp_); S.hand.write_joint_state_to_sim(qq, torch.zeros_like(qq), env_ids=br)
                tgt = pp_
            tt = S.hand.data.joint_pos_target.clone(); tt[br[:, None], ia[None]] = torch.max(torch.min(tgt, hi), lo); S.hand.set_joint_position_target(tt)
            S.scene.write_data_to_sim(); S.sim.step(render=False); S.scene.update(S.sim.get_physics_dt())
        Cs.append(S.cube_state()[br].view(NT, NB, 13)); Ps.append(S.hand.data.joint_pos[br][:, ia].view(NT, NB, 6))
        if args.profile and t < 2: torch.cuda.synchronize(); print(f"  step {t}: {1000 * (time.time() - tp):.1f} ms", flush=True); tp = time.time()
    return torch.stack(Cs, 2), torch.stack(Ps, 2)


def cost(Cb, U, goal, up):
    yw = wrap(yaw(Cb[..., 3:7]) - goal[:, None, None]).abs(); off, drop = off_palm(Cb)
    w = torch.full((H,), 1.0 / H, device=dev); w[-1] += 2.0
    J = (yw * w).sum(-1) + 10.0 * drop.any(-1).float() + (torch.clamp(off - 0.03, min=0) / 0.03 * w).sum(-1)
    dU = torch.diff(torch.cat([up[:, None, None].expand(-1, NB, 1, 6), U], 2), dim=2)
    return J + 1e-3 * (dU / C.D2R).square().mean((2, 3))


def run(arm, seed):
    mode = "oracle" if arm == "oracle" else arm.split("_")[0]; th = TH[arm.split("_")[1]] if "_" in arm else TH["CL"]
    use_cbf = arm.endswith("_cbf") or arm.endswith("_stack"); use_tv = arm.endswith("_stack")
    set_branch_hand(mode, th); f = C.ModelStep(th, lo, hi); cbf = C.CBF(kin, f, lod, hid) if use_cbf else None; il_ = C.ILQR(f, lod, hid, H=H) if use_tv else None
    g = torch.Generator(device=dev).manual_seed(1000 + seed); rng = np.random.default_rng(seed)
    goal = torch.tensor([goals[i % len(goals)] for i in rng.permutation(NT)], dtype=torch.float32, device=dev)
    S.reset(); cs = S.cube.data.default_root_state.clone(); cs[:, :3] += S.scene.env_origins
    S.cube.write_root_state_to_sim(cs)                       # fresh cube every episode (the hand reset alone left it where the last arm put it)
    for _ in range(30): S.advance(torch.zeros(N, 6, device=dev))
    c0 = S.cube_state()[live]; y0 = yaw(c0[:, 3:7]); goal_abs = wrap(y0 + goal)
    prev = torch.zeros(NT, 6, device=dev); mean = None; queue = None; plan_t = []; X_hist = []; Cube_hist = []; Q_hist = []
    pl_prev = S.state()["qa"][live]; reached = torch.zeros(NT, dtype=torch.bool, device=dev); t_reach = torch.full((NT,), float("nan"))
    dropped = torch.zeros(NT, dtype=torch.bool, device=dev); Xref = None
    for k in range(steps):
        st = S.state(); pl = st["qa"][live]; vl = (pl - pl_prev) / CONTROL_DT
        if k % RP == 0:
            torch.cuda.synchronize(); t0 = time.time()
            liveS = (st["q"][live].clone(), st["qd"][live].clone(), S.cube_state()[live].clone(), pl.clone(), vl.clone())
            tgt_saved = S.hand.data.joint_pos_target[live].clone()
            if mean is None: mean = prev[:, None].expand(NT, 4, 6).clone()
            std = ((hi - lo) * 0.25).expand(NT, 4, 6); best = (torch.full((NT,), float("inf"), device=dev), mean.clone(), None, None)
            for it in range(args.iters):
                Kn = mean[:, None] + std[:, None] * torch.randn(NT, NB, 4, 6, generator=g, device=dev)
                Kn[:, 0] = prev[:, None]; Kn[:, 1] = mean; Kn = torch.maximum(torch.minimum(Kn, hi), lo); U = knots_to_U(Kn)
                Cb, Pb = rollout_branches(mode, th, U, liveS); J = cost(Cb, U, goal_abs, prev)
                el = J.topk(4, dim=1, largest=False).indices; KE = torch.gather(Kn, 1, el[..., None, None].expand(-1, -1, 4, 6))
                mean = KE.mean(1); std = KE.std(1).clamp_min(0.02)
                jb, ib = J.min(1); imp = jb < best[0]; sel = torch.arange(NT, device=dev)
                best = (torch.where(imp, jb, best[0]), torch.where(imp[:, None, None], Kn[sel, ib], best[1]),
                        torch.where(imp[:, None, None], U[sel, ib], best[2]) if best[2] is not None else U[sel, ib],
                        torch.where(imp[:, None, None], Pb[sel, ib], best[3]) if best[3] is not None else Pb[sel, ib])
            if args.profile: torch.cuda.synchronize(); print(f" replan rollouts {1000 * (time.time() - t0):.0f} ms", flush=True)
            # restore the live envs (they were stepped along with the branches)
            S.write_hand_state(liveS[0], liveS[1], tgt_saved[:, ia], live); S.write_cube_state(liveS[2], live)
            queue = best[2]; mean = torch.cat([best[1][:, 1:], best[1][:, -1:]], 1)
            if use_tv:   # TVLQR about the chosen candidate and its branch-measured (contact-aware) servo trajectory
                x0 = torch.cat([pl, vl], -1) / C.D2R; Pd = best[3] / C.D2R; Vd = torch.diff(torch.cat([pl[:, None] / C.D2R, Pd], 1), dim=1) / CONTROL_DT
                Xref = torch.cat([x0[:, None], torch.cat([Pd, Vd], -1)], 1); il_.U = queue / C.D2R; il_.X = Xref
                fx, fu = il_._jac(Xref, il_.U); il_.mu = torch.full((NT,), 1e-2, device=dev); il_.K, _ = il_._backward(Xref, il_.U, prev / C.D2R, Pd, fx, fu, il_.mu)
            torch.cuda.synchronize(); plan_t.append(time.time() - t0)
        j = k % RP; cmd = queue[:, j]
        if use_tv: cmd = il_.feedback(j, torch.cat([pl, vl], -1) / C.D2R) * C.D2R
        if cbf is not None: cmd = cbf.filter(torch.cat([pl, vl], -1) / C.D2R, cmd / C.D2R, torch.full((NT,), 5e-4, device=dev)) * C.D2R
        full = S.hand.data.joint_pos_target[:, ia].clone(); full[live] = cmd
        S.advance(full); prev = cmd; pl_prev = pl
        c = S.cube_state()[live]; err = wrap(yaw(c[:, 3:7]) - goal_abs).abs(); _, dr = off_palm(c); dropped |= dr
        newly = (err < args.tol) & ~dropped & ~reached; t_reach[newly.cpu()] = (k + 1) * CONTROL_DT; reached |= newly
        Cube_hist.append(c.cpu()); Q_hist.append(S.state()["q"][live].cpu())
    c = S.cube_state()[live]; fin_err = wrap(yaw(c[:, 3:7]) - goal_abs).abs()
    r = dict(goal_deg=np.degrees(goal.cpu().numpy()), reached=reached.cpu().numpy(), final_ok=((fin_err < args.tol) & ~dropped).cpu().numpy(),
             dropped=dropped.cpu().numpy(), final_err_deg=np.degrees(fin_err.cpu().numpy()), t_reach=t_reach.numpy())
    summ = dict(success=float(r["reached"].mean()), final_success=float(r["final_ok"].mean()), drop=float(r["dropped"].mean()),
                plan_ms_median=float(np.median(plan_t) * 1000), t_reach_median=float(np.nanmedian(r["t_reach"])) if np.isfinite(r["t_reach"]).any() else None)
    if cbf is not None: summ["cbf"] = dict(cbf.stats)
    return summ, r, torch.stack(Q_hist, 1).numpy(), torch.stack(Cube_hist, 1).numpy(), goal_abs.cpu().numpy()


results, traces = {}, {}
for seed in [int(s) for s in args.seeds.split(",")]:
    for arm in args.arms.split(","):
        t0 = time.time(); summ, r, Q, Cu, ga = run(arm, seed); summ["seconds"] = time.time() - t0
        results[f"{arm}|{seed}"] = dict(summ, per_task={k: np.where(np.isfinite(v.astype(float)), v.astype(float), -1.0).tolist() for k, v in r.items()})
        traces[f"{arm}__{seed}__q"] = Q.astype(np.float32); traces[f"{arm}__{seed}__cube"] = Cu.astype(np.float32); traces[f"{arm}__{seed}__goal"] = ga
        print(f"RUN {arm:14s} seed {seed}: success {summ['success']:.3f} final {summ['final_success']:.3f} drop {summ['drop']:.3f} "
              f"plan {summ['plan_ms_median']:.0f} ms [{summ['seconds']:.0f} s]", flush=True)
        json.dump(dict(args=vars(args), runs=results), open(out / "results.json", "w"), indent=1)
np.savez_compressed(out / "traces.npz", **traces, joint_names=np.array(S.names), dt=CONTROL_DT)
print("EXP04_COMPLETE"); app.close()
