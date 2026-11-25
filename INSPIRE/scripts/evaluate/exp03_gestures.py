"""Experiment 03: gesture benchmark on the Inspire hand (hand 0 of experiment 02) with the controllers of SHADOW exp 50.

Tasks per seed (--task-set standard): 12 gestures x 10 starts (5 other library gestures + 5 random contact-free poses) = 120;
--task-set crossing: the 6 library pairs whose straight line collides + random contact-free starts whose straight line to a library
goal penetrates > 2 mm, 120 per seed; run in parallel (one env each);
3 seeds = 360 tasks per controller. Episode 4 s at 20 Hz. Success: final 6-servo RMSE < 2 deg (also: mean fingertip error < 5 mm).
Contact: fraction of steps whose measured pose has a checked link pair closer than 0 (conservative polytopes; Isaac prevents real
penetration, so this counts finger-on-finger contact). Every controller uses the model identified by closed-loop self-exploration
(exp 02, arm CL, 3 min, hand 0) unless the run name ends in _N0 (spec model) or _TRUE (true parameters).
Latency: wall time per re-solve (batched over the 120 tasks of a seed) and, after the runs, single-hand (E = 1, CPU and GPU) and\nShadow-convention batch (E = 150, GPU) timings of every layer.

Usage: ./run_python.sh scripts/evaluate/exp03_gestures.py --headless --device cuda:0 --output artifacts/exp03/bench --fits artifacts/exp02/explore/fits.pt
"""
import argparse, json, time
from pathlib import Path
from omni.isaac.lab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); ap.add_argument("--fits", required=True)
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--seconds", type=float, default=4.0); ap.add_argument("--replan", type=int, default=3)
ap.add_argument("--runs", default="direct,cem,ilqr,rrt_direct,rrt_cem,rrt_ilqr,cem_cbf,full,full_N0")
ap.add_argument("--lookahead", type=float, default=20.0); ap.add_argument("--stall-s", type=float, default=1.0)
ap.add_argument("--task-set", default="standard", choices=["standard", "crossing"]); ap.add_argument("--joint-tol", type=float, default=2.0); ap.add_argument("--tip-tol", type=float, default=5.0)
AppLauncher.add_app_launcher_args(ap); args = ap.parse_args(); app = AppLauncher(args).app

import numpy as np, torch
from inspire_dynamics.sim import InspireSim, true_hand_params, spec_params, CONTROL_DT
from inspire_dynamics.torch_kin import InspireKin
from inspire_dynamics import control as C
from shadow_dynamics.rrt_connect import PathFollower

out = Path(args.output); out.mkdir(parents=True, exist_ok=False); dev = args.device; HZ = 20
LIB = json.load(open(Path(__file__).resolve().parents[2] / "assets/inspire_hand/gesture_library.json"))["gestures"]; GN = list(LIB)
F = torch.load(args.fits); TH = dict(CL=F["fits"]["CL"][-1]["theta"][0:1].to(dev), N0=F["th_N0"][None].to(dev), TRUE=F["th_true"][0:1].to(dev))
kin = InspireKin(dev)
CROSS = []
if args.task_set == "crossing":   # every ordered library pair whose straight servo-space line collides (> 0.5 mm, checked every 1 deg)
    for a_ in GN:
        for b_ in GN:
            if a_ == b_: continue
            A0, B0 = torch.tensor(LIB[a_]["a"], device=dev), torch.tensor(LIB[b_]["a"], device=dev)
            n_ = int(max(2, (B0 - A0).abs().max() / 0.0175)); L_ = A0 + torch.linspace(0, 1, n_, device=dev)[:, None] * (B0 - A0)
            if kin.penetration(L_).max() > 5e-4: CROSS.append((a_, b_))
    print("crossing pairs", len(CROSS), CROSS, flush=True)
E = len(GN) * 10; S = InspireSim(num_envs=E, device=dev); lo, hi = S.lo, S.hi
hand = true_hand_params(0); S.set_servo_params(**{k: v[None].expand(E, 6) for k, v in hand.items()})
lod, hid = lo / C.D2R, hi / C.D2R; steps = int(args.seconds * HZ); runs = args.runs.split(",")


def tasks_for(seed):
    if args.task_set == "crossing":   # the colliding library pairs, then random contact-free starts whose straight line to a goal hits > 2 mm
        S0 = [LIB[a_]["a"] for a_, b_ in CROSS]; G0 = [LIB[b_]["a"] for a_, b_ in CROSS]; meta = [(b_, a_) for a_, b_ in CROSS]
        rng = np.random.default_rng(500 + seed)
        while len(S0) < E:
            a = torch.tensor(rng.uniform(lo.cpu(), hi.cpu()), dtype=torch.float32, device=dev); gname = GN[int(rng.integers(len(GN)))]; b = torch.tensor(LIB[gname]["a"], device=dev)
            if -kin.depth(a[None]).amax() < 0.002: continue
            L_ = a + torch.linspace(0, 1, 60, device=dev)[:, None] * (b - a)
            if kin.penetration(L_).max() > 2e-3: S0.append(a.tolist()); G0.append(LIB[gname]["a"]); meta.append((gname, "random_crossing"))
        return torch.tensor(S0, dtype=torch.float32, device=dev), torch.tensor(G0, dtype=torch.float32, device=dev), meta
    rng = np.random.default_rng(100 + seed); S0, G0, meta = [], [], []
    for gi, g in enumerate(GN):
        others = rng.choice([o for o in GN if o != g], 5, replace=False)
        for o in others: S0.append(LIB[o]["a"]); G0.append(LIB[g]["a"]); meta.append((g, str(o)))
        k = 0
        while k < 5:
            a = rng.uniform(lo.cpu(), hi.cpu())
            if -kin.depth(torch.tensor(a, dtype=torch.float32, device=dev)[None]).amax() >= 0.002: S0.append(a.tolist()); G0.append(LIB[g]["a"]); meta.append((g, "random")); k += 1
    return torch.tensor(S0, dtype=torch.float32, device=dev), torch.tensor(G0, dtype=torch.float32, device=dev), meta


def run(name, seed, S0, G0):
    mlab = "N0" if name.endswith("_N0") else ("TRUE" if name.endswith("_TRUE") else "CL"); base = name.rsplit("_", 1)[0] if mlab != "CL" else name
    f = C.ModelStep(TH[mlab], lo, hi); glob = base.startswith("rrt") or base == "full"; loc = "ilqr" if ("ilqr" in base or base == "full") else ("cem" if "cem" in base else "direct")
    use_cbf = base in ("cem_cbf", "full"); goal = G0 / C.D2R
    S.reset(S0); x = torch.cat([S0 / C.D2R, torch.zeros_like(S0)], -1); prev = x[:, :6].clone()
    ctl = C.CEM(f, lod, hid, seed=seed) if loc == "cem" else (C.ILQR(f, lod, hid) if loc == "ilqr" else None)
    cbf = C.CBF(kin, f, lod, hid) if use_cbf else None
    tol = torch.tensor(np.maximum(kin.penetration(S0), kin.penetration(G0)) + 0.5e-3, dtype=torch.float32, device=dev)
    info = dict(plan_s=[], cbf_s=[], rrt_s=0.0, rrt_replans=0)
    if glob:
        rr = C.make_rrt(kin, lod.cpu().numpy(), hid.cpu().numpy(), seed=seed); t0 = time.time()
        paths, pinfo = rr.plan(S0.cpu().numpy() / C.D2R, goal.cpu().numpy()); info["rrt_s"] = time.time() - t0
        info.update(rrt_needed=int((~pinfo["straight"]).sum()), rrt_unsolved=int((~pinfo["solved"]).sum()))
        fol = PathFollower(paths, goal.cpu().numpy(), args.lookahead); last_err = np.full(E, np.inf)
    Q = [S.state()["q"].cpu()]; QA = [S0.cpu()]; queue = None
    for k in range(steps):
        p_deg = x[:, :6]
        if glob:
            sr = int(args.stall_s * HZ); err = (p_deg - goal).square().mean(1).sqrt().cpu().numpy()
            fb_now = cbf.last_fallback.cpu().numpy() if (cbf is not None and cbf.last_fallback is not None) else np.zeros(E, bool)
            if (k and k % sr == 0) or fb_now.any():
                st = np.where((last_err - err < 0.5) & (err > args.joint_tol) & (fol.progress() < 0.98))[0] if (k and k % sr == 0) else np.zeros(0, int)
                st = np.union1d(st, np.where(fb_now)[0]).astype(int); info["fallback_replans"] = info.get("fallback_replans", 0) + int(fb_now.sum())
                if len(st):
                    t0 = time.time(); newp, _ = rr.plan(p_deg[st].cpu().numpy(), goal[st].cpu().numpy()); info["rrt_s"] += time.time() - t0; info["rrt_replans"] += len(st)
                    for j, i in enumerate(st): fol.set_path(int(i), newp[j], goal[i].cpu().numpy())
                if k and k % sr == 0: last_err = err
            ref = torch.tensor(fol.reference(p_deg.cpu().numpy()), dtype=torch.float32, device=dev)
        else: ref = goal
        if loc == "direct": cmd = ref
        elif loc == "cem":
            if k % args.replan == 0:
                torch.cuda.synchronize(); t0 = time.time(); queue = ctl.plan(x, prev, ref); torch.cuda.synchronize(); info["plan_s"].append(time.time() - t0)
            cmd = queue[:, k % args.replan]
        else:
            if k % args.replan == 0:
                torch.cuda.synchronize(); t0 = time.time(); ctl.plan(x, prev, ref, shift=args.replan if k else 0); torch.cuda.synchronize(); info["plan_s"].append(time.time() - t0)
            cmd = ctl.feedback(k % args.replan, x)
        if cbf is not None:
            torch.cuda.synchronize(); t0 = time.time(); cmd = cbf.filter(x, cmd, tol); torch.cuda.synchronize(); info["cbf_s"].append(time.time() - t0)
        S.advance(cmd * C.D2R); st_ = S.state(); pn = st_["qa"] / C.D2R
        x = torch.cat([pn, (pn - x[:, :6]) / CONTROL_DT], -1); prev = cmd
        Q.append(st_["q"].cpu()); QA.append(st_["qa"].cpu())
    QA = torch.stack(QA, 1); Q = torch.stack(Q, 1)                                                     # (E, T+1, 6), (E, T+1, 12)
    jerr = ((QA - G0.cpu()[:, None]) / C.D2R).square().mean(-1).sqrt()
    tq = kin.tips(QA.reshape(-1, 6).to(dev)).view(E, -1, 5, 3); tg = kin.tips(G0); terr = ((tq - tg[:, None]).norm(dim=-1).mean(-1) * 1000).cpu()
    pen = torch.tensor(kin.penetration(QA.reshape(-1, 6).to(dev)).reshape(E, -1))
    ok = jerr < args.joint_tol; t_tol = torch.where(ok.any(1), ok.float().argmax(1).float() / HZ, torch.full((E,), float("nan")))
    r = dict(final_rmse=jerr[:, -1].numpy(), final_tip_mm=terr[:, -1].numpy(), contact_frac=(pen > 0).float().mean(1).numpy(), deep_frac=(pen > 2e-3).float().mean(1).numpy(),
             max_pen_mm=(pen.amax(1) * 1000).numpy(), time_to_tol=t_tol.numpy())
    summ = dict(success=float((r["final_rmse"] < args.joint_tol).mean()), success_tip=float((r["final_tip_mm"] < args.tip_tol).mean()),
                rmse_mean=float(r["final_rmse"].mean()), contact_frac=float(r["contact_frac"].mean()), deep_frac=float(r["deep_frac"].mean()), time_to_tol_median=float(np.nanmedian(r["time_to_tol"])),
                plan_ms_median=float(np.median(info["plan_s"]) * 1000) if info["plan_s"] else 0.0, cbf_ms_median=float(np.median(info["cbf_s"]) * 1000) if info["cbf_s"] else 0.0,
                rrt_s=info["rrt_s"], rrt_replans=info["rrt_replans"], rrt_needed=info.get("rrt_needed", 0), rrt_unsolved=info.get("rrt_unsolved", 0))
    if cbf is not None: summ["cbf"] = dict(cbf.stats); summ["fallback_replans"] = info.get("fallback_replans", 0)
    return summ, r, Q.numpy().astype(np.float32)


results, traces = {}, {}
for seed in [int(s) for s in args.seeds.split(",")]:
    S0, G0, meta = tasks_for(seed)
    for name in runs:
        t0 = time.time(); summ, r, Q = run(name, seed, S0, G0); summ["seconds"] = time.time() - t0
        results[f"{name}|{seed}"] = dict(summ, per_task={k: np.where(np.isfinite(v), v, -1.0).tolist() for k, v in r.items()}, meta=meta)
        traces[f"{name}|{seed}"] = Q
        print(f"RUN {name:12s} seed {seed}: success {summ['success']:.3f} tip {summ['success_tip']:.3f} rmse {summ['rmse_mean']:.2f} contact {summ['contact_frac']:.3f} "
              f"plan {summ['plan_ms_median']:.1f} ms cbf {summ['cbf_ms_median']:.1f} ms rrt {summ['rrt_s']:.1f} s [{summ['seconds']:.0f} s]", flush=True)
        json.dump(dict(args=vars(args), gestures=GN, runs=results), open(out / "results.json", "w"), indent=1)

# single-task latency (E = 1) and Shadow-convention batch (E = 150), from random states toward random library goals
lat = {}
for ldev, Eb in (("cpu", 1), (dev, 1), (dev, 150)):
    kl = kin if ldev == dev else InspireKin(ldev); lol, hil = lo.to(ldev), hi.to(ldev); lodl, hidl = lol / C.D2R, hil / C.D2R
    f = C.ModelStep(TH["CL"].to(ldev), lol, hil); sync = (lambda: torch.cuda.synchronize()) if ldev.startswith("cuda") else (lambda: None)
    rng = np.random.default_rng(5); x0 = torch.tensor(np.concatenate([rng.uniform(lodl.cpu(), hidl.cpu(), (Eb, 6)), np.zeros((Eb, 6))], 1), dtype=torch.float32, device=ldev)
    g = torch.tensor([LIB[GN[i % len(GN)]]["a"] for i in range(Eb)], dtype=torch.float32, device=ldev) / C.D2R
    tag = f"{'cpu' if ldev == 'cpu' else 'gpu'}|E{Eb}"
    for name in ("cem", "ilqr_first", "ilqr_warm", "cbf"):
        ts = []
        for rep in range(12):
            if name == "ilqr_warm":
                il = C.ILQR(f, lodl, hidl); il.plan(x0, x0[:, :6], g); sync(); t0 = time.time(); il.plan(x0, x0[:, :6], g, shift=3)
            elif name == "cbf":
                cb = C.CBF(kl, f, lodl, hidl); sync(); t0 = time.time(); cb.filter(x0, g, torch.full((Eb,), 5e-4, device=ldev))
            elif name == "cem":
                c = C.CEM(f, lodl, hidl); sync(); t0 = time.time(); c.plan(x0, x0[:, :6], g)
            else:
                c = C.ILQR(f, lodl, hidl); sync(); t0 = time.time(); c.plan(x0, x0[:, :6], g)
            sync(); ts.append(time.time() - t0)
        lat[f"{name}|{tag}"] = float(np.median(ts[2:]) * 1000)
    rr = C.make_rrt(kl, lodl.cpu().numpy(), hidl.cpu().numpy(), seed=0); t0 = time.time(); rr.plan(x0[:, :6].cpu().numpy(), g.cpu().numpy()); lat[f"rrt|{tag}"] = float((time.time() - t0) * 1000)
print("LATENCY_MS", json.dumps(lat))
json.dump(dict(args=vars(args), gestures=GN, runs=results, latency_ms=lat), open(out / "results.json", "w"), indent=1)
np.savez_compressed(out / "traces.npz", **{k.replace("|", "__"): v for k, v in traces.items()}, joint_names=np.array(S.names))
print("EXP03_COMPLETE"); app.close()
