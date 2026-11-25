"""Experiment 49: the hand chooses its next exploration programs with its current Li-form model (D-optimal), or at random (baseline).

1. Pool: --pool candidates, half from the structured generator (all families, --speeds), half random multisines (3-20 joints, 1-5 components,
   10-45% of range); kinematic repair and FK check (the allowed prior), failures dropped.
2. Prediction: every candidate's slew-limited command path is rolled through the current model (--model, a fit_tendon_hand.py model.json)
   at 120 Hz from rest at its first target; samples every --stride substeps give q, v and the implicit-Euler acceleration.
3. Information: the equation-error residual is linear in the plant's linear parameters (standard inertial parameters of every moving body,
   armature, h = k0 + k1 q + k2 q^2 + d qd + Fc tanh(qd / 0.02), passive spring gains). Its regressor rows, scaled per joint like the fit and
   masked where a joint is near its limit or a finger's J0 is pinned, give F = sum Y^T Y per candidate. Data already collected (--collected
   folders) enters as the starting information.
4. Greedy selection of --select programs maximising log det(F_sel + eps diag(F_pool)); --random draws uniformly from the checked pool instead.
Writes <output>.npz (commanded, speed, family, seed) for collect_exp49.py --programs, and <output>.json.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics import structured_excitation as sx
from shadow_dynamics.hand_geometry import HandGeometry, TorchFK
from shadow_dynamics.kinematics_only import load_kinematics
from shadow_dynamics.tendon_hand_model import TendonHand, V_EPS

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True); p.add_argument("--output", required=True); p.add_argument("--select", type=int, required=True)
p.add_argument("--collected", nargs="*", default=[]); p.add_argument("--random", action="store_true")
p.add_argument("--pool", type=int, default=2048); p.add_argument("--multisine-share", type=float, default=0.5); p.add_argument("--speeds", nargs="+", type=float, default=[1, 5, 10, 20, 30])
p.add_argument("--seconds", type=float, default=20.0); p.add_argument("--stride", type=int, default=12); p.add_argument("--seed", type=int, default=0)
p.add_argument("--first-program-seed", type=int, required=True); p.add_argument("--precheck-mm", type=float, default=1.0); p.add_argument("--eps", type=float, default=1e-3)
p.add_argument("--chunk", type=int, default=128); p.add_argument("--device", default="cuda:0")
p.add_argument("--manifest", default="artifacts/shadow_structured_v3/train_speeds/manifest.json", help="action names and joint limits only")
p.add_argument("--poses", default="shadow_dynamics/data/shadow_pose_library.json")
p.add_argument("--j0-margin-deg", type=float, default=1.0); p.add_argument("--limit-margin-deg", type=float, default=0.5); p.add_argument("--passive-scale", type=float, default=2e-3)
args = p.parse_args(); dev = torch.device(args.device); HZ = 20
J0N = ("robot0_FFJ0", "robot0_MFJ0", "robot0_RFJ0", "robot0_LFJ0")


def build_pool(rng, spec, gen, T):
    progs, infos = [], []
    shares = np.array([sx.SPEED_SHARES.get(int(s), 0.2) for s in args.speeds]); shares /= shares.sum()
    for i in range(args.pool):
        sp = float(rng.choice(args.speeds, p=shares))
        if rng.uniform() < args.multisine_share:
            start = spec.poses[list(spec.poses)[rng.integers(len(spec.poses))]].copy(); prog = np.repeat(start[None], T, 0)
            for c in rng.choice(20, int(rng.integers(3, 21)), replace=False):
                amp = rng.uniform(0.1, 0.45) * spec.range[c]; center = np.clip(rng.uniform(spec.low[c], spec.high[c]), spec.low[c] + amp, spec.high[c] - amp)
                prog[:, c], _ = sx.sines(rng, T, center, amp, sp, n_max=5)
            progs.append(np.clip(prog, spec.low, spec.high).astype(np.float32)); infos.append(("designed_multisine", sp))
        else:
            fam = sx.FAMILIES[int(rng.integers(len(sx.FAMILIES)))]; prog, _ = gen.generate(rng, fam, sp); progs.append(prog.astype(np.float32)); infos.append((fam, sp))
    return np.stack(progs), infos


class Info:
    """Regressor information of the plant's linear parameters, scaled per joint and masked like fit_tendon_hand.py."""
    def __init__(self, model, scale):
        self.m = model; self.scale = scale; mv = torch.nonzero(model.moving)[:, 0]; self.mv = mv; nj = model.nj
        self.names = model.names; self.j0 = [self.names.index(n) for n in J0N]; self.j1 = [self.names.index(n.replace("J0", "J1")) for n in J0N]
        self.P = 10 * len(mv) + nj + 5 * nj + len(model.passive_t)
    @torch.no_grad()
    def rows(self, q, v, a):
        """q, v, a (N, nj) -> scaled masked regressor (N * nj, P)."""
        m = self.m; nj = m.nj; N = len(q)
        Yi = m.inertial_regressor(q, v, a, True)[:, :, self.mv].reshape(N, nj, -1)                       # inertial
        Ya = torch.diag_embed(a)                                                                          # armature
        eye = torch.eye(nj, device=q.device)
        Yh = torch.cat([eye.expand(N, nj, nj), torch.diag_embed(q), torch.diag_embed(q * q), torch.diag_embed(v), torch.diag_embed(torch.tanh(v / V_EPS))], -1)
        C = m.C()[m.passive_t]; Cf = m.Cf()[m.passive_t]; L = q @ C.T
        Yp = (-(L - m.pt_L0))[:, None, :] * Cf.T[None]                                                    # tau_p = Cf^T (-k (L - L0)): column per passive tendon, minus sign of the residual
        Y = torch.cat([Yi, Ya, Yh, -Yp], -1) / self.scale[None, :, None]
        lo, hi = m.kin.lo, m.kin.hi; lm = np.deg2rad(args.limit_margin_deg); jm = np.deg2rad(args.j0_margin_deg)
        mask = (q > lo + lm) & (q < hi - lm); pin = (q[:, self.j0] < lo[self.j0] + jm) | (q[:, self.j0] > hi[self.j0] - jm)
        mask[:, self.j0] &= ~pin; mask[:, self.j1] &= ~pin
        return (Y * mask[..., None]).reshape(N * nj, -1)


def main():
    t0 = time.time(); rng = np.random.default_rng(args.seed); kin = load_kinematics()
    cfg = json.loads(Path(args.model).read_text()); model = TendonHand(cfg, dev).eval(); nj = model.nj; act = kin["active"]
    man = json.loads(Path(args.manifest).read_text())["mapping"]; lib = json.loads(Path(args.poses).read_text()); lim = np.array(man["action_limits_deg"])
    spec = sx.Spec(man["action_names"], lim[:, 0], lim[:, 1], lib["poses_deg"]); T = int(round(args.seconds * HZ)); gen = sx.Generator(spec, args.seconds, HZ)
    progs, infos = build_pool(rng, spec, gen, T)
    geo = HandGeometry.from_viz_model("artifacts/viz_hand_model"); fk = TorchFK("artifacts/viz_hand_model", dev)
    checker = sx.ProgramChecker(geo, fk, man["joint_names"], man["action_names"], args.precheck_mm / 1000, dev); progs, _ = checker.repair(progs)
    paths = np.stack([sx.applied_path(pg, pg[0], sp, spec.low, spec.high) for pg, (_, sp) in zip(progs, infos)])
    ok = (checker.check_paths(paths) <= checker.threshold) & (checker.max_penetration(paths[:, 0]) <= checker.threshold); cand = np.where(ok)[0]
    print(f"pool {len(progs)}, {len(cand)} pass the kinematic check ({time.time() - t0:.0f}s)", flush=True)
    # per-joint residual scale: RMS drive torque of the data collected so far (as in the fit); passive joints fixed
    scale = torch.full((nj,), args.passive_scale, device=dev); Fsel = None
    coll = [np.load(Path(c) / "episodes.npz") for c in args.collected]
    if coll:
        tau = np.concatenate([z["tau"].reshape(-1, 20) for z in coll]); scale[act] = torch.as_tensor(np.sqrt((tau ** 2).mean(0)), device=dev).float().clamp_min(1e-4)
    info = Info(model, scale)
    if coll:   # information already collected (contact-free substeps, every stride-th)
        Fsel = torch.zeros(info.P, info.P, device=dev, dtype=torch.float64)
        for z in coll:
            for i in range(len(z["n_valid_states"])):
                n = int(z["n_valid_states"][i]) - 1; S = n * 6
                if n < 2: continue
                q = torch.deg2rad(torch.as_tensor(z["q_deg"][i, :S + 1], device=dev)); v = torch.deg2rad(torch.as_tensor(z["qd_deg"][i, :S + 1], device=dev))
                k = torch.arange(0, S, args.stride, device=dev); c = (k // 6).cpu().numpy(); val = z["validity"][i]
                free = torch.as_tensor((val[c] == 0) & (val[np.minimum(c + 1, len(val) - 1)] == 0), device=dev); k = k[free]
                if len(k): Y = info.rows(q[k], v[k + 1], (v[k + 1] - v[k]) * 120); Fsel += (Y.T @ Y).double()
    if args.random:
        chosen = list(rng.choice(len(cand), args.select, replace=False)); logdet = None
    else:
        Fc = []
        for b0 in range(0, len(cand), args.chunk):
            ids = cand[b0:b0 + args.chunk]; P_ = torch.as_tensor(paths[ids], dtype=torch.float32, device=dev); n = len(ids)
            s0 = torch.zeros(n, 48, device=dev); s0[:, act] = P_[:, 0]
            for f in range(4): s0[:, info.j0[f]] = 0.0
            with torch.no_grad(): X = model.rollout(s0, fa=P_, substep_states=True)                    # (n, T*6, 48) deg
            Xr = torch.deg2rad(X); k = torch.arange(1, X.shape[1] - 1, args.stride, device=dev)
            q, v0, v1 = Xr[:, k - 1, :nj], Xr[:, k - 1, nj:], Xr[:, k, nj:]
            fin = torch.isfinite(Xr[:, k]).all(-1) & (v1.abs() < 50).all(-1)                          # predicted samples that stay physical
            for j in range(n):
                f_ = fin[j]; Y = info.rows(q[j, f_], v1[j, f_], (v1[j, f_] - v0[j, f_]) * 120); Fc.append((Y.T @ Y).double().cpu())
            print(f"information for {min(b0 + args.chunk, len(cand))}/{len(cand)} candidates ({time.time() - t0:.0f}s)", flush=True)
        Fc = torch.stack(Fc).to(dev)                                                                  # (C, P, P) float64
        reg = args.eps * Fc.mean(0).diagonal().clamp_min(1e-30); base = Fsel if Fsel is not None else torch.zeros_like(Fc[0])
        def logdet(F): return torch.linalg.slogdet(F + torch.diag(reg))[1]
        chosen = []; cur = base.clone(); logdet0 = float(logdet(cur))
        for _ in range(args.select):
            best, bi = -float("inf"), -1
            for b0 in range(0, len(Fc), 256):
                g = logdet(cur[None] + Fc[b0:b0 + 256])
                if chosen:
                    for c_ in chosen:
                        if b0 <= c_ < b0 + 256: g[c_ - b0] = -float("inf")
                v_, i_ = g.max(0)
                if float(v_) > best: best, bi = float(v_), b0 + int(i_)
            chosen.append(bi); cur = cur + Fc[bi]
        logdet = dict(start=logdet0, final=float(logdet(cur)))
    sel = cand[np.array(chosen)]
    np.savez(Path(args.output).with_suffix(".npz"), commanded=progs[sel], speed=np.array([infos[i][1] for i in sel]), family=np.array([infos[i][0] for i in sel]),
             seed=np.arange(args.first_program_seed, args.first_program_seed + len(sel)))
    fam = {}
    for i in sel: fam[infos[i][0]] = fam.get(infos[i][0], 0) + 1
    rep = dict(model=args.model, collected=args.collected, pool=len(progs), pool_checked=int(len(cand)), selected=int(len(sel)), random=bool(args.random), params=info.P,
               logdet=logdet, families=fam, speeds={str(s): int(c) for s, c in zip(*np.unique([infos[i][1] for i in sel], return_counts=True))}, seconds=time.time() - t0)
    Path(args.output).with_suffix(".json").write_text(json.dumps(rep, indent=1)); print("DESIGN49_COMPLETE", json.dumps({k: v for k, v in rep.items() if k != "collected"}), flush=True)


main()
