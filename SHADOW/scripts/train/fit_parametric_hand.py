"""Identify the parametric hand model (shadow_dynamics/parametric_hand.py) from recorded hand data (experiment 36).

Minimises the 12-step open-loop rollout error (positions deg, plus 1e-3 x velocities deg/s) by differentiable simulation, starting from the
nominal export. Absolute stiffness K is weakly regularised toward the export (it is not identifiable without a known torque; the Fisher
information shows this). Writes <output>/identified.json (values, Fisher std, eigen-analysis, validation error) and <output>/prior.json
(the prior config with an 'identified' block; usable as --prior of train_shadow_flow_v3.py, including --kind physics).

    python3 scripts/tools/cpu_budget.py .venv/bin/python scripts/train/fit_parametric_hand.py --data DIR [DIR ...] --output OUT --device cuda:0 [--minutes 9]
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics.flow_data import StructuredDataset, HISTORY, PAST_ACTIONS, HORIZON
from shadow_dynamics.parametric_hand import ParametricHand
from shadow_dynamics.allegro_hand_model import AllegroHand

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data", nargs="+", required=True); p.add_argument("--output", required=True); p.add_argument("--device", default="cuda:0")
p.add_argument("--prior", default="artifacts/physics_prior_v2/prior.json", help="nominal prior config (structure, export, initial values)")
p.add_argument("--init", default=None, help="start from this identified prior.json instead of the export")
p.add_argument("--minutes", type=float, default=None, help="use only the first train episodes (seed order, shuffled once) covering this many minutes of valid data")
p.add_argument("--episode-seed", type=int, default=0, help="shuffle of the train episodes before the minutes cut (repeat draws)")
p.add_argument("--max-windows", type=int, default=400_000); p.add_argument("--steps", type=int, default=4000); p.add_argument("--batch", type=int, default=4096)
p.add_argument("--lr", type=float, default=0.02); p.add_argument("--vel-weight", type=float, default=1e-3)
p.add_argument("--k-reg", type=float, default=1e-3, help="weight of sum (log K - log K_nominal)^2")
p.add_argument("--init-reg", type=float, default=0.0, help="experiment 42: weight of sum (theta - theta_init)^2 over the free raw parameters (adaptive refits stay near the previous estimate)")
p.add_argument("--probe", action="store_true", help="use the recorded feed-forward torque (field effort_ff, N m) as a known input")
p.add_argument("--observer", action="store_true", help="ablation: fit with the load observer on")
p.add_argument("--no-kappa", action="store_true", help="ablation: no tendon reaction terms")
p.add_argument("--fix", nargs="*", default=[], help="parameter groups held at their initial value: tau K arm fc c b kappa (Shadow); tau K arm fc g umax vmax (Allegro; umax vmax always fixed unless --free-limits)")
p.add_argument("--free-limits", action="store_true", help="experiment 46: also fit the Allegro effort and velocity limits")
p.add_argument("--af-param", choices=("log", "lin"), default="log", help="experiment 47 (Allegro): lin = armature and friction on a linear physical scale from the export values")
p.add_argument("--val-data", nargs="*", default=None, help="take validation windows from these datasets' validation split (designed data has none)")
p.add_argument("--test-data", nargs="*", default=None, help="also report the error on these datasets' test split")
p.add_argument("--anchor-export", action="store_true", help="experiment 44: --init-reg pulls toward the export (--prior) values, not the --init start")
p.add_argument("--tau-bound", type=float, default=0.0, help="experiment 44: hinge penalty (weight 10) on |log tau - log tau_export| beyond log(this); 0 = off")
p.add_argument("--loss", default="mse", choices=["mse", "window_rms"], help="experiment 44: window_rms = mean over windows of the per-window position RMS (typical-window accuracy)")
p.add_argument("--extra-data", nargs="*", default=[], help="experiment 44: more train data, filtered by --extra-speeds and capped at --extra-episodes")
p.add_argument("--extra-speeds", nargs="*", type=float, default=None); p.add_argument("--extra-episodes", type=int, default=None)
p.add_argument("--val-windows", type=int, default=20000); p.add_argument("--fim-windows", type=int, default=2048); p.add_argument("--seed", type=int, default=0)
args = p.parse_args()


def main():
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False); dev = torch.device(args.device); torch.manual_seed(args.seed); t0 = time.time()
    fields = ("q", "qdot", "action_applied", "validity") + (("effort_ff",) if args.probe else ())
    ds = StructuredDataset(list(args.data) + list(args.extra_data), fields=fields)
    tr = np.where(ds.split == "train")[0]; NJ = len(ds.mapping["joint_names"])   # experiment 46: 24 (Shadow) or 16 (Allegro) joints
    if args.extra_data:   # experiment 44: episodes of the extra datasets, speed-filtered, first N in seed order
        n_main = len(StructuredDataset(args.data, fields=("validity",)).episodes); extra = tr[tr >= n_main]; tr = tr[tr < n_main]
        if args.extra_speeds is not None: extra = extra[np.isin(ds.speed[extra], args.extra_speeds)]
        extra = extra[np.argsort(ds.seed[extra], kind="stable")][:args.extra_episodes]
        print(f"extra: {len(extra)} episodes, speeds {np.unique(ds.speed[extra], return_counts=True)}", flush=True); tr = np.sort(np.concatenate([tr, extra]))
    if args.minutes is not None:   # data budget: whole episodes until the valid steps cover the minutes
        tr = tr[np.random.default_rng(args.episode_seed).permutation(len(tr))]; need = args.minutes * 60 * 20; keep, tot = [], 0
        for e in tr:
            if tot >= need: break
            keep.append(e); tot += int((ds.episodes[e]["validity"] <= 1).sum())
        tr = np.array(sorted(keep))
    Wall = ds.windows("train"); Wtr = Wall[np.isin(Wall[:, 0], tr)]
    rng = np.random.default_rng(args.seed)
    if len(Wtr) > args.max_windows: Wtr = Wtr[rng.choice(len(Wtr), args.max_windows, replace=False)]
    dva = ds if not args.val_data else StructuredDataset(args.val_data, fields=fields)
    Wva = dva.windows("validation"); Wva = Wva[rng.choice(len(Wva), min(args.val_windows, len(Wva)), replace=False)]
    minutes = sum(int((ds.episodes[e]["validity"] <= 1).sum()) for e in tr) / 1200
    print(f"data: {len(tr)} train episodes ({minutes:.1f} min valid), {len(Wtr)} train windows, {len(Wva)} val windows", flush=True)

    def tensors(W, ds=ds):
        b = ds.gather(W); T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
        d = dict(s=T(b["past_states"]), pa=T(b["past_actions"]), fa=T(b["future_actions"]), y=T(b["future_states"]))
        if args.probe: d["ff"] = T(np.stack([ds.episodes[e]["effort_ff"][t:t + HORIZON] for e, t in W]))
        return d
    TR, VA = tensors(Wtr), tensors(Wva, dva)
    cfg = json.loads(Path(args.init or args.prior).read_text()); cfg["disturbance"] = bool(args.observer)
    allegro = cfg.get("hand") == "allegro"; Hand = AllegroHand if allegro else ParametricHand
    if args.af_param == "lin": assert allegro, "--af-param lin is Allegro only"; cfg["af_param"] = "lin"
    def groups_of(m):
        if allegro: return dict(tau=[m.log_tau], K=[m.log_K], arm=[m.log_arm], fc=[m.raw_fc], g=[m.log_gscale], umax=[m.log_umax], vmax=[m.log_vmax])
        return dict(tau=[m.log_tau], K=[m.log_K], arm=[m.log_arm], fc=[m.raw_fc], c=[m.c], b=[m.b], kappa=[m.kappa], lam=[m.lam])
    m = Hand(cfg, dev); groups = groups_of(m)
    if allegro:
        assert not args.probe; args.fix = list(args.fix) + ([] if args.free_limits else ["umax", "vmax"])
    else:
        if not args.probe: args.fix = list(args.fix) + ["lam"]   # transmission of known torques: only identifiable with the probe
        if args.no_kappa: args.fix = list(args.fix) + ["kappa"]
    params = [q for g, ps in groups.items() if g not in args.fix for q in ps]
    for g in args.fix:
        for q in groups[g]: q.requires_grad_(False)
    opt = torch.optim.Adam(params, lr=args.lr); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    logK0 = m.K_nom.log().clone(); theta0 = [q.detach().clone() for q in params]
    if args.anchor_export or args.tau_bound > 0:
        excfg = json.loads(Path(args.prior).read_text())
        if args.af_param == "lin": excfg["af_param"] = "lin"
        ex = Hand(excfg, dev); exg = groups_of(ex)
        if args.anchor_export: theta0 = [q.detach().clone() for g, ps in exg.items() if g not in args.fix for q in ps]
        log_tau_ex = ex.log_tau.detach().clone(); del ex

    def predict(D, idx=None, grad=False):
        sel = (lambda x: x[idx]) if idx is not None else (lambda x: x)
        return m.rollout(sel(D["s"]), sel(D["fa"]), sel(D["pa"]), sel(D["ff"]) if args.probe else None, grad=grad)

    def evaluate(D, n=None):
        errs = []
        with torch.no_grad():
            for i in range(0, len(D["s"]), 8192):
                idx = torch.arange(i, min(i + 8192, len(D["s"])), device=dev); P = predict(D, idx); errs.append((P - D["y"][idx]) ** 2)
        e = torch.cat(errs); return dict(pos_rmse_deg=float(e[..., :NJ].mean().sqrt()), pos_rmse_h12=float(e[:, -1, :NJ].mean().sqrt()),
                                         vel_rmse_deg_s=float(e[..., NJ:].mean().sqrt()), per_joint_h12=e[:, -1, :NJ].mean(0).sqrt().tolist())
    init_val = evaluate(VA); print("initial (nominal) val", {k: v for k, v in init_val.items() if k != "per_joint_h12"}, flush=True)
    best, best_state, log = float("inf"), None, open(out / "fit_log.jsonl", "w")
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(TR["s"]), (args.batch,), device=dev); P = predict(TR, idx, grad=True); Y = TR["y"][idx]
        if args.loss == "window_rms": lpos = (((P[..., :NJ] - Y[..., :NJ]) ** 2).mean((1, 2)) + 1e-6).sqrt().mean()
        else: lpos = ((P[..., :NJ] - Y[..., :NJ]) ** 2).mean()
        loss = lpos + args.vel_weight * ((P[..., NJ:] - Y[..., NJ:]) ** 2).mean()
        loss = loss + args.k_reg * ((m.log_K - logK0)[m.act_mask] ** 2).sum()
        if args.tau_bound > 0: loss = loss + 10 * (((m.log_tau - log_tau_ex).abs() - math.log(args.tau_bound)).clamp_min(0) ** 2)[m.act_mask].sum()
        if args.init_reg > 0: loss = loss + args.init_reg * sum(((q - q0) ** 2).sum() for q, q0 in zip(params, theta0))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == args.steps:
            v = evaluate(VA); rec = dict(step=step, loss=float(loss), **{k: x for k, x in v.items() if k != "per_joint_h12"}, elapsed_s=time.time() - t0)
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            if v["pos_rmse_deg"] < best: best = v["pos_rmse_deg"]; best_state = {k: x.detach().clone() for k, x in m.state_dict().items()}
    m.load_state_dict(best_state); val = evaluate(VA); test = None
    if args.test_data:
        dte = StructuredDataset(args.test_data, fields=fields); Wte = dte.windows("test"); Wte = Wte[rng.choice(len(Wte), min(50000, len(Wte)), replace=False)]
        test = evaluate(tensors(Wte, dte)); print("test", {k: v for k, v in test.items() if k != "per_joint_h12"}, flush=True)

    # Fisher information at the optimum (finite differences in the raw parameters; residual = positions deg over the horizon)
    spec = m.param_spec(with_lam=args.probe); fw = torch.as_tensor(rng.choice(len(Wva), min(args.fim_windows, len(Wva)), replace=False), device=dev)
    with torch.no_grad():
        base = predict(VA, fw)[..., :NJ]; r0 = (base - VA["y"][fw][..., :NJ]).flatten(); sigma2 = float((r0 ** 2).mean())
        J = torch.zeros(r0.numel(), len(spec), device=dev, dtype=torch.float64)
        for k, (name, ten, flat) in enumerate(spec):
            eps = 1e-3 if not name.startswith(("kappa", "b[")) else 1e-3
            old = ten.view(-1)[flat].item()
            ten.view(-1)[flat] = old + eps; up = predict(VA, fw)[..., :NJ].flatten()
            ten.view(-1)[flat] = old - eps; dn = predict(VA, fw)[..., :NJ].flatten(); ten.view(-1)[flat] = old
            J[:, k] = ((up - dn) / (2 * eps)).double()
    F = (J.T @ J / sigma2).cpu().numpy(); prior_prec = np.zeros(len(spec))
    for k, (name, *_ ) in enumerate(spec):
        if name.startswith("log_K"): prior_prec[k] = 2 * args.k_reg * r0.numel() / sigma2   # the regulariser's curvature on the same scale
    names = [s_[0] for s_ in spec]; free = np.array([not any(n.startswith(g if g not in ("tau", "K", "arm", "fc", "g") else {"tau": "log_tau", "K": "log_K", "arm": "log_arm", "fc": "raw_fc", "g": "log_gscale"}[g]) for g in args.fix if g not in ("lam", "umax", "vmax")) for n in names])
    Ff = F[np.ix_(free, free)]; ev, evec = np.linalg.eigh(Ff); cov = np.linalg.pinv(Ff + np.diag(prior_prec[free]))
    std = np.full(len(spec), np.nan); std[free] = np.sqrt(np.clip(np.diag(cov), 0, None))
    fn = [n for n, f in zip(names, free) if f]; weakest = []
    for i in range(min(8, len(ev))):
        vec = evec[:, i]; top = np.argsort(-np.abs(vec))[:4]; weakest.append(dict(eigenvalue=float(ev[i]), components={fn[t]: round(float(vec[t]), 3) for t in top}))
    vals = m.values(); act_names = [n[7:] if n.startswith("robot0_") else n for n in vals["joints"]]
    rel = lambda pre: [float(std[names.index(f"{pre}[{j}]")]) for j in act_names]   # std of a log parameter = relative std
    unc = dict(tau_rel_std=rel("log_tau"), K_rel_std=rel("log_K"), armature_rel_std=rel("log_arm"),
               kappa_std={n: float(std[k]) for k, n in enumerate(names) if n.startswith("kappa")}, c_std={n: float(std[k]) for k, n in enumerate(names) if n.startswith("c[")})
    summary = dict(values=vals, uncertainty=unc, fisher=dict(n_params_free=int(free.sum()), eigenvalues=ev.tolist(), condition_number=float(ev.max() / max(ev.min(), 1e-30)),
                                                              weakest_directions=weakest, residual_sigma_deg=math.sqrt(sigma2), windows=len(fw)),
                   validation=val, validation_nominal=init_val, test=test, data=dict(dirs=args.data, train_episodes=int(len(tr)), minutes=minutes, train_windows=int(len(Wtr)),
                   probe=args.probe, episode_seed=args.episode_seed), args=vars(args), seconds=time.time() - t0)
    (out / "identified.json").write_text(json.dumps(summary, indent=1))
    pc = m.to_prior_config(dict(source=str(out), minutes=minutes, probe=args.probe, val_pos_rmse_deg=val["pos_rmse_deg"]))
    (out / "prior.json").write_text(json.dumps(pc))
    print("FIT_COMPLETE", json.dumps(dict(val_pos_rmse_deg=val["pos_rmse_deg"], val_h12=val["pos_rmse_h12"], nominal_h12=init_val["pos_rmse_h12"], minutes=minutes,
                                          train_episodes=int(len(tr)), seconds=time.time() - t0)), flush=True)


main()
