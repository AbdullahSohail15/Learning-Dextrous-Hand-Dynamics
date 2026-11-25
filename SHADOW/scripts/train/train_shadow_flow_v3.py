"""Train a whole-horizon hand dynamics model (flow_tcn / flow_mlp / det_mlp) on schema-3 structured data.

Episodes are padded into GPU tensors; batches are gathered by advanced indexing (no per-step CPU work). Normalization
comes from train windows only and is stored in the checkpoint with a data_report. The EMA weights are evaluated every
--eval-every steps on a fixed validation subset (mean of K=4 samples for flow models, 4 Euler steps); best.pt is the
checkpoint with the lowest validation position RMSE (deg).

  python3 scripts/tools/cpu_budget.py .venv/bin/python scripts/train/train_shadow_flow_v3.py --kind flow_tcn --output artifacts/flow_v3/flow_tcn --device cuda:0
"""
import argparse, copy, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics.flow_data import StructuredDataset
from shadow_dynamics.flow_model import HandDynamics, count_parameters, H, HIST
from shadow_dynamics.cube_world_data import corrupt_history, occlusion_cfg, rotation_distance, sixd_to_matrix

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--kind", required=True, choices=["flow_tcn", "flow_mlp", "det_mlp", "physics", "res_det", "res_flow_tcn"])
p.add_argument("--prior", default="artifacts/physics_prior_v1_obs/prior.json", help="physics-prior config (physics / res_* kinds; scripts/fit_shadow_physics_prior.py)")
p.add_argument("--data", nargs="+", default=["artifacts/shadow_structured_v3/train_speeds", "artifacts/shadow_structured_v3/topup_balance"])
p.add_argument("--dataset", choices=["structured", "cube"], default="structured", help="cube: hand + cube shards from collect_shadow_cube.py (63-dim state, experiment 12)")
p.add_argument("--init", default=None, help="initialize weights from this checkpoint (same kind and state size; Dyna rounds)")
p.add_argument("--init-keep-norm", action="store_true", help="experiment 23 fine-tuning: also keep the checkpoint's normalization statistics")
p.add_argument("--output", required=True); p.add_argument("--device", default="cuda:0")
p.add_argument("--steps", type=int, default=100000); p.add_argument("--batch-size", type=int, default=2048)
p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--ema", type=float, default=0.999); p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--width", type=int, default=512); p.add_argument("--depth", type=int, default=6); p.add_argument("--hist-width", type=int, default=256); p.add_argument("--cond-dim", type=int, default=256)
p.add_argument("--train-fraction", type=float, default=1.0, help="fraction of training episodes (by seed, nested: 0.25 is a subset of 0.5)")
p.add_argument("--eval-every", type=int, default=2500); p.add_argument("--val-windows", type=int, default=8192)
p.add_argument("--seed", type=int, default=0); p.add_argument("--no-amp", action="store_true")
# experiment 24: partially observed cube (cube_world_data.corrupt_history) on the history inputs, fresh draws per batch; targets stay true
p.add_argument("--obs-noise-pos-mm", type=float, default=0.0); p.add_argument("--obs-noise-rot-deg", type=float, default=0.0)
p.add_argument("--obs-occl-enter", type=float, default=0.0); p.add_argument("--obs-occl-stay", type=float, default=0.0)
# experiment 28: --prior also accepts model:<checkpoint> (a frozen learned model's prediction as the prior) and simcache:<npz> (nominal-simulator
# futures per window from scripts/compute_cube_sim_prior.py); --val-fraction subsamples validation episodes like --train-fraction; --select-metric cube
# picks best.pt by the validation cube rotation error at the end of the window
p.add_argument("--fraction-seed", type=int, default=12345, help="experiment 32: rng seed of the --train-fraction episode subset (default: the nested subsets of exps 26-28)")
p.add_argument("--oversample-family", nargs="*", default=[], help="experiment 41: motion_family prefixes whose windows are repeated to --oversample-frac of the train rows (and half of the validation rows)")
p.add_argument("--oversample-frac", type=float, default=0.25)
p.add_argument("--val-only-data", nargs="*", default=[], help="experiment 42: datasets used for validation windows only (their train / test episodes are excluded, also from the normalization)")
p.add_argument("--val-fraction", type=float, default=1.0); p.add_argument("--select-metric", choices=["joint", "cube"], default="joint")
p.add_argument("--contact", action="store_true", help="experiment 34: append the 78 whole-hand contact channels to the cube state (141)")
p.add_argument("--contact-loss", choices=["predict", "input"], default="predict", help="input: contact channels are inputs only (loss on the first 63 channels)")
p.add_argument("--contact-loss-weight", type=float, default=None, help="loss weight of the contact channels (with --contact --contact-loss predict)")
p.add_argument("--no-mask-invalid", action="store_true", help="keep windows with hand self-penetration states (experiment 34 masks them by default)")
p.add_argument("--prior-on-the-fly", action="store_true", help="experiment 24: compute the physics prior per batch instead of caching it for every window (same values; the cache needs 23 GB)")
args = p.parse_args()


class GPUWindows:
    def __init__(self, ds, device):
        E = len(ds.episodes); T = max(len(e["action_applied"]) for e in ds.episodes); D = getattr(ds, "state_dim", None) or ds.state(0).shape[-1]   # experiment 46: 32 on the Allegro
        X = np.zeros((E, T + 1, D), np.float32); U = np.zeros((E, T, ds.episodes[0]["action_applied"].shape[-1]), np.float32)   # experiment 46: 16 actions on the Allegro
        for i, e in enumerate(ds.episodes):
            n = len(e["action_applied"]); X[i, :n + 1] = ds.state(i); U[i, :n] = e["action_applied"]
        self.X = torch.as_tensor(X, device=device); self.U = torch.as_tensor(U, device=device)
        self.ho = torch.arange(-HIST + 1, 1, device=device); self.pa = torch.arange(-3, 0, device=device); self.fu = torch.arange(0, H, device=device)

    def batch(self, W, obs=None, gen=None):
        e, t = W[:, 0:1], W[:, 1:2]
        s = self.X[e, t + self.ho] if obs is None else corrupt_history(self.X, e, t, obs, gen)[0]
        return s, self.U[e, t + self.pa], self.U[e, t + self.fu], self.X[e, t + 1 + self.fu]


def main():
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device(args.device); torch.set_num_threads(1)
    if args.dataset == "cube":
        from shadow_dynamics.cube_world_data import CubeDataset
        t0 = time.time(); ds = CubeDataset(args.data, contact=args.contact, mask_invalid=not args.no_mask_invalid)
    else:
        t0 = time.time(); ds = StructuredDataset(list(args.data) + list(args.val_only_data))
        if args.val_only_data:   # experiment 42: fixed external validation set; training uses only --data
            vo = np.array([any(str(Path(m["source"].split(":")[0]).parent.resolve()) == str(Path(d).resolve()) for d in args.val_only_data) for m in ds.meta])
            ds.split = np.where(vo & (ds.split != "validation"), "excluded", np.where(~vo & (ds.split == "validation"), "excluded", ds.split))
            print(f"val-only data: {int(vo.sum())} episodes ({int((vo & (ds.split == 'validation')).sum())} validation); own validation episodes dropped", flush=True)
    Wtr = ds.windows("train"); Wva = ds.windows("validation")
    if args.train_fraction < 1:
        seeds = np.unique(ds.seed[ds.split == "train"]); order = np.random.default_rng(args.fraction_seed).permutation(len(seeds))
        keep = set(seeds[order[:int(round(args.train_fraction * len(seeds)))]].tolist())
        Wtr = Wtr[np.isin(ds.seed[Wtr[:, 0]], list(keep))]
    if args.oversample_family:   # experiment 41: the scarce closed-loop windows would be ~1% of the data; repeat them
        def over(W, frac):
            m = np.array([any(str(fm).startswith(pf) for pf in args.oversample_family) for fm in ds.family])[W[:, 0]]
            if not m.any(): return W, 0
            k = max(1, int(round(frac / (1 - frac) * (~m).sum() / m.sum()))); return np.concatenate([W[~m]] + [W[m]] * k), k
        n0 = len(Wtr); Wtr, ktr = over(Wtr, args.oversample_frac); Wva, kva = over(Wva, 0.5)
        print(f"oversampling {args.oversample_family}: train windows {n0} -> {len(Wtr)} (x{ktr}), validation x{kva}", flush=True)
    stats = ds.normalization("train") if args.train_fraction == 1 else None
    if stats is None:   # normalization from the kept training episodes only
        sub = copy.copy(ds); sub.split = np.where(np.isin(ds.seed, list(keep)), ds.split, "excluded"); stats = sub.normalization("train")
    gw = GPUWindows(ds, dev); Wtr_t = torch.as_tensor(Wtr, device=dev)
    NJ = len(ds.mapping["joint_names"]); AD = gw.U.shape[-1]   # hand joints (Shadow 24, Allegro 16) and actions (20 / 16)
    obs = occlusion_cfg(args.obs_noise_pos_mm, args.obs_noise_rot_deg, args.obs_occl_enter, args.obs_occl_stay); gtr = gval = None
    if obs is not None:
        assert args.dataset == "cube" and gw.X.shape[-1] == 63 and not args.contact
        gtr = torch.Generator(device=dev).manual_seed(args.seed + 7); gst = torch.Generator(device=dev).manual_seed(12345)
        # input / delta normalization from corrupted histories (the targets are relative to the last observed state)
        Ws = Wtr_t[torch.as_tensor(np.random.default_rng(0).choice(len(Wtr), min(400000, len(Wtr)), replace=False), device=dev)]; acc = dict(n_s=0, n_d=0); st = []
        def add(k, x):   # running sums in float64 (memory-light: the GPU is shared)
            acc[k] = acc.get(k, 0) + x.sum(0); acc[k + "2"] = acc.get(k + "2", 0) + (x * x).sum(0); acc["n_" + k] += x.shape[0]
        with torch.no_grad():
            for i in range(0, len(Ws), 16384):
                W = Ws[i:i + 16384]; s_, age = corrupt_history(gw.X, W[:, 0:1], W[:, 1:2], obs, gst); fut = gw.X[W[:, 0:1], W[:, 1:2] + 1 + gw.fu]
                add("s", s_.reshape(-1, s_.shape[-1]).double()); add("d", (fut - s_[:, -1:]).double()); st.append((age > 0).float().mean())
        ms = lambda k: (acc[k] / acc["n_" + k], ((acc[k + "2"] - acc[k] ** 2 / acc["n_" + k]) / (acc["n_" + k] - 1)).clamp_min(0).sqrt())
        (sm, ssd), (dm, dsd) = ms("s"), ms("d")
        stats.update(state_mean=sm.float().cpu().numpy(), state_std=ssd.clamp_min(1e-3).float().cpu().numpy(), delta_mean=dm.float().cpu().numpy(), delta_std=dsd.clamp_min(1e-3).float().cpu().numpy())
        print(f"OBS partial observability {obs}: stale fraction of history steps {float(torch.stack(st).mean()):.3f}; normalization from {acc['n_d']} corrupted windows", flush=True)
        del acc
    if args.val_fraction < 1:
        vs = np.unique(ds.seed[ds.split == "validation"]); vk = set(vs[np.random.default_rng(54321).permutation(len(vs))[:max(1, int(round(args.val_fraction * len(vs))))]].tolist())
        Wva = Wva[np.isin(ds.seed[Wva[:, 0]], list(vk))]; print(f"validation restricted to {len(vk)} episodes ({len(Wva)} windows)", flush=True)
    rng = np.random.default_rng(0); Wv = torch.as_tensor(Wva[rng.choice(len(Wva), min(args.val_windows, len(Wva)), replace=False)], device=dev)
    print(f"data loaded in {time.time() - t0:.0f}s: {len(ds.episodes)} episodes, train windows {len(Wtr)}, val windows {len(Wva)}", flush=True)
    def parse_prior(spec):
        if spec.startswith("model:"): return dict(source="model", path=spec[len("model:"):])
        if spec.startswith("simcache:"): return dict(source="sim", cache=spec[len("simcache:"):])
        return json.loads(Path(spec).read_text())
    prior = parse_prior(args.prior) if args.kind in ("physics", "res_det", "res_flow_tcn") else None
    def get_phys(W, s, a, fa):   # nominal trajectory for a batch whose (possibly corrupted) history is s
        if cache is None: return None
        if isinstance(cache, str) and cache == "sim":   # nominal-simulator futures of these windows (experiment 28)
            r = sim_idx[W[:, 0], W[:, 1]]; assert bool((r >= 0).all()), "window without a simulator prior"; return simY[r]
        if isinstance(cache, str): return probe.physics(s, a, fa)   # cube channels = persistence of the observed last state
        return obs_prior(cache[widx[W[:, 0], W[:, 1]]], s, obs)
    def obs_prior(ph, s, obs):   # the cached prior's cube channels are persistence of the clean last state; with corrupted inputs use the observed one
        if obs is None or ph is None: return ph
        ph = ph.clone(); ph[..., 48:] = s[:, -1:, 48:]; return ph
    cache = None
    if prior is not None and prior.get("source") == "sim":
        z = np.load(prior["cache"]); seed2e = {int(sd): e for e, sd in enumerate(ds.seed)}; rows = np.array([seed2e.get(int(sd), -1) for sd in z["seed"]])
        ok = rows >= 0; simY = torch.as_tensor(z["Y"][ok], dtype=torch.float32, device=dev)
        sim_idx = torch.full((gw.X.shape[0], gw.X.shape[1]), -1, dtype=torch.long, device=dev); sim_idx[torch.as_tensor(rows[ok], device=dev), torch.as_tensor(z["t"][ok], device=dev).long()] = torch.arange(int(ok.sum()), device=dev)
        cache = "sim"; print(f"simulator prior: {int(ok.sum())} windows from {prior['cache']} ({int((~ok).sum())} not in this data)", flush=True)
    if prior is not None and args.kind != "physics" and cache != "sim":
        # the prior has no learned parameters: compute it once for every training window (E, T, 12, 48), looked up per batch
        probe = HandDynamics("physics", prior=prior, state_dim=gw.X.shape[-1], action_dim=AD).to(dev); t1 = time.time()
        # one row per training window (experiment 17: the dense (E, T) layout needed 33 GB with round-5 data); widx maps (e, t) -> row
        if not args.prior_on_the_fly:
            widx = torch.full((gw.U.shape[0], gw.U.shape[1]), -1, dtype=torch.long, device=dev); widx[Wtr_t[:, 0], Wtr_t[:, 1]] = torch.arange(len(Wtr_t), device=dev)
            cache = torch.zeros(len(Wtr_t), H, gw.X.shape[-1], device=dev)
            with torch.no_grad():
                for i in range(0, len(Wtr_t), 16384):
                    W = Wtr_t[i:i + 16384]; s, a, fa, fut = gw.batch(W); cache[i:i + len(W)] = probe.physics(s, a, fa)
            print(f"prior cached for {len(Wtr_t)} train windows in {time.time() - t1:.0f}s ({cache.numel() * 4 / 1e9:.1f} GB)", flush=True)
        else: cache = "fly"; print("prior computed per batch (--prior-on-the-fly)", flush=True)
    if prior is not None and args.kind != "physics":
        # residual normalization from train windows only: R = Y_true - Y_physics
        Ws = Wtr_t[torch.as_tensor(np.random.default_rng(0).choice(len(Wtr), min(400000, len(Wtr)), replace=False), device=dev)]; r1 = r2 = 0; nR = 0
        with torch.no_grad():
            for i in range(0, len(Ws), 8192):
                W = Ws[i:i + 8192]; s, a, fa, fut = gw.batch(W, obs, gst if obs is not None else None); R = (fut - get_phys(W, s, a, fa)).double()
                r1 = r1 + R.sum(0); r2 = r2 + (R * R).sum(0); nR += len(R)
        rm = r1 / nR; rsd = ((r2 - r1 ** 2 / nR) / (nR - 1)).clamp_min(0).sqrt(); stats["res_mean"] = rm.float().cpu().numpy(); stats["res_std"] = rsd.clamp_min(1e-3).float().cpu().numpy()
        print(f"residual stats from {nR} train windows: |mean| {float(rm.abs().mean()):.4f}, std pos {float(rsd[..., :NJ].pow(2).mean().sqrt()):.3f} deg (rms of per-channel std), vel {float(rsd[..., NJ:2 * NJ].pow(2).mean().sqrt()):.3f} deg/s", flush=True)
    model = HandDynamics(args.kind, hist_width=args.hist_width, cond_dim=args.cond_dim, width=args.width, depth=args.depth, stats=stats, prior=prior, state_dim=gw.X.shape[-1],
                         loss_channels=63 if (args.contact and args.contact_loss == "input") else None,
                         contact_loss_weight=args.contact_loss_weight if args.contact else None, action_dim=AD).to(dev)
    if args.init:   # warm start (weights only; the normalization of this run is kept)
        init = torch.load(args.init, map_location=dev, weights_only=False)["model"]; keep = init if args.init_keep_norm else {k: v for k, v in init.items() if not k.endswith(("_mean", "_std"))}
        missing = model.load_state_dict(keep, strict=False); print("initialized from", args.init, "missing", [k for k in missing.missing_keys if not k.endswith(("_mean", "_std"))], flush=True)
    ema = copy.deepcopy(model).eval(); [q.requires_grad_(False) for q in ema.parameters()]
    n_params = 0 if args.kind == "physics" else count_parameters(model); print(f"{args.kind}: {n_params:,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    report = dict(joint_names=ds.mapping["joint_names"], action_names=ds.mapping["action_names"], active_indices=ds.mapping["active_indices"],
                  passive_indices=ds.mapping["passive_indices"], control_hz=20, history=HIST, past_actions=3, horizon=H,
                  units=dict(state=f"q deg ({NJ}) + qdot deg/s ({NJ})", action="deg, applied joint-position target"), data=args.data,
                  train_windows=int(len(Wtr)), train_episodes=int(len(np.unique(Wtr[:, 0]))), train_fraction=args.train_fraction)
    def save(name, step, metrics):
        torch.save(dict(config=model.config, model=ema.state_dict(), stats={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()},
                        data_report=report, args=vars(args), parameters=n_params, step=step, val=metrics), out / name)
    best = float("inf"); log = open(out / "train_log.jsonl", "w"); started = time.time(); amp = not args.no_amp
    def validate():
        with torch.no_grad():
            errs = []; cube = []; gv = torch.Generator(device=dev).manual_seed(999) if obs is not None else None
            for i in range(0, len(Wv), 2048):
                Wb_ = Wv[i:i + 2048]; s, a, fa, fut = gw.batch(Wb_, obs, gv); ph = get_phys(Wb_, s, a, fa) if cache == "sim" else None
                pred = ema.sample(s, a, fa, K=4 if ema.flow else 1, steps=4, phys=ph).mean(0)
                errs.append((pred - fut) ** 2)
                if gw.X.shape[-1] >= 63: cube.append(torch.rad2deg(rotation_distance(sixd_to_matrix(pred[:, -1, 51:57]), sixd_to_matrix(fut[:, -1, 51:57]))))
            e = torch.cat(errs); cube_deg = float(torch.cat(cube).mean()) if cube else float("nan")
        return float(e[..., :NJ].mean().sqrt()), float(e[..., NJ:2 * NJ].mean().sqrt()), e[..., :NJ].mean((0, 2)).sqrt().tolist(), cube_deg
    if args.kind == "physics":   # no learned parameters: validate and save the prior as a checkpoint
        pos, vel, pos_h, cdeg = validate(); rec = dict(step=0, val_pos_rmse_deg=pos, val_vel_rmse_deg_s=vel, val_pos_rmse_h1=pos_h[0], val_pos_rmse_h12=pos_h[-1], val_cube_rot_end_deg=cdeg)
        save("best.pt", 0, rec); save("last.pt", 0, rec)
        (out / "summary.json").write_text(json.dumps(dict(kind=args.kind, parameters=n_params, best_val_pos_rmse_deg=pos, steps=0, data_report=report, args=vars(args)), indent=1))
        print("TRAINING_COMPLETE", json.dumps(dict(kind=args.kind, best_val_pos_rmse_deg=pos, seconds=0)), flush=True); return
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(Wtr_t), (args.batch_size,), device=dev); Wb = Wtr_t[idx]; s, a, fa, fut = gw.batch(Wb, obs, gtr)
        phys = get_phys(Wb, s, a, fa)   # nominal trajectory, fp32 (cached or per batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = model.loss(s, a, fa, model.normalize_target(s, fut, phys), phys)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip); opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - args.ema)
        if step % args.eval_every == 0 or step == args.steps or step == 200:
            pos, vel, pos_h, cdeg = validate()
            rec = dict(step=step, loss=float(loss), grad_norm=float(gn), lr=sched.get_last_lr()[0], val_pos_rmse_deg=pos, val_vel_rmse_deg_s=vel,
                       val_pos_rmse_h1=pos_h[0], val_pos_rmse_h12=pos_h[-1], val_cube_rot_end_deg=cdeg, elapsed_s=time.time() - started)
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            crit = cdeg if args.select_metric == "cube" else pos
            if crit < best: best = crit; save("best.pt", step, rec)
    save("last.pt", args.steps, rec)
    (out / "summary.json").write_text(json.dumps(dict(kind=args.kind, parameters=n_params, best_val_pos_rmse_deg=best, select_metric=args.select_metric, steps=args.steps,
                                                      train_seconds=time.time() - started, data_report=report, args=vars(args)), indent=1))
    print("TRAINING_COMPLETE", json.dumps(dict(kind=args.kind, best_val_pos_rmse_deg=best, seconds=time.time() - started)), flush=True)


main()
