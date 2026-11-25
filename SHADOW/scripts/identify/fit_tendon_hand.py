"""Experiment 49: identify the Li-form TendonHand from self-explored data only, starting from nothing but the kinematics.

Inputs: one or more collect_exp49.py folders (episodes.npz). No URDF dynamics, no earlier recording, no truth file is read.
  stage 1  actuator: per commanded joint, least squares of the measured drive torque f_t on (a - l, -ldot) over unsaturated substeps -> K, D;
           u_max = the largest |f_t| seen if the data shows saturation there (> 1% of substeps within 0.5% of it), else unbounded (100x).
  stage 0  measured tendons' C is kinematic (Li et al. ID.2: tendon excursion vs joint angle). In simulation the actuator's excursion is the
           joint angle, so C = the config's identity; it is held fixed (force data alone cannot fix C's scale: C and the whole plant scale together).
  stage 2  plant by equation error (inverse dynamics) at every usable substep k (120 Hz):
             r = M(q_k) qdd_k + N(q_k, v_k+1) + G(q_k) + h(q_k, v_k+1) - Cf_p^T f_p(q_k, v_k+1) - Cf_m^T (f_t,k - f_f)
           qdd_k = (v_k+1 - v_k) * 120 (the implicit-Euler acceleration). Usable: both control boundaries free of self-contact, the
           episode valid; a joint within --limit-margin of its own limit is masked in r (its limit force is not modelled), and so are J1 and
           J0 of a finger whose J0 is within --j0-margin of a limit (the simulator's pinned-J0 lock). Residuals are scaled per joint by the RMS
           drive torque (passive joints by --passive-scale). Adam on all plant parameters from the generic start.
  stage 3  (--rollout-steps > 0) rollout refinement: actuator from the commands, robust per-window RMS loss (exp 44) over 0.2 s windows
           on the same usable set.
Outputs: model.json (config + identified values), fit_log.jsonl, report.json. Fit-quality numbers here use held-out episodes of the same
data (every 5th episode); comparisons with the simulator's truth live in analyze_exp49_fit.py.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics.kinematics_only import load_kinematics
from shadow_dynamics.tendon_hand_model import TendonHand, sim_tendon_config

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data", nargs="+", required=True); p.add_argument("--output", required=True); p.add_argument("--device", default="cuda:0")
p.add_argument("--init", default=None, help="model.json of an earlier round to start from (adaptive rounds); default: from scratch")
p.add_argument("--eq-steps", type=int, default=4000); p.add_argument("--eq-batch", type=int, default=16384); p.add_argument("--lr", type=float, default=0.01)
p.add_argument("--rollout-steps", type=int, default=0); p.add_argument("--rollout-batch", type=int, default=256); p.add_argument("--rollout-substeps", type=int, default=24)
p.add_argument("--j0-margin-deg", type=float, default=1.0); p.add_argument("--limit-margin-deg", type=float, default=0.5); p.add_argument("--passive-scale", type=float, default=2e-3)
p.add_argument("--holdout-every", type=int, default=5); p.add_argument("--seed", type=int, default=0)
p.add_argument("--passive-k0", type=float, default=0.1); p.add_argument("--passive-d0", type=float, default=1e-3); p.add_argument("--mu-reg", type=float, default=1e-3)
p.add_argument("--kel-reg", type=float, default=0.0, help="ridge on the joint elastic polynomial tau_k (k_el / per-joint torque scale), training loss only; separates tau_k from gravity")
args = p.parse_args()
dev = torch.device(args.device); torch.manual_seed(args.seed)
J0N = ("robot0_FFJ0", "robot0_MFJ0", "robot0_RFJ0", "robot0_LFJ0")


def load():
    eps = []
    for d in args.data:
        z = np.load(Path(d) / "episodes.npz"); man = json.loads((Path(d) / "manifest.json").read_text())
        for i in range(len(z["n_valid_states"])):
            n = int(z["n_valid_states"][i]) - 1                                  # usable control steps
            if n < 2: continue
            S = n * 6
            eps.append(dict(q=z["q_deg"][i, :S + 1], qd=z["qd_deg"][i, :S + 1], tg=z["target_deg"][i, :S], tau=z["tau"][i, :S], val=z["validity"][i, :n + 1], g=man["gravity_root"]))
    return eps


def tensors(eps, kin):
    """Stack all substeps: q_k, v_k, v_k+1, a_k, tau_k (rad, rad/s, N m) and the per-substep usable mask (contact-free control boundaries)."""
    lo, hi = np.deg2rad(kin["lower_deg"]), np.deg2rad(kin["upper_deg"]); j0 = [kin["joint_names"].index(n) for n in J0N]
    Q, V0, V1, A, F, U, EP = [], [], [], [], [], [], []
    for e_i, e in enumerate(eps):
        q = np.deg2rad(e["q"]); v = np.deg2rad(e["qd"]); S = len(e["tau"]); c = np.arange(S) // 6
        free = (e["val"][c] == 0) & (e["val"][np.minimum(c + 1, len(e["val"]) - 1)] == 0)
        Q.append(q[:S]); V0.append(v[:S]); V1.append(v[1:S + 1]); A.append(np.deg2rad(e["tg"])); F.append(e["tau"]); U.append(free); EP.append(np.full(S, e_i))
    t = lambda x, dt=torch.float32: torch.as_tensor(np.concatenate(x), dtype=dt, device=dev)
    return t(Q), t(V0), t(V1), t(A), t(F), torch.as_tensor(np.concatenate(U), device=dev), torch.as_tensor(np.concatenate(EP), device=dev)


def main():
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False); t0 = time.time(); log = open(out / "fit_log.jsonl", "w")
    kin = load_kinematics(); eps = load(); assert eps, "no usable episodes"
    g = eps[0]["g"]; assert all(np.allclose(e["g"], g) for e in eps)
    if args.init:
        cfg = json.loads(Path(args.init).read_text()); cfg["kinematics"] = kin
    else:
        cfg = dict(kinematics=kin, physics_hz=120, control_hz=20, gravity_root=g, tendons=sim_tendon_config(kin), actuator=dict(kind="pd_joint"))
    model = TendonHand(cfg, dev); act = kin["active"]; names = kin["joint_names"]
    Q, V0, V1, A, F, USE, EP = tensors(eps, kin); n_ep = len(eps)
    hold = (EP % args.holdout_every) == (args.holdout_every - 1); train = ~hold
    rep = dict(episodes=n_ep, substeps=int(len(Q)), usable_fraction=float(USE.float().mean()), train_substeps=int((USE & train).sum()), holdout_substeps=int((USE & hold).sum()),
               minutes=float(len(Q) / 120 / 60), init=args.init or "scratch")
    print("DATA", json.dumps(rep), flush=True)
    # ---- stage 1: actuator (C of the actuator tendons = their moment arms; from scratch the config's 1 on each own joint)
    with torch.no_grad():
        Ca = model.C()[model.meas]; l = Q @ Ca.T; ld = V0 @ Ca.T; K, D, Um = [], [], []
        for j in range(model.n_meas):
            f = F[:, j]; um = f.abs().max(); sat = (f.abs() > 0.995 * um).float().mean() > 0.01   # saturation seen: >1% of substeps at the maximum
            ok = (f.abs() < 0.99 * um) if sat else torch.ones_like(f, dtype=torch.bool); um = um if sat else um * 100.0
            X = torch.stack([A[ok, j] - l[ok, j], -ld[ok, j]], 1); w = torch.linalg.lstsq(X.double(), f[ok, None].double()).solution[:, 0].float()
            K.append(w[0]); D.append(w[1]); Um.append(um)
        model.log_K.copy_(torch.log(torch.stack(K).clamp_min(1e-4))); model.log_D.copy_(torch.log(torch.stack(D).clamp_min(1e-5))); model.log_umax.copy_(torch.log(torch.stack(Um)))
        pred = (model.K() * (A - l) - model.D() * ld).clamp(-model.umax(), model.umax()); r = (pred - F)
        rep["actuator"] = dict(K=model.K().tolist(), D=model.D().tolist(), umax=model.umax().tolist(), rel_rms=float(r.square().mean().sqrt() / F.square().mean().sqrt()))
    print("ACTUATOR", json.dumps(dict(rel_rms=rep["actuator"]["rel_rms"], K_median=float(model.K().median()), D_median=float(model.D().median()))), flush=True)
    # ---- stage 2: plant by equation error
    lo, hi = model.kin.lo, model.kin.hi; lm = np.deg2rad(args.limit_margin_deg)
    scale = torch.full((model.nj,), args.passive_scale, device=dev); scale[act] = F.square().mean(0).sqrt().clamp_min(1e-4)
    acc = (V1 - V0) * 120
    j0i = [names.index(n) for n in J0N]; j1i = [names.index(n.replace("J0", "J1")) for n in J0N]
    frozen = {"log_K", "log_D", "log_umax", "pt_logkl", "pt_lo", "pt_hi", "pin_logk", "pin_L0"}
    params = [p_ for n_, p_ in model.named_parameters() if n_ not in frozen]
    for n_, p_ in model.named_parameters(): p_.requires_grad_(n_ not in frozen)
    j1_of = {i: names.index(model.tendons[i]["joints"][0]) for i in model.passive_t}   # the first moment arm of each passive tendon sets its length scale
    with torch.no_grad():
        for i in model.passive_t: model.pt_logkl[model.passive_t.index(i)] = np.log(1e-6); model.pt_lo[model.passive_t.index(i)] = -10.0; model.pt_hi[model.passive_t.index(i)] = 10.0
        if not args.init:   # neutral start for the passive spring and damping (log parameters near zero get no gradient)
            model.pt_logk.fill_(np.log(args.passive_k0)); model.pt_logd.fill_(np.log(args.passive_d0))
    opt = torch.optim.Adam(params, lr=args.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.eq_steps)
    idx_tr = torch.nonzero(USE & train)[:, 0]; idx_ho = torch.nonzero(USE & hold)[:, 0]; gen = torch.Generator(device=dev).manual_seed(args.seed)
    def eq_resid(ii):
        q, v1, a_, f = Q[ii], V1[ii], acc[ii], F[ii]
        bodies = model._bodies(q); M = model.mass_matrix(q, bodies); bias = model.rnea(q, v1, torch.zeros_like(q), True, bodies)
        fp, _, _, Cp, Cpf = model.passive_force(q, v1); Cfm = model.Cf()[model.meas]
        r = (M @ a_[..., None])[..., 0] + bias + model.h_terms(q, v1) - fp @ Cpf - (f - model.tendon_friction(f, v1)) @ Cfm
        mask = ((q > lo + lm) & (q < hi - lm))
        jm = np.deg2rad(args.j0_margin_deg); pin = (q[:, j0i] < lo[j0i] + jm) | (q[:, j0i] > hi[j0i] - jm)   # simulator's pinned-J0 lock: mask J1 and J0 of that finger
        mask[:, j0i] &= ~pin; mask[:, j1i] &= ~pin
        return r / scale, mask.float()
    def eq_loss(ii):
        r, mask = eq_resid(ii); return ((r.square() * mask).sum(0) / mask.sum(0).clamp_min(1)).mean() + args.mu_reg * model.mu.square().sum()
    for it in range(args.eq_steps):
        ii = idx_tr[torch.randint(0, len(idx_tr), (min(args.eq_batch, len(idx_tr)),), generator=gen, device=dev)]
        loss = eq_loss(ii)
        if args.kel_reg > 0: loss = loss + args.kel_reg * (model.k_el / scale[:, None]).square().sum(1).mean()
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            for i, j in j1_of.items(): model.C_raw.grad[i, j] = 0.0                  # length scale of the passive tendons fixed by their first arm
            model.C_raw.grad[model.meas] = 0.0   # measured tendons: C is kinematic (excursion vs joint angle, Li et al. ID.2), not fitted to forces
        opt.step(); sched.step()
        if it % 250 == 0 or it == args.eq_steps - 1:
            with torch.no_grad(): ho = float(eq_loss(idx_ho[:65536])) if len(idx_ho) else float("nan")
            rec = dict(stage="eq", it=it, loss=float(loss), holdout=ho, t=time.time() - t0); log.write(json.dumps(rec) + "\n"); log.flush(); print("EQ", json.dumps(rec), flush=True)
    rep["eq"] = rec
    # ---- stage 3: rollout refinement (optional)
    if args.rollout_steps:
        H = args.rollout_substeps
        same_ep = torch.zeros(len(USE), dtype=torch.bool, device=dev); same_ep[: len(USE) - H] = (EP[: len(USE) - H] == EP[H:]) & USE[: len(USE) - H] & USE[H:]
        starts = torch.nonzero(same_ep & train)[:, 0]
        for p_ in model.parameters(): p_.requires_grad_(True)
        opt = torch.optim.Adam([p_ for n_, p_ in model.named_parameters() if n_ not in {"pt_logkl", "pt_lo", "pt_hi", "pin_logk", "pin_L0"}], lr=args.lr * 0.1)
        for it in range(args.rollout_steps):
            ii = starts[torch.randint(0, len(starts), (args.rollout_batch,), generator=gen, device=dev)]; q, v = Q[ii], V0[ii]; outq = []
            for k in range(H): q, v = model.step(q, v, A[ii + k]); outq.append(q)
            P_ = torch.stack(outq, 1); Y = torch.stack([Q[ii + k + 1] for k in range(H)], 1)
            loss = torch.rad2deg(P_ - Y).square().mean((1, 2)).clamp_min(1e-12).sqrt().mean(); opt.zero_grad(); loss.backward()
            for p_ in model.parameters():
                if p_.grad is not None: p_.grad = torch.nan_to_num(p_.grad)
            opt.step()
            if it % 100 == 0 or it == args.rollout_steps - 1:
                rec = dict(stage="rollout", it=it, loss_deg=float(loss), t=time.time() - t0); log.write(json.dumps(rec) + "\n"); log.flush(); print("RO", json.dumps(rec), flush=True)
        rep["rollout"] = rec
    cfg_out = {k: v for k, v in cfg.items() if k != "kinematics"}; cfg_out["kinematics"] = kin["source"]; cfg_out["identified"] = model.values()
    (out / "model.json").write_text(json.dumps(cfg_out)); rep["seconds"] = time.time() - t0; rep["data"] = args.data
    (out / "report.json").write_text(json.dumps(rep, indent=1)); print("FIT_TENDON_COMPLETE", out, flush=True)


main()
