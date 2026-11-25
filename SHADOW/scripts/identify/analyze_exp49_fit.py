"""Experiment 49: score identified TendonHand models against the simulator (evaluation only; the fits never read the truth).

Function space at 1,000 random poses / velocities (joints interior): M (Frobenius, without armature as PhysX reports it), G and N relative
errors, per-joint G error; actuator K, D, u_max relative errors per commanded joint; 0.6 s open-loop prediction (actuator from the commands)
on the fixed gate-0 test recording (artifacts/exp49/gate0/run3: fresh gravity-on programs), contact-free windows split into J0 interior and
J0 pinned. --models label=model.json ... (the label TRUE loads the truth with the gate-0 least-squares tendon spring).
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics.kinematics_only import load_kinematics
from shadow_dynamics.tendon_hand_model import TendonHand
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--models", nargs="+", required=True); p.add_argument("--output", required=True); p.add_argument("--device", default="cuda:0")
p.add_argument("--test", default="artifacts/exp49/gate0/run3"); p.add_argument("--windows", type=int, default=1500)
p.add_argument("--save-true", default=None, help="also write the TRUE reference as a TendonHand model.json (for the planner)")
args = p.parse_args(); dev = torch.device(args.device)
kin = load_kinematics(); names = kin["joint_names"]; act = kin["active"]; J0 = [names.index(f"robot0_{f}J0") for f in ("FF", "MF", "RF", "LF")]
truth = json.loads((Path(args.test) / "truth.json").read_text())
def true_model():
    """Truth (rigid body, drive, armature) with the passive tendons as measured at gate 0 (least-squares spring on the stock PhysX tendon)."""
    from shadow_dynamics.tendon_hand_model import sim_tendon_config
    ls = json.loads((Path(args.test) / "tendon_ls.json").read_text()); FG = ("FF", "MF", "RF", "LF")
    tc = sim_tendon_config(kin, coupling={f: tuple(ls[f]["c"]) for f in FG})
    m = TendonHand(dict(kinematics=kin, physics_hz=120, control_hz=20, gravity_root=truth["gravity_root"], tendons=tc, actuator=dict(kind="pd_joint")), dev).eval()
    v = copy.deepcopy(truth["values"]); v["C"] = m.C_raw.tolist(); v["Cf"] = m.Cf_raw.tolist(); m.load_values(v)
    with torch.no_grad():
        for n_, i in enumerate(m.passive_t):
            f = tc["tendons"][i]["name"][4:]; m.pt_logk[n_] = np.log(ls[f]["k"]); m.pt_L0[n_] = ls[f]["L0"]; m.pt_logkl[n_] = np.log(1e-6); m.pt_lo[n_] = -10; m.pt_hi[n_] = 10; m.pt_logd[n_] = np.log(1e-9)
    return m
def load(spec):
    lab, path = spec.split("=")
    if path == "TRUE":
        return lab, true_model()
    cfg = json.loads(Path(path).read_text()); m = TendonHand(cfg, dev).eval(); return lab, m
T = true_model()
if args.save_true:
    tc = copy.deepcopy(T.cfg); tc["kinematics"] = kin["source"]; tc["identified"] = T.values(); Path(args.save_true).write_text(json.dumps(tc))
gen = torch.Generator(device="cpu").manual_seed(4901); lo, hi = T.kin.lo.cpu(), T.kin.hi.cpu(); mg = 0.05
Qs = (lo + mg + (hi - lo - 2 * mg) * torch.rand(1000, 24, generator=gen)).to(dev); Vs = (torch.randn(1000, 24, generator=gen) * 1.5).to(dev)
with torch.no_grad():
    Mt = T.mass_matrix(Qs) - torch.diag_embed(T.arm())[None]; Gt = T.gravity_torque(Qs); GKt = Gt + T.h_terms(Qs, torch.zeros_like(Qs)); Mat = T.mass_matrix(Qs); Nt = T.coriolis(Qs, Vs)
d = np.load(Path(args.test) / "recording.npz"); q = torch.tensor(d["q_deg"], device=dev); qd = torch.tensor(d["qd_deg"], device=dev); tg = torch.tensor(d["target_deg"], device=dev)
V = torch.tensor(d["validity"], device=dev); E_ = q.shape[0]; Tc = V.shape[1] - 1; H = 12
idx = [(e, s) for e in range(E_) for s in range(0, Tc - H, 6) if bool((V[e, s:s + H + 1] == 0).all())]
rs = np.random.default_rng(0); idx = [idx[i] for i in rs.permutation(len(idx))[:args.windows]]
es = torch.tensor([i[0] for i in idx], device=dev); ss = torch.tensor([i[1] for i in idx], device=dev)
LO = torch.tensor(kin["lower_deg"], device=dev); HI = torch.tensor(kin["upper_deg"], device=dev)
qw = torch.stack([q[es, ss * 6 + k] for k in range(0, 73, 6)], 1); pinned = ~((qw[..., J0] > LO[J0] + 1) & (qw[..., J0] < HI[J0] - 1)).all(-1).all(-1)
res = {}
for spec in args.models:
    lab, m = load(spec)
    with torch.no_grad():
        M = m.mass_matrix(Qs) - torch.diag_embed(m.arm())[None]; G = m.gravity_torque(Qs); GK = G + m.h_terms(Qs, torch.zeros_like(Qs)); Ma = m.mass_matrix(Qs); N = m.coriolis(Qs, Vs)
        rel = lambda a, b: (a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1).clamp_min(1e-12)
        gj = ((G - Gt).square().mean(0).sqrt() / Gt.square().mean(0).sqrt().clamp_min(1e-9))
        r = dict(M_rel_median=float(rel(M, Mt).median()), G_rel_median=float(rel(G, Gt).median()), N_rel_median=float(rel(N, Nt).median()),
                 GK_rel_median=float(rel(GK, GKt).median()), Marm_rel_median=float(rel(Ma, Mat).median()),
                 G_rel_per_joint={names[j][7:]: round(float(gj[j]), 3) for j in act},
                 K_rel_median=float(((m.K() - T.K()).abs() / T.K()).median()), D_rel_median=float(((m.D() - T.D()).abs() / T.D()).median()),
                 umax_rel_median=float(((m.umax() - T.umax()).abs() / T.umax()).median()))
        qq, vv = torch.deg2rad(q[es, ss * 6]), torch.deg2rad(qd[es, ss * 6])
        for k in range(72): qq, vv = m.step(qq, vv, torch.deg2rad(tg[es, ss * 6 + k]))
        e = torch.rad2deg(qq) - q[es, ss * 6 + 72]; per = e.square().mean(1).sqrt(); fin = torch.isfinite(per)
        for nm, sel in (("interior", ~pinned), ("pinned", pinned)):
            sl = sel & fin; r[f"pred_{nm}_rms"] = float(e[sl].square().mean().sqrt()); r[f"pred_{nm}_median_window"] = float(per[sl].median())
        r["pred_nonfinite"] = int((~fin).sum())
    res[lab] = r; print(lab, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "G_rel_per_joint"}), flush=True)
Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(dict(models=args.models, test=args.test, windows=len(idx), pinned_windows=int(pinned.sum()), results=res), indent=1))
print("ANALYZE_EXP49_COMPLETE", args.output)
