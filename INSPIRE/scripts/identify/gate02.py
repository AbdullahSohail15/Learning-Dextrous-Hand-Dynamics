"""Experiment 02 gate 0: (1) per-env servo parameters reach PhysX (readback) and the speed limit set at runtime is the one honoured;
(2) M_bar = J_c^T M J_c and g_bar = J_c^T G from PhysX at random poses (how large, how much they vary: decides the constant-m model);
(3) TRUE model (true K, D, v_max, u_max; m from (2) + armature; gravity fitted to (2)) vs Isaac on a held-out free program: 0.6 s
open-loop error (H1 < 0.3 deg). Output: <output>/gate.json, traces.npz.

Usage: ./run_python.sh scripts/identify/gate02.py --headless --device cuda:0 --output artifacts/exp02/gate
"""
import argparse, json, math
from pathlib import Path
from omni.isaac.lab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args(); app = AppLauncher(args).app

import numpy as np, torch
from inspire_dynamics.sim import InspireSim, spec_params, true_hand_params, ACTUATED, CONTROL_DT
from inspire_dynamics.torch_kin import InspireKin
from inspire_dynamics import servo_model as sm, explore as ex

out = Path(args.output); out.mkdir(parents=True, exist_ok=False); dev = args.device
NH = 3; S = InspireSim(num_envs=2 + NH, device=dev); kin = InspireKin(dev); lo, hi = S.lo, S.hi
spec = spec_params(); hands = [true_hand_params(h) for h in range(NH)]
P = {k: torch.stack([spec[k], spec[k]] + [h[k] for h in hands]) for k in spec}
P["vmax"][1] = spec["vmax"] * 0.5                                                   # env 1: speed limit halved at runtime
S.set_servo_params(**P); rb = S.servo_params()
rep = dict(readback_max_rel_err={k: float(((rb[k].cpu() - P[k]).abs() / P[k].abs()).max()) for k in P})

# (1) speed limit honoured: each servo 0 -> 90 % range, slope over the middle of the move
S.reset(); slopes = []
for i in range(6):
    tgt = torch.zeros(S.N, 6, device=dev); tgt[:, i] = lo[i] + 0.9 * (hi[i] - lo[i]); tr = []
    for _ in range(int(2.5 / CONTROL_DT)): S.advance(tgt); tr.append(S.state()["qa"][:, i].cpu())
    tr = torch.stack(tr, 1); x = tr - lo[i].cpu(); span = 0.9 * float(hi[i] - lo[i]); sl = []
    for e in range(S.N):
        k0 = int((x[e] > 0.2 * span).float().argmax()); k1 = int((x[e] > 0.7 * span).float().argmax())
        sl.append(float((x[e, k1] - x[e, k0]) / max(k1 - k0, 1) / CONTROL_DT))
    slopes.append(sl); S.reset()
slopes = np.array(slopes).T                                                         # (N, 6) rad/s
rep["speed_measured_over_set"] = (slopes / P["vmax"].numpy()).round(3).tolist()

# (2) PhysX mass matrix and gravity through the linkage at random poses
Jc = torch.zeros(len(S.names), 6, device=dev)
for i, j in enumerate(S.ia): Jc[j, i] = 1.0
for l, j in enumerate(S.il): Jc[j, int(S.l_src[l])] = float(S.l_k[l])
rng = np.random.default_rng(7); Mb, Gb, Ps = [], [], []
for r in range(16):
    qa = torch.tensor(rng.uniform(lo.cpu(), hi.cpu(), (S.N, 6)), dtype=torch.float32)
    qa[:, 2:] = qa[:, 2:]; ok = kin.penetration(qa) <= 0
    S.reset(qa); S.advance(qa.to(dev))
    M = S.hand.root_physx_view.get_mass_matrices().to(dev); G = S.hand.root_physx_view.get_generalized_gravity_forces().to(dev)
    Mb.append(Jc.T @ M @ Jc); Gb.append(G @ Jc); Ps.append(S.state()["qa"])
Mb, Gb, Ps = torch.cat(Mb), torch.cat(Gb), torch.cat(Ps)
d = Mb.diagonal(dim1=1, dim2=2); off = (Mb - torch.diag_embed(d)).abs().amax((1, 2))
rep["Mbar_diag_mean"] = d.mean(0).tolist(); rep["Mbar_diag_rel_range"] = ((d.amax(0) - d.amin(0)) / d.mean(0)).tolist()
rep["Mbar_max_offdiag_over_diag"] = float((off / d.amin(1)).max())
rep["gbar_max_abs_Nm"] = Gb.abs().amax(0).tolist(); rep["gbar_max_over_umax"] = float((Gb.abs().amax(0).cpu() / spec["effort"]).max())
# gravity basis fit g = g1 cos p + g2 sin p per servo (only the servo's own angle; the thumb couples yaw/pitch, residual reported)
gfit, gres = [], []
for i in range(6):
    X = torch.stack([torch.cos(Ps[:, i]), torch.sin(Ps[:, i])], 1); c = torch.linalg.lstsq(X, Gb[:, i:i + 1]).solution[:, 0]
    gfit.append(c.tolist()); gres.append(float((X @ c - Gb[:, i]).abs().max()))
rep["gravity_fit_1e-3Nm"] = (np.array(gfit) * 1e3).tolist(); rep["gravity_fit_max_resid_Nm"] = gres

# (3) TRUE model vs Isaac on a held-out free (screened) random program, 30 s per true hand
m_link = d.mean(0).cpu()                                                           # PhysX mass matrix excludes the drive armature (diag < armature)
def true_theta(h, armature_in_M=True):
    a = hands[h]; m = m_link + a["armature"]
    return sm.pack(a["K"], a["D"], m, a["vmax"], a["effort"], torch.tensor(gfit)[:, 0] * 1e3, torch.tensor(gfit)[:, 1] * 1e3)
rng = np.random.default_rng(99); S.reset(); T = 600
A = torch.zeros(S.N, T, 6)
for e in range(S.N):
    chunks = []; start = None
    while len(chunks) < T // ex.CHUNK:
        c = torch.tensor(ex.gen_chunks(rng, lo.cpu().numpy(), hi.cpu().numpy(), 16, start), dtype=torch.float32)
        pen = kin.penetration(c.reshape(-1, 6)).reshape(16, -1).max(1); good = np.where(pen <= 0)[0]
        if len(good): chunks.append(c[good[0]]); start = c[good[0], -1].numpy()
    A[e] = torch.cat(chunks)
Q = []
for t in range(T): S.advance(A[:, t].to(dev)); Q.append(S.state()["qa"].cpu())
Pm = torch.cat([torch.zeros(S.N, 1, 6), torch.stack(Q, 1)], 1)                      # p[0] = reset pose, p[t+1] after step t
pen = kin.penetration(Pm.reshape(-1, 6)).reshape(S.N, -1)
TH = torch.stack([true_theta(h) for h in range(NH)]).to(dev)
data = []
for h in range(NH):
    e = 2 + h; p = Pm[e].to(dev); a = A[e].to(dev); valid = torch.tensor(pen[e] <= 0.0005)
    s = sm.windows(p, a, valid); data.append(sm.gather(p, a, s))
med, per = sm.window_error(TH, data, lo, hi)
spec_th = sm.pack(spec["K"], spec["D"], m_link + spec["armature"], spec["vmax"], spec["effort"]).to(dev)[None].expand(NH, 6, 7).contiguous()
med0, per0 = sm.window_error(spec_th, data, lo, hi)
rep["H1_true_model_window_err_deg"] = med; rep["true_per_servo_final_deg"] = per; rep["N0_window_err_deg"] = med0; rep["N0_per_servo_final_deg"] = per0
rep["contact_frac"] = (pen[2:] > 0.0005).mean(1).tolist()
rep["m_link"] = m_link.tolist(); rep["hands"] = [{k: v.tolist() for k, v in h.items()} for h in hands]; rep["spec"] = {k: v.tolist() for k, v in spec.items()}
json.dump(rep, open(out / "gate.json", "w"), indent=1)
np.savez(out / "traces.npz", p=Pm.numpy(), a=A.numpy(), slopes=slopes, Mbar=Mb.cpu().numpy(), gbar=Gb.cpu().numpy(), poses=Ps.cpu().numpy())
print(json.dumps(rep, indent=1)); print("GATE02_COMPLETE"); app.close()
