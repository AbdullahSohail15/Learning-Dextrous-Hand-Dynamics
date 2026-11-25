"""Experiment 02: closed-loop self-exploration vs open-loop exploration, servo-linkage model identification.

Envs (one Isaac process, lockstep): for each of NH unknown hands (true_hand_params(h)): arm CL (closed loop), arm D0 (designed once from
the spec model N0), arm R (random), and one test env running a held-out, contact-free random program (fresh seed, 60 s).
Rounds: cumulative exploration 15, 30, 60, 120, 180 s (chunks of 3 s). After every round all arms are refitted on their data so far
(warm start, 400 Adam steps); CL designs its next round from its new fit. Contact windows (predicted penetration of the measured pose
> 0.5 mm) are masked for every arm. Evaluation (end): 0.6 s open-loop window error on the test program, parameter errors.
Output: <output>/{explore.npz, fits.pt, report.json}.

Usage: ./run_python.sh scripts/identify/exp02_explore.py --headless --device cuda:0 --output artifacts/exp02/explore --gate artifacts/exp02/gate
"""
import argparse, json, math, time
from pathlib import Path
from omni.isaac.lab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); ap.add_argument("--gate", required=True)
ap.add_argument("--hands", type=int, default=3); ap.add_argument("--rounds", default="5,5,10,20,20"); ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(ap); args = ap.parse_args(); app = AppLauncher(args).app

import numpy as np, torch
from inspire_dynamics.sim import InspireSim, spec_params, true_hand_params, CONTROL_DT
from inspire_dynamics.torch_kin import InspireKin
from inspire_dynamics import servo_model as sm, explore as ex

out = Path(args.output); out.mkdir(parents=True, exist_ok=False); dev = args.device
gate = json.load(open(Path(args.gate) / "gate.json")); gfit = torch.tensor(gate["gravity_fit_1e-3Nm"])
m_link = torch.tensor(gate["Mbar_diag_mean"])   # link inertia through the linkage; PhysX M excludes armature (gate)
NH = args.hands; ARMS = ["CL", "D0", "R"]; rounds = [int(x) for x in args.rounds.split(",")]; NCH = sum(rounds)
env_of = {(a, h): i for i, (a, h) in enumerate((a, h) for h in range(NH) for a in ARMS)}; test_env = {h: len(env_of) + h for h in range(NH)}
S = InspireSim(num_envs=len(env_of) + NH, device=dev); kin = InspireKin(dev); lo, hi = S.lo, S.hi; lon, hin = lo.cpu().numpy(), hi.cpu().numpy()
spec = spec_params(); hands = [true_hand_params(h) for h in range(NH)]
hand_of_env = [h for h in range(NH) for _ in ARMS] + list(range(NH))
S.set_servo_params(**{k: torch.stack([hands[h][k] for h in hand_of_env]) for k in spec})
th_N0 = sm.pack(spec["K"], spec["D"], m_link + spec["armature"], spec["vmax"], spec["effort"]).to(dev)
th_true = torch.stack([sm.pack(h["K"], h["D"], m_link + h["armature"], h["vmax"], h["effort"], gfit[:, 0], gfit[:, 1]) for h in hands]).to(dev)
rng = np.random.default_rng(args.seed)

# programs known up front: R (random), D0 (designed from N0, predicted information), test (contact-free under the true geometry)
prog = {}
for h in range(NH):
    prog[("R", h)] = np.concatenate(list(ex.gen_chunks(rng, lon, hin, NCH, None)))
    z = torch.zeros(6, device=dev)
    A, st = ex.design_round(kin, th_N0[None], z, z, torch.zeros(6, 7, 7, device=dev), NCH, rng, lo, hi); prog[("D0", h)] = A
    chunks, start = [], None
    while len(chunks) < 20:
        c = ex.gen_chunks(rng, lon, hin, 16, start); pen = kin.penetration(torch.tensor(c, dtype=torch.float32).reshape(-1, 6)).reshape(16, -1).max(1)
        g = np.where(pen <= 0)[0]
        if len(g): chunks.append(c[g[0]]); start = c[g[0], -1]
    prog[("T", h)] = np.concatenate(chunks)
print("D0 designed", flush=True)

T_total = NCH * ex.CHUNK; N = S.N
Ptr = np.zeros((N, T_total + 1, 6), np.float32); Atr = np.zeros((N, T_total, 6), np.float32); Qtr = np.zeros((N, T_total + 1, 12), np.float32)
S.reset(); st = S.state(); Ptr[:, 0] = st["qa"].cpu().numpy(); Qtr[:, 0] = st["q"].cpu().numpy()
th_fit = {a: th_N0[None].expand(NH, 6, 7).clone() for a in ARMS}; fits = {a: [] for a in ARMS}; design_stats = []; t_fit = []
cl_prog = np.zeros((NH, T_total, 6), np.float32); t0 = 0


def data_for(env, T):
    p = torch.tensor(Ptr[env, :T + 1], device=dev); a = torch.tensor(Atr[env, :T], device=dev)
    valid = torch.tensor(kin.penetration(p) <= 0.0005)
    s = sm.windows(p, torch.cat([a, a[-1:]]), valid); return sm.gather(p, torch.cat([a, a[-1:]]), s), float((~valid).float().mean())


for r, nch in enumerate(rounds):
    T1 = t0 + nch * ex.CHUNK
    # closed loop: design this round from the current fit and the information of the data measured so far
    tic = time.time()
    for h in range(NH):
        e = env_of[("CL", h)]; p0 = torch.tensor(Ptr[e, t0], device=dev); v0 = torch.tensor((Ptr[e, t0] - Ptr[e, max(t0 - 1, 0)]) / CONTROL_DT, device=dev)
        F0 = torch.zeros(6, 7, 7, device=dev)
        if t0 > 0:
            (dp0, dv0, dA, dY), _ = data_for(e, t0); F0 = sm.window_info(th_fit["CL"][h:h + 1], dp0[::3], dv0[::3], dA[::3], lo, hi)
        A, st = ex.design_round(kin, th_fit["CL"][h:h + 1], p0, v0, F0, nch, rng, lo, hi); cl_prog[h, t0:T1] = A
        design_stats.append(dict(round=r, hand=h, **{k: v if not isinstance(v, list) else v[-1] for k, v in st.items()}, seconds=time.time() - tic))
    for t in range(t0, T1):
        tgt = torch.zeros(N, 6)
        for (a, h), e in env_of.items(): tgt[e] = torch.tensor(cl_prog[h, t] if a == "CL" else prog[(a, h)][t])
        for h in range(NH): tgt[test_env[h]] = torch.tensor(prog[("T", h)][t % len(prog[("T", h)])])
        S.advance(tgt.to(dev)); st = S.state(); Atr[:, t] = tgt.numpy(); Ptr[:, t + 1] = st["qa"].cpu().numpy(); Qtr[:, t + 1] = st["q"].cpu().numpy()
    # refit every arm on all its data so far
    tic = time.time()
    for a in ARMS:
        D = []; cf = []
        for h in range(NH): d, c = data_for(env_of[(a, h)], T1); D.append(d); cf.append(c)
        th_fit[a], loss = sm.fit(th_fit[a], D, lo, hi, iters=args.iters, seed=r)
        fits[a].append(dict(seconds=T1 * CONTROL_DT, theta=th_fit[a].cpu(), loss=loss.tolist(), contact_frac=cf, windows=[int(d[0].shape[0]) for d in D]))
    t_fit.append(time.time() - tic); t0 = T1
    print(f"round {r} done: {T1 * CONTROL_DT:.0f} s explored, fit {t_fit[-1]:.1f} s, loss " +
          " ".join(f"{a}={np.mean(fits[a][-1]['loss']):.3f}" for a in ARMS), flush=True)

# evaluation on the held-out test program of each hand (first 60 s of the test env)
Ttest = len(prog[("T", 0)]); test = []
for h in range(NH):
    e = test_env[h]; p = torch.tensor(Ptr[e, :Ttest + 1], device=dev); a = torch.tensor(Atr[e, :Ttest], device=dev)
    valid = torch.tensor(kin.penetration(p) <= 0.0005); s = sm.windows(p, torch.cat([a, a[-1:]]), valid); test.append(sm.gather(p, torch.cat([a, a[-1:]]), s))
rep = dict(rounds=rounds, hands=NH, true_err=sm.window_error(th_true, test, lo, hi)[0],
           N0_err=sm.window_error(th_N0[None].expand(NH, 6, 7).contiguous(), test, lo, hi)[0], arms={}, design=design_stats, fit_seconds=t_fit)
tv = sm.unpack(th_true.cpu())
for a in ARMS:
    rows = []
    for f in fits[a]:
        err, per = sm.window_error(f["theta"].to(dev), test, lo, hi); u = sm.unpack(f["theta"])
        rel = {k: ((u[k] - tv[k]) / tv[k]).abs().tolist() for k in ("K", "D", "m", "vmax", "umax")}
        rows.append(dict(seconds=f["seconds"], err=err, per_servo=per, rel_err=rel, contact_frac=f["contact_frac"], windows=f["windows"], loss=f["loss"]))
    rep["arms"][a] = rows
# unplanned contact over the whole exploration, per arm
rep["contact_frac_total"] = {a: [float((kin.penetration(torch.tensor(Ptr[env_of[(a, h)]], device=dev)) > 0.0005).mean()) for h in range(NH)] for a in ARMS}
np.savez(out / "explore.npz", P=Ptr, A=Atr, Q=Qtr, names=np.array(S.names), env_of=np.array([[a, h, e] for (a, h), e in env_of.items()]), test_env=np.array(list(test_env.values())))
torch.save(dict(fits=fits, th_true=th_true.cpu(), th_N0=th_N0.cpu(), hands=hands, spec=spec), out / "fits.pt")
json.dump(rep, open(out / "report.json", "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("true_err", "N0_err", "contact_frac_total")}, indent=1))
for a in ARMS: print(a, [(r_["seconds"], np.round(r_["err"], 3).tolist()) for r_ in rep["arms"][a]])
print("EXP02_COMPLETE"); app.close()
