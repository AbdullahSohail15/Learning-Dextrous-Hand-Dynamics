"""Experiment 04b: MAP (regularised) refit of the closed-loop exploration data of experiment 02 (arm CL, all 180 s, every hand).
Prior = the spec model N0 in log space: sigma = log 1.5 for K, D, m, u_max (not identifiable in free motion), log 3 for v_max (the data
sets it), 5 (x 1e-3 N m) for gravity. Fresh fit from N0, 800 Adam steps. Reports the held-out window error (same test program as exp 02)
and the parameter errors; writes <output>/fits_reg.pt with the same layout the cube script reads (fits["REG"][-1]["theta"]).
Usage: ./run_python.sh scripts/identify/fit_regularised.py --explore artifacts/exp02/explore --output artifacts/exp04b/fit --weight 1e-3
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inspire_dynamics.torch_kin import InspireKin
from inspire_dynamics import servo_model as sm

ap = argparse.ArgumentParser(); ap.add_argument("--explore", required=True); ap.add_argument("--output", required=True)
ap.add_argument("--weight", type=float, default=1e-3); ap.add_argument("--iters", type=int, default=800); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=False); dev = a.device
z = np.load(Path(a.explore) / "explore.npz"); F = torch.load(Path(a.explore) / "fits.pt", weights_only=False)
env_of = {(r[0], int(r[1])): int(r[2]) for r in z["env_of"]}; test_env = [int(e) for e in z["test_env"]]; NH = len(test_env)
kin = InspireKin(dev); lo, hi = kin.lo, kin.hi; P, A = z["P"], z["A"]


def data(e, T):
    p = torch.tensor(P[e, :T + 1], device=dev); u = torch.tensor(A[e, :T], device=dev); u = torch.cat([u, u[-1:]])
    s = sm.windows(p, u, torch.tensor(kin.penetration(p) <= 0.0005)); return sm.gather(p, u, s)


train = [data(env_of[("CL", h)], A.shape[1]) for h in range(NH)]; test = [data(e, 1200) for e in test_env]
th0 = F["th_N0"].to(dev)[None].expand(NH, 6, 7).contiguous(); sig = torch.tensor([math.log(1.5)] * 3 + [math.log(3.0), math.log(1.5), 5.0, 5.0], device=dev)
th_reg, loss = sm.fit(th0, train, lo, hi, iters=a.iters, prior=th0, prior_sigma=sig, prior_weight=a.weight)
th_cl = F["fits"]["CL"][-1]["theta"].to(dev); tv = sm.unpack(F["th_true"])
rep = dict(weight=a.weight, iters=a.iters)
for name, th in (("REG", th_reg), ("CL", th_cl), ("TRUE", F["th_true"].to(dev)), ("N0", th0)):
    err, _ = sm.window_error(th, test, lo, hi); u = sm.unpack(th.cpu())
    rep[name] = dict(window_err_deg=err, rel_err={k: float(np.median(((u[k] - tv[k]) / tv[k]).abs().numpy())) for k in ("K", "D", "m", "vmax", "umax")},
                     hand0={k: u[k][0].numpy().round(5).tolist() for k in ("K", "D", "m", "vmax", "umax")})
print(json.dumps(rep, indent=1))
json.dump(rep, open(out / "report.json", "w"), indent=1)
torch.save(dict(fits=dict(REG=[dict(seconds=180.0, theta=th_reg.cpu())], CL=F["fits"]["CL"]), th_true=F["th_true"], th_N0=F["th_N0"], hands=F["hands"], spec=F["spec"]), out / "fits_reg.pt")
print("FITREG_COMPLETE")
