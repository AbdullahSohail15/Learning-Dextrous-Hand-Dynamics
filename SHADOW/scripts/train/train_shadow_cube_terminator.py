"""Train the drop predictor (shadow_dynamics.cube_terminator) on hand + cube data (experiment 12, round 2b).

Label for window (episode e, anchor t): the episode ended by a drop and its last recorded step is within 12 steps of t.
The classifier needs only past states and the future applied targets, so its windows run up to the drop itself (the
world model's windows need 12 future states and therefore never contain a drop before their last step); targets after
an episode's last action are padded with that action.
Weighted BCE (positives up-weighted to balance the classes), AdamW, model selection on validation F1; precision /
recall / F1 on the test split at thresholds 0.3 / 0.5 / 0.7. Saves best.pt.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics.cube_world_data import CubeDataset, corrupt_history, occlusion_cfg
from shadow_dynamics.cube_terminator import DropClassifier

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data", nargs="+", required=True); p.add_argument("--output", required=True); p.add_argument("--device", default="cuda:0")
p.add_argument("--steps", type=int, default=20000); p.add_argument("--batch-size", type=int, default=4096); p.add_argument("--lr", type=float, default=3e-4)
# experiment 24: partially observed cube on the history inputs (cube_world_data.corrupt_history), fresh draws per training batch, fixed draws for val / test
p.add_argument("--obs-noise-pos-mm", type=float, default=0.0); p.add_argument("--obs-noise-rot-deg", type=float, default=0.0)
p.add_argument("--obs-occl-enter", type=float, default=0.0); p.add_argument("--obs-occl-stay", type=float, default=0.0)
a = p.parse_args(); torch.set_num_threads(1); torch.manual_seed(0); dev = torch.device(a.device); out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
ds = CubeDataset(a.data); stats = ds.normalization("train")
E = len(ds.episodes); T = max(len(e["action_applied"]) for e in ds.episodes)
X = np.zeros((E, T + 1, 63), np.float32); U = np.zeros((E, T + 12, 20), np.float32); L = np.zeros(E, np.int64); D = np.zeros(E, bool); DST = np.zeros(E, np.int64)
for i, e in enumerate(ds.episodes):
    n = len(e["action_applied"]); X[i, :n + 1] = ds.state(i); U[i, :n] = e["action_applied"]; U[i, n:] = e["action_applied"][-1]; L[i] = n; D[i] = ds.meta[i]["dropped"]; DST[i] = ds.meta[i]["drop_step"]
X, U = torch.as_tensor(X, device=dev), torch.as_tensor(U, device=dev)
def windows(split):
    end = np.where(D, DST, L)   # anchors up to the drop (states after it are not valid anchors)
    W = np.concatenate([np.stack([np.full(end[e] - 3, e), np.arange(3, end[e])], 1) for e in range(E) if ds.split[e] == split and end[e] > 3])
    y = D[W[:, 0]] & (DST[W[:, 0]] - W[:, 1] <= 12)
    return torch.as_tensor(W, device=dev), torch.as_tensor(y, dtype=torch.float32, device=dev)
ho, pa, fu = torch.arange(-3, 1, device=dev), torch.arange(-3, 0, device=dev), torch.arange(0, 12, device=dev)
OBS = occlusion_cfg(a.obs_noise_pos_mm, a.obs_noise_rot_deg, a.obs_occl_enter, a.obs_occl_stay); G = torch.Generator(device=dev).manual_seed(7); print("OBS", OBS, flush=True)
def batch(W, gen=None):
    e, t = W[:, 0:1], W[:, 1:2]
    s_ = X[e, t + ho] if OBS is None else corrupt_history(X, e, t, OBS, gen if gen is not None else G)[0]
    return s_, U[e, t + pa], U[e, t + fu]
Wtr, ytr = windows("train"); Wva, yva = windows("validation"); Wte, yte = windows("test")
pos = float(ytr.mean()); print(f"train windows {len(Wtr)} positives {pos:.3%}; val {len(Wva)} ({float(yva.mean()):.3%}); test {len(Wte)} ({float(yte.mean()):.3%})", flush=True)
m = DropClassifier(stats=stats).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4); w_pos = (1 - pos) / max(pos, 1e-6)
def scores(W):
    g_ = torch.Generator(device=dev).manual_seed(999)
    with torch.no_grad(): return torch.cat([torch.sigmoid(m.logits(*batch(W[i:i + 16384], g_))) for i in range(0, len(W), 16384)])
def prf(pr, y, thr):
    pd_ = pr > thr; tp = float((pd_ & (y > 0.5)).sum()); P = tp / max(1.0, float(pd_.sum())); R = tp / max(1.0, float((y > 0.5).sum())); return P, R, 2 * P * R / max(1e-9, P + R)
best = -1; t0 = time.time()
for step in range(1, a.steps + 1):
    i = torch.randint(len(Wtr), (a.batch_size,), device=dev); lg = m.logits(*batch(Wtr[i])); y = ytr[i]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, y, pos_weight=torch.tensor(w_pos, device=dev))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        m.eval(); P, R, F = prf(scores(Wva), yva, 0.5); m.train()
        print(json.dumps(dict(step=step, loss=round(float(loss), 4), val_precision=round(P, 3), val_recall=round(R, 3), val_f1=round(F, 3), s=round(time.time() - t0))), flush=True)
        if F > best: best = F; torch.save(dict(config=m.config, model=m.state_dict(), stats={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()}, step=step, val_f1=F, data=a.data, obs=OBS), out / "best.pt")
from shadow_dynamics.cube_terminator import load_terminator
m = load_terminator(out / "best.pt", dev); pr = scores(Wte)
rep = dict(test_windows=int(len(Wte)), test_positives=int(yte.sum()), thresholds={str(t): dict(zip(("precision", "recall", "f1"), prf(pr, yte, t))) for t in (0.3, 0.5, 0.7)})
(out / "metrics.json").write_text(json.dumps(rep, indent=1)); print("TERMINATOR_COMPLETE", json.dumps(rep), flush=True)
