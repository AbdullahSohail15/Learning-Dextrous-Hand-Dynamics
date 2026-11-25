"""Experiment 03: the Inspire gesture library (12 goals in the 6-D servo space), solved against the collision geometry (no Isaac).

Fixed servos are given as fractions of the range; free servos are found by GPU random search (40k samples) + local refinement:
  wrap  thumb (yaw in a range, bend) over the closed fingers: maximise bend subject to clearance >= 2 mm (fist, point, victory, rock, three, four)
  pinch thumb tip to a finger tip (or to the index-middle midpoint for the tripod, middle = index): minimise tip distance subject to clearance >= 1 mm
Output: assets/inspire_hand/gesture_library.json (rad, ACTUATED order) with the clearance and tip distance of each goal.
Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:../SHADOW $ISAAC_PYTHON scripts/evaluate/build_gesture_library.py   (any Python with torch + numpy)
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inspire_dynamics.torch_kin import InspireKin, ACT

dev = "cuda:0"; K = InspireKin(dev); lo, hi = K.lo, K.hi; g = torch.Generator(device=dev).manual_seed(0)
C = 0.95   # "closed" finger fraction
# name: (fixed fractions (None = free), mode, extra)
SPEC = dict(
    open=([0, 0, 0, 0, 0, 0], None, None),
    thumbs_up=([0, 0, C, C, C, C], None, None),
    call=([0, 0, C, C, C, 0], None, None),
    fist=([None, None, C, C, C, C], "wrap", (0.5, 1.0)),
    point=([None, None, 0, C, C, C], "wrap", (0.3, 1.0)),
    victory=([None, None, 0, 0, C, C], "wrap", (0.3, 1.0)),
    rock=([None, None, 0, C, C, 0], "wrap", (0.3, 1.0)),
    three=([None, None, 0, 0, 0, C], "wrap", (0.3, 1.0)),
    four=([None, None, 0, 0, 0, 0], "wrap", (0.6, 1.0)),
    pinch_index=([None, None, None, 0, 0, 0], "pinch", ["index_tip"]),
    cylinder=([1.0, 0.25, 0.5, 0.5, 0.5, 0.5], None, None),
    tripod=([None, None, None, None, 0, 0], "pinch", ["index_tip", "middle_tip"]),
)
TIP = {n: i for i, n in enumerate(["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"])}


def evaluate(F, mode, extra):
    a = lo + F * (hi - lo); clr = -K.depth(a).amax(1)
    if mode == "wrap": return -F[:, 1], clr >= 0.002, clr                              # maximise bend
    tips = K.tips(a); target = tips[:, [TIP[t] for t in extra]].mean(1)
    return (tips[:, 0] - target).norm(dim=1), clr >= 0.001, clr


lib = {}
for name, (fix, mode, extra) in SPEC.items():
    f = torch.tensor([0.0 if v is None else v for v in fix], device=dev); free = torch.tensor([v is None for v in fix], device=dev)
    if mode is None:
        a = lo + f * (hi - lo); lib[name] = dict(a=a.tolist(), clearance_mm=float(-K.depth(a[None]).amax() * 1000)); continue
    n = 40000; F = f.expand(n, 6).clone(); R = torch.rand(n, 6, generator=g, device=dev)
    if mode == "wrap": R[:, 0] = extra[0] + (extra[1] - extra[0]) * R[:, 0]
    F[:, free] = R[:, free]
    tie = name == "tripod"                                                              # tripod: middle follows index
    for it in range(6):   # random search, then shrink around the best feasible point
        if tie: F[:, 3] = F[:, 2]
        obj, ok, clr = evaluate(F, mode, extra); obj = torch.where(ok, obj, torch.full_like(obj, 1e9)); b = int(obj.argmin()); best = F[b].clone()
        sig = 0.1 * 0.5 ** it; F = best.expand(n, 6).clone(); F[:, free] = (best[free] + sig * torch.randn(n, int(free.sum()), generator=g, device=dev)).clamp(0, 1)
        F[0] = best
        if tie: F[:, 3] = F[:, 2]
        if mode == "wrap": F[:, 0] = F[:, 0].clamp(*extra)
    obj, ok, clr = evaluate(best[None], mode, extra); a = lo + best * (hi - lo)
    lib[name] = dict(a=a.tolist(), frac=best.tolist(), clearance_mm=float(clr[0] * 1000), objective=float(obj[0]) * (1000 if mode == "pinch" else 1))
    print(name, np.round(best.cpu().numpy(), 3), f"clearance {lib[name]['clearance_mm']:.2f} mm", f"obj {lib[name]['objective']:.3f}")
out = Path(__file__).resolve().parents[2] / "assets/inspire_hand/gesture_library.json"
json.dump(dict(servo_order=ACT, gestures=lib, note="pinch objective = thumb tip to target tip distance (mm); wrap objective = -bend fraction"), open(out, "w"), indent=1)
print("wrote", out)
