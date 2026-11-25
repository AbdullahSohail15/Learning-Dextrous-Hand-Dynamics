"""One cube-task reward for PPO training in imagination and for MPC planning (experiment 14).

Before experiment 14 the imagined environment (imagination.ImaginedCubeEnv.step) and the planner (cube_mpc.plan_reward)
had separate copies of the same reward. Both now call `cube_reward`, so what PPO is trained on and what the planner
optimises cannot drift apart.

Per predicted step, while the cube is held (no imagined fall yet):
    -10 * distance to the in-hand point (m) + 1 / (rotation error + 0.1)                      (Isaac's shaped terms)
    - dist_weight  * relu(distance_mm - dist_safe_mm) / 50                                   (drop precursor: cube drifting off)
    - speed_weight * relu(cube_speed_mm_s - speed_safe_mm_s) / 200                           (drop precursor: cube moving fast)
averaged over the steps, then per trajectory:
    + success_bonus on reaching the goal (< 0.1 rad while held)
    - drop_penalty  if an imagined fall (distance >= fall_mm) or the drop predictor fires (p > term_threshold)
    - risk_weight * p_drop                                                                   (continuous drop risk)
    - action_penalty * |x|^2                                                                 (raw policy command)
p_drop is the drop predictor's output (trained with an up-weighted positive class, so it ranks risk; it is not a
calibrated probability). Preset "r3" reproduces the experiment-12 round-3 / experiment-13 reward exactly.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import torch

from .cube_world_data import sixd_to_matrix, rotation_distance

SUCCESS_RAD = 0.1


@dataclass(frozen=True)
class CubeCost:
    fall_mm: float = 130.0
    term_threshold: float = 0.3
    success_bonus: float = 50.0
    drop_penalty: float = 50.0
    risk_weight: float = 0.0
    dist_weight: float = 0.0
    dist_safe_mm: float = 80.0
    speed_weight: float = 0.0
    speed_safe_mm_s: float = 250.0
    action_penalty: float = 0.1
    reduce: str = "mean"   # "sum": shaped terms summed over the predicted steps (Isaac's native per-step reward, experiment 21)

    def as_dict(self): return asdict(self)


PRESETS = {
    "r3": CubeCost(),
    # experiment 14: a drop costs 3x a success, continuous risk below the termination threshold, dense precursors
    "safe": CubeCost(term_threshold=0.5, drop_penalty=150.0, risk_weight=50.0, dist_weight=2.0, speed_weight=1.0),
    # experiment 15 ablation: "safe" with the round-3 drop penalty (50): do the continuous risk and precursors carry the effect?
    "safe_p50": CubeCost(term_threshold=0.5, drop_penalty=50.0, risk_weight=50.0, dist_weight=2.0, speed_weight=1.0),
    # experiment 21: Isaac's native in-hand reward as P was trained with it (train_shadow_ppo_isaac.py): shaped terms summed
    # over the steps, +250 per success, 0.0002 |x|^2 per step (x 6 steps = 0.0012 per decision), no drop penalty, termination
    # on a drop (240 mm, or the sampled hazard); the predictor threshold is disabled (2.0 > any p)
    "native": CubeCost(fall_mm=240.0, term_threshold=2.0, success_bonus=250.0, drop_penalty=0.0, action_penalty=0.0012, reduce="sum"),
}


def get_cost(cost):
    if isinstance(cost, CubeCost): return cost
    if isinstance(cost, str): return PRESETS[cost]
    return replace(PRESETS["r3"], **cost)


def cube_reward(Y, Rg, x, p_drop, cost, drop_event=None):
    """Y (..., N, T, 63) predicted states (optional leading sample dims), Rg (N, 3, 3), x (N, 20) raw command,
    p_drop (N,) or None. drop_event (N,) bool, if given, replaces the thresholded predictor (p > term_threshold) as the
    predicted drop (experiment 15: drops sampled from the calibrated hazard).
    Returns reward (..., N), success (..., N), drop (..., N), alive (..., N, T)."""
    c = get_cost(cost)
    R = sixd_to_matrix(Y[..., 51:57]); rot = rotation_distance(R, Rg[:, None]); dist_mm = Y[..., 48:51].norm(dim=-1); speed = Y[..., 57:60].norm(dim=-1)
    fell = dist_mm >= c.fall_mm
    alive = (torch.cumsum(fell.float(), -1) == 0).float()
    step = -10 * dist_mm / 1000 + 1.0 / (rot + 0.1)
    if c.dist_weight: step = step - c.dist_weight * torch.relu(dist_mm - c.dist_safe_mm) / 50
    if c.speed_weight: step = step - c.speed_weight * torch.relu(speed - c.speed_safe_mm_s) / 200
    shaped = (step * alive).sum(-1) if c.reduce == "sum" else (step * alive).mean(-1)
    success = ((rot < SUCCESS_RAD) & (alive > 0)).any(-1)
    drop = fell.any(-1)
    r = shaped + c.success_bonus * success.float() - c.action_penalty * x.square().sum(-1)
    if p_drop is not None:
        drop = drop | ((p_drop > c.term_threshold) if drop_event is None else drop_event)
        if c.risk_weight: r = r - c.risk_weight * p_drop
    r = r - c.drop_penalty * drop.float()
    return r, success, drop, alive
