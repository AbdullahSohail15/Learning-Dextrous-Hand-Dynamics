"""Imagined environments: PPO trains inside a learned whole-horizon dynamics model (no simulator in the loop).

Decision chunks: the policy acts every CHUNK = 6 control steps (0.3 s at 20 Hz). Its action x in [-1, 1]^20 sets a
command pose g = prev + MAX_DELTA * x (deg, clipped to the joint limits); the 12 applied targets of the model's horizon
move from the previous target toward g at the slew limit (the exact same rule is used when deploying in Isaac, see
`chunk_targets`). The model predicts 12 states; the first CHUNK become the transition. Histories (4 states, 3 applied
targets) are carried exactly as in the training windows. Episodes start from real recorded states (never from model
outputs) and are short, which limits model exploitation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .cube_reward import cube_reward, get_cost

HZ, CHUNK, H, HIST = 20, 6, 12, 4
FINGERS = ("FF", "MF", "RF", "LF")


def chunk_targets(prev, g, slew_deg_s, low, high, steps=H):
    """prev (N, 20), g (N, 20) deg -> (N, steps, 20) applied targets moving toward g at the slew limit."""
    out = []; p = prev; m = slew_deg_s / HZ
    for _ in range(steps):
        p = (p + (g - p).clamp(-m, m)).clamp(low, high); out.append(p)
    return torch.stack(out, 1)


def pose_obs(q, qd, goal, prev):
    """Observation shared by imagination and Isaac deployment (all deg / deg/s)."""
    return torch.cat([q / 90, qd / 200, goal / 90, (goal - q) / 30, prev / 90], -1)


class ImaginedPoseEnv:
    """rsl_rl VecEnv-compatible pose-reaching task inside a learned hand model."""

    def __init__(self, model, starts, goals, low, high, act_idx, num_envs=4096, episode_chunks=10, slew_deg_s=80.0, max_delta=25.0,
                 tol_deg=2.0, flow_steps=4, device="cuda:0", seed=0):
        self.model, self.device = model, torch.device(device)
        self.S0, self.A0 = starts                     # (M, 4, 48), (M, 3, 20) real recorded windows (deg)
        self.goals = goals                            # (G, 24) deg
        self.low, self.high, self.act = low, high, act_idx
        self.num_envs, self.num_actions, self.max_episode_length = num_envs, 20, episode_chunks
        self.slew, self.max_delta, self.tol, self.flow_steps = slew_deg_s, max_delta, tol_deg, flow_steps
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.cfg = dict(task="imagined_pose", episode_chunks=episode_chunks, chunk=CHUNK, slew_deg_s=slew_deg_s, max_delta=max_delta)
        self.S = torch.zeros(num_envs, HIST, 48, device=self.device); self.A = torch.zeros(num_envs, 3, 20, device=self.device)
        self.goal = torch.zeros(num_envs, 24, device=self.device)
        self._reset(torch.arange(num_envs, device=self.device))
        self.num_obs = self.obs().shape[1]; self.num_privileged_obs = None

    def _reset(self, ids):
        n = len(ids)
        if n == 0: return
        i = torch.randint(len(self.S0), (n,), device=self.device, generator=self.gen); j = torch.randint(len(self.goals), (n,), device=self.device, generator=self.gen)
        self.S[ids] = self.S0[i]; self.A[ids] = self.A0[i]; self.goal[ids] = self.goals[j]; self.episode_length_buf[ids] = 0

    def obs(self):
        x = self.S[:, -1]; return pose_obs(x[:, :24], x[:, 24:], self.goal, self.A[:, -1])

    def get_observations(self):
        return self.obs(), {"observations": {}}

    def reset(self):
        self._reset(torch.arange(self.num_envs, device=self.device)); return self.get_observations()

    @torch.no_grad()
    def step(self, actions):
        x = actions.clamp(-1, 1); prev = self.A[:, -1]
        g = (prev + self.max_delta * x).clamp(self.low, self.high)
        fa = chunk_targets(prev, g, self.slew, self.low, self.high)
        Y = self.model.sample(self.S, self.A, fa, K=1, steps=self.flow_steps)[0]   # (N, 12, 48); flow kinds draw one sample
        Y = Y[:, :CHUNK]
        err = (Y[..., :24] - self.goal[:, None]).square().mean(-1).sqrt()            # (N, CHUNK) joint RMSE, deg
        reward = -err.mean(1) / 10 + (err[:, -1] < self.tol).float() - 0.01 * x.square().mean(1)
        self.S = torch.cat([self.S, Y], 1)[:, -HIST:]; self.A = torch.cat([self.A, fa[:, :CHUNK]], 1)[:, -3:]
        self.episode_length_buf += 1
        timeout = self.episode_length_buf >= self.max_episode_length
        bad = ~torch.isfinite(Y).all(-1).all(-1) | (Y[..., :24].abs() > 200).any(-1).any(-1)   # model blew up: end, no bootstrap
        reward = torch.where(bad, torch.full_like(reward, -10.0), reward)
        done = timeout | bad
        info = {"observations": {}, "time_outs": timeout & ~bad,
                "log": {"final_joint_rmse": err[:, -1].mean(), "success": (err[:, -1] < self.tol).float().mean()}}
        self._reset(done.nonzero(as_tuple=False).squeeze(-1))
        return self.obs(), reward, done.long(), info


def load_starts_and_goals(data_dirs, split="train", max_windows=400000, max_goals=100000, library=None, act_idx=None, mid=None, pas=None, seed=0, device="cuda:0"):
    """Real recorded history windows (start states) and goal poses from a schema-3 dataset split."""
    from .flow_data import StructuredDataset
    ds = StructuredDataset(data_dirs); W = ds.windows(split); rng = np.random.default_rng(seed)
    W = W[rng.choice(len(W), min(max_windows, len(W)), replace=False)]; b = ds.gather(W)
    S = torch.as_tensor(b["past_states"], dtype=torch.float32, device=device); A = torch.as_tensor(b["past_actions"], dtype=torch.float32, device=device)
    G = torch.as_tensor(b["future_states"][rng.choice(len(W), min(max_goals, len(W)), replace=False), -1, :24], dtype=torch.float32, device=device)
    if library is not None:
        lib = json.loads(Path(library).read_text())["poses_deg"]; L = torch.zeros(len(lib), 24, device=device)
        L[:, act_idx] = torch.tensor(list(lib.values()), dtype=torch.float32, device=device); L[:, pas] = (1.143 * L[:, mid]).clamp(0, 90)
        G = torch.cat([G, L.repeat(max(1, len(G) // (4 * len(L))), 1)])   # ~20% library poses
    return (S, A), G


# ---------------------------------------------------------------- cube reorientation (experiment 12)
FALL_MM, SUCCESS_RAD = 240.0, 0.1


def _cube_features(q, qd, pos, R, lin, ang, Rg, prev):
    from .cube_world_data import rotation_distance
    rel = R @ Rg.transpose(-1, -2); d = rotation_distance(R, Rg)[..., None]
    return torch.cat([q / 90, qd / 200, pos / 50, R.flatten(-2), lin / 200, ang / 360, Rg.flatten(-2), rel.flatten(-2), d / np.pi, prev / 90], -1)


def cube_obs(q, qd, cube13, goal_quat, prev):
    """Isaac-side observation: cube13 = pos mm (rel. in-hand point), quat wxyz, linvel mm/s, angvel deg/s."""
    from .cube_world_data import quat_to_matrix
    return _cube_features(q, qd, cube13[:, :3], quat_to_matrix(cube13[:, 3:7]), cube13[:, 7:10], cube13[:, 10:13], quat_to_matrix(goal_quat), prev)


def random_goal_quat(n, gen, device):
    """Isaac's goal distribution: qx(pi u0) * qy(pi u1), u ~ U(-1, 1)."""
    u = (torch.rand(n, 2, generator=gen, device=device) * 2 - 1) * np.pi / 2   # half angles
    qx = torch.stack([torch.cos(u[:, 0]), torch.sin(u[:, 0]), torch.zeros_like(u[:, 0]), torch.zeros_like(u[:, 0])], -1)
    qy = torch.stack([torch.cos(u[:, 1]), torch.zeros_like(u[:, 1]), torch.sin(u[:, 1]), torch.zeros_like(u[:, 1])], -1)
    w1, x1, y1, z1 = qx.unbind(-1); w2, x2, y2, z2 = qy.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def clamp_contact(Y):
    """Experiment 34: keep predicted contact channels (63: log-forces, 89: flags, 115: log self-forces) in their recorded range."""
    if Y.shape[-1] <= 63: return Y
    return torch.cat([Y[..., :63], Y[..., 63:89].clamp_min(0), Y[..., 89:115].clamp(0, 1), Y[..., 115:].clamp_min(0)], -1)


class ImaginedCubeEnv:
    """rsl_rl VecEnv-compatible cube reorientation inside a learned hand + cube world model (63-dim state, or 141 with the
    experiment-34 contact channels; the policy observation uses the first 63).

    Reward per chunk = cube_reward.cube_reward on the chunk's 6 predicted steps (Isaac's shaped terms, success bonus with
    a new goal after it as in Isaac, drop penalty with the episode ending without bootstrap; experiment 14 adds the
    continuous drop risk and drop precursors), divided by 10. The cost is a CubeCost / preset name; without one it is
    built from the legacy arguments (drop penalty 50). Episodes are short branches (default 10 chunks = 3 s) from real
    recorded hand + cube windows; time-outs bootstrap, so the value function carries the long-horizon return.
    Experiment 15: with drop_hazard, a predicted drop is sampled each chunk with the calibrated probability (the
    deterministic threshold made imagined drops almost never happen); info["terminal_obs"] is the pre-reset observation.
    """

    def __init__(self, model, starts, low, high, num_envs=4096, episode_chunks=10, slew_deg_s=80.0, max_delta=25.0, flow_steps=4, device="cuda:0", seed=0, fall_mm=FALL_MM,
                 success_bonus=250.0, action_penalty=0.01, terminator=None, term_threshold=0.5, cost=None, drop_hazard=None, settled_starts=None, settled_frac=0.0, guard=None):
        self.cost = get_cost(cost if cost is not None else dict(fall_mm=fall_mm, term_threshold=term_threshold, success_bonus=success_bonus, action_penalty=action_penalty))
        fall_mm, success_bonus, action_penalty, term_threshold = self.cost.fall_mm, self.cost.success_bonus, self.cost.action_penalty, self.cost.term_threshold
        from .cube_world_data import sixd_to_matrix, quat_to_matrix, rotation_distance
        self.R6, self.Rq, self.rdist = sixd_to_matrix, quat_to_matrix, rotation_distance
        self.model, self.device = model, torch.device(device); self.S0, self.A0 = starts
        self.low, self.high = low, high; self.num_envs, self.num_actions, self.max_episode_length = num_envs, 20, episode_chunks
        self.slew, self.max_delta, self.flow_steps = slew_deg_s, max_delta, flow_steps
        self.success_bonus, self.action_penalty, self.terminator, self.term_threshold = success_bonus, action_penalty, terminator, term_threshold
        self.settled, self.settled_frac = settled_starts, settled_frac   # experiment 21: part of the resets from the settled start of recorded episodes (Isaac-like task starts)
        self.guard = guard   # experiment 21: dict(u90, lam, trunc) with an EnsembleWorldModel; penalty lam * relu(u - u90) per step, time-out when u > trunc * u90
        self.drop_hazard = drop_hazard   # experiment 15: cube_terminator.DropHazard; predicted drops are sampled from it instead of p > threshold
        self.fall_mm = fall_mm   # imagined drop radius: the models predict a cube starting to leave but not the fall itself (exp 12 round 0)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.cfg = dict(task="imagined_cube", fall_mm=fall_mm, episode_chunks=episode_chunks, chunk=CHUNK, slew_deg_s=slew_deg_s, max_delta=max_delta)
        self.S = torch.zeros(num_envs, HIST, self.S0.shape[-1], device=self.device); self.A = torch.zeros(num_envs, 3, 20, device=self.device); self.Rg = torch.zeros(num_envs, 3, 3, device=self.device)
        self._reset(torch.arange(num_envs, device=self.device)); self.num_obs = self.obs().shape[1]; self.num_privileged_obs = None

    def _reset(self, ids):
        n = len(ids)
        if n == 0: return
        i = torch.randint(len(self.S0), (n,), device=self.device, generator=self.gen)
        self.S[ids] = self.S0[i]; self.A[ids] = self.A0[i]
        if self.settled is not None and self.settled_frac > 0:
            use = torch.rand(n, device=self.device, generator=self.gen) < self.settled_frac
            if use.any():
                j = torch.randint(len(self.settled[0]), (int(use.sum()),), device=self.device, generator=self.gen); sel = ids[use]
                self.S[sel] = self.settled[0][j]; self.A[sel] = self.settled[1][j]
        self.Rg[ids] = self.Rq(random_goal_quat(n, self.gen, self.device)); self.episode_length_buf[ids] = 0

    def obs(self):
        x = self.S[:, -1]
        return _cube_features(x[:, :24], x[:, 24:48], x[:, 48:51], self.R6(x[:, 51:57]), x[:, 57:60], x[:, 60:63], self.Rg, self.A[:, -1])

    def get_observations(self):
        return self.obs(), {"observations": {}}

    def reset(self):
        self._reset(torch.arange(self.num_envs, device=self.device)); return self.get_observations()

    @torch.no_grad()
    def step(self, actions):
        x = actions.clamp(-1, 1); prev = self.A[:, -1]
        g = (prev + self.max_delta * x).clamp(self.low, self.high); fa = chunk_targets(prev, g, self.slew, self.low, self.high)
        with torch.no_grad(): p_logit = self.terminator.logits(self.S, self.A, fa) if self.terminator is not None else None   # learned "drop within 12 steps" (round 2b)
        p_drop = torch.sigmoid(p_logit) if p_logit is not None else None
        Y = self.model.sample(self.S, self.A, fa, K=1, steps=self.flow_steps)[0][:, :CHUNK]   # (N, 6, 63)
        R = self.R6(Y[..., 51:57]); Y = clamp_contact(torch.cat([Y[..., :51], R[..., :, 0], R[..., :, 1], Y[..., 57:]], -1))   # keep the rotation orthonormal
        rot = self.rdist(R, self.Rg[:, None])                                                                  # (N, 6)
        # shared with the MPC planner; action penalty on the RAW policy output: on the clamped action it is flat beyond +-1 and cannot pull a saturated mean back (round 2a)
        # a predicted drop (p > threshold) ends the episode at the chunk end; experiment 14 adds the continuous risk and precursors
        hazard = self.drop_hazard.from_logits(p_logit) if self.drop_hazard is not None and p_logit is not None else None
        event = (torch.rand(hazard.shape, generator=self.gen, device=self.device) < hazard) if hazard is not None else None
        r, success, fell_any, _ = cube_reward(Y, self.Rg, actions, p_drop, self.cost, drop_event=event)
        trunc = torch.zeros_like(success)
        if self.guard is not None:
            u = self.model.last_u; over = torch.relu(u - self.guard["u90"])
            r = r - self.guard["lam"] * CHUNK * over                       # per-step penalty, summed over the chunk's steps
            trunc = u > self.guard["trunc"] * self.guard["u90"]
        reward = r / 10
        self.S = torch.cat([self.S, Y], 1)[:, -HIST:]; self.A = torch.cat([self.A, fa[:, :CHUNK]], 1)[:, -3:]
        new_goal = success & ~fell_any
        if new_goal.any(): self.Rg[new_goal] = self.Rq(random_goal_quat(int(new_goal.sum()), self.gen, self.device))
        self.episode_length_buf += 1
        bad = ~torch.isfinite(Y).all(-1).all(-1) | (Y[..., :24].abs() > 200).any(-1).any(-1)
        timeout = ((self.episode_length_buf >= self.max_episode_length) | trunc) & ~fell_any & ~bad
        reward = torch.where(bad, torch.full_like(reward, -10.0), reward)
        done = timeout | fell_any | bad
        terminal_obs = self.obs()   # before the reset: lets the trainer bootstrap time-outs with V(s_{t+1}) (experiment 15)
        info = {"observations": {}, "time_outs": timeout, "success": success & ~bad, "drop": fell_any & ~bad, "terminal_obs": terminal_obs, "p_drop": p_drop, "p_drop_logit": p_logit, "hazard": hazard,
                "log": {"rot_err": rot[:, -1].mean(), "success": success.float().mean(), "drop": fell_any.float().mean(), "trunc": trunc.float().mean(),
                        "u": self.model.last_u.mean() if self.guard is not None else torch.zeros((), device=self.device)}}
        self._reset(done.nonzero(as_tuple=False).squeeze(-1))
        return self.obs(), reward, done.long(), info


class EnsembleWorldModel:
    """Experiment 21: several world models of one kind (different seeds) as one imagined model. sample() predicts with every
    member and returns, per environment, the prediction of a randomly drawn member (MBPO-style); last_u (N,) is the mean
    pairwise rotation distance (rad) between the members' predicted cube orientations at the end of the decision chunk
    (step CHUNK), the disagreement used by ImaginedCubeEnv's guard. predict_mean() is the members' mean (planner use)."""

    def __init__(self, models, generator):
        self.models, self.gen, self.last_u = models, generator, None
        self.flow, self.sd = models[0].flow, models[0].sd

    @torch.no_grad()
    def all(self, s, a, fa, steps=4, generator=None):
        return torch.stack([m.sample(s, a, fa, K=1, steps=steps, generator=generator)[0] for m in self.models])   # (M, N, 12, 63)

    @staticmethod
    def disagreement(Ys, step=CHUNK - 1):
        from .cube_world_data import sixd_to_matrix, rotation_distance
        R = sixd_to_matrix(Ys[:, :, step, 51:57]); M = len(Ys)
        return torch.stack([rotation_distance(R[i], R[j]) for i in range(M) for j in range(i + 1, M)]).mean(0)

    @torch.no_grad()
    def sample(self, s, a, fa, K=1, steps=4, generator=None):
        assert K == 1
        Ys = self.all(s, a, fa, steps, generator); self.last_u = self.disagreement(Ys)
        i = torch.randint(len(self.models), (s.shape[0],), device=s.device, generator=self.gen)
        return Ys[i, torch.arange(s.shape[0], device=s.device)][None]
