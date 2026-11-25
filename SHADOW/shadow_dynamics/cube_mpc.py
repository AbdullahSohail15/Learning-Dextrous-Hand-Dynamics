"""PPO-proposed model-predictive control for cube reorientation (experiment 13).

One planner for every forward model. At each decision a PPO policy proposes K commands x (candidate 0 = policy mean,
the rest = mean + Gaussian noise), each held over the 12-step horizon exactly as in imagination (`chunk_targets`).
A forward model predicts the 12 future hand + cube states; the candidates are scored with the imagined-training reward
on those predictions (cube_reward.py), and the first 6 steps of the best one are executed. Forward models: the learned world models
(with the drop predictor) and the oracle (Isaac branched from the live state, implemented in the evaluation script).
With K = 1 the controller is PPO alone.
"""
from __future__ import annotations
import torch

from .cube_world_data import quat_to_6d
from .imagination import chunk_targets, cube_obs, H
from .cube_reward import cube_reward


def state63(q, qd, c13):
    """Isaac readings (deg, deg/s, cube13 = pos mm rel. in-hand, quat wxyz, linvel mm/s, angvel deg/s) -> world-model state."""
    return torch.cat([q, qd, c13[:, :3], quat_to_6d(c13[:, 3:7]), c13[:, 7:10], c13[:, 10:13]], -1)


def plan_reward(Y, Rg, x, p_drop=None, cost="r3", lcb_beta=0.0):
    """Task reward of predicted futures (shared with imagined PPO training, see cube_reward.py).

    Y (S, N, T, 63) predicted states (S samples), Rg (N, 3, 3) goal, x (N, 20) raw command, p_drop (N,) or None.
    Returns (N,) mean over samples (minus lcb_beta x their standard deviation, experiment 22), and the per-sample success / drop flags.
    """
    r, success, drop, _ = cube_reward(Y, Rg, x, p_drop, cost)
    if lcb_beta and r.shape[0] > 1: return r.mean(0) - lcb_beta * r.std(0), success, drop
    return r.mean(0), success, drop


class LearnedForward:
    """World model (+ optional drop predictor) as a planner forward model. lcb_beta > 0 (flow models, experiment 22) scores
    candidates by the lower confidence bound mean - beta * std of the reward over the sampled futures."""

    def __init__(self, model, terminator=None, samples=4, lcb_beta=0.0, steps=4):
        self.model, self.terminator, self.lcb_beta, self.steps = model, terminator, lcb_beta, steps
        self.samples = samples if model.flow else 1

    @torch.no_grad()
    def __call__(self, S, A, fa):
        Y = self.model.sample(S, A, fa, K=self.samples, steps=self.steps)   # (S, N, 12, 63); ensembles return one per member
        p = self.terminator(S, A, fa) if self.terminator is not None else None
        return Y, p


def finger_groups(action_names):
    """(G, 20) 0/1 masks of the actuator groups: wrist, thumb, first, middle, ring, little finger (by name prefix)."""
    keys = ("WR", "TH", "FF", "MF", "RF", "LF"); names = [n.split("_", 1)[1] for n in action_names]
    G = torch.tensor([[float(n.startswith(k)) for n in names] for k in keys])
    assert (G.sum(0) == 1).all(), action_names
    return G


class PolicyProposer:
    """Candidates around the policy mean (experiment 18 adds structured noise):
    gauss   N(0, std) on all 20 commands (experiments 13-17);
    finger  each candidate perturbs one actuator group (wrist, thumb or one finger) with N(0, group_std), the rest at the mean;
    pair    each candidate perturbs the thumb plus one other group (an opposition pair) with N(0, group_std)."""

    def __init__(self, policy, std=0.3, mode="gauss", groups=None, group_std=0.6):
        self.ac, self.norm, self.std = policy["actor_critic"], policy["obs_normalizer"], std
        self.mode, self.groups, self.group_std = mode, groups, group_std
        assert mode == "gauss" or groups is not None

    @torch.no_grad()
    def mean(self, q, qd, c13, goal_quat, prev):
        return self.ac.act_inference(self.norm(cube_obs(q, qd, c13, goal_quat, prev)))

    @torch.no_grad()
    def propose(self, q, qd, c13, goal_quat, prev, K, generator):
        m = self.mean(q, qd, c13, goal_quat, prev)                                        # (N, 20)
        if K == 1: return m[:, None]
        noise = torch.randn(m.shape[0], K - 1, m.shape[1], generator=generator, device=m.device)
        if self.mode == "gauss": noise = noise * self.std
        else:
            G = self.groups.to(m.device); ng = G.shape[0]
            if self.mode == "finger": mask = G[torch.randint(ng, (m.shape[0], K - 1), generator=generator, device=m.device)]
            else: mask = (G[1] + G[torch.tensor([0, 2, 3, 4, 5], device=m.device)[torch.randint(5, (m.shape[0], K - 1), generator=generator, device=m.device)]]).clamp(max=1)
            noise = noise * mask * self.group_std
        return torch.cat([m[:, None], m[:, None] + noise], 1).clamp(-1, 1)                 # (N, K, 20)


def candidate_targets(prev, X, max_delta, slew, low, high):
    """prev (N, 20), X (N, K, 20) -> (N*K, 12, 20) applied targets, candidate-major within each case."""
    N, K, _ = X.shape; p = prev[:, None].expand(N, K, 20).reshape(N * K, 20)
    g = (p + max_delta * X.reshape(N * K, 20)).clamp(low, high)
    return chunk_targets(p, g, slew, low, high, steps=H)


@torch.no_grad()
def score_candidates(forward, S, A, fa, X, Rg, cost="r3", with_drop=False):
    """Score K candidates per case with a forward model. S (N,4,63), A (N,3,20), fa (N*K,12,20), X (N,K,20), Rg (N,3,3).
    Returns (N, K) scores; with_drop also returns the predicted drop frequency over samples (N, K)."""
    N, K, _ = X.shape
    Sk, Ak = S.repeat_interleave(K, 0), A.repeat_interleave(K, 0)
    Y, p = forward(Sk, Ak, fa)
    r, _, drop = plan_reward(Y, Rg.repeat_interleave(K, 0), X.reshape(N * K, 20), p, cost, getattr(forward, "lcb_beta", 0.0))
    if with_drop: return r.view(N, K), drop.float().mean(0).view(N, K)
    return r.view(N, K)
