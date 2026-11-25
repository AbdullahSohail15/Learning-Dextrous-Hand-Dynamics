"""Learned drop predictor for the imagined cube environment (experiment 12, round 2b).

A small MLP on the same inputs the world model sees (4 past hand + cube states, 3 past applied targets, 12 future
applied targets; normalized with train statistics) that predicts "the cube is dropped within the next 12 control
steps". It is the continuation / termination head of Dreamer-style world models: the regression models predict a cube
starting to leave the hand but not the fall, so a policy can live in states where they miss drops.
Labels come from episode metadata (episode ended by a drop and the drop step is within 12 steps of the anchor).
"""
from __future__ import annotations
import torch
from torch import nn


class DropClassifier(nn.Module):
    def __init__(self, sd=63, ad=20, width=512, stats=None):
        super().__init__()
        self.config = dict(sd=sd, ad=ad, width=width)
        self.net = nn.Sequential(nn.Linear(4 * sd + 3 * ad + 12 * ad, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 1))
        for k, n in (("state_mean", sd), ("state_std", sd), ("action_mean", ad), ("action_std", ad)):
            self.register_buffer(k, torch.zeros(n) if k.endswith("mean") else torch.ones(n))
        if stats is not None:
            for k in ("state_mean", "state_std", "action_mean", "action_std"): getattr(self, k).copy_(torch.as_tensor(stats[k], dtype=torch.float32))

    def logits(self, s, a, fa):
        s = (s - self.state_mean) / self.state_std; a = (a - self.action_mean) / self.action_std; fa = (fa - self.action_mean) / self.action_std
        return self.net(torch.cat([s.flatten(1), a.flatten(1), fa.flatten(1)], -1))[:, 0]

    @torch.no_grad()
    def forward(self, s, a, fa):
        return torch.sigmoid(self.logits(s, a, fa))


def load_terminator(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = DropClassifier(**ck["config"]).to(device); m.load_state_dict(ck["model"]); m.eval()
    return m


class DropHazard:
    """Calibrated per-chunk drop hazard (experiment 15): maps the drop predictor's score p to the empirical probability
    that the cube is dropped within the next `steps` control steps (one decision chunk). The predictor is trained with an
    up-weighted positive class, so p ranks risk but is not a probability. The map is a monotone step function on
    held-out data (scripts/calibrate_shadow_cube_terminator.py), linearly interpolated in logit space."""

    def __init__(self, path, device="cpu"):
        import json
        from pathlib import Path
        d = json.loads(Path(path).read_text()); self.path, self.steps = str(path), d["steps"]
        self.x = torch.tensor(d["bin_logit"], dtype=torch.float32, device=device); self.y = torch.tensor(d["bin_hazard"], dtype=torch.float32, device=device)

    def from_logits(self, z):
        """Hazard from the predictor's logits (DropClassifier.logits). Use this: the logits span about -90..+45."""
        z = z.clamp(self.x[0], self.x[-1])
        i = torch.searchsorted(self.x, z.contiguous()).clamp(1, len(self.x) - 1); x0, x1, y0, y1 = self.x[i - 1], self.x[i], self.y[i - 1], self.y[i]
        return y0 + (y1 - y0) * (z - x0) / (x1 - x0).clamp_min(1e-9)

    def __call__(self, p):
        """Hazard from probabilities. Before experiment 16 p was clamped to [1e-6, 1 - 1e-6], i.e. logits to +-13.8, which
        mapped every safe state to the hazard of the logit -13.8 bin (bug). Float32 sigmoid still saturates above a logit
        of about 17, so prefer from_logits."""
        return self.from_logits(torch.logit(p.clamp(1e-37, 1 - 6e-8)))
