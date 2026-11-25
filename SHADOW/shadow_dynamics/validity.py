"""Per-step physical-validity monitor for the Shadow Hand in Isaac Lab (import after AppLauncher starts).

Two independent measures, both on GPU:
  * visual-link interpenetration (shadow_dynamics.hand_geometry) from the simulator's body poses;
  * PhysX self-contact: one RigidContactView per hand link (that link against every other link of the same hand),
    giving per link pair the minimum contact separation and the summed normal force.
Visual penetration is judged separately for finger-finger pairs and finger-palm pairs (the palm's convex hull over-covers
its concave front, so a fingertip pressed into the palm in a fist already reads ~3 mm).
Classes: VALID_NO_CONTACT (0), VALID_CONTACT (1), INVALID_PENETRATION (2), INVALID_NUMERICAL (3).
"""
from __future__ import annotations
import torch

VALID_NO_CONTACT, VALID_CONTACT, INVALID_PENETRATION, INVALID_NUMERICAL = 0, 1, 2, 3
CLASS_NAMES = ["VALID_NO_CONTACT", "VALID_CONTACT", "INVALID_PENETRATION", "INVALID_NUMERICAL"]
FINGER_PREFIX = {"thumb": "robot0_th", "index": "robot0_ff", "middle": "robot0_mf", "ring": "robot0_rf", "little": "robot0_lf"}


class ValidityMonitor:
    def __init__(self, env, geometry, thresholds, max_contacts_per_env=256):
        self.env = env; self.g = geometry; self.th = dict(thresholds); hand = env.hand; N = env.num_envs
        self.body_names = list(hand.body_names); self.geo_idx = torch.tensor([self.body_names.index(k) for k in geometry.links], device=env.device)
        links = list(self.body_names); L = len(links); pat = "/World/envs/env_*/Robot/"
        # one view per link: sensor = that link in every env, filters = every other link (a filter pattern must match one body per sensor)
        self.views = []
        for k, name in enumerate(links):
            others = [j for j in range(L) if j != k]
            v = env.sim.physics_sim_view.create_rigid_contact_view(pat + name, [pat + links[j] for j in others], max_contact_data_count=N * max_contacts_per_env)
            assert v.sensor_count == N and v.filter_count == L - 1, (name, v.sensor_count, v.filter_count)
            assert all(p.split("/")[3] == f"env_{i}" for i, p in enumerate(v.sensor_paths)), "contact-view sensors are not in env order"
            self.views.append((k, torch.tensor(others, device=env.device), v))
        self.L = L; self.N = N
        finger_of = torch.full((L,), -1, dtype=torch.long, device=env.device)
        for k, (f, pre) in enumerate(FINGER_PREFIX.items()):
            for i, b in enumerate(self.body_names):
                if b.startswith(pre): finger_of[i] = k
        self.finger_of = finger_of
        palm = torch.tensor([("robot0_palm" in pr) for pr in geometry.pairs], device=env.device)
        self.palm_pairs = palm
        self.limits = hand.data.soft_joint_pos_limits[0]   # (24, 2) rad

    def contacts(self):
        """Per env: min separation (N, L, L) metres (inf where no contact) and summed normal force (N, L, L), body order."""
        dt = self.env.physics_dt; N, L = self.N, self.L; dev = self.finger_of.device
        minsep = torch.full((N, L, L), float("inf"), device=dev); fsum = torch.zeros(N, L, L, device=dev)
        for k, others, v in self.views:
            force, _, _, sep, count, start = v.get_contact_data(dt)
            count = count.reshape(-1).long(); start = start.reshape(-1).long(); total = int(count.sum())
            if not total: continue
            pair = torch.repeat_interleave(torch.arange(len(count), device=dev), count)
            offs = torch.arange(total, device=dev) - torch.repeat_interleave(torch.cumsum(count, 0) - count, count)
            rows = start[pair] + offs
            ms = torch.full((N * (L - 1),), float("inf"), device=dev).scatter_reduce_(0, pair, sep.reshape(-1)[rows], reduce="amin")
            fs = torch.zeros(N * (L - 1), device=dev).scatter_add_(0, pair, force.reshape(-1)[rows].abs())
            minsep[:, k, others] = ms.reshape(N, L - 1); fsum[:, k, others] = fs.reshape(N, L - 1)
        return torch.minimum(minsep, minsep.transpose(1, 2)), fsum + fsum.transpose(1, 2)

    def measure(self):
        """Dict of per-env tensors for the current physics state."""
        d = self.env.hand.data; q, qd = d.joint_pos, d.joint_vel
        pose = d.body_state_w; pos, quat = pose[..., :3], pose[..., 3:7]
        pen = self.g.penetration_from_world(self.body_names, pos, quat)          # (N, n_pairs)
        hull_max, hull_arg = pen.max(1)
        fing = pen.masked_fill(self.palm_pairs, 0).amax(1); palm = pen.masked_fill(~self.palm_pairs, 0).amax(1)
        minsep, fsum = self.contacts()
        touching = (minsep <= self.th["touch_separation_m"]) | (fsum > self.th["touch_force_n"])
        link_contact = touching.any(2)
        fo = self.finger_of; fc = []
        for k in range(5):
            m = fo == k; other = (fo != k)
            fc.append((touching[:, m][:, :, other]).flatten(1).any(1))
        finger_contact = torch.stack(fc, 1)
        sep_min = minsep.flatten(1).amin(1)
        finite = torch.isfinite(q).all(1) & torch.isfinite(qd).all(1) & torch.isfinite(pos).flatten(1).all(1)
        excess = torch.maximum(self.limits[:, 0] - q, q - self.limits[:, 1]).amax(1)
        numerical = ~finite | (excess > self.th["limit_excess_rad"]) | (qd.abs().amax(1) > self.th["velocity_rad_s"])
        invalid_pen = (fing > self.th["hull_invalid_m"]) | (palm > self.th["hull_palm_invalid_m"]) | (sep_min < -self.th["physx_penetration_m"])
        contact = touching.flatten(1).any(1) | (hull_max > self.th["hull_contact_m"])
        cls = torch.where(numerical, INVALID_NUMERICAL, torch.where(invalid_pen, INVALID_PENETRATION, torch.where(contact, VALID_CONTACT, VALID_NO_CONTACT)))
        near = torch.deg2rad(torch.tensor(0.5, device=q.device))
        limit_flags = (q <= self.limits[:, 0] + near) | (q >= self.limits[:, 1] - near)
        return dict(validity=cls.to(torch.uint8), hull_penetration_mm=hull_max * 1000, hull_finger_penetration_mm=fing * 1000, hull_palm_penetration_mm=palm * 1000,
                    hull_worst_pair=hull_arg.to(torch.int16),
                    physx_min_separation_mm=torch.where(torch.isfinite(sep_min), sep_min * 1000, torch.full_like(sep_min, 1e3)),
                    touching_pairs=(touching.flatten(1).sum(1) // 2).to(torch.int16), link_contact=link_contact, finger_contact=finger_contact,
                    joint_limit_flags=limit_flags, touching_matrix=touching, link_self_force_n=fsum.sum(2))
