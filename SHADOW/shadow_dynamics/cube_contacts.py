"""Whole-hand contact with the cube (experiment 34; import after AppLauncher starts).

One PhysX RigidContactView: sensor = the cube in every env, filters = every hand link (26). Per physics substep
`substep()` accumulates the force on the cube from each link (get_contact_force_matrix); at the end of a control step
`read()` returns the control-step summary and the contact geometry of the last substep (get_contact_data):

  link_cube_force      (N, L, 3)  N, world frame: force on the cube from each link, mean over the control step's substeps
  link_cube_force_max  (N, L)     N, max |force| over the substeps
  link_cube_contact    (N, L)     bool: |force| > force_threshold in any substep, or separation <= touch_mm at the end
  link_cube_min_sep_mm (N, L)     mm, minimum contact separation at the end (+1000 where the pair has no contact point;
                                  negative = penetration)
  link_cube_point      (N, L, 3)  mm, force-weighted mean contact point in the CUBE frame (centre = 0; NaN = no contact)
  link_cube_normal     (N, L, 3)  unit, mean contact normal in the cube frame (NaN = no contact)
"""
from __future__ import annotations
import torch
from omni.isaac.lab.utils.math import quat_rotate_inverse


class CubeContactMonitor:
    def __init__(self, env, force_threshold_n=0.02, touch_mm=0.5, max_contacts_per_env=128):
        self.env = env; N = env.num_envs; self.N = N; self.dev = env.device
        self.links = list(env.hand.body_names); self.L = len(self.links); pat = "/World/envs/env_*/"
        self.view = env.sim.physics_sim_view.create_rigid_contact_view(pat + "object", [pat + "Robot/" + b for b in self.links], max_contact_data_count=N * max_contacts_per_env)
        assert self.view.sensor_count == N and self.view.filter_count == self.L, (self.view.sensor_count, self.view.filter_count)
        assert all(p.split("/")[3] == f"env_{i}" for i, p in enumerate(self.view.sensor_paths)), "contact-view sensors are not in env order"
        self.fth = force_threshold_n; self.touch = touch_mm / 1000; self.dt = env.physics_dt; self.begin()

    def begin(self):
        self._sum = torch.zeros(self.N, self.L, 3, device=self.dev); self._max = torch.zeros(self.N, self.L, device=self.dev); self._n = 0

    def substep(self):
        F = self.view.get_contact_force_matrix(self.dt).view(self.N, self.L, 3)
        self._sum += F; self._max = torch.maximum(self._max, F.norm(dim=-1)); self._n += 1

    def read(self):
        """Summary of the substeps since begin(); call begin() again for the next control step."""
        N, L, dev = self.N, self.L, self.dev
        force, pts, nrm, sep, count, start = self.view.get_contact_data(self.dt)
        count = count.reshape(-1).long(); start = start.reshape(-1).long(); total = int(count.sum())
        minsep = torch.full((N * L,), float("inf"), device=dev); wsum = torch.zeros(N * L, device=dev)
        psum = torch.zeros(N * L, 3, device=dev); nsum = torch.zeros(N * L, 3, device=dev)
        if total:
            pair = torch.repeat_interleave(torch.arange(N * L, device=dev), count)
            offs = torch.arange(total, device=dev) - torch.repeat_interleave(torch.cumsum(count, 0) - count, count); rows = start[pair] + offs
            f = force.reshape(-1)[rows].abs() + 1e-6; s = sep.reshape(-1)[rows]
            minsep.scatter_reduce_(0, pair, s, reduce="amin"); wsum.scatter_add_(0, pair, f)
            psum.index_add_(0, pair, pts.reshape(-1, 3)[rows] * f[:, None]); nsum.index_add_(0, pair, nrm.reshape(-1, 3)[rows] * f[:, None])
        minsep, wsum, psum, nsum = minsep.view(N, L), wsum.view(N, L), psum.view(N, L, 3), nsum.view(N, L, 3)
        has = wsum > 0
        o = self.env.object.data.root_state_w; q = o[:, None, 3:7].expand(N, L, 4).reshape(-1, 4)
        p_w = psum / wsum.clamp_min(1e-9)[..., None]
        p_c = quat_rotate_inverse(q, (p_w - o[:, None, :3]).reshape(-1, 3)).view(N, L, 3) * 1000
        n_c = torch.nn.functional.normalize(quat_rotate_inverse(q, nsum.reshape(-1, 3)).view(N, L, 3), dim=-1)
        nan = torch.full_like(p_c, float("nan"))
        n = max(self._n, 1); mean_f = self._sum / n
        contact = (self._max > self.fth) | (minsep <= self.touch)
        out = dict(link_cube_force=mean_f, link_cube_force_max=self._max.clone(), link_cube_contact=contact,
                   link_cube_min_sep_mm=torch.where(torch.isfinite(minsep), minsep * 1000, torch.full_like(minsep, 1000.0)),
                   link_cube_point=torch.where(has[..., None], p_c, nan), link_cube_normal=torch.where(has[..., None], n_c, nan))
        self.begin(); return out
