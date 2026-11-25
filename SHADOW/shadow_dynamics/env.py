"""Isaac Lab 1.2 environment. Import only after AppLauncher starts."""
from __future__ import annotations

import numpy as np
import torch
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab_tasks.direct.inhand_manipulation import InHandManipulationEnv
from omni.isaac.lab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg

FINGERS = ("FF", "MF", "RF", "LF")


class ResearchShadowEnv(InHandManipulationEnv):
    """No automatic resets/goals; collection controls episode boundaries explicitly.

    Actions supplied to advance are physical joint-position targets in runtime
    actuator order. Distal joints are never commanded. The original gym task
    remains registered and unchanged.
    """

    tendon_override = None   # experiment 49: dict applied to the fixed-tendon USD attributes before PhysX parses the stage (see apply_tendon_override)

    def _setup_scene(self):
        super()._setup_scene()
        if self.tendon_override: self.tendon_override_applied = apply_tendon_override(self.tendon_override)
        for finger in (*FINGERS, "TH"):
            key = finger.lower()
            self.scene.sensors[key] = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/robot0_{key}distal",
                update_period=0.0,
                filter_prim_paths_expr=["/World/envs/env_.*/object"],
            ))

    def _get_dones(self):
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return zeros, zeros.clone()

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_observations(self):
        return {"policy": torch.cat((self.hand.data.joint_pos, self.hand.data.joint_vel), -1)}

    def gravity_root(self):
        """Experiment 49: gravity in the hand's root (mount) frame, zero when hand gravity is off. Returns a list (m/s^2)."""
        from omni.isaac.lab.utils.math import quat_rotate_inverse
        g = torch.tensor(self.cfg.sim.gravity, dtype=torch.float32, device=self.device); rq = self.hand.data.root_state_w[0, 3:7]
        return (quat_rotate_inverse(rq[None], g[None])[0] * (0.0 if self.cfg.robot_cfg.spawn.rigid_props.disable_gravity else 1.0)).tolist()

    def configure_research(self, mode="free"):
        self.mode = mode
        self.goal_markers.set_visibility(False)
        self.active = list(self.actuated_dof_indices)
        self.passive = [self.hand.joint_names.index(f"robot0_{f}J0") for f in FINGERS]
        self.middle = [self.hand.joint_names.index(f"robot0_{f}J1") for f in FINGERS]
        assert self.hand.num_joints == 24 and len(self.active) == 20
        assert not set(self.active).intersection(self.passive)
        self.action_names = [self.hand.joint_names[i] for i in self.active]
        self.low = self.hand_dof_lower_limits[0, self.active].clone()
        self.high = self.hand_dof_upper_limits[0, self.active].clone()
        self.sim_time = 0.0
        self.last_target = torch.zeros((self.num_envs, 20), device=self.device)
        self.reset_physical()

    def set_cube_physics(self, mass_scale=1.0, friction=None, env_ids=None):
        """Experiment 23 physics shift: scale the cube's mass and inertia, and set its static and dynamic friction (PhysX combines it with
        the hand's material, default mode average). Applied once after configure_research; persists across reset_physical. Returns the new values.
        Experiment 28: env_ids (CPU long tensor) limits the shift to those environments (the others keep the nominal cube)."""
        view = self.object.root_physx_view; ids = torch.arange(self.num_envs, device="cpu") if env_ids is None else torch.as_tensor(env_ids, device="cpu").long()
        if not hasattr(self, "_cube_nominal"): self._cube_nominal = (view.get_masses().clone(), view.get_inertias().clone(), view.get_material_properties().clone())
        if mass_scale != 1.0:
            m, I = view.get_masses().clone(), view.get_inertias().clone(); m[ids] *= mass_scale; I[ids] *= mass_scale
            view.set_masses(m, ids); view.set_inertias(I, ids)
        if friction is not None:
            mat = view.get_material_properties().clone(); mat[ids, ..., 0] = friction; mat[ids, ..., 1] = friction; view.set_material_properties(mat, ids)
        i0 = int(ids[0]); return dict(mass_kg=float(view.get_masses()[i0].sum()), friction=view.get_material_properties()[i0, :, :2].tolist(),
                                      nominal_mass_kg=float(view.get_masses().sum(-1).min()), n_shifted=len(ids))

    def set_cube_physics_abs(self, mass_scale=1.0, friction=None, env_ids=None):
        """Experiment 29: set the cube's mass / inertia to mass_scale x NOMINAL and its static and dynamic friction to `friction` (None = nominal)
        in env_ids, independent of earlier calls (the nominal values are captured on the first call, before any change)."""
        view = self.object.root_physx_view
        if not hasattr(self, "_cube_nominal"): self._cube_nominal = (view.get_masses().clone(), view.get_inertias().clone(), view.get_material_properties().clone())
        m0, I0, mat0 = self._cube_nominal; ids = torch.arange(self.num_envs, device="cpu") if env_ids is None else torch.as_tensor(env_ids, device="cpu").long()
        m, I, mat = view.get_masses().clone(), view.get_inertias().clone(), view.get_material_properties().clone()
        m[ids] = m0[ids] * mass_scale; I[ids] = I0[ids] * mass_scale; mat[ids] = mat0[ids]
        if friction is not None: mat[ids, ..., 0] = friction; mat[ids, ..., 1] = friction
        view.set_masses(m, ids); view.set_inertias(I, ids); view.set_material_properties(mat, ids)
        return dict(nominal_mass_kg=float(m0[0].sum()), nominal_friction=mat0[0, :, :2].tolist())

    def set_hand_physics(self, spec):
        """Experiment 36: one perturbed hand for every environment. stiffness_scale / damping_scale are relative to the stock asset, armature is
        absolute (kg m^2); the fixed-tendon limit stiffness is applied at spawn (hand_physics_spawn_cfg). Returns the values read back from PhysX."""
        h = self.hand; v = h.root_physx_view; names = h.joint_names; dev = self.device
        K = v.get_dof_stiffnesses().clone().to(dev); D = v.get_dof_dampings().clone().to(dev); A = v.get_dof_armatures().clone().to(dev)
        for j, sc in spec.get("stiffness_scale", {}).items(): K[:, names.index(j)] *= sc
        for j, sc in spec.get("damping_scale", {}).items(): D[:, names.index(j)] *= sc
        for j, a in spec.get("armature", {}).items(): A[:, names.index(j)] = a
        h.write_joint_stiffness_to_sim(K); h.write_joint_damping_to_sim(D); h.write_joint_armature_to_sim(A)
        for act in h.actuators.values():   # keep the actuator's torque estimate (recorded 'effort') consistent with the new gains
            ji = act.joint_indices; act.stiffness[:] = K if isinstance(ji, slice) else K[:, ji]; act.damping[:] = D if isinstance(ji, slice) else D[:, ji]
        g = lambda x: np.asarray(x[0].cpu()).tolist()
        return dict(stiffness=g(v.get_dof_stiffnesses()), damping=g(v.get_dof_dampings()), armature=g(v.get_dof_armatures()),
                    tendon_limit_stiffness=g(v.get_fixed_tendon_limit_stiffnesses()), spec=spec)

    def reset_physical(self, q=None):
        self.reset()
        q = self.hand.data.default_joint_pos.clone() if q is None else q.clone()
        if q.ndim == 1:
            q = q.repeat(self.num_envs, 1)
        self.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.last_target = q[:, self.active].clone()
        self.hand.set_joint_position_target(self.last_target, joint_ids=self.active)
        obj = self.object.data.default_root_state.clone()
        obj[:, :3] += self.scene.env_origins
        if self.mode == "free":
            obj[:, 0] += 0.45
        self.object.write_root_state_to_sim(obj)
        self.scene.reset()
        self.scene.write_data_to_sim()
        self.sim_time = 0.0

    def snapshot(self):
        """CPU arrays for all environments; time is the completed physics boundary."""
        def cpu(x):
            return x.detach().cpu().numpy().copy()
        net = torch.stack([self.scene[f.lower()].data.net_forces_w[:, 0] for f in (*FINGERS, "TH")], 1)
        obj_force = torch.stack([
            self.scene[f.lower()].data.force_matrix_w[:, 0, 0] for f in (*FINGERS, "TH")
        ], 1)
        return {
            "joint_pos": cpu(self.hand.data.joint_pos),
            "joint_vel": cpu(self.hand.data.joint_vel),
            "object_state": cpu(self.object.data.root_state_w),
            "hand_root_state": cpu(self.hand.data.root_state_w),
            "fingertip_state": cpu(self.hand.data.body_state_w[:, self.finger_bodies]),
            "fingertip_net_force": cpu(net),
            "fingertip_object_force": cpu(obj_force),
            "time": np.full(self.num_envs, self.sim_time, dtype=np.float64),
        }

    def advance(self, target, record_physics=False, on_substep=None):
        if target.ndim == 1:
            target = target.repeat(self.num_envs, 1)
        target = torch.maximum(torch.minimum(target, self.high), self.low)
        assert torch.isfinite(target).all()
        self.last_target = target.clone()
        self.hand.set_joint_position_target(target, joint_ids=self.active)
        records = []
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            self.sim_time += self.physics_dt
            if on_substep is not None:   # experiment 34: per-substep contact accumulation
                on_substep()
            if record_physics:
                records.append(self.snapshot())
        return self.snapshot(), records


def apply_tendon_override(spec, fingers=FINGERS):
    """Experiment 49: rewrite the four J1-J0 fixed tendons in every environment before simulation starts. spec: gearing / force_coefficient
    = (J1, J0) pairs, and the tendon's stiffness, damping, limit_stiffness, lower / upper limit, rest_length (PhysX fixed-tendon attributes).
    force_coefficient = gearing makes the tendon reciprocal (tau = C^T f, Li et al.). Returns the attribute values read back from env_0."""
    import omni.usd
    from pxr import Vt
    stage = omni.usd.get_context().get_stage(); done = {}
    keys = dict(stiffness="stiffness", damping="damping", limit_stiffness="limitStiffness", lower_limit="lowerLimit", upper_limit="upperLimit", rest_length="restLength")
    for prim in stage.Traverse():
        pth = str(prim.GetPath())
        if not pth.startswith("/World/envs/env_") or "/Robot/" not in pth: continue
        for f in fingers:
            tn = f"physxTendon:robot0_T_{f}J1c"; k = {f"robot0_{f}J1": 0, f"robot0_{f}J0": 1}.get(prim.GetName())
            if k is None: continue
            for field, attr in (("gearing", "gearing"), ("force_coefficient", "forceCoefficient")):
                if field in spec:
                    a = prim.GetAttribute(f"{tn}:{attr}"); assert a.IsValid(), (pth, attr); a.Set(Vt.FloatArray([float(spec[field][k])]))
            if k == 0:
                for field, attr in keys.items():
                    if field in spec:
                        a = prim.GetAttribute(f"{tn}:{attr}"); assert a.IsValid(), (pth, attr); a.Set(float(spec[field]))
            if pth.startswith("/World/envs/env_0/"):
                done[f"{f}{'J1' if k == 0 else 'J0'}"] = {a.GetName().split(":")[-1]: str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith(tn)}
    assert len(done) == 2 * len(fingers), done
    return done


def hand_physics_spawn_cfg(cfg, spec):
    """Experiment 36: the fixed-tendon limit stiffness must be set at spawn (the Isaac Lab 1.2 runtime tendon setters fail or do not apply)."""
    if spec and "tendon_limit_stiffness_scale" in spec:
        t = cfg.robot_cfg.spawn.fixed_tendons_props; t.limit_stiffness = t.limit_stiffness * spec["tendon_limit_stiffness_scale"]
    return cfg


def make_cfg(args):
    cfg = ShadowHandEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.num_observations = 48
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.sim.dt = 1.0 / args.physics_hz
    control_hz = getattr(args, "control_hz", 60)
    if args.physics_hz % control_hz:
        raise ValueError("Physics rate must be an integer multiple of the control rate")
    cfg.decimation = args.physics_hz // control_hz
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = 100000
    cfg.reset_position_noise = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.reset_dof_vel_noise = 0.0
    cfg.robot_cfg.spawn.activate_contact_sensors = True
    cfg.robot_cfg.spawn.rigid_props.disable_gravity = not getattr(args, "gravity", False)  # baseline asset: off; experiment 49 runs with --gravity (manifest records it)
    if getattr(args, "mode", "free") == "fixture":
        cfg.object_cfg.spawn.rigid_props.kinematic_enabled = True
        cfg.object_cfg.init_state.pos = (0.0, -0.425, 0.54)
    cfg.viewer.eye = (0.55, -0.8, 0.9)
    cfg.viewer.lookat = (0.0, -0.32, 0.6)
    cfg.viewer.resolution = (800, 640)
    return cfg
