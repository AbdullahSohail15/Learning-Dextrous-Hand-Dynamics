"""Isaac Lab 1.2 simulation of the Inspire RH56 right hand (6 linear servos, 12 joints). Import only after AppLauncher starts.

Actuation (see experiments/01_*.md):
  - 6 actuated joints (thumb yaw, thumb pitch, index/middle/ring/pinky proximal) carry a stiff PhysX implicit position drive with the
    spec velocity limit and a torque cap = rated tip force x lever. This stands in for the real hand's geared (non-backdrivable)
    linear servo with its speed setting and force threshold.
  - 6 linkage joints (thumb intermediate/distal, finger intermediates) follow their actuated joint through the URDF <mimic> coupling.
    They are coupled by the PhysX mimic-joint constraint the URDF importer creates (rigid linkage, no drive on these joints).
    Gate 0: the importer drops the URDF mimic offset, so the sim URDF uses offset 0 (q_link = multiplier * q_act). Driving the
    linkage joints as well ("drive" coupling) diverges to NaN against the mimic constraint and was removed.
  - Self-collision on, except hand_base_link <-> thumb_proximal: the thumb's base mechanism lies inside the palm shell in the dex-urdf
    collision geometry (10.6 mm overlap at the open pose, 92% of random poses; artifacts/exp01/self_collision_pairs.json).
Control at 20 Hz, physics at 120 Hz (6 substeps), as in SHADOW.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg
from omni.isaac.lab.utils import configclass

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets/inspire_hand"
META = json.load(open(ASSET / "inspire_hand_right_meta.json"))
ACTUATED = META["actuated"]                       # 6 actuated joints, in this order everywhere (actions, model)
COUPLING = META["coupling"]                       # linkage joint -> {source, multiplier, offset}
LINKAGE = list(COUPLING)
SERVO = dict(stiffness=10.0, damping=0.07, armature=1e-4)   # N m/rad, N m s/rad, kg m^2 (assumed reflected rotor inertia)
CONTROL_DT, SUBSTEPS = 1 / 20, 6


def spec_params():
    """The values a user knows before any identification (N0): spec speed, spec force threshold x lever, nominal drive (6,) each."""
    lim = META["spec_limits"]
    return dict(K=torch.full((6,), SERVO["stiffness"]), D=torch.full((6,), SERVO["damping"]), armature=torch.full((6,), SERVO["armature"]),
                effort=torch.tensor([lim[n]["effort"] for n in ACTUATED]), vmax=torch.tensor([lim[n]["velocity"] for n in ACTUATED]))


def true_hand_params(seed):
    """Experiments 02-04: an 'unknown' hand = the spec values perturbed (as a real unit differs from its data sheet).
    Drive K, D x U(0.6, 1.6); armature x U(0.5, 2); speed: fingers x U(0.7, 1.0), thumb bend x U(0.7, 1.1), thumb yaw (not published)
    x U(0.5, 0.9); torque cap x U(0.7, 1.2). Hand 0 is the hand of experiments 03 and 04."""
    g = torch.Generator().manual_seed(1000 + seed); s = spec_params(); u = lambda lo, hi, n=6: lo + (hi - lo) * torch.rand(n, generator=g)
    vs = torch.cat([u(0.5, 0.9, 1), u(0.7, 1.1, 1), u(0.7, 1.0, 4)])
    return dict(K=s["K"] * u(0.6, 1.6), D=s["D"] * u(0.6, 1.6), armature=s["armature"] * u(0.5, 2.0), effort=s["effort"] * u(0.7, 1.2), vmax=s["vmax"] * vs)


def ensure_usd(force=False):
    """Convert the simulation URDF to USD once (assets/inspire_hand/usd/). PhysX mimic joints come from the importer's parse_mimic default."""
    out = ASSET / "usd/inspire_hand_right.usd"
    if out.exists() and not force: return str(out)
    UrdfConverter(UrdfConverterCfg(asset_path=str(ASSET / "inspire_hand_right.urdf"), usd_dir=str(out.parent), usd_file_name=out.name,
                                   fix_base=True, merge_fixed_joints=False, make_instanceable=False, import_inertia_tensor=True,
                                   self_collision=True, default_drive_type="position", force_usd_conversion=True))
    return str(out)


FILTERED_PAIRS = [("hand_base_link", "thumb_proximal")]


def hand_cfg(usd, root_rot=(1.0, 0.0, 0.0, 0.0), root_pos=(0.0, 0.0, 0.5), gravity=True, self_collision=True):
    lim = META["spec_limits"]
    act = {f"a_{n}": ImplicitActuatorCfg(joint_names_expr=[n], stiffness=SERVO["stiffness"], damping=SERVO["damping"], armature=SERVO["armature"],
                                         effort_limit=lim[n]["effort"], velocity_limit=lim[n]["velocity"]) for n in ACTUATED}
    act["linkage"] = ImplicitActuatorCfg(joint_names_expr=LINKAGE, stiffness=0.0, damping=0.0, armature=1e-5)   # mimic-constrained, no drive
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=not gravity, max_depenetration_velocity=1000.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=self_collision, solver_position_iteration_count=8,
                                                                       solver_velocity_iteration_count=0, sleep_threshold=0.005,
                                                                       stabilization_threshold=0.0005, fix_root_link=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=root_pos, rot=root_rot, joint_pos={".*": 0.0}),
        actuators=act, soft_joint_pos_limit_factor=1.0)


def cube_cfg(size=0.05, mass=0.05, pos=(0.0, 0.0, 0.6), friction=1.0):
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/object", spawn=sim_utils.CuboidCfg(
        size=(size,) * 3, rigid_props=sim_utils.RigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.1))), init_state=RigidObjectCfg.InitialStateCfg(pos=pos))


class InspireSim:
    """N parallel hands (optionally each with a cube). advance(target) applies 6 actuated targets (rad) for one 20 Hz control step."""

    def __init__(self, num_envs=1, device="cuda:0", cube=False, root_rot=(1.0, 0.0, 0.0, 0.0), gravity=True, spacing=0.5, self_collision=True,
                 cube_size=0.05, cube_mass=0.05, cube_pos=(0.0, 0.0, 0.6), cube_friction=1.0):
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=CONTROL_DT / SUBSTEPS, device=device, gravity=(0.0, 0.0, -9.81)))
        usd = ensure_usd()

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = hand_cfg(usd, root_rot, gravity=gravity, self_collision=self_collision)
            light = None
        if cube: SceneCfg.object = cube_cfg(cube_size, cube_mass, cube_pos, cube_friction)
        self.scene = InteractiveScene(SceneCfg(num_envs=num_envs, env_spacing=spacing, replicate_physics=True))
        from pxr import UsdPhysics
        import omni.isaac.core.utils.stage as stage_utils
        stage = stage_utils.get_current_stage()
        names = {n for pair in FILTERED_PAIRS for n in pair}; found = {}          # one stage traversal (per-env traversals were quadratic)
        for p in stage.Traverse():
            ps = str(p.GetPath())
            if ps.startswith("/World/envs/env_") and "/Robot/" in ps and p.GetName() in names and p.HasAPI(UsdPhysics.RigidBodyAPI):
                found.setdefault((ps.split("/")[3], p.GetName()), p.GetPath())
        for i in range(num_envs):
            for a, b in FILTERED_PAIRS:
                pa, pb = found[(f"env_{i}", a)], found[(f"env_{i}", b)]
                UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(pa)).CreateFilteredPairsRel().AddTarget(pb)
        self.sim.reset(); self.hand: Articulation = self.scene["robot"]; self.cube: RigidObject | None = self.scene["object"] if cube else None
        names = self.hand.joint_names; self.names = names
        self.ia = [names.index(n) for n in ACTUATED]; self.il = [names.index(n) for n in LINKAGE]
        src = [ACTUATED.index(COUPLING[n]["source"]) for n in LINKAGE]
        self.l_src = torch.tensor(src, device=device); dev = device
        self.l_k = torch.tensor([COUPLING[n]["multiplier"] for n in LINKAGE], device=dev); self.l_b = torch.tensor([COUPLING[n]["offset"] for n in LINKAGE], device=dev)
        self.lo = self.hand.data.soft_joint_pos_limits[0, self.ia, 0].clone(); self.hi = self.hand.data.soft_joint_pos_limits[0, self.ia, 1].clone()
        self.N, self.device, self.t = num_envs, device, 0.0

    def linkage_of(self, qa):
        """URDF linkage law: linkage joint angles from the 6 actuated angles (..., 6) -> (..., 6 linkage)."""
        return qa[..., self.l_src] * self.l_k + self.l_b

    def reset(self, qa=None):
        qa = torch.zeros(self.N, 6, device=self.device) if qa is None else qa.to(self.device)
        q = torch.zeros(self.N, self.hand.num_joints, device=self.device); q[:, self.ia] = qa; q[:, self.il] = self.linkage_of(qa)
        self.hand.write_joint_state_to_sim(q, torch.zeros_like(q)); self._set_targets(qa); self.scene.write_data_to_sim()
        self.sim.step(render=False); self.scene.update(self.sim.get_physics_dt()); self.t = 0.0

    def _set_targets(self, qa):
        qa = torch.max(torch.min(qa, self.hi), self.lo)
        self.hand.set_joint_position_target(qa, joint_ids=self.ia)
        return qa

    def advance(self, target, on_substep=None):
        tgt = self._set_targets(target.to(self.device))
        for _ in range(SUBSTEPS):
            self.scene.write_data_to_sim(); self.sim.step(render=False); self.scene.update(self.sim.get_physics_dt())
            if on_substep: on_substep(self)
        self.t += CONTROL_DT; return tgt

    def set_servo_params(self, K, D, armature, effort, vmax, env_ids=None):
        """Per-env servo parameters (N, 6) each, ACTUATED order: drive stiffness/damping, armature, torque cap, speed limit (rad/s).
        Used to make each env a different 'unknown' hand (true_hand_params). The speed limit goes to PhysX directly (max joint velocity)."""
        ids = torch.arange(self.N, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        h = self.hand
        h.write_joint_stiffness_to_sim(K.to(self.device), joint_ids=self.ia, env_ids=ids)
        h.write_joint_damping_to_sim(D.to(self.device), joint_ids=self.ia, env_ids=ids)
        h.write_joint_armature_to_sim(armature.to(self.device), joint_ids=self.ia, env_ids=ids)
        h.write_joint_effort_limit_to_sim(effort.to(self.device), joint_ids=self.ia, env_ids=ids)
        mv = h.root_physx_view.get_dof_max_velocities().clone()                       # (N, dof) on CPU in Isaac 4.2
        mv[ids.cpu()[:, None], torch.tensor(self.ia)[None]] = vmax.to(mv.device, mv.dtype)
        h.root_physx_view.set_dof_max_velocities(mv, ids.cpu())

    def servo_params(self):
        """Read back what PhysX holds (N, 6) each."""
        v = self.hand.root_physx_view
        return dict(K=v.get_dof_stiffnesses()[:, self.ia], D=v.get_dof_dampings()[:, self.ia], armature=v.get_dof_armatures()[:, self.ia],
                    effort=v.get_dof_max_forces()[:, self.ia], vmax=v.get_dof_max_velocities()[:, self.ia])

    def state(self):
        d = self.hand.data
        return dict(q=d.joint_pos.clone(), qd=d.joint_vel.clone(), qa=d.joint_pos[:, self.ia].clone(), torque=d.applied_torque[:, self.ia].clone())

    def cube_state(self):
        """(N, 13): position relative to the env origin (m), quaternion wxyz, linear and angular velocity (world)."""
        d = self.cube.data.root_state_w.clone(); d[:, :3] -= self.scene.env_origins; return d

    def write_cube_state(self, s13, env_ids=None):
        ids = torch.arange(self.N, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        s = s13.clone().to(self.device); s[:, :3] += self.scene.env_origins[ids]; self.cube.write_root_state_to_sim(s, env_ids=ids)

    def write_hand_state(self, q, qd, target_qa, env_ids):
        """Full joint state (n, 12) + servo targets (n, 6) for a subset of envs (branch rollouts)."""
        ids = torch.as_tensor(env_ids, device=self.device)
        self.hand.write_joint_state_to_sim(q, qd, env_ids=ids)
        tgt = self.hand.data.joint_pos_target.clone(); tgt[ids[:, None], torch.tensor(self.ia, device=self.device)[None]] = torch.max(torch.min(target_qa, self.hi), self.lo)
        self.hand.set_joint_position_target(tgt)

    def body_pos(self, names):
        ids = [self.hand.body_names.index(n) for n in names]
        return self.hand.data.body_pos_w[:, ids] - self.scene.env_origins[:, None]
