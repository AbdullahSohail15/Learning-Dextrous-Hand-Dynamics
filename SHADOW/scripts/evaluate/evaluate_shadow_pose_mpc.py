"""Closed-loop hand-pose reaching in Isaac with CEM-MPC on learned whole-horizon models (experiment 10).

Each task (start pose, target) runs in its own environment; all tasks run in parallel and the planner is batched over
them. Controllers: mpc_<model> for every --models entry, mpc_tracking (CEM with a "reaches its commands" model),
direct (command the target joints, slew-limited), hold. Objectives: joint (24-joint error) and tip (fingertip
position error through FK). Units: deg, deg/s, mm; Isaac in radians internally.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics.resources import restrict_simulator_gpu
PHYSICAL_GPU = restrict_simulator_gpu()
from omni.isaac.lab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", required=True)
p.add_argument("--models", nargs="*", default=["det_mlp=artifacts/flow_v4/det_mlp/best.pt", "res_det=artifacts/flow_v4/res_det/best.pt", "res_flow_tcn=artifacts/flow_v4/res_flow_tcn/best.pt"])
p.add_argument("--ppo", nargs="*", default=[], help="label=policy.pt from train_shadow_ppo_imagined.py (controller ppo_<label>)")
p.add_argument("--controllers", nargs="*", default=None, help="subset of controller names (default: all)")
p.add_argument("--objectives", nargs="+", default=["joint", "tip"])
p.add_argument("--data", default="artifacts/shadow_structured_v3/train_speeds"); p.add_argument("--poses", default="shadow_dynamics/data/shadow_pose_library.json")
p.add_argument("--recorded-targets", type=int, default=29); p.add_argument("--task-seed", type=int, default=610000)
p.add_argument("--seconds", type=float, default=6.0); p.add_argument("--slew-deg", type=float, default=80.0)
p.add_argument("--candidates", type=int, default=128); p.add_argument("--iterations", type=int, default=4); p.add_argument("--elites", type=int, default=8)
p.add_argument("--samples", type=int, default=8); p.add_argument("--risk", type=float, default=0.25); p.add_argument("--replan", type=int, default=3)
p.add_argument("--joint-tol", type=float, default=2.0); p.add_argument("--tip-tol", type=float, default=5.0)
# experiment 27: gesture benchmark. --starts-per-target n: neutral + (n - 1) recorded starts per library target (experiment 10: 2);
# --gestures: subset of library names; --sequences N: N tasks that each go through --sequence-length random gestures, --segment-seconds each
p.add_argument("--starts-per-target", type=int, default=2); p.add_argument("--gestures", nargs="*", default=None)
p.add_argument("--sequences", type=int, default=0); p.add_argument("--sequence-length", type=int, default=4); p.add_argument("--segment-seconds", type=float, default=3.0)
# experiment 30: inverse-model proposers. --proposers label=checkpoint (train_inverse_proposer.py); --proposer-runs model|mode|proposer|objective,
# mode: init (iteration 1 of every replan uses proposer candidates: flow = samples, det = Gaussian around its output), only (one scoring pass over
# proposer samples, no CEM), direct (execute the proposer's command, no planning)
# experiment 33: PPO-proposal planner. --ppo-planner-runs policy|forward (forward = isaac or a --models label); --oracle-candidates K branch environments per task
# (the Isaac forward model: branches are set to the live state, run each candidate for 12 steps and are discarded); --ppo-candidates, --ppo-noise
p.add_argument("--ppo-planner-runs", nargs="*", default=[]); p.add_argument("--oracle-candidates", type=int, default=0); p.add_argument("--ppo-candidates", type=int, default=32); p.add_argument("--ppo-noise", type=float, default=0.3)
p.add_argument("--proposers", nargs="*", default=[]); p.add_argument("--proposer-runs", nargs="*", default=[])
p.add_argument("--recognisable-tol", type=float, default=5.0, help="joint RMSE (deg) for the looser 'recognisable gesture' criterion"); p.add_argument("--gap-tol", type=float, default=8.0, help="pinch-gap error (mm) for pinch gestures")
p.add_argument("--hand-physics", default=None, help="experiment 36: JSON of a perturbed hand (env.set_hand_physics) applied to every environment")
p.add_argument("--gravity", action="store_true", help="experiment 49: hand gravity on (make_cfg)")
p.add_argument("--record-exp49", default=None, help="experiment 49 stage B: record the (single) controller's run at 120 Hz in collect_exp49.py format to this folder")
p.add_argument("--tendon-models", nargs="*", default=[], help="experiment 49: label=model.json of a TendonHand (Li-form) model; controller mpc_<label>")
p.add_argument("--exp50-runs", nargs="*", default=[], help="experiment 50: name|tendon-model label|local (direct, cem, ilqr)|global (none, rrt)|cbf (0, 1)")
p.add_argument("--measure-validity", action="store_true", help="experiment 50: per run, fraction of control steps with self-contact (validity 1) and invalid penetration (>= 2)")
p.add_argument("--rrt-lookahead", type=float, default=20.0); p.add_argument("--stall-replan-s", type=float, default=1.0)
p.add_argument("--cbf-n", type=int, default=4); p.add_argument("--cbf-gamma", type=float, default=0.2); p.add_argument("--cbf-dact", type=float, default=0.02)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.physics_hz = 120; args.control_hz = 20; args.mode = "free"; args.seed = 0
launcher = AppLauncher(args, multi_gpu=False); app = launcher.app
import numpy as np
import torch
from shadow_dynamics.env import ResearchShadowEnv, make_cfg, hand_physics_spawn_cfg
from shadow_dynamics.flow_data import StructuredDataset
from shadow_dynamics.flow_model import load_model, H, HIST
from shadow_dynamics.hand_geometry import TorchFK
from shadow_viz.kinematics import HandModel

HZ, BLOCK, KNOTS = 20, 3, 4   # knots every 3 steps over the 12-step horizon
TIPS = ("th", "ff", "mf", "rf", "lf")


def make_tasks(names, act, pas, mid):
    """Targets: library poses + recorded held-out states; starts: neutral and a recorded state. Returns lists of (24,) deg."""
    rng = np.random.default_rng(args.task_seed); lib = json.loads(Path(args.poses).read_text())
    assert [f"robot0_{n}" for n in lib["action_names"]] == [names[i] for i in act]
    def full(active):   # passive J0 from the tendon coupling
        q = np.zeros(24); q[act] = active; q[pas] = np.clip(1.143 * q[mid], 0, 90); return q
    ds = StructuredDataset([args.data]); test = [e for e in range(len(ds.episodes)) if ds.split[e] == "test"]
    def recorded():
        while True:
            e = test[int(rng.integers(len(test)))]; ep = ds.episodes[e]; ok = np.where(ep["validity"] <= 1)[0]
            if len(ok): t = int(rng.choice(ok)); return ep["q"][t].astype(np.float64), f"{ds.family[e]}:{ds.seed[e]}:{t}"
    gest = {k: v for k, v in lib["poses_deg"].items() if args.gestures is None or k in args.gestures}
    if args.sequences:   # experiment 27: each task visits sequence-length random gestures (no immediate repeats), from a recorded start
        names_ = [k for k in gest if k != "neutral"]; tasks = []
        for _ in range(args.sequences):
            seq = []
            while len(seq) < args.sequence_length:
                k = names_[int(rng.integers(len(names_)))]
                if not seq or seq[-1] != k: seq.append(k)
            sq, sname = recorded(); tasks.append(dict(target=np.stack([full(np.array(gest[k], float)) for k in seq]), target_name="sequence:" + ">".join(seq), start=sq, start_name=sname))
        return tasks
    targets = [(full(np.array(v, float)), f"library:{k}") for k, v in gest.items()] + [recorded() for _ in range(args.recorded_targets)]
    tasks = []
    for tq, tname in targets:
        tasks.append(dict(target=tq, target_name=tname, start=np.zeros(24), start_name="neutral"))
        for _ in range(args.starts_per_target - 1):
            sq, sname = recorded(); tasks.append(dict(target=tq, target_name=tname, start=sq, start_name=sname))
    return tasks


class Planner:
    """Batched CEM over (envs, candidates); predict(s, a, fa) -> (K, N, 12, 48) deg."""
    def __init__(self, predict, objective, target, low, high, act, pas, mid, fk_tips, device):
        self.predict, self.obj, self.dev = predict, objective, device
        self.low, self.high, self.act = low, high, act; E = target.shape[0]; self.E = E; self.C = args.candidates
        self.target = target; self.step_max = args.slew_deg / HZ; self.fk_tips = fk_tips
        self.target_tips = fk_tips(target) if objective == "tip" else None
        self.mean = None; self.gen = torch.Generator(device=device).manual_seed(args.task_seed)
        self.std0 = 30.0   # deg, proposal spread of the command pose

    def actions(self, knots, prev):   # knots (E, C, 4, 20) -> slew-limited (E, C, 12, 20)
        a = knots.repeat_interleave(BLOCK, 2); out = []; p_ = prev[:, None].expand_as(a[:, :, 0])
        for t in range(H):
            p_ = (p_ + (a[:, :, t] - p_).clamp(-self.step_max, self.step_max)).clamp(self.low, self.high); out.append(p_)
        return torch.stack(out, 2)

    def cost(self, pred, acts, prev):   # pred (K, E, C, 12, 48)
        q = pred[..., :24]
        if self.obj == "joint":
            e = (q - self.target[None, :, None, None]).square().mean(-1)                                       # (K, E, C, 12)
        else:
            tips = self.fk_tips(q.reshape(-1, 24)).reshape(q.shape[:-1] + (5, 3))
            e = (tips - self.target_tips[None, :, None, None]).norm(dim=-1).mean(-1).square() * 1e6           # mm^2
        c = e.mean(-1) + 2 * e[..., -1]
        da = torch.diff(torch.cat([prev[:, None, None].expand(-1, acts.shape[1], 1, -1), acts], 2), dim=2)
        c = c + 1e-3 * da.square().sum((-1, -2))[None]
        return c.mean(0) + (args.risk * c.std(0) if c.shape[0] > 1 else 0)                                   # (E, C)

    @torch.no_grad()
    def propose(self, s, a, prev, K):
        """Experiment 30: K proposer commands (E, K, 20) for the current targets (T: fingertips mm; J: joints)."""
        m, var, clip = self.proposer
        if var == "T":   # desired fingertips one 0.6 s step toward the target: per-tip displacement clipped to the training 90th percentile
            cur = self.fk_tips(s[:, -1, :24]) * 1000; dlt = self.fk_tips(self.target) * 1000 - cur; n = dlt.norm(dim=-1, keepdim=True)
            d = (cur + dlt * torch.clamp(clip.view(1, 5, 1) / n.clamp_min(1e-6), max=1.0)).flatten(1)
        else:
            cur = s[:, -1, :24]; d = cur + (self.target - cur).clamp(-clip, clip)
        u_phys = prev if var == "T" else self.target[:, self.act]
        return m.sample(s, a, d, u_phys, K=K, generator=self.gen).permute(1, 0, 2).clamp(self.low, self.high)

    def plan(self, s, a, prev):   # s (E, 4, 48), a (E, 3, 20), prev (E, 20) applied target (deg)
        """Search over one command pose g (20-D) per plan; the hand moves toward g at the slew limit (all knots = g),
        plus small per-knot noise. Warm start from the previous plan's mean."""
        E, C = self.E, self.C; mode = getattr(self, "mode", None)
        if mode == "direct":   # the proposer's command, held through the slew limit, no planning
            g = self.propose(s, a, prev, 1)[:, 0]; return self.actions(g[:, None, None].expand(E, 1, KNOTS, 20), prev)[:, 0]
        mean = prev.clone() if self.mean is None else self.mean
        std = torch.full_like(mean, self.std0) if self.mean is None else (2 * self.std).clamp(1.0, self.std0)   # warm-started spread
        for it in range(1 if mode == "only" else args.iterations):
            g = mean[:, None] + std[:, None] * torch.randn((E, C, 20), device=self.dev, generator=self.gen)
            if it == 0 and mode in ("init", "only"):
                if self.proposer[0].flow: g[:, 2:] = self.propose(s, a, prev, C - 2)
                else: g = self.propose(s, a, prev, 1) + 10.0 * torch.randn((E, C, 20), device=self.dev, generator=self.gen); g[:, 2] = self.propose(s, a, prev, 1)[:, 0]
            g[:, 0] = mean; g[:, 1] = prev
            knots = g[:, :, None].expand(E, C, KNOTS, 20) + 0.15 * self.std0 * torch.randn((E, C, KNOTS, 20), device=self.dev, generator=self.gen) * (torch.arange(C, device=self.dev) >= 2)[None, :, None, None]
            knots = knots.clamp(self.low, self.high)
            acts = self.actions(knots, prev)
            pred = self.predict(s[:, None].expand(E, C, HIST, 48).reshape(E * C, HIST, 48), a[:, None].expand(E, C, 3, 20).reshape(E * C, 3, 20), acts.reshape(E * C, H, 20))
            pred = pred.view(pred.shape[0], E, C, H, 48); cst = self.cost(pred, acts, prev)
            idx = cst.topk(args.elites, dim=1, largest=False).indices; el = torch.gather(g, 1, idx[..., None].expand(-1, -1, 20))
            mean, std = el.mean(1), el.std(1).clamp_min(0.5)
        self.mean = mean; self.std = std; best = torch.gather(acts, 1, idx[:, :1, None, None].expand(-1, 1, H, 20))[:, 0]
        return best   # (E, 12, 20) applied-target sequence


def main():
    torch.set_num_threads(1); assert sorted(os.sched_getaffinity(0)) == __import__('shadow_dynamics.resources', fromlist=['budget_cpus']).budget_cpus()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    names = json.loads(Path("artifacts/viz_hand_model/model.json").read_text())["joint_names"]
    act = [i for i, n in enumerate(names) if not any(n == f"robot0_{f}J0" for f in ("FF", "MF", "RF", "LF"))]; pas = [names.index(f"robot0_{f}J0") for f in ("FF", "MF", "RF", "LF")]; mid = [names.index(f"robot0_{f}J1") for f in ("FF", "MF", "RF", "LF")]
    tasks = make_tasks(names, act, pas, mid); E = len(tasks); KB = args.oracle_candidates; args.num_envs = E * (1 + KB)
    if args.ppo_planner_runs and any(r.split("|")[1] == "isaac" for r in args.ppo_planner_runs): assert KB == args.ppo_candidates, "Isaac planner needs --oracle-candidates == --ppo-candidates"
    cfg = make_cfg(args); cfg.object_cfg.spawn.rigid_props.kinematic_enabled = True
    hp = json.loads(Path(args.hand_physics).read_text()) if args.hand_physics else None
    if hp: hand_physics_spawn_cfg(cfg, hp)
    env = ResearchShadowEnv(cfg); dev = env.device
    try:
        env.configure_research("free")
        if hp: Path(args.output).mkdir(parents=True, exist_ok=True); (Path(args.output) / "hand_physics_applied.json").write_text(json.dumps(env.set_hand_physics(hp), indent=1))
        assert env.hand.joint_names == names and [names.index(n) for n in env.action_names] == act
        low, high = torch.rad2deg(env.low), torch.rad2deg(env.high)
        fk = TorchFK("artifacts/viz_hand_model", dev); hm = HandModel("artifacts/viz_hand_model")
        tb = torch.tensor([fk.body_names.index(f"robot0_{f}distal") for f in TIPS], device=dev); to = torch.tensor(np.stack([hm.tips[f"robot0_{f}distal"] for f in TIPS]), dtype=torch.float32, device=dev)
        def fk_tips(q):   # (N, 24) deg -> (N, 5, 3) m
            T = fk(q)[:, tb]; return torch.einsum("nkij,kj->nki", T[..., :3, :3], to) + T[..., :3, 3]
        TS = torch.tensor(np.stack([np.atleast_2d(t["target"]) for t in tasks]), dtype=torch.float32, device=dev)   # (E, segments, 24)
        NSEG = TS.shape[1]; target = TS[:, 0].clone(); start = torch.tensor(np.stack([t["start"] for t in tasks]), dtype=torch.float32, device=dev)
        PINCH = dict(thumb_index_pinch=["ff"], thumb_middle_pinch=["mf"], thumb_ring_pinch=["rf"], tripod=["ff", "mf"], thumb_opposition=["lf"], ok_sign=["ff"])
        models = {}
        for spec in args.models:
            lab, path = spec.split("="); m, ck = load_model(path, dev); assert list(ck["data_report"]["joint_names"]) == names; models[lab] = m
        def tracking(s, a, fa):   # "reaches its commands": q_active = command, J0 = 1.143 J1, velocities from differences
            q = torch.zeros(fa.shape[0], H, 24, device=dev); q[..., act] = fa; q[..., pas] = (1.143 * q[..., mid]).clamp(0, 90)
            v = torch.diff(torch.cat([s[:, -1:, :24], q], 1), dim=1) * HZ; return torch.cat([q, v], -1)[None]
        predictors = {f"mpc_{lab}": (lambda m: (lambda s, a, fa: torch.cat([m.sample(s[i:i + 8192], a[i:i + 8192], fa[i:i + 8192], K=args.samples if m.flow else 1, steps=4) for i in range(0, len(s), 8192)], 1)))(m) for lab, m in models.items()}
        predictors["mpc_tracking"] = tracking
        from shadow_dynamics.tendon_hand_model import TendonHand
        tmodels = {}
        for spec in args.tendon_models:   # experiment 49: Li-form model, actuator from the commands
            lab, path = spec.split("="); tm = TendonHand(json.loads(Path(path).read_text()), dev).eval(); assert tm.names == names
            def tendon_pred(s, a, fa, tm=tm):
                with torch.no_grad(): return torch.cat([tm.rollout(s[i:i + 4096, -1], fa=fa[i:i + 4096]) for i in range(0, len(s), 4096)], 0)[None]
            predictors[f"mpc_{lab}"] = tendon_pred; tmodels[lab] = tm
        from shadow_dynamics.imagination import pose_obs, chunk_targets
        import types; sys.modules.setdefault("git", types.ModuleType("git"))   # unpickling rsl_rl modules imports GitPython (not installed)
        policies = {}
        for spec in args.ppo:
            lab, path = spec.split("="); pol = torch.load(path, map_location=dev, weights_only=False); pol["actor_critic"].to(dev).eval(); pol["obs_normalizer"].to(dev).eval(); policies[f"ppo_{lab}"] = pol
        from shadow_dynamics.inverse_proposer import load_proposer
        proposers = {}
        for spec in args.proposers:
            lab, path = spec.split("="); pm, pck = load_proposer(path, dev); proposers[lab] = (pm, pck["args"]["variant"], torch.tensor(pck["step_clip"], device=dev))
        prop_runs = {}
        for spec in args.proposer_runs:
            mlab, mode, plab, obj_ = spec.split("|"); prop_runs[f"mpc_{mlab}@{mode}_{plab}"] = (f"mpc_{mlab}", mode, plab, obj_)
        ppo_plan = {f"ppoplan_{r.split('|')[0]}@{r.split('|')[1]}": tuple(r.split("|")) for r in args.ppo_planner_runs}
        exp50 = {}; monv = None
        for spec in args.exp50_runs:   # experiment 50: explicit global planning (RRT-Connect), local control (CEM / iLQR-MPC + TVLQR), CBF safety filter
            name, mlab, loc, glob, cbf = spec.split("|"); assert loc in ("direct", "cem", "ilqr") and glob in ("none", "rrt"); exp50[name] = (mlab, loc, glob, cbf == "1")
        if exp50 or args.measure_validity:
            from shadow_dynamics.hand_cspace import HandCSpace
            from shadow_dynamics.rrt_connect import BatchRRTConnect, PathFollower
            from shadow_dynamics.ilqr import ILQR
            from shadow_dynamics.cbf import CBFFilter
            from shadow_dynamics.validity import ValidityMonitor
            from shadow_dynamics.hand_geometry import HandGeometry
            cspace = HandCSpace(device=dev)
            monv = ValidityMonitor(env, cspace.geo, dict(touch_separation_m=0.0, touch_force_n=1e-3, limit_excess_rad=0.5, velocity_rad_s=200.0, hull_invalid_m=0.003,
                                   hull_palm_invalid_m=0.008, physx_penetration_m=0.002, hull_contact_m=0.0005)) if args.measure_validity else None
        runs = [("hold", None), ("direct", None)] + [(c, None) for c in exp50] + [(c, o) for o in args.objectives for c in predictors] + [(c, None) for c in policies] + [(c, v[3]) for c, v in prop_runs.items()] + [(c, None) for c in ppo_plan]
        live = torch.arange(E, device=dev); branch = torch.arange(E, E * (1 + KB), device=dev); case_of = live.repeat_interleave(KB) if KB else None
        rec49 = None
        if args.record_exp49:   # experiment 49 stage B: closed-loop data in the exploration collector's format
            assert not KB, "recording needs no oracle branches"
            from shadow_dynamics.validity import ValidityMonitor
            from shadow_dynamics.hand_geometry import HandGeometry
            mon49 = ValidityMonitor(env, HandGeometry.from_viz_model("artifacts/viz_hand_model"), dict(touch_separation_m=0.0, touch_force_n=1e-3, limit_excess_rad=0.5,
                                    velocity_rad_s=200.0, hull_invalid_m=0.003, hull_palm_invalid_m=0.008, physx_penetration_m=0.002, hull_contact_m=0.0005))
        def sub49():
            rec49["q"].append(env.hand.data.joint_pos[:E].clone()); rec49["qd"].append(env.hand.data.joint_vel[:E].clone()); rec49["tau"].append(env.hand.data.applied_torque[:E][:, env.active].clone())
        def adv(cmd_deg):   # live commands only; branch environments keep their last target
            if KB: tgt = env.last_target.clone(); tgt[:E] = torch.deg2rad(cmd_deg); env.advance(tgt)
            elif rec49 is not None:
                env.advance(torch.deg2rad(cmd_deg), on_substep=sub49)
                for _ in range(6): rec49["target"].append(env.last_target[:E].clone())
                rec49["val"].append(mon49.measure()["validity"][:E].clone())
            else: env.advance(torch.deg2rad(cmd_deg))
        def kin_update():
            psv = getattr(env.sim, "physics_sim_view", None)
            if psv is not None and hasattr(psv, "update_articulations_kinematic"): psv.update_articulations_kinematic()
        @torch.no_grad()
        def isaac_forward(fa):   # fa (E*KB, 12, 20) deg, case-major -> (E*KB, 12, 48) true futures from the live states
            jp, jv, lt = env.hand.data.joint_pos[:E].clone(), env.hand.data.joint_vel[:E].clone(), env.last_target[:E].clone()
            env.hand.write_joint_state_to_sim(jp[case_of], jv[case_of], env_ids=branch); env.last_target[branch] = lt[case_of]; kin_update(); out_ = []
            for t in range(H):
                tgt = env.last_target.clone(); tgt[:E] = lt; tgt[branch] = torch.deg2rad(fa[:, t]); env.advance(tgt)
                out_.append(torch.cat([torch.rad2deg(env.hand.data.joint_pos[branch]), torch.rad2deg(env.hand.data.joint_vel[branch])], -1))
            env.hand.write_joint_state_to_sim(jp, jv, env_ids=live); env.last_target[:E] = lt; kin_update(); return torch.stack(out_, 1)
        if args.controllers: runs = [r for r in runs if r[0] in args.controllers or r[0] in prop_runs or r[0] in ppo_plan or r[0] in exp50]
        steps = int(args.seconds * HZ) if NSEG == 1 else int(round(args.segment_seconds * HZ)) * NSEG; seg_steps = steps // NSEG; results = {}; traces = {}
        for ctrl, obj in runs:
            key = ctrl if obj is None else f"{ctrl}:{obj}"; t0 = time.time()
            env.reset_physical(torch.deg2rad(torch.cat([start, start[case_of]]) if KB else start)); prev = start[:, act].clone()
            for _ in range(2 * HZ): adv(prev)   # settle at the start pose
            def state(): return torch.cat([torch.rad2deg(env.hand.data.joint_pos[:E]), torch.rad2deg(env.hand.data.joint_vel[:E])], -1)
            S = [state()] * HIST; A = [prev.clone()] * 3; qs = [S[-1][:, :24].clone()]; plan_t = []; queue = None; gppo = torch.Generator(device=dev).manual_seed(args.task_seed + 7)
            planner = Planner(predictors[prop_runs[ctrl][0] if ctrl in prop_runs else ctrl], obj, target, low, high, act, pas, mid, fk_tips, dev) if ctrl.startswith("mpc_") else None
            if ctrl in prop_runs: planner.mode = prop_runs[ctrl][1]; planner.proposer = proposers[prop_runs[ctrl][2]]
            target = TS[:, 0].clone()
            if planner is not None: planner.target = target; planner.target_tips = fk_tips(target) if obj == "tip" else None
            e50 = exp50.get(ctrl); info50 = {}; vrec = []
            if e50:
                mlab, loc, glob, use_cbf = e50; tmod = tmodels[mlab]; slew = args.slew_deg / HZ
                fstep = lambda x_, u_, tm=tmod: tm.rollout(x_, fa=u_[:, None])[:, 0]
                goals20 = target[:, act].clone(); g_np = goals20.cpu().numpy()
                tol = torch.tensor(np.maximum(cspace.penetration(prev), cspace.penetration(goals20)) + 0.5e-3, dtype=torch.float32, device=dev)
                if loc == "cem": planner = Planner(predictors[f"mpc_{mlab}"], "joint", target, low, high, act, pas, mid, fk_tips, dev); planner.target = target; planner.target_tips = None
                if loc == "ilqr": il = ILQR(fstep, low, high, act, slew=slew, H=H)
                flt = CBFFilter(cspace, fstep, low, high, slew=slew, n=args.cbf_n, gamma=args.cbf_gamma, d_act=args.cbf_dact) if use_cbf else None
                if glob == "rrt":
                    tg = time.time(); rr = BatchRRTConnect(cspace, low.cpu().numpy(), high.cpu().numpy(), seed=args.task_seed); paths, pinfo = rr.plan(prev.cpu().numpy(), g_np)
                    follower = PathFollower(paths, g_np, args.rrt_lookahead); last_err = np.full(E, np.inf)
                    info50.update(global_plan_s=time.time() - tg, rrt_needed=int((~pinfo["straight"]).sum()), rrt_unsolved=int((~pinfo["solved"]).sum()),
                                  rrt_length_deg_median=float(np.nanmedian(pinfo["length_deg"])), rrt_replans=0)
                    print("RRT", ctrl, json.dumps({k_: (round(v_, 2) if isinstance(v_, float) else v_) for k_, v_ in info50.items()}), flush=True)
            if args.record_exp49:
                rec49 = dict(q=[env.hand.data.joint_pos[:E].clone()], qd=[env.hand.data.joint_vel[:E].clone()], tau=[], target=[], val=[mon49.measure()["validity"][:E].clone()])
            for k in range(steps):
                if NSEG > 1 and k % seg_steps == 0 and k:   # next gesture of the sequence; the planner restarts its search
                    target = TS[:, k // seg_steps].clone()
                    if planner is not None: planner.target = target; planner.target_tips = fk_tips(target) if obj == "tip" else None; planner.mean = None
                if e50:
                    x_now = S[-1]
                    if glob == "rrt":
                        sr = int(args.stall_replan_s * HZ)
                        err = (x_now[:, act] - goals20).square().mean(1).sqrt().cpu().numpy()
                        if k and k % sr == 0:   # replanning (week 3): tasks whose measured error to the goal did not fall by 0.5 deg in stall_replan_s, far from the goal
                            st = np.where((last_err - err < 0.5) & (err > args.joint_tol) & (follower.progress() < 0.98))[0]
                            if len(st):
                                newp, _ = rr.plan(x_now[st][:, act].cpu().numpy(), g_np[st]); info50["rrt_replans"] += int(len(st))
                                for j_, i_ in enumerate(st): follower.set_path(i_, newp[j_], g_np[i_])
                            last_err = err
                        # the reference advances along the path from the commanded configuration (a moving carrot, like a time-parameterised path):
                        # tying it to the measured pose stalls fingers stuck in contact, whose drive then pulls only `lookahead` deg
                        ref20 = torch.tensor(follower.reference(prev.cpu().numpy()), dtype=torch.float32, device=dev).clamp(low, high); ref24 = cspace.full(ref20)
                    else: ref20, ref24 = goals20, target
                    if loc == "direct": cmd = (prev + (ref20 - prev).clamp(-slew, slew)).clamp(low, high)
                    elif loc == "cem":
                        if k % args.replan == 0:
                            planner.target = ref24; torch.cuda.synchronize(); tp = time.time(); queue = planner.plan(torch.stack(S[-HIST:], 1), torch.stack(A[-3:], 1), prev); torch.cuda.synchronize(); plan_t.append(time.time() - tp)
                        cmd = queue[:, k % args.replan]
                    else:
                        j_ = k % args.replan
                        if j_ == 0:
                            torch.cuda.synchronize(); tp = time.time(); il.plan(x_now, prev, ref24, shift=args.replan if k else 0); up0p = prev.clone(); torch.cuda.synchronize(); plan_t.append(time.time() - tp)
                        cmd = il.feedback(j_, x_now, prev, up0p)
                    if flt is not None: cmd = flt.filter(x_now, prev, cmd, tol)
                    fin = torch.isfinite(cmd).all(1); cmd = torch.where(fin[:, None], cmd, prev); info50["nonfinite_cmd"] = info50.get("nonfinite_cmd", 0) + int((~fin).sum())
                elif ctrl == "hold": cmd = prev
                elif ctrl == "direct": cmd = (prev + (target[:, act] - prev).clamp(-args.slew_deg / HZ, args.slew_deg / HZ)).clamp(low, high)
                elif ctrl in ppo_plan:   # experiment 33: the policy proposes, a forward model (Isaac branches or learned) chooses
                    plab, fwd = ppo_plan[ctrl]; pol = policies[f"ppo_{plab}"]
                    if k % pol["chunk"] == 0:
                        torch.cuda.synchronize(); tp = time.time(); K_ = args.ppo_candidates
                        with torch.no_grad():
                            xm = pol["actor_critic"].act_inference(pol["obs_normalizer"](pose_obs(S[-1][:, :24], S[-1][:, 24:], target, prev))).clamp(-1, 1)
                            X = torch.cat([xm[:, None], (xm[:, None] + args.ppo_noise * torch.randn(E, K_ - 1, 20, device=dev, generator=gppo)).clamp(-1, 1)], 1)
                            pr = prev[:, None].expand(E, K_, 20).reshape(-1, 20); g = (pr + pol["max_delta"] * X.reshape(-1, 20)).clamp(low, high)
                            fa = chunk_targets(pr, g, pol["slew_deg_s"], low, high, steps=H)                                  # (E*K, 12, 20)
                            if fwd == "isaac": Y = isaac_forward(fa)
                            else:
                                m_ = models[fwd]; Y = m_.sample(torch.stack(S[-HIST:], 1).repeat_interleave(K_, 0), torch.stack(A[-3:], 1).repeat_interleave(K_, 0), fa)[0]
                            e_ = (Y[..., :24] - target.repeat_interleave(K_, 0)[:, None]).square().mean(-1); c_ = (e_.mean(-1) + 2 * e_[:, -1]).view(E, K_)
                            j_ = c_.argmin(1); queue = fa.view(E, K_, H, 20)[live, j_][:, :pol["chunk"]]
                        torch.cuda.synchronize(); plan_t.append(time.time() - tp)
                    cmd = queue[:, k % pol["chunk"]]
                elif ctrl.startswith("ppo_"):   # same chunk semantics as the imagined training env
                    pol = policies[ctrl]
                    if k % pol["chunk"] == 0:
                        with torch.no_grad():
                            x = pol["actor_critic"].act_inference(pol["obs_normalizer"](pose_obs(S[-1][:, :24], S[-1][:, 24:], target, prev))).clamp(-1, 1)
                        g = (prev + pol["max_delta"] * x).clamp(low, high); queue = chunk_targets(prev, g, pol["slew_deg_s"], low, high, steps=pol["chunk"])
                    cmd = queue[:, k % pol["chunk"]]
                else:
                    if k % args.replan == 0:
                        torch.cuda.synchronize(); tp = time.time(); queue = planner.plan(torch.stack(S[-HIST:], 1), torch.stack(A[-3:], 1), prev); torch.cuda.synchronize(); plan_t.append(time.time() - tp)
                    cmd = queue[:, k % args.replan]
                adv(cmd); prev = torch.rad2deg(env.last_target[:E]); A.append(prev.clone()); S.append(state()); qs.append(S[-1][:, :24].clone())
                if monv is not None: vrec.append(monv.measure()["validity"][:E].clone())
            if rec49 is not None:
                Vr = torch.stack(rec49["val"], 1).cpu().numpy(); bad = Vr >= 2; n_ok = np.where(bad.any(1), bad.argmax(1), Vr.shape[1])
                cat = lambda k_: torch.rad2deg(torch.stack(rec49[k_], 1)).cpu().numpy().astype(np.float32); ro = Path(args.record_exp49); ro.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(ro / "episodes.npz", q_deg=cat("q"), qd_deg=cat("qd"), target_deg=cat("target"), tau=torch.stack(rec49["tau"], 1).cpu().numpy().astype(np.float32),
                                    validity=Vr.astype(np.uint8), n_valid_states=n_ok, speed=np.full(E, args.slew_deg), family=np.array([t["target_name"] for t in tasks]), seed=np.full(E, args.task_seed))
                (ro / "manifest.json").write_text(json.dumps(dict(schema="exp49-collect-1", source=f"closed-loop {ctrl} task-seed {args.task_seed}", gravity=bool(args.gravity), gravity_root=env.gravity_root(),
                                    physics_hz=120, control_hz=20, programs_run=int(E), valid_minutes=float(n_ok.sum() / 20 / 60),
                                    validity_fraction={n_: float((Vr == i).mean()) for i, n_ in enumerate(["free", "contact", "invalid_pen", "invalid_num"])}), indent=1))
                rec49 = None; print("RECORD_EXP49", ro, flush=True)
            Q = torch.stack(qs, 1)                                                         # (E, T+1, 24)
            TGT = TS.repeat_interleave(seg_steps, 1); TGT = torch.cat([TGT[:, :1], TGT], 1)[:, :Q.shape[1]]   # target in force at each recorded step
            jerr = (Q - TGT).square().mean(-1).sqrt(); jact = (Q[..., act] - TGT[..., act]).square().mean(-1).sqrt(); jpas = (Q[..., pas] - TGT[..., pas]).square().mean(-1).sqrt()
            tipsQ = fk_tips(Q.reshape(-1, 24)).reshape(E, -1, 5, 3); tipsT = fk_tips(TGT.reshape(-1, 24)).reshape(E, -1, 5, 3)
            terr = (tipsQ - tipsT).norm(dim=-1).mean(-1) * 1000
            # experiment 27: per segment (the last 0.5 s of each), joint / tip error and the pinch gap error (thumb to the pinched fingertips, achieved vs target)
            last = int(0.5 * HZ); TI = {f: i for i, f in enumerate(TIPS)}; seg_end = [(j + 1) * seg_steps for j in range(NSEG)]
            seg_joint = torch.stack([jerr[:, e - last + 1:e + 1].mean(1) for e in seg_end], 1); seg_tip = torch.stack([terr[:, e - last + 1:e + 1].mean(1) for e in seg_end], 1)
            seg_names = [t["target_name"].split(":", 1)[1].split(">") for t in tasks]; gap = torch.full((E, NSEG), float("nan"), device=dev)
            for i_, sn in enumerate(seg_names):
                for j, g_ in enumerate(sn):
                    if g_ in PINCH and j < NSEG:
                        e = seg_end[j]; ids = [TI[f] for f in PINCH[g_]]
                        dq = (tipsQ[i_, e - last + 1:e + 1, 0:1] - tipsQ[i_, e - last + 1:e + 1, ids]).norm(dim=-1).mean(); dt = (tipsT[i_, e, 0:1] - tipsT[i_, e, ids]).norm(dim=-1).mean()
                        gap[i_, j] = (dq - dt).abs() * 1000
            recog = (seg_joint < args.recognisable_tol) & (torch.isnan(gap) | (gap < args.gap_tol))
            last = int(0.5 * HZ)
            def first_below(x, tol):
                ok = x < tol; t = ok.float().argmax(1).float() / HZ; return torch.where(ok.any(1), t, torch.full_like(t, float("nan")))
            r = dict(final_joint_rmse=jerr[:, -last:].mean(1), final_active_rmse=jact[:, -last:].mean(1), final_passive_rmse=jpas[:, -last:].mean(1), final_tip_mm=terr[:, -last:].mean(1),
                     time_joint_tol=first_below(jerr[:, :seg_steps + 1], args.joint_tol), time_tip_tol=first_below(terr[:, :seg_steps + 1], args.tip_tol),
                     seg_joint_rmse=seg_joint, seg_tip_mm=seg_tip, seg_gap_err_mm=gap, seg_recognisable=recog.float())
            if vrec:
                Vv = torch.stack(vrec, 1); r["contact_frac"] = (Vv == 1).float().mean(1); r["invalid_frac"] = (Vv >= 2).float().mean(1)
            r = {k: v.cpu().numpy() for k, v in r.items()}
            summ = dict(objective=obj, seconds=time.time() - t0, plan_ms=1000 * float(np.median(plan_t)) if plan_t else 0.0,
                        joint_rmse_mean=float(r["final_joint_rmse"].mean()), joint_rmse_median=float(np.median(r["final_joint_rmse"])), active_rmse_mean=float(r["final_active_rmse"].mean()), passive_rmse_mean=float(r["final_passive_rmse"].mean()),
                        tip_mm_mean=float(r["final_tip_mm"].mean()), tip_mm_median=float(np.median(r["final_tip_mm"])),
                        success_joint=float((r["final_joint_rmse"] < args.joint_tol).mean()), success_tip=float((r["final_tip_mm"] < args.tip_tol).mean()),
                        time_joint_tol_median=float(np.nanmedian(r["time_joint_tol"])) if np.isfinite(r["time_joint_tol"]).any() else None,
                        time_tip_tol_median=float(np.nanmedian(r["time_tip_tol"])) if np.isfinite(r["time_tip_tol"]).any() else None)
            if "contact_frac" in r: summ.update(contact_frac=float(r["contact_frac"].mean()), invalid_frac=float(r["invalid_frac"].mean()))
            if e50:
                summ.update(info50)
                if flt is not None: summ["cbf"] = dict(flt.stats)
                if loc == "ilqr": summ["ilqr_fallback"] = int(getattr(il, "n_fallback", 0))
            lib = np.array([t["target_name"].startswith("library") for t in tasks])
            summ.update(segment_success_joint=float((r["seg_joint_rmse"] < args.joint_tol).mean()), segment_recognisable=float(r["seg_recognisable"].mean()),
                        pinch_gap_err_mm=float(np.nanmean(r["seg_gap_err_mm"])) if np.isfinite(r["seg_gap_err_mm"]).any() else None)
            summ["library"] = dict(joint_rmse_mean=float(r["final_joint_rmse"][lib].mean()), tip_mm_mean=float(r["final_tip_mm"][lib].mean()), success_joint=float((r["final_joint_rmse"][lib] < args.joint_tol).mean()))
            summ["recorded"] = dict(joint_rmse_mean=float(r["final_joint_rmse"][~lib].mean()), tip_mm_mean=float(r["final_tip_mm"][~lib].mean()), success_joint=float((r["final_joint_rmse"][~lib] < args.joint_tol).mean()), success_tip=float((r["final_tip_mm"][~lib] < args.tip_tol).mean()))
            results[key] = dict(summ, per_task={k_: np.where(np.isfinite(v), v, -1.0).tolist() for k_, v in r.items()})   # experiment 27: paired analysis (NaN stored as -1)
            traces[key] = dict(joint_rmse=jerr.cpu().numpy().astype(np.float32), tip_mm=terr.cpu().numpy().astype(np.float32), q=Q.cpu().numpy().astype(np.float32), per_task={k: v.tolist() for k, v in r.items()})
            print(f"RUN {key:28s} joint {summ['joint_rmse_mean']:.2f} deg (succ {summ['success_joint']:.2f})  tip {summ['tip_mm_mean']:.2f} mm (succ {summ['success_tip']:.2f})  plan {summ['plan_ms']:.0f} ms  [{summ['seconds']:.0f}s]"
                  + (f"  contact {summ['contact_frac']:.3f} invalid {summ['invalid_frac']:.3f}" if "contact_frac" in summ else ""), flush=True)
            (out / "results.json").write_text(json.dumps(dict(args=vars(args), tasks=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in t.items()} for t in tasks], runs=results), indent=1))
        np.savez_compressed(out / "traces.npz", targets=TS.cpu().numpy().astype(np.float32), segment_steps=seg_steps, control_hz=HZ, target_names=np.array([t["target_name"] for t in tasks]),
                            start_names=np.array([t["start_name"] for t in tasks]), **{f"{k}/{n}": v for k, d in traces.items() for n, v in d.items() if n != "per_task"})
        print("POSE_MPC_COMPLETE", out, flush=True)
    finally:
        env.close()


try: main()
except Exception:
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
finally: app.close()
