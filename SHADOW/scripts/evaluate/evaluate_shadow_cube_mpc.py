"""Closed-loop cube reorientation in Isaac: PPO alone and PPO-proposed MPC with learned or oracle forward models (experiment 13).

Cases are built exactly as in evaluate_shadow_cube_ppo.py (same seed -> same 120 cases): random start orientation settled
on the palm for 1 s (redrawn if it rolls off), goal stratified by initial error, 10 s, continuing goals, success
< 0.1 rad without a drop (> 240 mm). Arms:
  hold | random | ppo:<policy> | mpc:<policy>:<model>[:<cost>[:<proposal>]]   with model a --models label or oracle, cost a
  cube_reward preset (default r3, the experiment-13 cost), proposal gauss (default) | finger | pair (cube_mpc.PolicyProposer).
  Experiment 18 model variants: oracle+<label> = true Isaac futures scored with that label's drop predictor;
  <label>-noterm = the learned model without a drop predictor (only predicted falls past fall_mm count).
  Experiment 22: <label>-lcb = the same model scored by mean - --lcb-beta x std of the reward over the --flow-samples futures.
  --models entries are label=path[,terminator_path]; a model
  without its own drop predictor uses --terminator. Decisions also store the chosen candidate's predicted drop.
The oracle needs --oracle-candidates K > 0: the scene then holds the E live environments plus K branch environments per
case; at each decision the live states are written into the branches, the branches run the candidates for 12 steps,
and the live states are restored. In oracle arms all three learned models also score the same candidates (ranking data).
Every arm stores the full 63-number state and applied targets per step (traces.npz) for the on-policy model analysis.
Experiment 34: --contact-sensors all also stores the 78 contact channels of every state (cube_world_data.contact_channels:
log(1 + max |F| hand->cube), cube-contact flag, log(1 + self-contact force), per hand link) as start_C (E, 4, 78) and
contact (E, T+1, 78), and the hand validity class as validity (E, T+1), for the 141-dim contact models (no oracle arms).
141-dim models (and ensembles of them) can then be planner models: their history gets the live contact channels.
Experiment 35 labelling arm dagger:<proposer>:<model>:<cost>:<student|none>:<beta>: the mpc planner labels every decision; per case, with
probability beta the planner's action is executed, otherwise the student policy's mean. Saves labels_<arm>.npz: obs (D, E, 105) policy
observation, X (D, E, K, 20) candidates, scores (D, E, K), chosen (D, E), executed (D, E, 20), expert_acts (E,), alive (D, E).
Experiment 39: arms cem:<policy>:<model>:<cost> = CEM seeded by the policy (--cem-iters rounds of --candidates; candidate 0 stays the policy
mean, candidate 1 the best so far, the rest resampled from the top --cem-elites); model hyb_<label> = hybrid forward model: the 48-dim hand
model --hand-models <label> predicts every candidate's 12-step joint trajectory, which is imposed on the hand of the branch environments
(joint state written before each physics substep, linearly interpolated, PD target = the predicted joints) while the cube is simulated;
the hand part of the returned future is the model's, the cube part the simulator's. Needs --oracle-candidates. In mpc:<policy>:oracle arms
every hand model also scores the oracle's candidates (rank_hyb_<label>) and stores its future of the chosen one (hyb_chosen_pred_<label>).
Experiment 51: --tendon-hand-models label=model.json adds exp-49 Li-form hand models (Markov, from the last state) usable as hyb_<label>.
Arms stk:<policy>:hyb_<tendon label>:<cost>:<layers> = the cem arm as planner on top, then the exp-50 stack executes: the chosen candidate's
predicted joint trajectory is the reference; layers ('+'-joined) rrt (self-collision check of the chosen command path, RRT-Connect to its
end where it penetrates), ilqr (iLQR-MPC + TVLQR tracking the reference with the same tendon model, replanned every --stack-replan steps; variants: ilqrc = reference is
the commanded joints instead of the predicted ones, ilqrnf = no TVLQR feedback between replans),
cbf (discrete CBF self-collision filter on every command). Every arm records per-step self-penetration of the measured hand (k-DOP).
"""
import argparse, json, math, os, sys, time, traceback, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics.resources import restrict_simulator_gpu
PHYSICAL_GPU = restrict_simulator_gpu()
from omni.isaac.lab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", required=True); p.add_argument("--arms", nargs="+", required=True)
p.add_argument("--hand-physics", default=None, help="experiment 43: JSON of a perturbed hand (env.set_hand_physics) applied to every environment")
p.add_argument("--policies", nargs="*", default=["det_mlp=artifacts/ppo_cube_r3/det_mlp/policy.pt", "res_det=artifacts/ppo_cube_r3/res_det/policy.pt", "res_flow_tcn=artifacts/ppo_cube_r3/res_flow_tcn/policy.pt"])
p.add_argument("--models", nargs="*", default=["det_mlp=artifacts/cube_world_r3/det_mlp/best.pt", "res_det=artifacts/cube_world_r3/res_det/best.pt", "res_flow_tcn=artifacts/cube_world_r3/res_flow_tcn/best.pt"])
p.add_argument("--terminator", default="artifacts/cube_terminator_r3/best.pt")
p.add_argument("--candidates", type=int, default=32); p.add_argument("--proposal-std", type=float, default=0.3); p.add_argument("--flow-samples", type=int, default=4); p.add_argument("--lcb-beta", type=float, default=1.0)
p.add_argument("--oracle-candidates", type=int, default=0, help="branch environments per case (must equal --candidates when an oracle arm is run)")
p.add_argument("--cases-per-bin", type=int, default=40); p.add_argument("--bins", nargs="+", type=float, default=[0.4, 1.0, 2.0, 3.1416])
p.add_argument("--seconds", type=float, default=10.0); p.add_argument("--seed", type=int, default=790000)
p.add_argument("--fall-dist", type=float, default=0.24); p.add_argument("--success", type=float, default=0.1)
# experiment 23: physics shift of the cube and corrupted observations for the controller (success and drops are judged on the true state)
p.add_argument("--cube-mass-scale", type=float, default=1.0); p.add_argument("--cube-friction", type=float, default=None)
p.add_argument("--obs-noise-pos-mm", type=float, default=0.0); p.add_argument("--obs-noise-rot-deg", type=float, default=0.0); p.add_argument("--obs-noise-joint-deg", type=float, default=0.0)
p.add_argument("--obs-delay", type=int, default=0, help="control steps (50 ms each) between the true state and what the controller sees")
p.add_argument("--obs-occl-enter", type=float, default=0.0, help="experiment 24: cube occlusion chain (cube_world_data.CubeOcclusionObserver; uses --obs-noise-pos-mm/--obs-noise-rot-deg for the cube, joints clean)")
p.add_argument("--obs-occl-stay", type=float, default=0.0)
p.add_argument("--inverse", nargs="*", default=[], help="experiment 30: label=checkpoint of a cube inverse proposer (train_inverse_proposer.py --task cube); "
               "arms mpc:<policy>:<model>:<cost>:<flowonly|mix>:<label>")
p.add_argument("--flow-steps", type=int, default=4, help="experiment 25: Euler steps of the flow sampler in the planner")
p.add_argument("--branch-physics", choices=["same", "nominal"], default="same", help="experiment 28: 'nominal' applies --cube-mass-scale/--cube-friction to the live "
               "environments only, so the oracle branches simulate the nominal cube (the oracle arm becomes 'nominal simulator as the model', and models with a "
               "simulator prior get their prior from these branches)")
p.add_argument("--contact-sensors", choices=["none", "all"], default="none", help="experiment 34: record whole-hand cube contact + hand validity in the traces")
p.add_argument("--viz-model", default="artifacts/viz_hand_model")
p.add_argument("--hand-models", nargs="*", default=[], help="experiment 39: label=path of 48-dim hand models for hyb_<label> forward models")
p.add_argument("--explore-delta", type=float, default=0.25, help="experiment 41, cemx arms: among candidates scoring >= best - delta * (best - median), execute the one with the largest hand-ensemble disagreement")
p.add_argument("--true-hand", action="store_true", help="experiment 52: hand model label 'true' = the true future hand from the oracle branches (perfect hand model)")
p.add_argument("--soft-hand-models", nargs="*", default=[], help="experiment 52: labels of hand models (or 'true') to also replay softly as soft_<label>: the predicted joints become the branch hand's drive target (stiffness x --track-gain, damping x sqrt) instead of being written kinematically")
p.add_argument("--track-gain", type=float, default=10.0, help="experiment 52: soft replay stiffness multiplier on the actuated joints (exp 40b: 10)")
p.add_argument("--tendon-hand-models", nargs="*", default=[], help="experiment 51: label=model.json of an exp-49 TendonHand; hyb_<label> and stk arms")
p.add_argument("--pen-weight", type=float, default=2.0, help="experiment 54: layer pen, CEM cost per mm of predicted finger overlap beyond the start's own (max over the horizon)")
p.add_argument("--stack-cbf-tol-mm", type=float, default=0.5, help="experiment 54: overlap the CBF tolerates beyond the start's own (exp 51: 0.5; light filter: 3)")
p.add_argument("--stack-replan", type=int, default=3); p.add_argument("--cbf-n", type=int, default=4); p.add_argument("--cbf-gamma", type=float, default=0.2); p.add_argument("--cbf-dact", type=float, default=0.02)
p.add_argument("--cem-iters", type=int, default=3); p.add_argument("--cem-elites", type=int, default=8); p.add_argument("--cem-min-std", type=float, default=0.05)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.physics_hz = 120; args.control_hz = 20; args.mode = "cube"
launcher = AppLauncher(args, multi_gpu=False); app = launcher.app
import numpy as np
import torch
from shadow_dynamics.env import ResearchShadowEnv, make_cfg, hand_physics_spawn_cfg
from shadow_dynamics.imagination import chunk_targets, random_goal_quat, cube_obs, CHUNK, H
from shadow_dynamics.cube_world_data import quat_to_matrix, rotation_distance
from shadow_dynamics.cube_mpc import state63, LearnedForward, PolicyProposer, candidate_targets, score_candidates, plan_reward, finger_groups

HZ = 20


def cube13(env, in_hand_w, ids=None):
    o = env.object.data.root_state_w if ids is None else env.object.data.root_state_w[ids]
    ih = in_hand_w if ids is None else in_hand_w[ids]
    return torch.cat([(o[:, :3] - ih) * 1000, o[:, 3:7], o[:, 7:10] * 1000, torch.rad2deg(o[:, 10:13])], -1)


def wilson(k, n, z=1.96):
    if n == 0: return [float("nan")] * 2
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def main():
    torch.set_num_threads(1); assert sorted(os.sched_getaffinity(0)) == __import__('shadow_dynamics.resources', fromlist=['budget_cpus']).budget_cpus()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    nb = len(args.bins) - 1; E = nb * args.cases_per_bin; Kb = args.oracle_candidates
    if any(a.split(":")[0] in ("mpc", "cem", "cemx") and a.split(":")[2].startswith(("oracle", "hyb_")) for a in args.arms) or args.hand_models or args.true_hand:
        assert Kb == args.candidates > 1, "oracle and hybrid arms need --oracle-candidates == --candidates"
    args.num_envs = E * (1 + Kb)
    cfg = make_cfg(args); cfg.object_cfg.spawn.rigid_props.kinematic_enabled = False
    if args.contact_sensors == "all": assert Kb == 0, "contact recording needs --oracle-candidates 0"; cfg.object_cfg.spawn.activate_contact_sensors = True
    hp = json.loads(Path(args.hand_physics).read_text()) if args.hand_physics else None
    if hp: hand_physics_spawn_cfg(cfg, hp)
    env = ResearchShadowEnv(cfg); dev = env.device; live = torch.arange(E, device=dev)
    try:
        env.configure_research("cube")
        if hp: (Path(args.output) / "hand_physics_applied.json").write_text(json.dumps(env.set_hand_physics(hp), indent=1)); print("HAND_PHYSICS applied", args.hand_physics, flush=True)
        if args.cube_mass_scale != 1.0 or args.cube_friction is not None:
            print("cube physics shift:", env.set_cube_physics(args.cube_mass_scale, args.cube_friction, env_ids=torch.arange(E) if args.branch_physics == "nominal" else None), flush=True)
        low, high = torch.rad2deg(env.low), torch.rad2deg(env.high); in_hand_w = env.in_hand_pos + env.scene.env_origins
        origins = env.scene.env_origins
        cmon = vmon = None; sub = None
        if args.contact_sensors == "all":   # experiment 34, thresholds as collect_shadow_cube.py
            from shadow_dynamics.cube_contacts import CubeContactMonitor
            from shadow_dynamics.hand_geometry import HandGeometry
            from shadow_dynamics.validity import ValidityMonitor
            cmon = CubeContactMonitor(env); sub = cmon.substep
            vmon = ValidityMonitor(env, HandGeometry.from_viz_model(args.viz_model), dict(touch_separation_m=0.0, touch_force_n=1e-3, limit_excess_rad=0.5, velocity_rad_s=200.0,
                                   hull_invalid_m=0.003, hull_palm_invalid_m=0.008, physx_penetration_m=0.002, hull_contact_m=0.0005))
        live_contact = {}
        class ContactForward:
            """Planner forward model on [63-dim state, 78 live contact channels] histories (experiment 34)."""
            def __init__(self, inner, ctx): self.inner, self.ctx, self.model, self.terminator, self.lcb_beta = inner, ctx, inner.model, inner.terminator, inner.lcb_beta
            def __call__(self, S_, A_, fa_):
                C_ = self.ctx["C"]; return self.inner(torch.cat([S_, C_.repeat_interleave(S_.shape[0] // C_.shape[0], 0)], -1), A_, fa_)
        def contact78():
            c_ = cmon.read(); v_ = vmon.measure()
            return torch.cat([torch.log1p(c_["link_cube_force_max"]), c_["link_cube_contact"].float(), torch.log1p(v_["link_self_force_n"])], -1)[:E], v_["validity"][:E]
        sys.modules.setdefault("git", types.ModuleType("git"))
        from shadow_dynamics.flow_model import load_forward
        from shadow_dynamics.cube_terminator import load_terminator
        term = load_terminator(args.terminator, dev)
        models = {}; ck0_names = torch.load(args.models[0].split("=")[1].split(",")[0].split("+")[0], map_location="cpu", weights_only=False)["data_report"]["joint_names"]
        for spec in args.models:
            lab, path = spec.split("="); path, *tp = path.split(","); m, _ = load_forward(path, dev)   # path1+path2+... = deterministic ensemble (experiment 25)
            models[lab] = LearnedForward(m, load_terminator(tp[0], dev) if tp else term, args.flow_samples, steps=args.flow_steps)
            sd = getattr(m, "sd", None) or m.members[0].sd
            if sd > 63:   # experiment 34: contact model, needs the recorded contact history
                assert cmon is not None, f"{lab}: a 141-dim model needs --contact-sensors all"; models[lab] = ContactForward(models[lab], live_contact)
        hand_models = {}
        for spec in args.hand_models:   # experiment 39
            lab, path = spec.split("="); hm, hck = load_forward(path, dev); assert hck["data_report"]["joint_names"] == ck0_names, path; hand_models[lab] = hm
        from shadow_dynamics.tendon_hand_model import TendonHand
        class TendonSampler:
            """Experiment 51: exp-49 Li-form hand model behind the hybrid's sample() interface (Markov: only the last state is used)."""
            flow = False
            def __init__(self, tm): self.tm = tm
            def sample(self, S_, A_, fa_, K=1, **kw):
                with torch.no_grad(): return torch.cat([self.tm.rollout(S_[i:i + 4096, -1, :48], fa=fa_[i:i + 4096]) for i in range(0, len(S_), 4096)], 0)[None]
        tendon = {}
        for spec in args.tendon_hand_models:
            lab, path = spec.split("="); cfg_t = json.loads(Path(path).read_text()); tm = TendonHand(cfg_t, dev).eval()
            assert list(tm.names) == list(ck0_names), (path, tm.names[:3], ck0_names[:3]); tendon[lab] = tm; hand_models[lab] = TendonSampler(tm)
            print(f"TENDON {lab} gravity_root {cfg_t.get('gravity_root')}", flush=True)
        if args.true_hand:   # experiment 52: the perfect hand model; sample() is bound to oracle_forward once the branches exist
            class TrueHand:
                fwd = None
                def sample(self, S_, A_, fa_, K=1, generator=None):
                    assert self.fwd is not None and len(fa_) == E * Kb, "true hand needs one call per decision over all branches"
                    return self.fwd(S_, A_, fa_)[0][..., :48]
            hand_models["true"] = TrueHand()
        for lab in args.soft_hand_models:   # experiment 52: same model, soft replay
            assert lab in hand_models, lab; hand_models[f"soft_{lab}"] = hand_models[lab]
        from shadow_dynamics.hand_cspace import HandCSpace
        from shadow_dynamics.rrt_connect import BatchRRTConnect
        from shadow_dynamics.ilqr import ILQR
        from shadow_dynamics.cbf import CBFFilter
        cspace = HandCSpace(device=dev)
        def self_pen(q24):   # (N, 24) deg measured -> (N,) max link penetration (m, >= 0), k-DOP
            with torch.no_grad(): return cspace._signed_depth(q24.float()).clamp_min(0).amax(1)
        def is_sim_prior(m):
            pr = getattr(m, "config", {}).get("prior"); return isinstance(pr, dict) and pr.get("source") == "sim"
        policies = {}
        for spec in args.policies:
            lab, path = spec.split("="); pol = torch.load(path, map_location=dev, weights_only=False); pol["actor_critic"].to(dev).eval(); pol["obs_normalizer"].to(dev).eval()
            assert pol["chunk"] == CHUNK and pol["slew_deg_s"] == 80.0 and pol["max_delta"] == 25.0   # imag_fall_mm is a training setting (experiment 21 native: 240), not a deployment one
            policies[lab] = pol
        groups = finger_groups(torch.load(args.models[0].split("=")[1].split(",")[0].split("+")[0], map_location="cpu", weights_only=False)["data_report"]["action_names"])
        P0 = next(iter(policies.values())); SLEW, MAXD = P0["slew_deg_s"], P0["max_delta"]
        assert all(pol["slew_deg_s"] == SLEW and pol["max_delta"] == MAXD for pol in policies.values())

        # ---- paired cases (identical construction to evaluate_shadow_cube_ppo.py; branch envs are parked meanwhile)
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        S0q = random_goal_quat(E, gen, dev); bins = np.repeat(np.arange(nb), args.cases_per_bin)
        def place_cubes(q_live):
            env.reset_physical(); obj = env.object.data.default_root_state.clone(); obj[:, :3] += origins
            obj[:E, 3:7] = q_live; env.object.write_root_state_to_sim(obj)
        for attempt in range(10):
            place_cubes(S0q); hold = torch.rad2deg(env.last_target).clone()
            for _ in range(HZ): env.advance(torch.deg2rad(hold))
            off = cube13(env, in_hand_w, live)[:, :3].norm(dim=-1) >= 120
            if not off.any(): break
            S0q[off] = random_goal_quat(int(off.sum()), gen, dev)
        print(f"start orientations: {attempt} redraw rounds, {int(off.sum())} still off the palm", flush=True)
        Rs = quat_to_matrix(env.object.data.root_state_w[:E, 3:7].clone()); G0 = torch.zeros(E, 4, device=dev)
        lo_b = torch.tensor([args.bins[b] for b in bins], device=dev); hi_b = torch.tensor([args.bins[b + 1] for b in bins], device=dev)
        for i in range(E):
            cand = random_goal_quat(4000, gen, dev); d = rotation_distance(Rs[i:i + 1].expand(4000, 3, 3), quat_to_matrix(cand)); ok = ((d >= lo_b[i]) & (d < hi_b[i])).nonzero().squeeze(-1)
            assert len(ok), f"no goal candidate for env {i}"; G0[i] = cand[ok[0]]

        # ---- oracle branches
        if Kb:
            branch = torch.arange(E, E * (1 + Kb), device=dev); case_of = live.repeat_interleave(Kb)
        oracle_stats = dict(calls=0, seconds=0.0)
        def kin_update():
            psv = getattr(env.sim, "physics_sim_view", None)
            if psv is not None and hasattr(psv, "update_articulations_kinematic"): psv.update_articulations_kinematic()
        @torch.no_grad()
        def oracle_forward(S, A, fa):
            """fa (E*Kb, 12, 20) deg, case-major -> true futures (1, E*Kb, 12, 63) from the live states."""
            t0 = time.time()
            jp, jv = env.hand.data.joint_pos[:E].clone(), env.hand.data.joint_vel[:E].clone()
            ob = env.object.data.root_state_w[:E].clone(); lt = env.last_target[:E].clone()
            env.hand.write_joint_state_to_sim(jp[case_of], jv[case_of], env_ids=branch)
            obb = ob[case_of].clone(); obb[:, :3] += origins[branch] - origins[case_of]; env.object.write_root_state_to_sim(obb, env_ids=branch)
            env.last_target[branch] = lt[case_of]; kin_update()
            frames = []; tgt = env.last_target.clone()
            for t in range(H):
                tgt[:E] = lt; tgt[branch] = torch.deg2rad(fa[:, t]); env.advance(tgt)
                frames.append(state63(torch.rad2deg(env.hand.data.joint_pos[branch]), torch.rad2deg(env.hand.data.joint_vel[branch]), cube13(env, in_hand_w, branch)))
            env.hand.write_joint_state_to_sim(jp, jv, env_ids=live); env.object.write_root_state_to_sim(ob, env_ids=live); env.last_target[:E] = lt; kin_update()
            oracle_stats["calls"] += 1; oracle_stats["seconds"] += time.time() - t0
            Y = torch.stack(frames, 1); oracle_stats["last"] = Y
            return Y[None], None

        hyb_last = {}; hyb_dis = {}
        def hybrid_forward(lab):
            """Experiment 39: hand from the hand model, cube from Isaac (kinematic replay in the branches). fa (E*Kb, 12, 20) deg, case-major."""
            hm = hand_models[lab]; act = env.active; soft = lab.startswith("soft_")
            @torch.no_grad()
            def f(S_, A_, fa_):
                t0 = time.time()
                Ys = torch.cat([hm.sample(S_[i:i + 8192, :, :48], A_[i:i + 8192], fa_[i:i + 8192], K=1) for i in range(0, len(S_), 8192)], 1)   # (M, N, 12, 48) deg
                Yh = Ys[0]; hyb_dis[lab] = Ys[..., :24].std(0).mean((-1, -2)) if Ys.shape[0] > 1 else torch.zeros(Ys.shape[1], device=Ys.device)   # experiment 41: member 0 is imposed
                jp, jv = env.hand.data.joint_pos[:E].clone(), env.hand.data.joint_vel[:E].clone()
                ob = env.object.data.root_state_w[:E].clone(); lt = env.last_target[:E].clone()
                env.hand.write_joint_state_to_sim(jp[case_of], jv[case_of], env_ids=branch)
                obb = ob[case_of].clone(); obb[:, :3] += origins[branch] - origins[case_of]; env.object.write_root_state_to_sim(obb, env_ids=branch)
                env.last_target[branch] = lt[case_of]; kin_update()
                q0, v0 = torch.deg2rad(S_[:, -1, :24]), torch.deg2rad(S_[:, -1, 24:48]); n_sub = env.cfg.decimation; frames = []; tgt = env.last_target.clone()
                if soft:   # experiment 52: stiffer drives on the branch hands only (restored below); the hand state stays Isaac's
                    hh = env.hand; K0 = hh.data.joint_stiffness.clone(); D0 = hh.data.joint_damping.clone(); Ks, Ds = K0[branch].clone(), D0[branch].clone()
                    Ks[:, act] *= args.track_gain; Ds[:, act] *= args.track_gain ** 0.5
                    hh.write_joint_stiffness_to_sim(Ks, env_ids=branch); hh.write_joint_damping_to_sim(Ds, env_ids=branch)
                    vt = torch.zeros_like(env.hand.data.joint_vel); hq = []
                for t in range(H) if soft else []:
                    q1, v1 = torch.deg2rad(Yh[:, t, :24]), torch.deg2rad(Yh[:, t, 24:48]); k = [1]
                    def soft_after():
                        k[0] += 1; w = min(k[0] / n_sub, 1.0); tg_ = env.last_target.clone(); tg_[branch] = (q0 + w * (q1 - q0))[:, act]
                        vt[branch] = v0 + w * (v1 - v0); env.hand.set_joint_position_target(tg_, joint_ids=act); env.hand.set_joint_velocity_target(vt)
                    vt[branch] = v0 + (v1 - v0) / n_sub; env.hand.set_joint_velocity_target(vt)
                    tgt[:E] = lt; tgt[branch] = (q0 + (q1 - q0) / n_sub)[:, act]; env.advance(tgt, on_substep=soft_after)
                    Sb = state63(torch.rad2deg(env.hand.data.joint_pos[branch]), torch.rad2deg(env.hand.data.joint_vel[branch]), cube13(env, in_hand_w, branch))
                    frames.append(Sb); hq.append(Sb[:, :24]); q0, v0 = q1, v1
                if soft:
                    vt.zero_(); env.hand.set_joint_velocity_target(vt); hh.write_joint_stiffness_to_sim(K0[branch], env_ids=branch); hh.write_joint_damping_to_sim(D0[branch], env_ids=branch)
                    oracle_stats.setdefault("soft_track_deg", []).append(float((torch.stack(hq, 1) - Yh[:, :, :24]).square().mean().sqrt()))
                for t in [] if soft else range(H):
                    q1, v1 = torch.deg2rad(Yh[:, t, :24]), torch.deg2rad(Yh[:, t, 24:48]); k = [0]
                    def put():
                        w = k[0] / n_sub; env.hand.write_joint_state_to_sim(q0 + w * (q1 - q0), v0 + w * (v1 - v0), env_ids=branch); kin_update()
                    def after():
                        k[0] += 1
                        if k[0] < n_sub: put()
                    put(); tgt[:E] = lt; tgt[branch] = q1[:, act]; env.advance(tgt, on_substep=after)
                    frames.append(torch.cat([Yh[:, t], state63(Yh[:, t, :24], Yh[:, t, 24:], cube13(env, in_hand_w, branch))[:, 48:]], -1)); q0, v0 = q1, v1
                env.hand.write_joint_state_to_sim(jp, jv, env_ids=live); env.object.write_root_state_to_sim(ob, env_ids=live); env.last_target[:E] = lt; kin_update()
                oracle_stats["hyb_seconds"] = oracle_stats.get("hyb_seconds", 0.0) + time.time() - t0
                Y = torch.stack(frames, 1); hyb_last[lab] = Y
                return Y[None], None
            return f
        hyb_fwd = {lab: hybrid_forward(lab) for lab in hand_models}
        if args.true_hand: hand_models["true"].fwd = oracle_forward

        def quat_mul(a_, b_):
            w1, x1, y1, z1 = a_.unbind(-1); w2, x2, y2, z2 = b_.unbind(-1)
            return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
        class Observer:
            """What the controller sees: the true readings delayed by --obs-delay steps plus Gaussian noise; velocity noise is that of a finite
            difference of the noisy positions (sqrt(2) * sigma * 20 Hz). With all settings at 0 it returns the true readings unchanged."""
            def __init__(self, gen): self.buf, self.g = [], gen
            def __call__(self, q, qd, c):
                self.buf.append((q.clone(), qd.clone(), c.clone())); self.buf = self.buf[-(args.obs_delay + 1):]
                q, qd, c = (x_.clone() for x_ in self.buf[0]); k = math.sqrt(2) * HZ
                n = lambda shape, sd: torch.randn(shape, generator=self.g, device=dev) * sd
                if args.obs_noise_joint_deg: q += n(q.shape, args.obs_noise_joint_deg); qd += n(qd.shape, args.obs_noise_joint_deg * k)
                if args.obs_noise_pos_mm: c[:, :3] += n((len(c), 3), args.obs_noise_pos_mm); c[:, 7:10] += n((len(c), 3), args.obs_noise_pos_mm * k)
                if args.obs_noise_rot_deg:
                    v = n((len(c), 3), math.radians(args.obs_noise_rot_deg)); ang = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    c[:, 3:7] = quat_mul(torch.cat([torch.cos(ang / 2), torch.sin(ang / 2) * v / ang], -1), c[:, 3:7]); c[:, 10:13] += n((len(c), 3), args.obs_noise_rot_deg * k)
                return q, qd, c

        class OcclusionObserver:
            """Experiment 24: the cube as seen through cube_world_data.CubeOcclusionObserver (the training-time corruption, per step); hand clean."""
            def __init__(self, gen):
                from shadow_dynamics.cube_world_data import CubeOcclusionObserver, occlusion_cfg
                assert args.obs_delay == 0 and args.obs_noise_joint_deg == 0
                self.o = CubeOcclusionObserver(E, occlusion_cfg(args.obs_noise_pos_mm, args.obs_noise_rot_deg, args.obs_occl_enter, args.obs_occl_stay), gen, dev); self.ages = []
            def __call__(self, q, qd, c):
                oc, age = self.o(c); self.ages.append(age); return q.clone(), qd.clone(), oc
        T = int(args.seconds * HZ); results = {}; sim_unstable = [0]
        from shadow_dynamics.inverse_proposer import load_proposer
        inverse = {}
        for spec in args.inverse:
            lab, path = spec.split("="); im, ick = load_proposer(path, dev); inverse[lab] = (im, torch.tensor(ick["rot_mag_quantiles_deg"], device=dev))
        def inverse_proposals(inv, S_, A_, c13, goal_q, x_p, K_, g_):
            """K_ proposals per case: desired 0.6 s rotation = axis toward the goal, magnitude min(remaining angle, m), m ~ P's achieved 0.6 s rotations."""
            from shadow_dynamics.cube_world_data import matrix_to_quat
            m_, qs = inv; N_ = len(S_); Rc = quat_to_matrix(c13[:, 3:7]); Rr = quat_to_matrix(goal_q) @ Rc.transpose(-1, -2); qr = matrix_to_quat(Rr)
            v = qr[:, 1:]; n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9); ang = torch.rad2deg(2 * torch.atan2(n[:, 0], qr[:, 0])); axis = v / n
            u_ = torch.rand(N_, K_, generator=g_, device=dev) * 100; lo_ = u_.floor().long().clamp(max=99); mag = qs[lo_] + (u_ - lo_) * (qs[(lo_ + 1).clamp(max=100)] - qs[lo_])
            d = axis[:, None] * torch.minimum(mag, ang[:, None])[..., None]                                     # (N, K, 3) deg
            rep_ = lambda x_: x_.repeat_interleave(K_, 0)
            U_ = m_.sample(rep_(S_), rep_(A_), d.reshape(-1, 3), rep_(x_p), K=1, generator=g_)[0].view(N_, K_, 20)
            if not m_.flow: U_ = U_ + args.proposal_std * torch.randn(U_.shape, generator=g_, device=dev)
            return U_.clamp(-1, 1)
        for arm in args.arms:
            t_arm = time.time(); parts = arm.split(":"); kind = parts[0]; dagger = None
            explore = kind == "cemx"; kind = "cem" if explore else kind   # experiment 41: curiosity-driven execution (needs an ensemble hyb_ model)
            stack = None
            if kind == "stk":   # experiment 51: CEM planner on top, exp-50 stack below
                layers = set(parts[4].split("+")); assert layers and layers <= {"rrt", "rrtp", "ilqr", "ilqrc", "ilqrnf", "tvlqr", "cbf", "pen"} and len(layers & {"ilqr", "ilqrc", "ilqrnf", "tvlqr"}) <= 1 and not {"rrt", "rrtp"} <= layers, arm
                tlab = parts[2][4:].removeprefix("soft_"); assert parts[2].startswith("hyb_") and tlab in tendon, arm   # experiment 54: soft_<tendon> hybrids
                tm_s = tendon[tlab]; slew_s = SLEW / HZ; fstep = lambda x_, u_, tm=tm_s: tm.rollout(x_, fa=u_[:, None])[:, 0]
                stack = dict(layers=layers, il=ILQR(fstep, low, high, env.active, slew=slew_s, H=H, fb_scale=0.0 if "ilqrnf" in layers else 1.0) if layers & {"ilqr", "ilqrc", "ilqrnf", "tvlqr"} else None,
                             flt=CBFFilter(cspace, fstep, low, high, slew=slew_s, n=args.cbf_n, gamma=args.cbf_gamma, d_act=args.cbf_dact) if "cbf" in layers else None,
                             rr=BatchRRTConnect(cspace, low.cpu().numpy(), high.cpu().numpy(), seed=args.seed) if layers & {"rrt", "rrtp"} else None,
                             st=dict(decisions=0, rrt_needed=0, rrt_unsolved=0, rrtp_chosen=0, rrt_seconds=0.0, ilqr_seconds=0.0, ilqr_plans=0, cbf_seconds=0.0, nonfinite_cmd=0, track_err=[]))
                parts = parts[:4]; kind = "cem"
            if explore: assert parts[2].startswith("hyb_"), "cemx arms need a hybrid ensemble model (label=a+b+c)"
            if kind == "dagger":   # experiment 35: planner labels, executed by expert (prob beta per case) or student
                stud = None if parts[4] == "none" else policies[parts[4]]; beta = float(parts[5]); parts = parts[:4]; kind = "mpc"
                use_exp = torch.rand(E, generator=torch.Generator(device=dev).manual_seed(args.seed + 11), device=dev) < beta
                dagger = dict(student=stud, use_exp=use_exp if stud is not None else torch.ones(E, dtype=torch.bool, device=dev), obs=[], X=[], scores=[], chosen=[], executed=[], alive=[])
            g2 = torch.Generator(device=dev).manual_seed(args.seed + 1); gp = torch.Generator(device=dev).manual_seed(args.seed + 2)
            place_cubes(S0q); prev = torch.rad2deg(env.last_target[:E]).clone(); obs = (OcclusionObserver if args.obs_occl_enter > 0 else Observer)(torch.Generator(device=dev).manual_seed(args.seed + 3))
            Sh, Ah, Ch = [], [], []
            if cmon is not None: cmon.begin()
            for k in range(HZ):   # the cube settles on the palm; the last 4 states and 3 targets become the history
                env.advance(torch.deg2rad(torch.rad2deg(env.last_target).clone()), on_substep=sub)
                if cmon is not None: Ch.append(contact78())
                oq, oqd, oc = obs(torch.rad2deg(env.hand.data.joint_pos[:E]), torch.rad2deg(env.hand.data.joint_vel[:E]), cube13(env, in_hand_w, live))
                Sh.append(state63(oq, oqd, oc)); Ah.append(prev.clone())
            S = torch.stack(Sh[-4:], 1); A = torch.stack(Ah[-3:], 1); S_start, A_start = S.clone(), A.clone()
            goal = G0.clone(); c = cube13(env, in_hand_w, live); valid = c[:, :3].norm(dim=-1) < 120
            init_err = rotation_distance(quat_to_matrix(c[:, 3:7]), quat_to_matrix(goal))
            alive = torch.ones(E, dtype=torch.bool, device=dev); first_t = torch.full((E,), float("nan"), device=dev); goals_hit = torch.zeros(E, device=dev); best = init_err.clone()
            x = torch.zeros(E, 20, device=dev); scale = torch.rand(E, 1, generator=g2, device=dev) * 0.8 + 0.1; queue = None
            tr_S, tr_U, tr_G, tr_alive = [S[:, -1].clone()], [], [goal.clone()], [alive.clone()]
            tr_pen = [self_pen(torch.rad2deg(env.hand.data.joint_pos[:E]))]
            if stack is not None:
                stack["tol"] = tr_pen[0] + args.stack_cbf_tol_mm * 1e-3; stack["pen_tol"] = tr_pen[0] + 0.5e-3; stack["ref"] = None
                if stack["il"] is not None: stack["il"].U = None
            if cmon is not None: tr_C, tr_V = [Ch[-1][0]], [Ch[-1][1]]; live_contact["C"] = torch.stack([c_ for c_, _ in Ch[-4:]], 1)
            dec = dict(chosen=[], pred_best=[], pred_mean=[], xabs=[]); rank = {k_: [] for k_ in ["oracle"] + [l_ for l_, mf_ in models.items() if not is_sim_prior(mf_.model)] + [f"hyb_{l_}" for l_ in hand_models]}; rank_alive = []
            inv_mode = parts[4] if len(parts) > 5 and parts[4] in ("flowonly", "mix") else None; inv = inverse[parts[5]] if inv_mode else None
            proposer = PolicyProposer(policies[parts[1]], args.proposal_std, mode=parts[4] if len(parts) > 4 and not inv_mode else "gauss", groups=groups) if kind in ("ppo", "mpc", "cem") else None
            fwd = None; rkw = dict(cost=parts[3] if len(parts) > 3 else "r3")
            if kind == "cem": assert dagger is None and not inv_mode
            if kind in ("mpc", "cem"):
                ml = parts[2]
                if Kb and args.branch_physics == "nominal":   # experiment 29: branch cube physics per arm (oracle@<mass scale>_<friction|nom>; others nominal)
                    bm, bf = (1.0, None)
                    if ml.startswith("oracle@"): v = ml.split("@", 1)[1].split("_"); bm, bf = float(v[0]), (None if v[1] == "nom" else float(v[1]))
                    print(f"  branch cube physics for {arm}: mass x{bm}, friction {bf or 'nominal'}", flush=True); env.set_cube_physics_abs(bm, bf, env_ids=torch.arange(E, E * (1 + Kb)))
                    if ml.startswith("oracle@"): ml = "oracle"
                if ml == "oracle": fwd = oracle_forward
                elif ml.startswith("hyb_"): fwd = hyb_fwd[ml[4:]]
                elif ml.startswith("oracle+"):
                    tf = models[ml[len("oracle+"):]].terminator
                    fwd = lambda S_, A_, fa_, tf=tf: (oracle_forward(S_, A_, fa_)[0], tf(S_, A_, fa_))
                elif ml.endswith("-noterm"): fwd = LearnedForward(models[ml[:-len("-noterm")]].model, None, args.flow_samples, steps=args.flow_steps)
                elif ml.endswith("-lcb"): b = models[ml[:-len("-lcb")]]; assert b.model.flow, ml; fwd = LearnedForward(b.model, b.terminator, args.flow_samples, args.lcb_beta, steps=args.flow_steps)
                elif is_sim_prior(models[ml].model):   # experiment 28: residual model on the simulator's prediction from the (nominal) branches
                    assert Kb == args.candidates and args.branch_physics == "nominal", "simulator-prior models need --oracle-candidates == --candidates and --branch-physics nominal"
                    def sim_prior(S_, A_, fa_):   # nominal-branch futures; rare restore blow-ups get the persistence prior (as in compute_cube_sim_prior.py)
                        Y_ = oracle_forward(S_, A_, fa_)[0][0]; bad = ~torch.isfinite(Y_).all(-1).all(-1) | (Y_[..., :24].abs() > 360).any(-1).any(-1)
                        if bad.any(): Y_ = Y_.clone(); Y_[bad] = S_[bad, -1:].expand(-1, Y_.shape[1], -1); sim_unstable[0] += int(bad.sum())
                        return Y_
                    fwd = (lambda b: (lambda S_, A_, fa_: (b.model.sample(S_, A_, fa_, phys=sim_prior(S_, A_, fa_)), b.terminator(S_, A_, fa_))))(models[ml])
                else: fwd = models[ml]
            for t in range(T):
                if t % CHUNK == 0:
                    q, qd = oq, oqd   # observed (= true without --obs-* settings)
                    if kind == "hold": x = torch.zeros(E, 20, device=dev)
                    elif kind == "random": x = 0.7 * x + 0.7 * scale * torch.randn(E, 20, generator=g2, device=dev)
                    elif kind == "ppo": x = proposer.mean(q, qd, oc, goal, prev)
                    else:
                        X = proposer.propose(q, qd, oc, goal, prev, args.candidates, gp)                     # (E, K, 20)
                        if inv_mode:   # experiment 30: inverse-model proposals around P's action, toward the goal
                            K_ = args.candidates - 1 if inv_mode == "flowonly" else args.candidates // 2
                            X = torch.cat([X[:, :args.candidates - K_], inverse_proposals(inv, S, A, oc, goal, X[:, 0], K_, gp)], 1)
                        fa = candidate_targets(prev, X, MAXD, SLEW, low, high); Rg = quat_to_matrix(goal)
                        def score_fn(fa_, X_):   # experiment 54, layer pen: predicted finger overlap (soft branches) is a CEM cost
                            sc_, pd_ = score_candidates(fwd, S, A, fa_, X_, Rg, with_drop=True, **rkw)
                            if stack is not None and "pen" in stack["layers"]:
                                Yq = hyb_last[parts[2][4:]][..., :24].reshape(-1, 24); pq = self_pen(Yq).view(E, -1, H).amax(-1)
                                sc_ = sc_ - args.pen_weight * 1000 * (pq - stack["pen_tol"][:, None]).clamp_min(0)
                            return sc_, pd_
                        sc, pdrop = score_fn(fa, X)          # (E, K)
                        for _ in range(args.cem_iters - 1 if kind == "cem" else 0):   # experiment 39: CEM seeded by the policy
                            el = sc.topk(args.cem_elites, 1).indices; Xe = torch.gather(X, 1, el[..., None].expand(-1, -1, X.shape[-1]))
                            mu, sd = Xe.mean(1), Xe.std(1).clamp(args.cem_min_std, args.proposal_std); xb = X[live, sc.argmax(1)]
                            Xn = mu[:, None] + sd[:, None] * torch.randn(E, args.candidates - 2, X.shape[-1], generator=gp, device=dev)
                            X = torch.cat([X[:, :1], xb[:, None], Xn], 1).clamp(-1, 1)
                            fa = candidate_targets(prev, X, MAXD, SLEW, low, high); sc, pdrop = score_fn(fa, X)
                        j = sc.argmax(1)
                        if explore:   # experiment 41: near-best candidates only, then the most uncertain one for the hand ensemble
                            dis = hyb_dis[parts[2][4:]].view(E, -1); best_, med_ = sc.max(1).values, sc.median(1).values
                            ok_ = sc >= (best_ - args.explore_delta * (best_ - med_))[:, None]
                            j = torch.where(ok_, dis, torch.full_like(dis, -1e9)).argmax(1)
                            dec.setdefault("explore_dis", []).append(dis[live, j].cpu()); dec.setdefault("explore_dis_best", []).append(dis[live, sc.argmax(1)].cpu())
                        x = X[live, j]
                        if dagger is not None:
                            if dagger["student"] is not None:
                                st = dagger["student"]
                                with torch.no_grad(): xs = st["actor_critic"].act_inference(st["obs_normalizer"](cube_obs(q, qd, oc, goal, prev)))
                                x = torch.where(dagger["use_exp"][:, None], x, xs)
                            for k_, v_ in (("obs", cube_obs(q, qd, oc, goal, prev)), ("X", X.half()), ("scores", sc), ("chosen", j), ("executed", x), ("alive", alive.clone())): dagger[k_].append(v_.cpu())
                        dec["chosen"].append(j.cpu()); dec["pred_best"].append(sc[live, j].cpu()); dec["pred_mean"].append(sc.mean(1).cpu())
                        dec.setdefault("pred_drop", []).append(pdrop[live, j].cpu()); dec.setdefault("pred_drop_mean", []).append(pdrop.mean(1).cpu()); dec.setdefault("dec_alive", []).append(alive.cpu())
                        if kind == "mpc" and parts[2].startswith("oracle") and "+" not in parts[2]:   # ranking data: every learned model scores the same candidates
                            rank["oracle"].append(sc.cpu()); rank_alive.append(alive.cpu())
                            dec.setdefault("oracle_chosen_pred", []).append(oracle_stats["last"].view(E, args.candidates, H, -1)[live, j].cpu())
                            for lab, mf in models.items():
                                if not is_sim_prior(mf.model): rank[lab].append(score_candidates(mf, S, A, fa, X, Rg, **rkw).cpu())
                            for lab, hf in hyb_fwd.items():   # experiment 39
                                rank[f"hyb_{lab}"].append(score_candidates(hf, S, A, fa, X, Rg, **rkw).cpu())
                                dec.setdefault(f"hyb_chosen_pred_{lab}", []).append(hyb_last[lab].view(E, args.candidates, H, -1)[live, j].cpu())
                    g = (prev + MAXD * x.clamp(-1, 1)).clamp(low, high); queue = chunk_targets(prev, g, SLEW, low, high, steps=CHUNK)
                    dec["xabs"].append(float(x.clamp(-1, 1).abs()[alive].mean()) if alive.any() else float("nan"))
                    if stack is not None:   # experiment 51: reference = the chosen candidate's predicted joints (the hand model's future of fa)
                        st_ = stack["st"]; st_["decisions"] += 1; fa_c = fa.view(E, args.candidates, H, 20)[live, j]                      # (E, 12, 20)
                        hyb_last_dec = hyb_last[parts[2][4:]].view(E, args.candidates, H, -1).clone()
                        ref = hyb_last_dec[live, j][..., :24].clone()                         # (E, 12, 24)
                        if "ilqrc" in stack["layers"]: ref = cspace.full(fa_c)   # exp 51 follow-up: command-space reference (the commanded joints, not the model's free-motion response)
                        if stack["rr"] is not None:   # global layer: does the chosen command path penetrate (beyond the start's own contact)?
                            tq0 = time.time(); pts = torch.cat([prev[:, None], fa_c], 1); mids = 0.5 * (pts[:, 1:] + pts[:, :-1])
                            allp = torch.cat([pts[:, 1:], mids], 1); pen_ = torch.as_tensor(cspace.penetration(allp), device=dev).amax(1)
                            thr_ = torch.maximum(torch.as_tensor(cspace.penetration(prev), device=dev), torch.as_tensor(cspace.penetration(fa_c[:, -1]), device=dev)) + 0.5e-3
                            bad = ((pen_ > thr_) & (pen_ > cspace.threshold) & alive).nonzero().squeeze(-1)
                            if len(bad):
                                st_["rrt_needed"] += int(len(bad)); paths, pinfo = stack["rr"].plan(prev[bad].cpu().numpy(), fa_c[bad, -1].cpu().numpy())
                                st_["rrt_unsolved"] += int((~pinfo["solved"]).sum())
                                for b_, path_ in zip(bad.tolist(), paths):
                                    if path_ is None: continue
                                    seg = np.linalg.norm(np.diff(path_, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
                                    s_ = np.minimum(np.arange(1, H + 1) * slew_s, cum[-1]); wp = np.stack([np.interp(s_, cum, path_[:, d]) for d in range(20)], 1)
                                    wp_t = torch.as_tensor(wp, dtype=torch.float32, device=dev).clamp(low, high)
                                    ref[b_] = cspace.full(wp_t); queue[b_] = wp_t[:CHUNK]; fa_c[b_] = wp_t
                            st_["rrt_seconds"] += time.time() - tq0
                        if "rrtp" in stack["layers"]:   # experiment 54: the RRT path is one more CEM candidate (replaces the worst), scored in the branches
                            tq0 = time.time(); pts = torch.cat([prev[:, None], fa_c], 1); mids = 0.5 * (pts[:, 1:] + pts[:, :-1])
                            allp = torch.cat([pts[:, 1:], mids], 1); pen_ = torch.as_tensor(cspace.penetration(allp), device=dev).amax(1)
                            thr_ = torch.maximum(torch.as_tensor(cspace.penetration(prev), device=dev), torch.as_tensor(cspace.penetration(fa_c[:, -1]), device=dev)) + 0.5e-3
                            bad = ((pen_ > thr_) & (pen_ > cspace.threshold) & alive).nonzero().squeeze(-1)
                            if len(bad):
                                st_["rrt_needed"] += int(len(bad)); paths, pinfo = stack["rr"].plan(prev[bad].cpu().numpy(), fa_c[bad, -1].cpu().numpy())
                                st_["rrt_unsolved"] += int((~pinfo["solved"]).sum()); fa2 = fa.view(E, args.candidates, H, 20).clone(); w_ = sc.argmin(1); got = []
                                for b_, path_ in zip(bad.tolist(), paths):
                                    if path_ is None: continue
                                    seg = np.linalg.norm(np.diff(path_, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
                                    s_ = np.minimum(np.arange(1, H + 1) * slew_s, cum[-1]); wp = np.stack([np.interp(s_, cum, path_[:, d]) for d in range(20)], 1)
                                    fa2[b_, w_[b_]] = torch.as_tensor(wp, dtype=torch.float32, device=dev).clamp(low, high); got.append(b_)
                                if got:
                                    sc2, _ = score_fn(fa2.view(-1, H, 20), X); Y2 = hyb_last[parts[2][4:]].view(E, args.candidates, H, -1)
                                    for b_ in got:
                                        if sc2[b_, w_[b_]] > sc2[b_, j[b_]]:   # CEM's own choice re-scored in the same call (deterministic soft branches)
                                            st_["rrtp_chosen"] += 1; wp_t = fa2[b_, w_[b_]]; queue[b_] = wp_t[:CHUNK]; fa_c[b_] = wp_t; ref[b_] = Y2[b_, w_[b_], :, :24]
                                            if "tvlqr" in stack["layers"]: stack.setdefault("ref48_fix", {})[b_] = Y2[b_, w_[b_], :, :48].clone()
                            st_["rrt_seconds"] += time.time() - tq0
                        if "tvlqr" in stack["layers"]:   # experiment 54: gains about CEM's commands and their (soft, contact-aware) predicted states
                            X48 = hyb_last_dec[live, j][..., :48].clone()
                            for b_, v_ in stack.pop("ref48_fix", {}).items(): X48[b_] = v_
                            stack["il"].gains(torch.cat([oq, oqd], -1), prev, fa_c, X48); stack["up0p"] = prev.clone()
                        stack["ref"] = ref; stack["fa"] = fa_c
                if stack is not None:   # the stack turns the planner's chunk into the executed command
                    st_ = stack["st"]; x_now = torch.cat([oq, oqd], -1); o_ = t % CHUNK; cmd = queue[:, o_]
                    if "tvlqr" in stack["layers"]: cmd = stack["il"].feedback(o_, x_now, prev, stack["up0p"])   # experiment 54: CEM's commands + feedback only
                    elif stack["il"] is not None:
                        il = stack["il"]; jr = o_ % args.stack_replan
                        if jr == 0:
                            idx = torch.clamp(torch.arange(o_, o_ + H, device=dev), max=H - 1); torch.cuda.synchronize(); ti = time.time()
                            il.plan(x_now, prev, stack["ref"][:, idx], shift=args.stack_replan if t else 0, U_init=stack["fa"][:, idx]); stack["up0p"] = prev.clone(); torch.cuda.synchronize()
                            st_["ilqr_seconds"] += time.time() - ti; st_["ilqr_plans"] += 1
                        cmd = il.feedback(jr, x_now, prev, stack["up0p"])
                    if stack["flt"] is not None:
                        tc = time.time(); cmd = stack["flt"].filter(x_now, prev, cmd, stack["tol"]); st_["cbf_seconds"] += time.time() - tc
                    fin = torch.isfinite(cmd).all(1); st_["nonfinite_cmd"] += int((~fin).sum()); cmd = torch.where(fin[:, None], cmd, prev)
                    st_["track_err"].append(float((oq - stack["ref"][:, o_ - 1 if o_ else 0]).square().mean(1).sqrt()[alive].mean()) if (alive.any() and o_) else float("nan"))
                    queue = queue.clone(); queue[:, o_] = cmd
                tgt = env.last_target.clone(); tgt[:E] = torch.deg2rad(queue[:, t % CHUNK]); env.advance(tgt, on_substep=sub)
                if cmon is not None: c78, v78 = contact78(); tr_C.append(c78); tr_V.append(v78); live_contact["C"] = torch.cat([live_contact["C"][:, 1:], c78[:, None]], 1)
                prev = torch.rad2deg(env.last_target[:E]).clone(); c = cube13(env, in_hand_w, live)
                tq, tqd = torch.rad2deg(env.hand.data.joint_pos[:E]), torch.rad2deg(env.hand.data.joint_vel[:E]); s_new = state63(tq, tqd, c)   # true (traces)
                oq, oqd, oc = obs(tq, tqd, c); S = torch.cat([S[:, 1:], state63(oq, oqd, oc)[:, None]], 1); A = torch.cat([A[:, 1:], prev[:, None]], 1)
                err = rotation_distance(quat_to_matrix(c[:, 3:7]), quat_to_matrix(goal)); fell = c[:, :3].norm(dim=-1) >= args.fall_dist * 1000
                alive &= ~fell; hit = alive & (err < args.success)
                first_t = torch.where(hit & torch.isnan(first_t), torch.full_like(first_t, (t + 1) / HZ), first_t)
                best = torch.where(alive & (goals_hit == 0), torch.minimum(best, err), best); goals_hit += hit.float()
                if hit.any(): goal[hit] = random_goal_quat(int(hit.sum()), g2, dev)
                tr_S.append(s_new); tr_U.append(prev.clone()); tr_G.append(goal.clone()); tr_alive.append(alive.clone()); tr_pen.append(self_pen(tq))
            vm = valid.cpu().numpy(); ok = (~torch.isnan(first_t)).cpu().numpy() & vm
            r = dict(valid_start=vm, success=ok, time_to_success=first_t.cpu().numpy(), goals=np.where(vm, goals_hit.cpu().numpy(), np.nan), dropped=(~alive).cpu().numpy() & vm,
                     init_err=init_err.cpu().numpy(), best_err_first_goal=np.where(vm, best.cpu().numpy(), np.nan))
            n = int(vm.sum()); ks, kd = int(r["success"][vm].sum()), int(r["dropped"][vm].sum())
            summ = dict(valid_cases=n, success_rate=ks / n, success_ci95=wilson(ks, n), drop_rate=kd / n, drop_ci95=wilson(kd, n), goals_per_episode=float(np.nanmean(r["goals"])),
                        median_time_to_success=float(np.nanmedian(r["time_to_success"])) if r["success"].any() else None,
                        best_err_mean=float(np.nanmean(r["best_err_first_goal"])), success_within_0p4=float(np.nanmean(r["best_err_first_goal"][vm] < 0.4)),
                        per_bin={f"{args.bins[b]:.1f}-{args.bins[b + 1]:.1f}": dict(n=int((vm & (bins == b)).sum()), success=float(r["success"][vm & (bins == b)].mean()), drop=float(r["dropped"][vm & (bins == b)].mean()),
                                                                                   best_err=float(np.nanmean(r["best_err_first_goal"][vm & (bins == b)]))) for b in range(nb)},
                        mean_abs_action=float(np.nanmean(dec["xabs"])), seconds=time.time() - t_arm)
            pen_all = torch.stack(tr_pen[1:], 1); al_ = torch.stack(tr_alive[1:], 1) & valid[:, None]
            summ.update(self_contact_frac=float((pen_all > 0)[al_].float().mean()), self_pen_1mm_frac=float((pen_all > 1e-3)[al_].float().mean()),
                        self_pen_3mm_frac=float((pen_all > 3e-3)[al_].float().mean()), self_pen_mean_mm=float(pen_all[al_].mean() * 1000))
            if stack is not None:
                st_ = stack["st"]; te = np.array(st_.pop("track_err")); summ.update(stack_layers=sorted(stack["layers"]), stack=st_, track_err_deg=float(np.nanmean(te)) if np.isfinite(te).any() else None)
                if stack["il"] is not None: summ["stack"]["ilqr_fallback"] = int(getattr(stack["il"], "n_fallback", 0))
                if stack["flt"] is not None: summ["stack"]["cbf"] = dict(stack["flt"].stats)
            if kind in ("mpc", "cem"): summ.update(oracle_seconds=oracle_stats["seconds"] if parts[2].startswith("oracle") else None, hybrid_seconds=oracle_stats.pop("hyb_seconds", None), soft_track_deg=(lambda v_: float(np.mean(v_)) if v_ else None)(oracle_stats.pop("soft_track_deg", [])), sim_prior_unstable_candidates=sim_unstable[0]); sim_unstable[0] = 0
            if isinstance(obs, OcclusionObserver):   # stale-step statistics while the cube is in the hand (compare with the training corruption)
                ag = torch.stack(obs.ages[HZ:], 1)[:, :T]; al = torch.stack(tr_alive[1:], 1)[:, :ag.shape[1]] & valid[:, None]
                summ.update(stale_fraction=float((ag > 0)[al].float().mean()), mean_stale_age=float(ag[al & (ag > 0)].float().mean()))
            results[arm] = dict(summary=summ, per_case={k_: v.tolist() for k_, v in r.items()})
            tag = arm.replace(":", "__")
            arrays = dict(self_pen=torch.stack(tr_pen, 1), states=torch.stack(tr_S, 1), targets=torch.stack(tr_U, 1), goal_quat=torch.stack(tr_G, 1), alive=torch.stack(tr_alive, 1), start_S=S_start, start_A=A_start)
            extra = {}
            if cmon is not None: extra = dict(start_C=torch.stack([c_ for c_, _ in Ch[-4:]], 1), contact=torch.stack(tr_C, 1), validity=torch.stack(tr_V, 1))
            if dec["chosen"]: extra = dict(chosen=torch.stack(dec["chosen"]), pred_best=torch.stack(dec["pred_best"]), pred_mean=torch.stack(dec["pred_mean"]),
                                           pred_drop=torch.stack(dec["pred_drop"]), pred_drop_mean=torch.stack(dec["pred_drop_mean"]), dec_alive=torch.stack(dec["dec_alive"]))
            if "oracle_chosen_pred" in dec: extra["oracle_chosen_pred"] = torch.stack(dec["oracle_chosen_pred"])   # (D, E, 12, 63): restore-fidelity audit
            extra.update({k_: torch.stack(v) for k_, v in dec.items() if k_.startswith(("hyb_chosen_pred_", "explore_"))})
            if rank_alive: extra.update({f"rank_{k_}": torch.stack(v) for k_, v in rank.items()}, rank_alive=torch.stack(rank_alive))
            if dagger is not None:
                np.savez_compressed(out / f"labels_{tag}.npz", expert_acts=dagger["use_exp"].cpu().numpy(), valid=vm, **{k_: torch.stack(dagger[k_]).numpy() for k_ in ("obs", "X", "scores", "chosen", "executed", "alive")})
            _g = [k_ for k_, v in {**arrays, **extra}.items() if torch.is_tensor(v) and v.requires_grad]
            if _g: print(f"WARN requires_grad in traces: {_g}", flush=True)
            np.savez_compressed(out / f"traces_{tag}.npz", bins=bins, init_goal=G0.cpu().numpy(), success=r["success"], dropped=r["dropped"], valid=vm, time_to_success=r["time_to_success"],
                                **{k_: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k_, v in {**arrays, **extra}.items()})
            print(f"RUN {arm:28s} valid {n}  success {summ['success_rate']:.3f} {summ['success_ci95']}  drop {summ['drop_rate']:.2f}  goals/ep {summ['goals_per_episode']:.2f}  "
                  f"best err {summ['best_err_mean']:.2f}  <0.4 {summ['success_within_0p4']:.2f}  |x| {summ['mean_abs_action']:.2f}  pen>1mm {summ['self_pen_1mm_frac']:.4f}  {summ['seconds']:.0f}s", flush=True)
            (out / "results.json").write_text(json.dumps(dict(args=vars(args), bins=args.bins, initial_error_rad=init_err.cpu().tolist(), runs=results), indent=1))
        print("CUBE_MPC_EVAL_COMPLETE", out, flush=True)
    finally:
        env.close()


try: main()
except Exception:
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
finally: app.close()
