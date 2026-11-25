"""Train PPO (rsl_rl) for cube reorientation directly in Isaac, with no learned model anywhere (experiment 19).

The model-free ceiling of the pipeline's task: the same decision interface as the imagined training and the Isaac
deployment (shadow_dynamics.imagination: every CHUNK = 6 control steps at 20 Hz the policy's x in [-1, 1]^20 sets a
command g = prev + max_delta * x, the applied targets move toward g at the slew limit), the same observation
(imagination.cube_obs), the same goal distribution, and Isaac's native in-hand reward evaluated per control step and
summed over the chunk: -10 * distance to the in-hand point (m) + 1 / (rotation error + 0.1) - 0.0002 * |x|^2 per
step, +250 once when the error is < 0.1 rad (then a new goal, the episode continues), fall penalty 0 with termination
(no bootstrap) when the cube is >= fall_dist from the in-hand point; the sum is scaled by --reward-scale. Episodes are
--episode-chunks decisions; time-outs bootstrap with V(s_{t+1}) of the pre-reset observation (experiment 15 rule).
Resets are per environment (hand default pose + noise, cube at its default position with a random orientation, no
settling). Saves policy.pt in the pipeline's format (deployable with evaluate_shadow_cube_mpc.py / _ppo.py),
train_log.jsonl, and ckpt_last.pt every --save-every iterations (resume with --resume).

  ./run_python.sh scripts/train/train_shadow_ppo_isaac.py --headless --device cuda:0 --output artifacts/ppo_cube_isaac_v1 --num-envs 4096 --iterations 1000

Experiment 40, --hand-model <path>: the hybrid environment of experiment 39. A 48-dim hand model (e.g. the identified equation + residual
+ observer, artifacts/exp37a/res_phmro_f100/best.pt) predicts each decision's 6-step hand trajectory from the last 4 hand states and 3
commands; it is imposed on Isaac's hand (joint state written before every physics substep, linearly interpolated, PD target = predicted
joints) and only the cube is simulated. The policy observes the model's hand state; the command interface (prev = slewed command) is
unchanged. --frozen-policy <policy.pt>: no learning, roll that policy (sampled actions, as during training) for --iterations x
--steps-per-env decisions and log the episode statistics (the go/no-go check of P inside the hybrid).
--replay soft (experiment 40, after the kinematic hybrid dropped the cube 98% of the time): the model's predicted joint trajectory becomes the
position and velocity target of Isaac's hand drive, with the actuated joints' stiffness scaled by --track-gain and damping by its square root,
so the hand tracks the model in free space but contact can push back. The model's history is the true Isaac hand state (re-anchored
every decision, as in experiment 39), the residual observer sees the real discrepancies, and the policy observes the Isaac hand.
"""
import argparse, copy, json, os, sys, time, traceback, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from shadow_dynamics.resources import restrict_simulator_gpu
PHYSICAL_GPU = restrict_simulator_gpu()
from omni.isaac.lab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", required=True); p.add_argument("--num-envs", type=int, default=4096); p.add_argument("--iterations", type=int, default=1000)
p.add_argument("--steps-per-env", type=int, default=24, help="decisions per environment per PPO update"); p.add_argument("--episode-chunks", type=int, default=33, help="episode length in decisions (33 = 9.9 s)")
p.add_argument("--seed", type=int, default=0); p.add_argument("--resume", action="store_true", help="continue from <output>/ckpt_last.pt")
p.add_argument("--save-every", type=int, default=100)
p.add_argument("--hand-noise-deg", type=float, default=8.0, help="reset noise on the 20 actuated joints (the collection's start distribution)")
p.add_argument("--slew-deg", type=float, default=80.0); p.add_argument("--max-delta", type=float, default=25.0)
p.add_argument("--fall-dist", type=float, default=0.24); p.add_argument("--success", type=float, default=0.1)
p.add_argument("--dist-scale", type=float, default=-10.0); p.add_argument("--rot-scale", type=float, default=1.0); p.add_argument("--rot-eps", type=float, default=0.1)
p.add_argument("--action-penalty", type=float, default=-0.0002); p.add_argument("--goal-bonus", type=float, default=250.0); p.add_argument("--fall-penalty", type=float, default=0.0)
p.add_argument("--reward-scale", type=float, default=0.1)
p.add_argument("--hand-model", default=None, help="experiment 40: hybrid env, hand from this 48-dim hand model, cube from Isaac")
p.add_argument("--replay", choices=["kinematic", "soft"], default="kinematic", help="experiment 40: how the model hand is applied")
p.add_argument("--track-gain", type=float, default=10.0, help="experiment 40, soft replay: stiffness multiplier (damping x sqrt)")
p.add_argument("--frozen-policy", default=None, help="experiment 40: evaluate this policy.pt in the env (sampled actions), no training")
p.add_argument("--init-noise", type=float, default=0.5); p.add_argument("--entropy", type=float, default=0.001); p.add_argument("--gamma", type=float, default=0.99)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.physics_hz = 120; args.control_hz = 20; args.mode = "cube"
launcher = AppLauncher(args, multi_gpu=False); app = launcher.app
import numpy as np
import torch
sys.modules.setdefault("git", types.ModuleType("git"))   # rsl_rl imports GitPython only to log code state
from rsl_rl.runners import OnPolicyRunner
from shadow_dynamics.env import ResearchShadowEnv, make_cfg
from shadow_dynamics.imagination import chunk_targets, cube_obs, random_goal_quat, CHUNK
from shadow_dynamics.cube_world_data import quat_to_matrix, rotation_distance
from shadow_dynamics.flow_model import load_forward

HZ = 20


class IsaacCubeEnv:
    """rsl_rl VecEnv-compatible cube reorientation in Isaac with the pipeline's decision interface and the native reward."""

    def __init__(self, env, a, seed, hand_model=None):
        self.env, self.a, dev = env, a, env.device; N = env.num_envs; self.dev = dev; self.hm = hand_model
        self.S = torch.zeros(N, 4, 48, device=dev); self.A = torch.zeros(N, 3, 20, device=dev)   # experiment 40: model hand history (deg, deg/s) and commands
        self.soft = hand_model is not None and a.replay == "soft"; self.track = []   # soft replay: per-decision hand-vs-model RMSE (deg)
        self.num_envs, self.num_actions, self.num_privileged_obs, self.max_episode_length = N, 20, None, a.episode_chunks
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.active = env.active; self.low, self.high = torch.rad2deg(env.low), torch.rad2deg(env.high)
        self.origins = env.scene.env_origins; self.in_hand_w = env.in_hand_pos + self.origins
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=dev)
        self.goal = torch.zeros(N, 4, device=dev); self.prev = torch.rad2deg(env.last_target).clone()
        self.ep_success = torch.zeros(N, dtype=torch.bool, device=dev); self.ep_goals = torch.zeros(N, device=dev); self.ep_first = torch.full((N,), float("nan"), device=dev)
        self.finished = []   # (n, success_any, goals, dropped, timeout, first_time_s) sums of finished episodes since the last flush
        self.all_ids = torch.arange(N, device=dev)
        self._reset(self.all_ids); self.num_obs = self.obs().shape[1]

    def _kin_update(self):
        psv = getattr(self.env.sim, "physics_sim_view", None)
        if psv is not None and hasattr(psv, "update_articulations_kinematic"): psv.update_articulations_kinematic()

    def _reset(self, ids):
        n = len(ids)
        if n == 0: return
        env = self.env
        q = env.hand.data.default_joint_pos[ids].clone()
        q[:, self.active] = (q[:, self.active] + torch.deg2rad(torch.randn(n, 20, generator=self.gen, device=self.dev) * self.a.hand_noise_deg)).clamp(env.low, env.high)
        env.hand.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        obj = env.object.data.default_root_state[ids].clone(); obj[:, :3] += self.origins[ids]; obj[:, 3:7] = random_goal_quat(n, self.gen, self.dev); obj[:, 7:] = 0
        env.object.write_root_state_to_sim(obj, env_ids=ids)
        env.last_target[ids] = q[:, self.active]; env.hand.set_joint_position_target(env.last_target, joint_ids=self.active); self._kin_update()
        self.prev[ids] = torch.rad2deg(q[:, self.active]); self.goal[ids] = random_goal_quat(n, self.gen, self.dev)
        self.S[ids] = torch.cat([torch.rad2deg(q), torch.zeros_like(q)], -1)[:, None].expand(-1, 4, -1); self.A[ids] = self.prev[ids][:, None].expand(-1, 3, -1)
        self.episode_length_buf[ids] = 0; self.ep_success[ids] = False; self.ep_goals[ids] = 0; self.ep_first[ids] = float("nan")

    def cube13(self):
        o = self.env.object.data.root_state_w
        return torch.cat([(o[:, :3] - self.in_hand_w) * 1000, o[:, 3:7], o[:, 7:10] * 1000, torch.rad2deg(o[:, 10:13])], -1)

    def obs(self):
        env = self.env
        if self.hm is not None and not self.soft: return cube_obs(self.S[:, -1, :24], self.S[:, -1, 24:48], self.cube13(), self.goal, self.prev)
        return cube_obs(torch.rad2deg(env.hand.data.joint_pos), torch.rad2deg(env.hand.data.joint_vel), self.cube13(), self.goal, self.prev)

    def get_observations(self):
        return self.obs(), {"observations": {}}

    def reset(self):
        self._reset(self.all_ids); return self.get_observations()

    @torch.no_grad()
    def step(self, actions):
        a, env, N = self.a, self.env, self.num_envs
        x = actions.clamp(-1, 1); g = (self.prev + a.max_delta * x).clamp(self.low, self.high)
        queue = chunk_targets(self.prev, g, a.slew_deg, self.low, self.high, steps=CHUNK)
        rew = torch.zeros(N, device=self.dev); fell = torch.zeros(N, dtype=torch.bool, device=self.dev); hits = torch.zeros(N, device=self.dev)
        if self.hm is not None:   # experiment 40: the decision's hand trajectory from the model (12-step window, commands held after the chunk)
            fa = torch.cat([queue, queue[:, -1:].expand(-1, 12 - CHUNK, -1)], 1)
            Yh = self.hm.sample(self.S, self.A, fa, K=1)[0][:, :CHUNK]; n_sub = env.cfg.decimation
        for j in range(CHUNK):
            if self.soft:   # model trajectory as drive target (interpolated per substep); the hand state stays Isaac's
                qa, va = torch.deg2rad(self.S[:, -1, :24])[:, self.active] if j == 0 else q1, torch.deg2rad(self.S[:, -1, 24:48])[:, self.active] if j == 0 else v1
                q1, v1 = torch.deg2rad(Yh[:, j, :24])[:, self.active], torch.deg2rad(Yh[:, j, 24:48])[:, self.active]; k = [1]
                def after():
                    k[0] += 1; w = min(k[0] / n_sub, 1.0)
                    env.hand.set_joint_position_target(qa + w * (q1 - qa), joint_ids=self.active); env.hand.set_joint_velocity_target(va + w * (v1 - va), joint_ids=self.active)
                env.hand.set_joint_velocity_target(va + (v1 - va) / n_sub, joint_ids=self.active); env.advance(qa + (q1 - qa) / n_sub, on_substep=after)
                env.hand.set_joint_velocity_target(torch.zeros_like(v1), joint_ids=self.active)
                hq, hv = torch.rad2deg(env.hand.data.joint_pos), torch.rad2deg(env.hand.data.joint_vel)
                self.track.append(float((hq - Yh[:, j, :24]).square().mean().sqrt()))
                self.prev = queue[:, j].clone(); self.S = torch.cat([self.S[:, 1:], torch.cat([hq, hv], -1)[:, None]], 1); self.A = torch.cat([self.A[:, 1:], self.prev[:, None]], 1)
            elif self.hm is None: env.advance(torch.deg2rad(queue[:, j])); self.prev = torch.rad2deg(env.last_target).clone()
            else:
                q0, v0 = torch.deg2rad(self.S[:, -1, :24]), torch.deg2rad(self.S[:, -1, 24:48]); q1, v1 = torch.deg2rad(Yh[:, j, :24]), torch.deg2rad(Yh[:, j, 24:48]); k = [0]
                def put():
                    w = k[0] / n_sub; env.hand.write_joint_state_to_sim(q0 + w * (q1 - q0), v0 + w * (v1 - v0)); self._kin_update()
                def after():
                    k[0] += 1
                    if k[0] < n_sub: put()
                put(); env.advance(q1[:, self.active], on_substep=after)
                self.prev = queue[:, j].clone(); self.S = torch.cat([self.S[:, 1:], Yh[:, j][:, None]], 1); self.A = torch.cat([self.A[:, 1:], self.prev[:, None]], 1)
            c = self.cube13(); dist = c[:, :3].norm(dim=-1) / 1000; rot = rotation_distance(quat_to_matrix(c[:, 3:7]), quat_to_matrix(self.goal))
            held = ~fell
            step_r = a.dist_scale * dist + a.rot_scale / (rot + a.rot_eps) + a.action_penalty * actions.square().sum(-1)   # native per-step terms (penalty on the raw command)
            rew += torch.where(held, step_r, torch.zeros_like(step_r))
            hit = held & (rot < a.success)
            if hit.any():
                rew[hit] += a.goal_bonus; hits[hit] += 1; self.goal[hit] = random_goal_quat(int(hit.sum()), self.gen, self.dev)
                t_now = (self.episode_length_buf * CHUNK + j + 1) / HZ
                self.ep_first = torch.where(hit & torch.isnan(self.ep_first), t_now.float(), self.ep_first)
            new_fall = held & (dist >= a.fall_dist)
            rew[new_fall] += a.fall_penalty; fell |= new_fall
        rew = rew * a.reward_scale
        self.ep_success |= hits > 0; self.ep_goals += hits
        self.episode_length_buf += 1
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~fell
        done = timeout | fell
        terminal_obs = self.obs()   # pre-reset: the trainer bootstraps time-outs with V(s_{t+1})
        if done.any():
            d = done; ft = self.ep_first[d]
            self.finished.append((int(d.sum()), float(self.ep_success[d].float().sum()), float(self.ep_goals[d].sum()), float(fell[d].float().sum()), float(timeout[d].float().sum()),
                                  float(torch.nan_to_num(ft, nan=0.0).sum()), int((~torch.isnan(ft)).sum())))
        self._reset(done.nonzero(as_tuple=False).squeeze(-1))
        info = {"observations": {}, "time_outs": timeout, "terminal_obs": terminal_obs,
                "log": {"hits": hits.mean(), "fell": fell.float().mean(), "rot_err": rot.mean()}}
        return self.obs(), rew, done.long(), info

    def flush(self):
        if not self.finished: return dict(episodes=0)
        s = np.array(self.finished, dtype=np.float64).sum(0); self.finished = []
        n = s[0]; tr = float(np.mean(self.track)) if self.track else None; self.track = []
        return dict(track_rmse_deg=tr, episodes=int(n), success=s[1] / n, goals_per_episode=s[2] / n, dropped=s[3] / n, timeout=s[4] / n, mean_time_to_success=(s[5] / s[6]) if s[6] else None)


def main():
    torch.set_num_threads(1); assert sorted(os.sched_getaffinity(0)) == __import__('shadow_dynamics.resources', fromlist=['budget_cpus']).budget_cpus()
    out = Path(args.output); ck_path = out / "ckpt_last.pt"
    if args.resume: assert ck_path.exists(), f"--resume needs {ck_path}"
    else: out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    cfg = make_cfg(args); cfg.object_cfg.spawn.rigid_props.kinematic_enabled = False
    env = ResearchShadowEnv(cfg); dev = env.device
    try:
        env.configure_research("cube")
        hm = load_forward(args.hand_model, dev)[0] if args.hand_model else None
        if hm is not None and args.replay == "soft":   # stiffer drive on the actuated joints so the hand tracks the model's trajectory
            jn = env.hand.joint_names; g_ = args.track_gain
            print("soft replay gains", env.set_hand_physics(dict(stiffness_scale={jn[i]: g_ for i in list(env.active)}, damping_scale={jn[i]: g_ ** 0.5 for i in list(env.active)}))["stiffness"][:4], flush=True)
        venv = IsaacCubeEnv(env, args, args.seed, hm)
        if args.frozen_policy:   # experiment 40: roll a trained policy, no learning
            if args.frozen_policy == "zero":   # diagnostic: hold the current command
                ac = types.SimpleNamespace(act=lambda o: torch.zeros(len(o), 20, device=dev)); nm = lambda o: o
            else:
                pol = torch.load(args.frozen_policy, map_location=dev, weights_only=False); ac, nm = pol["actor_critic"].to(dev).eval(), pol["obs_normalizer"].to(dev).eval()
                assert pol["slew_deg_s"] == args.slew_deg and pol["max_delta"] == args.max_delta
            log = open(out / "frozen_log.jsonl", "a"); obs, _ = venv.get_observations(); t0 = time.time(); recs = []
            for it in range(1, args.iterations + 1):
                with torch.no_grad():
                    for _ in range(args.steps_per_env): obs, rew, dones, info = venv.step(ac.act(nm(obs)))
                rec = dict(iteration=it, elapsed_s=time.time() - t0, **{f"ep_{k}": v for k, v in venv.flush().items()}); recs.append(rec); log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            fin = [r for r in recs[2:] if r["ep_episodes"]]   # skip the first two iterations (all episodes start together)
            n = sum(r["ep_episodes"] for r in fin); summ = {k: sum(r[f"ep_{k}"] * r["ep_episodes"] for r in fin) / n for k in ("success", "dropped", "goals_per_episode")}
            summ.update(episodes=n, seconds_per_iteration=(time.time() - t0) / args.iterations, policy=args.frozen_policy, hand_model=args.hand_model)
            (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("PPO_FROZEN_COMPLETE", json.dumps(summ), flush=True); return
        train_cfg = dict(num_steps_per_env=args.steps_per_env, save_interval=10 ** 9, empirical_normalization=True,
                         policy=dict(class_name="ActorCritic", init_noise_std=args.init_noise, actor_hidden_dims=[512, 512, 256, 128], critic_hidden_dims=[512, 512, 256, 128], activation="elu"),
                         algorithm=dict(class_name="PPO", value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=args.entropy, num_learning_epochs=5, num_mini_batches=4,
                                        learning_rate=3e-4, schedule="adaptive", gamma=args.gamma, lam=0.95, desired_kl=0.016, max_grad_norm=1.0))
        runner = OnPolicyRunner(venv, copy.deepcopy(train_cfg), log_dir=None, device=str(dev)); alg, norm = runner.alg, runner.obs_normalizer
        start_it, elapsed0 = 1, 0.0
        if args.resume:
            ck = torch.load(ck_path, map_location=dev, weights_only=False)
            alg.actor_critic.load_state_dict(ck["actor_critic"]); alg.optimizer.load_state_dict(ck["optimizer"]); norm.load_state_dict(ck["obs_normalizer"])
            alg.learning_rate = ck["learning_rate"]
            for g_ in alg.optimizer.param_groups: g_["lr"] = ck["learning_rate"]
            start_it, elapsed0 = ck["iteration"] + 1, ck["elapsed_s"]; venv.gen.manual_seed(args.seed + ck["iteration"])
            print(f"resumed from iteration {ck['iteration']}", flush=True)
        log = open(out / "train_log.jsonl", "a"); started = time.time() - elapsed0; alg.actor_critic.train(); norm.train()
        obs, _ = venv.get_observations(); obs = norm(obs)
        decisions_per_it = args.steps_per_env * args.num_envs

        def save(it, rec):
            alg.actor_critic.eval(); norm.eval(); alg.actor_critic.distribution = None   # the last Normal holds non-leaf tensors that cannot be deep-copied
            policy = dict(actor_critic=copy.deepcopy(alg.actor_critic).cpu(), obs_normalizer=copy.deepcopy(norm).cpu(), model=None, trained_in=("hybrid:" + args.hand_model) if args.hand_model else "isaac", hand_model=args.hand_model, reward="native",
                          chunk=CHUNK, slew_deg_s=args.slew_deg, max_delta=args.max_delta, cost=None, terminator=None, bootstrap="next",
                          obs="cube_obs(q, qd, cube13, goal_quat, prev), see shadow_dynamics.imagination", episode_chunks=args.episode_chunks, args=vars(args), iterations=it,
                          reward_terms=dict(dist_scale=args.dist_scale, rot_scale=args.rot_scale, rot_eps=args.rot_eps, action_penalty=args.action_penalty, goal_bonus=args.goal_bonus,
                                            fall_penalty=args.fall_penalty, fall_dist_m=args.fall_dist, success_rad=args.success, reward_scale=args.reward_scale, per_step="summed over the 6 control steps of a decision"))
            torch.save(policy, out / "policy.pt"); torch.save(policy, out / f"policy_it{it}.pt")
            torch.save(dict(actor_critic=alg.actor_critic.state_dict(), optimizer=alg.optimizer.state_dict(), obs_normalizer=norm.state_dict(), learning_rate=alg.learning_rate,
                            iteration=it, elapsed_s=time.time() - started, last=rec), ck_path)
            alg.actor_critic.train(); norm.train()

        rec = None
        for it in range(start_it, args.iterations + 1):
            t_it = time.time(); rews = []
            with torch.no_grad():
                for _ in range(args.steps_per_env):
                    actions = alg.act(obs, obs); obs, rew, dones, info = venv.step(actions); obs = norm(obs); rews.append(float(rew.mean()))
                    to = info.pop("time_outs")   # rsl_rl would add gamma*V(s_t); use the true next state instead
                    if to.any():
                        o_t = (info["terminal_obs"][to] - norm._mean) / (norm._std + norm.eps)
                        rew = rew.clone(); rew[to] += alg.gamma * alg.actor_critic.evaluate(o_t).squeeze(-1)
                    alg.process_env_step(rew, dones, info)
                alg.compute_returns(obs)
            vloss, sloss = alg.update(); ep = venv.flush()
            rec = dict(iteration=it, reward=float(np.mean(rews)), value_loss=float(vloss), surrogate_loss=float(sloss), noise_std=float(alg.actor_critic.std.mean()), learning_rate=float(alg.learning_rate),
                       decisions=it * decisions_per_it, iteration_s=time.time() - t_it, elapsed_s=time.time() - started, **{f"ep_{k}": v for k, v in ep.items()})
            log.write(json.dumps(rec) + "\n"); log.flush()
            if it % 10 == 0 or it == start_it: print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            if it % args.save_every == 0 or it == args.iterations: save(it, rec)
        (out / "summary.json").write_text(json.dumps(dict(final=rec, args=vars(args)), indent=1))
        print("PPO_ISAAC_COMPLETE", json.dumps(rec), flush=True)
    finally:
        env.close()


try: main()
except Exception:
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
finally: app.close()
