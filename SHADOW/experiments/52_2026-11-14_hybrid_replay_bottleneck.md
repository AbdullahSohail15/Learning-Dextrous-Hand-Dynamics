# Experiment 52: is the hybrid's kinematic hand replay the cube bottleneck?

Date: 14 Nov 2026. Results: `result_summaries/2026-11-14_hybrid_replay_bottleneck_results.md` (when done). Follows exp 51
(`result_summaries/2026-11-14_full_stack_on_cube_results.md`).

## Why

The user asked for an approach that beats `cem:P:hyb_res_phmro` (exp 51: 74.6%, oracle CEM 87.1%). What is known:
- the planner is not the limit (exp 39: CEM ≈ one-shot; with the true simulator both reach ~85%);
- hand-model error is not the limit (exp 41: contact data halved cube-task hand error 4.9 → 2.7°, success unchanged);
- exp 41 left one suspect: the hybrid imposes the predicted hand **kinematically** (joint state written every substep), so in planning
  the fingers move through the cube with unbounded force and sometimes fling it (cube position error p90 142 mm at 0.6 s). Exp 40 showed
  the same: P in the kinematic hybrid drops 98%, with soft replay (drive target, stiffness × 10) 5.6%.

## Arms (evaluator `scripts/evaluate_shadow_cube_mpc.py`, new flags `--true-hand`, `--soft-hand-models`, `--track-gain`)

| arm | hand in the planner's branches | question |
|---|---|---|
| `ppo:P` | – | reference |
| `cem:P:oracle:safe` | Isaac (ceiling) | reference |
| `cem:P:hyb_res_phmro:safe` | exp-37a eq+res+observer, kinematic | exp 39/51 baseline |
| `cem:P:hyb_true:safe` | **true** future hand (from the oracle branches), kinematic | perfect hand model under kinematic replay |
| `cem:P:hyb_soft_res_phmro:safe` | res_phmro prediction as drive target, stiffness × 10, damping × √10 (branch hands only, restored after) | does compliant replay beat kinematic? |
| `cem:P:hyb_soft_true:safe` | true future hand, soft | soft replay with a perfect hand |
| `mpc:P:oracle:safe` | Isaac | ranking diagnostic: Spearman of every hybrid's candidate scores vs the oracle's |

`hyb_true` uses no learned model, so it is a diagnostic only (it runs the oracle branches first). `hyb_soft_res_phmro` is the candidate
method: it needs only the hand model, like `hyb_res_phmro`. Soft replay reports the branch hand's tracking error to the prediction.

## Hypotheses (pre-registered)

| # | hypothesis | prob. |
|---|---|---|
| H1 | `hyb_true` < oracle CEM − 5 (kinematic replay costs ≥ 5 points even with a perfect hand) | 65% |
| H2 | `hyb_soft_res_phmro` > `hyb_res_phmro` (paired, 360 cases, p < 0.05) | 40% |
| H3 | `hyb_soft_true` ≥ `hyb_true` | 60% |
| H4 | the soft hybrids' Spearman vs oracle exceeds the kinematic ones' | 55% |

Reading: H1 true and H2 true → the replay was the bottleneck and soft replay is the fix. H1 true, H2 false → replay matters but the soft
version or res_phmro's prediction under contact is not good enough (tune gain). H1 false → kinematic replay is fine and the gap is the
hand model; then contact-aware hand prediction is the lever.

## Benchmark, chain, budget

Seeds 790000/791000/792000 × 120 cases, 10 s, cost `safe`, 32 candidates, `--oracle-candidates 32`, as exps 39/51. GPU: physical 6 (user, 14 Nov; GPU 2, first chosen, was full with the user's YOLO training). Chain `logs/run_exp52.sh` (tmux `exp52`): smoke (3 cases per bin, 3 s), then the 3 seeds (all 7 arms in one process per seed, so
exactly paired), analysis `scripts/analyze_exp39.py` with pairs. Budget ≈ 25–35 min per seed. Nothing committed.

## Not in this experiment (later, if soft replay works)

Penetration cost inside CEM or a CBF candidate filter (safety without modifying the chosen commands); soft-replay gain sweep (5/10/20).
