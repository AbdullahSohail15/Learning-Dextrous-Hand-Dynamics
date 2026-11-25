# Experiment 52 results: is the hybrid's kinematic hand replay the cube bottleneck?

Plan: `experiments/52_2026-11-14_hybrid_replay_bottleneck.md`. Date: 14 Nov 2026. GPU 6. 3 seeds × 120 paired cases (all arms of a seed in
one process, so exactly paired), 10 s, success < 0.1 rad, cost `safe`, 32 candidates. Chain `logs/run_exp52.sh`, outputs
`artifacts/exp52/{smoke,eval_s790000,eval_s791000,eval_s792000,analysis}`. Nothing committed.

## Method

Kinematic replay (exps 39–51): the planner's branch hands have their joint state overwritten with the predicted trajectory every substep;
fingers move through the cube with unbounded force. Soft replay (new, `--soft-hand-models`, after exp 40b): the predicted trajectory is the
branch hand's drive target (position and velocity, interpolated per substep), stiffness × 10 and damping × √10 on the actuated joints of
the branch hands only, restored after each call; Isaac simulates hand and cube together, so contact pushes back. `hyb_true`
(`--true-hand`): the true future hand from the oracle branches, a perfect hand model (diagnostic, not a method). Soft tracking error of the
branch hand to the prediction (smoke): true hand 0.97°, `res_phmro` 3.2°.

## Results (360 cases)

| arm | success [95% CI] | per seed | drops |
|---|---|---|--:|
| `ppo:P` | 40.8 [35.9, 46.0] | 35.8 / 46.7 / 40.0 | 0.8 |
| `cem:P:oracle:safe` | 83.6 [79.4, 87.1] | 85.8 / 84.2 / 80.8 | 0.3 |
| `mpc:P:oracle:safe` | 81.7 [77.3, 85.3] | 83.3 / 81.7 / 80.0 | 0.6 |
| `cem:P:hyb_res_phmro:safe` (kinematic) | 75.0 [70.3, 79.2] | 77.5 / 71.7 / 75.8 | 1.7 |
| `cem:P:hyb_true:safe` (kinematic) | 77.8 [73.2, 81.8] | 75.8 / 79.2 / 78.3 | 1.1 |
| **`cem:P:hyb_soft_res_phmro:safe`** | **79.7 [75.3, 83.6]** | 80.8 / 83.3 / 75.0 | **0.0** |
| `cem:P:hyb_soft_true:safe` | 81.7 [77.3, 85.3] | 80.8 / 86.7 / 77.5 | 0.8 |

Paired (exact McNemar):

| comparison | Δ points | only A / only B | p |
|---|--:|--:|--:|
| true hand (kinematic) vs oracle | −5.8 | 17 / 38 | 0.006 |
| true hand vs `res_phmro` (both kinematic) | +2.8 | 41 / 31 | 0.29 |
| **soft `res_phmro` vs kinematic `res_phmro`** | **+4.7** | 45 / 28 | **0.060** |
| soft true vs kinematic true | +3.9 | 38 / 24 | 0.098 |
| soft `res_phmro` vs oracle | −3.9 | 17 / 31 | 0.059 |
| soft true vs oracle | −1.9 | 18 / 25 | 0.36 |

Candidate ranking vs the oracle (`mpc:P:oracle` states, 12,230 decisions, 32 candidates):

| model | Spearman median (mean) | top-1 agreement | median regret |
|---|--:|--:|--:|
| `hyb_res_phmro` (kinematic) | 0.663 (0.602) | 0.396 | 0.124 |
| `hyb_true` (kinematic) | 0.941 (0.851) | 0.619 | 0 |
| **`hyb_soft_res_phmro`** | **0.943 (0.880)** | 0.633 | 0 |
| `hyb_soft_true` | 0.972 (0.920) | 0.718 | 0 |
| `A_all` (learned 63-d cube world model, exp 34) | 0.943 (0.876) | 0.608 | 0 |

## Hypotheses

| # | hypothesis | outcome |
|---|---|---|
| H1 | `hyb_true` < oracle − 5 | **held** (−5.8, p = 0.006): even a perfect hand loses ~6 points under kinematic replay |
| H2 | soft `res_phmro` > kinematic (p < 0.05) | **narrowly failed**: +4.7, p = 0.060 (seeds +3.3 / +11.7 / −0.8) |
| H3 | soft true ≥ kinematic true | **held** (+3.9, p = 0.10) |
| H4 | soft hybrids rank better than kinematic | **held, strongly**: Spearman 0.94 vs 0.66 (res_phmro), 0.97 vs 0.94 (true) |

## Reading

- Soft replay fixes the hybrid's candidate ranking: with the same learned hand model, Spearman vs the oracle rises from 0.66 to 0.94, top-1
  agreement from 40 to 63%, and the median regret goes to 0. This is the clearest result of the experiment (exp 39 measured 0.66 too).
- It turns that into +4.7 points of success (79.7 vs 75.0%, p = 0.06) and zero drops; the best learned-model planner so far on this
  benchmark, 3.9 points below the oracle (p = 0.06). The success gain is smaller than the ranking gain suggests and varies by seed
  (seed 792000: −0.8).
- The hand model is not the limit: under soft replay the perfect hand adds 2.0 points over `res_phmro` (n.s.); under kinematic replay 2.8
  (n.s.). The kinematic replay was the main defect in the hybrid (exp 41's open question).
- Why the ranking gain does not fully reach success: ranking is measured on P's one-shot candidates (`mpc` states); good ranking is
  necessary but the soft branch hand still differs from the real one (stiffer drive, 1–3° tracking error), and `hyb_true` already ranks at
  0.94 yet reaches only 77.8%, so part of the remaining gap is how kinematic/soft branches differ from the real hand's dynamics during
  contact, not ranking alone.
- Runtime: soft replay costs the same as kinematic (≈ 120 s per 120-case seed on a free GPU).

## Next

1. Soft-replay gain sweep (5 / 10 / 20 / 40) and velocity-target on/off: the gain trades tracking of the prediction against compliance.
2. More seeds for the soft vs kinematic test (p = 0.06 on 360 cases).
3. Safety without modifying the chosen commands: penetration cost inside CEM, or a CBF candidate filter, on top of soft replay.
4. PPO trained in the soft hybrid (exp 40b soft replay trained a policy from scratch; with the better ranking, a P-initialized fine-tune).

## Code

`scripts/evaluate_shadow_cube_mpc.py`: `--true-hand` (label `true`, bound to the oracle branches), `--soft-hand-models` (labels
`soft_<label>`), `--track-gain`; `soft_track_deg` in run summaries. Backup of the pre-exp-52 evaluator in the session scratchpad.
