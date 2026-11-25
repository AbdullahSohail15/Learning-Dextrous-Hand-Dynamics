# Experiment 50 results: explicit planning and control on the self-identified hand model

Plan: `experiments/50_2026-11-05_explicit_planning_and_control.md`. Chain `logs/run_exp50.sh`, outputs `artifacts/exp50/gestures_s{610000,611000,612000}`, tables `scripts/analyze_exp50.py` → `artifacts/exp50/summary.json`.

## Setup

- Shadow Hand in Isaac, gravity on. Model: the exp-49 Li-form model identified from self-exploration only (A4, kinematics-only prior).
- Gesture benchmark on the v3 library: 15 gestures × 10 starts × 3 task seeds = 450 tasks per controller. Success = final 24-joint RMSE < 2°.
- New code:
  - `shadow_dynamics/hand_cspace.py` (configuration space, self-collision obstacles, clearance);
  - `rrt_connect.py` (batched RRT-Connect, shortcutting, path follower);
  - `ilqr.py` (iLQR-MPC with TVLQR feedback);
  - `cbf.py` (discrete-time CBF safety filter);
  - evaluator options `--exp50-runs` and `--measure-validity`.

## Results (3 seeds, 450 tasks each)

| controller | course topic | success | per seed | excl. thumb_middle_pinch | tip < 5 mm | mean joint err | self-contact steps | penetration steps |
|---|---|--:|---|--:|--:|--:|--:|--:|
| direct | – | 74.4% | 74.7 / 72.7 / 76.0 | 79.8% | 0.0% | 2.09° | 48.1% | 6.85% |
| cem (exp 49) | MPC, sampling-based | 85.3% | 84.7 / 84.0 / 87.3 | 91.4% | 63.1% | 1.66° | 41.8% | 4.92% |
| rrt_direct | sampling-based path | 76.2% | 76.0 / 74.7 / 78.0 | 81.7% | 0.0% | 1.91° | 47.1% | 6.13% |
| rrt_cem | path + CEM | 78.7% | 78.7 / 77.3 / 80.0 | 84.3% | 53.6% | 1.78° | 32.4% | 3.42% |
| ilqr | trajectory optimisation + TVLQR | 94.4% | 94.0 / 96.0 / 93.3 | 98.3% | 90.2% | 1.02° | 42.7% | 3.09% |
| **rrt_ilqr** | path + iLQR-MPC | **99.8%** | 100 / 100 / 99.3 | 99.8% | 96.4% | 0.72° | 41.2% | 0.91% |
| cem_cbf | CEM + safety filter | 86.2% | 86.0 / 84.7 / 88.0 | 92.4% | 64.9% | 1.58° | 39.7% | 3.76% |
| **rrt_ilqr_cbf** | full stack | **100.0%** | 100 / 100 / 100 | 100% | 99.6% | 0.68° | 40.1% | 0.77% |

Planning time is about 11.4–11.8 s per replan for all 150 tasks batched (CEM and iLQR alike, under heavy shared-CPU load). RRT takes 7–14 s per batch of 150 tasks; 26–39 tasks per seed need it (the straight line collides). iLQR fallbacks and non-finite commands: 0 in all runs.

### Per gesture (success %, 3 seeds)

| gesture | direct | cem | ilqr | rrt_ilqr | rrt_ilqr_cbf |
|---|--:|--:|--:|--:|--:|
| thumb_middle_pinch | 0 | 0 | 40 | **100** | **100** |
| thumbs_up | 0 | 23 | 97 | 97 | **100** |
| victory | 0 | 100 | 100 | 100 | 100 |
| thumb_ring_pinch | 73 | 87 | 83 | **100** | **100** |
| ok_sign | 70 | 93 | 97 | **100** | **100** |
| cylindrical_grasp | 87 | 87 | 100 | 100 | 100 |
| thumb_index_pinch | 87 | 90 | 100 | 100 | 100 |
| other 8 gestures | 100 | 100 | 100 | 100 | 100 |

The two gestures that capped exp 49 below 95% are solved:
- thumb_middle_pinch (path-dependent thumb–index collision) is solved by RRT + iLQR;
- thumbs_up (borderline about 2°) is solved by iLQR.

### Paired tests (exact McNemar, 450 tasks)

| comparison | only a succeeds | only b succeeds | p | Δ self-contact (pp) | Δ penetration (pp) |
|---|--:|--:|--:|--:|--:|
| ilqr vs cem | 47 | 6 | 6e-9 | +0.9 | −1.8 |
| rrt_ilqr vs ilqr | 25 | 1 | 8e-7 | −1.5 | −2.2 |
| rrt_ilqr vs cem | 65 | 0 | 5e-20 | −0.5 | −4.0 |
| rrt_direct vs direct | 8 | 0 | 0.008 | −1.0 | −0.7 |
| rrt_cem vs cem | 5 | 35 | 1e-6 | −9.4 | −1.5 |
| cem_cbf vs cem | 9 | 5 | 0.42 | −2.1 | −1.2 |
| rrt_ilqr_cbf vs rrt_ilqr | 1 | 0 | 1 | −1.1 | −0.1 |

## Hypotheses

- **H1 (RRT helps the local controller and lowers contact): partly held.**
  - It holds for iLQR (+25/−1, p 8e-7, penetration −2.2 pp) and for direct (+8/−0).
  - It fails for CEM: rrt_cem is worse than CEM (5 vs 35, p 1e-6), although it has the least self-contact (32%). CEM searches one held command pose with a warm-started mean; a reference moving along the path makes that search lag.
  - The self-contact reduction is small except with CEM.
- **H2 (iLQR within 3 points of CEM at equal time): exceeded.** iLQR is 9 points better (94.4 vs 85.3%, p 6e-9), with 0.6° lower error and 27 points more fingertip success, at the same planning time. iLQR optimises the whole 12-step command sequence with the model's gradients and corrects the measured state with TVLQR gains between replans. CEM samples one held command pose. The fine offsets (gravity sag, tendon coupling) are what decide the 2° threshold.
- **H3 (CBF cuts self-contact ≥ 50% at ≤ 3 points): failed on contact, no success cost.**
  - Self-contact falls only 2.1 pp (CEM) and 1.1 pp (full stack).
  - Penetration falls 24% for CEM (4.92 → 3.76%) and 15% for the full stack (0.91 → 0.77%).
  - Self-contact is the wrong measure here: pinch, fist and grasp goals require touching, so about 40% contact is intended. Penetration is the meaningful safety measure.
  - The filter is limited by the 80 °/s slew limit (gate G3).

## Caveats

- **Not real time.** One replan takes about 11.6 s for 150 tasks on a shared, heavily loaded machine; a 4-task smoke test took about 9.5 s for iLQR, so the cost is mostly sequential overhead (12 time steps of finite-difference Jacobians, 69 model calls each), not batch size. A real hand needs a replan in about 0.15 s. Options: analytic or autograd Jacobians, fewer iterations, a shorter horizon, or a compiled model.
- **Jacobians:** finite differences with about 15% step-size sensitivity (median).
- **Goals:** v3 replaced targets (exp 49) are poses the simulator can hold from neutral. 2 of 150 RRT problems per seed fell back to the straight line.
- **Model:** A4 (25 valid minutes). The minimum-data model (m1, 1 minute) was not run with iLQR.

## Next

- A real-time iLQR (autograd Jacobians through the TendonHand step, or a reduced horizon).
- rrt_ilqr with the 1-minute model m1.
- The exploration-safety use of the CBF (filtering exploration programs).
- The progress page: add the planning-and-control chapter.
