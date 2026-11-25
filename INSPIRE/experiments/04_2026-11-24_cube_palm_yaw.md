# Experiment 04: cube task on the Inspire hand — turn a cube lying on the palm to a goal yaw

Date: 24 Nov 2026 (plan, written before the runs). Replicates the cube part of SHADOW exps 39, 51, 52 and 54 (CEM with the Isaac cube in the
loop; hybrid planner = learned hand model + Isaac cube; soft replay; stack layers on the cube). Hand: hand 0 of experiment 02 (perturbed
"unknown" servo parameters). Model: the closed-loop self-explored model of experiment 02 (arm CL, 3 min).

## Why this task

The user asked for "the cube task". SHADOW's task (arbitrary 3-D reorientation in a fingers-up hand) needs ~20 independent joints; the
Inspire hand has 6 servos, one per finger, each curling its finger in one plane, and a 2-DoF thumb. The nearest task it can plausibly
do is **single-axis rotation of a cube lying on the palm** (roadmap option a): fingers curl over the cube and push on one side, the thumb
sweeps across the palm. Hand palm up (rotated −90° about y), cube 4 cm, 30 g, friction 1.0, dropped on the palm centre.

## Step 0: feasibility probe (`scripts/evaluate/exp04_probe.py`)

96 envs, 8 s: 90 random servo programs (experiment 02 generator, up to 80% closure) and 6 scripted single-servo curls. Measures how far
the cube's yaw moves while it stays on the palm and how often it drops. **Decision rule:** if the random programs never turn the cube by
more than 30° while keeping it on the palm, the task is not feasible for this hand at this mount, and that negative is reported (no
forcing of a task); otherwise the goal set is chosen inside the reachable range (e.g. ±20°, ±40°).

## Planner and arms (if feasible)

CEM-MPC over held servo targets (4 knots over a 0.6 s horizon, re-solved every 0.15 s), scored by rolling candidates out in Isaac
"branch" environments copied from the live state (hand joints + cube pose and velocity); cost = cube yaw error over the horizon (+2x
terminal) + drop penalty (cube off the palm) + small command smoothness. No PPO proposer exists for this hand, so the SHADOW slide's PPO
row has no counterpart; CEM starts from the hold-still candidate and its own shifted mean.

| arm | branch environments' hand | layers |
|---|---|---|
| oracle | true servo parameters (the live hand itself) | — |
| kin_CL | identified model, **kinematic** replay (joint state written every substep; fingers pass through the cube) | — |
| soft_CL | identified model, **soft** replay (prediction = stiff drive target, Isaac simulates the contact) | — |
| soft_N0 | spec model (thumb yaw speed wrong by 10–50%), soft replay | — |
| soft_CL_cbf | soft_CL | + CBF self-collision filter on the executed command |
| soft_CL_stack | soft_CL | + TVLQR tracking of the chosen candidate's predicted (contact-aware) states + CBF (the exp 54 cube-aware stack) |

Tasks: per seed, 30 goals (yaw targets from the feasible set, signs balanced) × start = the settled cube; 3 seeds = 90 per arm, paired.
Episode 10 s. Success: |yaw error| < 0.1 rad (5.7°) at any time with the cube on the palm (as SHADOW's 0.1 rad), and also final
success. Drop: cube centre below the palm surface or > 6 cm from the palm centre. Latency: wall time per CEM re-solve (all branches
batched) and per stack layer.

## Hypotheses (pre-registered)

| # | hypothesis |
|---|---|
| H1 | soft_CL is within 5 points of oracle (the identified model is near-true; exp 02) |
| H2 | soft_N0 is below soft_CL (wrong thumb speed mis-ranks thumb sweeps) |
| H3 | kinematic replay is below soft replay (SHADOW exp 52: fingers through the cube mis-rank candidates) |
| H4 | the CBF and TVLQR layers do not raise success on the cube (SHADOW exp 51/54: the cube is the bottleneck, not the hand) |

## Feasibility outcome and changes before the main run (decided before any non-oracle arm ran)

- Probe 1 (cube at the palm centre, x = −0.07 m): max yaw 3.7°, the fingers barely reach the cube. Probe 2 (cube 3.5 cm nearer the
  finger bases, x = −0.105 m, full closure): random programs turn it at most 14° (median 3.5°), never drop it. By the rule above that
  is "not feasible for random motion"; because random programs do not accumulate pushes, the rule was extended with an oracle-CEM pilot.
- Oracle pilot, 0.6 s horizon, 16 candidates: 7% of goals (±10–30°) reached — the horizon is shorter than a finger's reach-and-push.
- Oracle pilot, **1.2 s horizon (24 steps), 24 candidates**: **83%** (±10°: 88%, ±20°: 79%), no drops, 1.7 s per re-solve.
- Main run therefore uses H = 24, NB = 24, goals ±10, ±20, ±30° (30° untested by the pilot), 30 tasks × 3 seeds, all six arms.
- **Change after seed 0 of the first main run (`artifacts/exp04/cube`, stopped):** oracle 83.3%, kin_CL 10.0%, soft_CL 16.7% (8.5 s per
  re-solve, 12 min per run; the kin and soft numbers were later found invalid — see the cube-reset bug below — so the switch rests on
  run time and the stiffness argument only). Soft replay as in SHADOW (prediction as a ×10-stiff drive target) makes the planner's fingers push the cube
  with a 10× stiffer loop than the real K ≈ 10 servo, so it mis-ranks pushes; it is also too slow to finish 3 seeds. Replaced by
  **param replay**: the planner's simulated hand carries the *identified* parameters on its drives (K, D, v_max, u_max, armature =
  m − link inertia). Main run `artifacts/exp04/cube_param`: oracle, kin_CL, param_CL, param_N0, param_CL_cbf, param_CL_stack, 3 seeds.
  H1–H4 are read with param in place of soft.
- **Bug found and fixed (24 Nov, 13:45):** the episode reset restored the hand but not the cube, so every arm after the first in a
  process started from wherever the previous arm left the cube (oracle 16.7% when run 5th vs 83–90% when run 1st). Runs
  `artifacts/exp04/{cube,cube_param}_INVALID_cube_not_reset` are kept for the record and not used. The cube is now reset
  to its spawn state before settling. Valid main run: `artifacts/exp04/cube_main` (all six arms, 3 seeds, oracle first).

## Results (`artifacts/exp04/cube_main`, 30 tasks × 3 seeds = 90 paired tasks per arm; goals ±10/20/30°, 10 s)

| arm | success (< 5.7° at any time) | final | drop | median time | ms per CEM solve |
|---|---|---|---|---|---|
| oracle (true hand in the planner) | **87.8** | 87.8 | 0 | 4.1 s | 4,789 |
| param_N0 (spec parameters) | **81.1** | 77.8 | 1.1 | 3.3 s | 4,992 |
| param_CL (identified, closed-loop exploration) | 51.1 | 43.3 | 0 | 3.8 s | 5,443 |
| param_CL + CBF | 54.4 | 46.7 | 0 | 4.6 s | 5,180 |
| param_CL + TVLQR + CBF (cube-aware stack) | 44.4 | 35.6 | 0 | 4.6 s | 5,503 |
| kin_CL (identified, kinematic replay) | 33.3 | 27.8 | 0 | 5.7 s | 5,507 |

By goal (oracle / N0 / CL): 10° 87/97/80, 20° 90/77/47, 30° 87/70/27.
Paired (exact McNemar): oracle vs CL 36/3 (p = 4e-8); N0 vs CL 33/6 (p = 1e-5); N0 vs oracle 9/15 (p = 0.31); +CBF vs CL 13/10 (p = 0.68);
stack vs CL 9/15 (p = 0.31); kin vs CL 5/21 (p = 0.002). CBF fallback (hold) fired 100 (CBF) and 123 (stack) times in 6,000 steps.

| # | verdict |
|---|---|
| H1 (identified ≈ oracle) | **fail**: 51 vs 88%. The free-motion fit left damping ≈ 0 on 3 servos and the inertia ≈ 0 (unidentifiable at 20 Hz); free-motion prediction does not care, contact does |
| H2 (N0 below CL) | **reversed**: spec values 81% beat the identified model; the wrong thumb speed costs less than wrong contact dynamics |
| H3 (kinematic below param replay) | **pass**: 33 vs 51% (p = 0.002), as SHADOW exp 52 |
| H4 (layers do not help on the cube) | **pass**: CBF +3 (n.s.), cube-aware stack −7 (n.s.) |

Takeaway: on the cube, the model's contact parameters matter and free-motion self-exploration does not identify them. Next: keep
unidentifiable parameters at their prior (regularised fit) and add contact/press phases to exploration; then re-run param_CL.
