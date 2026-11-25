# Experiment 03: gestures and transitions on the Inspire hand — direct vs CEM-MPC vs RRT-Connect + iLQR-MPC/TVLQR + CBF

Date: 24 Nov 2026 (plan; runs after experiment 02 gives an identified model). Replicates SHADOW exp 27 (gesture benchmark) and exp 50
(explicit planning and control: RRT-Connect → iLQR-MPC with TVLQR → CBF), with the task adapted to a 6-servo hand.

## Why the Shadow benchmark cannot be copied as is

On Shadow, gestures were hard because the joints are weak, coupled by tendons and loaded by gravity (direct commands 74%, exp 50). The
Inspire servos are stiff and fast and the fingers weigh 4–5 g: in gate 0 a direct command reaches any free target to 0.01° within 0.3 s
(fingers) or 0.9 s (thumb yaw). What remains hard on this hand, and what the planning/control stack can show:

1. **Collisions on the way.** The thumb has to cross the index and middle fingers to change between gestures (fist → pinch, thumbs-up →
   tripod, …). A straight servo-space move drives the thumb into the fingers, and contact stalls it (gate 0: the fist leaves the thumb 3° short).
   This is SHADOW exp 50's thumb_middle_pinch failure (direct 0%, RRT 100%), and on Inspire it is the main failure mode.
2. **Contact force at speed.** Pinch gestures end in thumb–finger contact. The real hand overshoots its force threshold by up to +1618% at
   high speed (arXiv 2603.08988), so the approach speed near contact must be limited: a CBF on the predicted clearance, with the closing
   speed bounded as the clearance shrinks.

## Benchmark

- **Gesture library** (`inspire_dynamics/data/inspire_gestures_v1.json`, to write): open, fist, point, peace, three (index–middle–ring),
  rock (index + little), call (thumb + little), thumbs_up, pinch (thumb–index), thumb_middle_pinch, tripod (thumb–index–middle), little_up.
  Each target is in servo space (6 values). Pinch targets are made **reachable**: settled in Isaac from the commanded pose (SHADOW exp 49 lesson:
  5 Shadow targets could not be held).
- **Tasks:** transitions between gestures (start = another library gesture, plus random collision-free starts); every ordered pair with a
  thumb crossing is included, the rest sampled. 6 s per task, gravity on, task seeds 610000/611000/612000, 150 tasks per seed.
- **Success** (measurable on the real hand): servo-space RMSE over the last 0.5 s < 2° (6 servos). Also: time to success, fraction of steps
  in self-contact and with penetration > 1/3 mm, peak contact torque (sim) and steps above the force threshold, planning time per replan.

## Controllers (model = the experiment-02 identified model, D 1 min unless noted)

| name | global | local | filter |
|---|---|---|---|
| direct | – | servo command = target | – |
| cem | – | CEM-MPC (32 × 3, 0.6 s) | – |
| rrt_direct | RRT-Connect in servo space (6-D) | path following | – |
| ilqr | – | iLQR-MPC + TVLQR | – |
| rrt_ilqr | RRT-Connect | iLQR-MPC + TVLQR | – |
| rrt_ilqr_cbf | RRT-Connect | iLQR-MPC + TVLQR | CBF (clearance + approach speed) |

Code reuse: SHADOW's `rrt_connect.py`, `ilqr.py`, `cbf.py` take a configuration-space object and a model; the Inspire configuration space
(`inspire_dynamics/cspace.py`, to write) uses the dex-urdf convex hulls through `urdf_kin.py` (batched in torch for the GPU), with the
hand_base_link–thumb_proximal pair excluded as in the sim (experiment 01).

## Hypotheses (pre-registered)

| # | hypothesis |
|---|---|
| H1 | direct ≥ 95% on transitions without a thumb crossing, ≤ 60% on transitions with one |
| H2 | RRT (rrt_direct or rrt_ilqr) ≥ 95% on thumb-crossing transitions |
| H3 | iLQR/CEM without a global planner do not fix thumb crossings (≤ direct + 10 points) |
| H4 | the CBF halves steps above the force threshold on pinch goals at ≤ 0.3 s extra time to success |
| H5 | all controllers within 3 points of each other on the TRUE model vs the identified model (identification is not the bottleneck) |

## Videos

`scripts/render/render_traces.py` with the target gesture as a translucent ghost hand beside the real one; one clip per controller for
the hardest thumb-crossing transitions (as SHADOW's paired clips).

## What was run (24 Nov 2026)

`scripts/evaluate/exp03_gestures.py` (controllers in `inspire_dynamics/control.py`, RRT/PathFollower reused from SHADOW), hand 0 of
experiment 02 (unknown servo values), model = closed-loop exploration fit (arm CL, 180 s) unless `_N0`. Library: 12 goals solved against
the collision geometry (`scripts/evaluate/build_gesture_library.py`, `assets/inspire_hand/gesture_library.json`: open, thumbs_up, call,
fist, point, victory, rock, three, four, pinch_index, cylinder, tripod). Episode 4 s; success = final servo RMSE < 2°.
Added before the runs: a **fallback** in the CBF (Lecture 16): the filtered command is re-predicted; if an active pair still breaks the
barrier condition with negative barrier, the command becomes "hold the measured servo positions", and with RRT in the stack the task is
replanned from the measured pose at the next step. Added after the standard set: a **crossing** task set (`--task-set crossing`),
because the standard set is saturated.

## Results, standard set (`artifacts/exp03/bench`, 360 tasks per controller)

| controller | success | contact % steps | deep (> 2 mm) % | median time to 2° | solve ms (120 tasks, GPU) |
|---|---|---|---|---|---|
| direct | 99.7 | 1.06 | 0.30 | 0.65 s | — |
| CEM | 99.7 | 1.07 | 0.29 | 0.65 s | 59 |
| iLQR + TVLQR | 99.7 | 1.08 | 0.29 | 0.65 s | 135 |
| RRT + direct | 100 | 0.64 | 0.02 | 0.65 s | — |
| RRT + CEM | 100 | 0.35 | 0.00 | 0.80 s | 59 |
| RRT + iLQR | 100 | 0.35 | 0.00 | 0.80 s | 135 |
| CEM + CBF | **97.2** | 0.46 | 0.00 | 0.65 s | 59 + 42 |
| full (RRT + iLQR + CBF) | 100 | 0.22 | 0.00 | 0.80 s | 135 + 26 |
| full, spec model N0 | 100 | 0.09 | 0.00 | 0.80 s | 135 + 27 |

- Saturated: direct commands fail 1 of 360 (three ← random start, 33° short, blocked by the thumb). RRT removes deep contact.
- CEM + CBF fails 10 (McNemar vs CEM p = 0.004): the filter stops the thumb or fingers at the barrier and falls back to holding
  (254–329 fallbacks per seed) with no planner to find another route — a deadlock; 6 of 10 start from "four" (thumb across the palm).
  In the full stack the fallback never fired (0): the RRT path already avoids the barrier.
- Identified vs spec model: no difference on gestures (the speed error only changes timing, and 4 s is enough).
- Latency, one hand (`latency_ms`): CPU core — CEM 26 ms, iLQR first 106 / warm 54 ms, CBF 14 ms/step, RRT 31 ms; GPU — 58 / 259 / 133 /
  24 / 18 ms (kernel-launch bound); GPU, 150 tasks batched — 59 / 263 / 135 / 46 / 724 ms. Full stack per 0.15 s cycle on CPU ≈ 97 ms.

## Results, crossing set (`artifacts/exp03/crossing`, 360 tasks per controller)

Tasks: the 6 library pairs whose straight servo-space line collides + random contact-free starts whose straight line to a library goal
penetrates > 2 mm (120 per seed).

| controller | success | contact % | deep % | solve ms | fallbacks (→ RRT replans) |
|---|---|---|---|---|---|
| direct | 95.6 | 8.57 | 0.96 | — | — |
| CEM | 95.0 | 8.94 | 0.90 | 62 | — |
| iLQR + TVLQR | 96.7 | 7.67 | 0.78 | 141 | — |
| RRT + direct | 99.7 | 3.75 | 0.58 | — | — |
| RRT + CEM | 100 | 2.24 | 0.00 | 62 | — |
| RRT + iLQR | 100 | 2.29 | 0.00 | 142 | — |
| CEM + CBF | **88.6** | 3.01 | 0.02 | 61 + 43 | holds only |
| **full** | **100** | **1.42** | 0.00 | 143 + 28 | 9 (all 9 tasks succeed) |
| full, N0 | 100 | 0.53 | 0.00 | 141 + 27 | 5 |

Paired (exact McNemar): full vs direct 16/0 (p = 3e-5); RRT + iLQR vs iLQR 12/0 (p = 5e-4); RRT + CEM vs CEM 18/0 (p = 8e-6); iLQR vs
CEM 7/1 (p = 0.07); CEM + CBF vs CEM 5/28 (p = 7e-5); full vs full_N0 0/0.

| # | verdict (H1–H5 of the plan, read on the crossing set) |
|---|---|
| global planning | **RRT is the piece that matters**: local controllers end 20–35° short with a finger blocked by the thumb |
| local control | iLQR vs CEM ≈ tie (p = 0.07); with a path both reach 100% — the servo is precise by itself |
| safety | the CBF cuts contact ~40% with a path; without one it deadlocks (88.6%); fallback + replan recovers 9/9 |
| model | identified vs spec: no difference on gestures |
| latency | one hand on one CPU core: full stack ≈ 97 ms per 0.15 s cycle (real time); Shadow stack 9.5–11.6 s per solve |

Clips: `artifacts/exp03/clips/` (call ← crossing start: direct vs full; fist ← four: CBF only vs full), rendered to `artifacts/exp03/videos/`.
