# Experiment 50: explicit planning and control on the self-identified hand model

Date: 5 Nov 2026. Results: `result_summaries/2026-11-05_explicit_planning_and_control_results.md` (when done).

## Why

The professor's comment on the progress page: there is no explicit planning and control part of the kind taught in the motion planning and control course (weeks 1–8: foundations, representations, heuristic search and replanning, sampling-based planning, kinodynamic and feedback planning, trajectory optimisation, MPC and learning-based control, safety-critical control). The current controller is CEM-MPC, a sampling-based shooting method over one held command pose, on top of the hand's joint PD drives. This experiment adds the course's methods as explicit layers and tests each one on the same gesture benchmark:

global path (configuration space, sampling-based) → local trajectory (trajectory optimisation + feedback, MPC) → safety filter (control barrier function) → hand.

## Problem formulation (weeks 1, 5)

- State x = (q, q̇) ∈ R^48 (deg, deg/s; 24 joints, 4 passive J0). Control u ∈ R^20 = joint position targets at 20 Hz. Dynamics x_{k+1} = f(x_k, u_k) = the exp-49 Li-form model identified from self-exploration (`artifacts/exp49/ladder/A/f4/model.json`, 25 valid minutes, from a kinematics-only prior; implicit Euler, 6 substeps).
- Constraints: joint limits (box on u), slew limit |u_k − u_{k−1}| ≤ 4° per step (80 °/s), no link penetration (configuration-space obstacles from the link meshes, k-DOP hulls, `shadow_dynamics/hand_geometry.py`).
- Cost (as the existing CEM): J = (1/H) Σ_t ||q̂_t − r_t||²/24 + 2 ||q̂_H − r_H||²/24 + 1e-3 Σ ||u_t − u_{t−1}||², horizon H = 12 (0.6 s). r_t = target, or a point on the global path.

## Methods

1. **Configuration space and obstacles (week 2).** `shadow_dynamics/hand_cspace.py`: q(u) with J0 = clip(1.143 J1, 0, 90); validity = max link penetration ≤ 1 mm; a signed clearance per link pair (unclamped k-DOP depth, differentiable through FK) for the safety filter. Figure: a 2-D slice (thumb THJ3 × index FFJ2) of the obstacle region.
2. **RRT-Connect (week 4), plus shortcut smoothing.** `shadow_dynamics/rrt_connect.py`, batched in lockstep over all tasks (collision checks on GPU). For each task: straight line if valid, else RRT-Connect in the 20-D command space (step 10°, edge resolution 2°, goal bias 10%, ≤ 600 iterations), then 60 rounds of random shortcutting. Start and goal neighbourhoods (5°) are exempt from the check (starts and pinch goals may touch).
3. **Path following.** Reference r_k = the point a fixed 20° ahead (along the path's length) of the projection of the measured q onto the path (monotone), ending at the goal. Used by any local controller.
4. **iLQR / iLQR-MPC with TVLQR feedback (weeks 5–7).** `shadow_dynamics/ilqr.py`: augmented state (x, u_prev), control Δu (box ±4° = the slew limit; u box by projection), linearisation by batched finite differences of the identified model, Levenberg–Marquardt regularisation, per-task line search. Receding horizon: replan every 3 steps (warm start: shift), 8 iterations on the first plan, 3 after; between replans u = ū_k + K_k (x − x̄_k) (time-varying LQR feedback from the backward pass).
5. **Discrete-time CBF safety filter (week 8).** `shadow_dynamics/cbf.py`: for link pairs with clearance < 10 mm, h = clearance; require h(q_{k+2}) ≥ (1 − γ)² h(q_k), γ = 0.3, with q_{k+2}(u) predicted by the identified model (command held 2 steps), linearised in u (finite differences) and in q (autograd through FK). Minimal change ||u − u_nom||² by Hildreth's dual QP iterations, then box and slew projection.
6. **Heuristic search and replanning (week 3)**: the path-following reference is replanned (RRT from the current command) if progress stalls for 1 s (s does not advance and the error to the goal > 2°).

## Benchmark

Exp-49 gesture benchmark on the v3 library (`shadow_dynamics/data/shadow_pose_library_v3_exp49.json`), gravity on, 15 gestures × 10 starts, 6 s, task seeds 610000/611000/612000; success = final 24-joint RMSE < 2°; also reported excluding thumb_middle_pinch (exp 49 decision) and including it (the global planner targets exactly the path-dependent failures). Added measurements per run: fraction of control steps with self-contact (validity 1) and with invalid penetration (≥ 2), and planning time.

Controllers (model = A4 unless noted):

| name | global | local | filter |
|---|---|---|---|
| direct | – | slew-limited command | – |
| cem (exp 49 baseline) | – | CEM-MPC | – |
| rrt_direct | RRT-Connect | slew-limited follow | – |
| rrt_cem | RRT-Connect | CEM-MPC on the path reference | – |
| ilqr | – | iLQR-MPC + TVLQR | – |
| rrt_ilqr | RRT-Connect | iLQR-MPC + TVLQR | – |
| cem_cbf | – | CEM-MPC | CBF |
| rrt_ilqr_cbf | RRT-Connect | iLQR-MPC + TVLQR | CBF |

## Hypotheses (pre-registered)

- H1: RRT-based controllers raise success on the path-dependent gestures (thumb_middle_pinch and the curled-finger catches) over the same local controller without the path, and lower self-contact time.
- H2: iLQR-MPC matches CEM-MPC success (within 3 points) with equal or lower planning time; TVLQR feedback is the only difference between replans.
- H3: the CBF filter cuts self-contact time by ≥ 50% at a success cost ≤ 3 points.
- Paired comparisons: exact McNemar on the 450 paired tasks.

## Gates before the GPU runs

- G1 (CPU/GPU-light): RRT finds paths for all 150 tasks of seed 610000; reports how many need more than a straight line, path length, time.
- G2: iLQR on the identified model lowers the model's own predicted cost vs the warm start on a batch of tasks (sanity), and one-step finite-difference Jacobians match autograd on a few states.
- G3: CBF keeps the predicted clearance ≥ 0 on a synthetic approach (two fingers driven together) in the model.

Chain `logs/run_exp50.sh`, tmux `exp50`, output `artifacts/exp50`, ends with `EXP50_COMPLETE`. One GPU job at a time (GPU 3).

## Gate results and fixes (5 Nov)

- G1: 150/150 paths (12 s); 38 tasks need RRT (all 10 thumb_middle_pinch). Curled-finger goals touch (fist 3.2 mm, thumbs_up 2.2, pointing 1.7, rock 1.6 mm), so each problem tolerates the penetration of its own start or goal + 0.5 mm.
- G2: iLQR from "hold" ends worse than a plain slew (median predicted cost 955 vs 669); started from the slew it lowers cost on 98.7% of tasks (664 vs 669: the 80 °/s slew dominates the cost early). The initial guess is now the better of the shifted previous plan and a fresh slew. Finite-difference Jacobians vary about 15% (median) between step sizes. Indefinite Qvv: eigenvalue-clamped inverse. About 10 s per replan for 150 tasks.
- G3 (partial): on a deliberate collision approach, the CBF cuts the model-state penetration from a median of −6.9 to −2.6 mm (n = 4, γ = 0.2, d_act = 20 mm), with transients of 1–2 mm while the thumb sweeps across several links, settling to about −0.2 mm; the slew limit caps how fast it can brake.
- First full run, stopped: the path follower tied the reference to the projection of the *measured* pose. Fingers starting curled in contact then stayed stuck (6 tasks about 60° off; rrt_direct 69% vs direct 75%), because the drive pulled only 20° ahead. Fixed: the reference advances from the *commanded* pose (a moving carrot); replanning triggers when the measured error to the goal falls by < 0.5° in 1 s. Check on open/neutral/thumb_middle_pinch (30 tasks): rrt_direct 2.32° vs direct 2.78°, no stuck tasks. Full chain restarted.
- iLQR produced a non-finite command on seed 610000 (crash after rrt_cem). Guards added: non-finite rollout cost = inf (never selected), NaN gains/Jacobians zeroed, non-finite feedback command → open-loop plan → hold; counts reported (ilqr_fallback, nonfinite_cmd). Chain restarted from seed 610000.
