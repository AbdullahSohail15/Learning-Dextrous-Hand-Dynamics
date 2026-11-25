# Experiment 04 results: turning a cube on the palm of the Inspire hand (24 Nov 2026)

Plan, deviations and tables: `experiments/04_2026-11-24_cube_palm_yaw.md`.

- Task: palm up, 4 cm cube near the finger bases, turn it ±10/20/30° in yaw; CEM-MPC, 1.2 s horizon, 24 candidates, Isaac cube in
  the planner. Feasible only with the 1.2 s horizon (0.6 s: 7%).
- 90 paired tasks per arm: oracle 87.8%, spec parameters 81.1%, identified parameters 51.1%, + CBF 54.4%, + TVLQR + CBF 44.4%,
  kinematic replay 33.3%. No drops (≤ 1%).
- Free-motion closed-loop exploration identifies speeds but drives the unidentifiable damping/inertia to ~0, which breaks contact
  prediction: the spec model beats the identified one on the cube (p = 1e-5). Stack layers do not help (as SHADOW exps 51/54).
- A cube-reset bug invalidated the first two runs (kept as `*_INVALID_cube_not_reset`).
- Latency: ~5 s per CEM solve (48 branch envs × 24 steps of Isaac per task, 2 iterations) — not real time; the gesture stack is.
