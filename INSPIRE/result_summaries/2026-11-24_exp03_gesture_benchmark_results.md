# Experiment 03 results: gestures with RRT + iLQR-MPC/TVLQR + CBF on the Inspire hand (24 Nov 2026)

Plan and tables: `experiments/03_2026-11-24_gestures_planning_control.md`.

- Standard set (12 gestures × 10 starts × 3 seeds): saturated — direct commands 99.7%, every RRT variant and the full stack 100%.
- Crossing set (straight move collides): direct 95.6, CEM 95.0, iLQR 96.7, RRT + iLQR 100, full 100% with the least finger contact
  (1.4% of steps vs 8.6% for direct, deep contact 0). CBF without a planner deadlocks (88.6%).
- Fallback (Lecture 16): re-predicted filtered command still unsafe → hold servos → RRT replan; fired 9 times in the full stack, all recovered.
- Latency, one hand, one CPU core: CEM 26 ms, iLQR warm 54 ms, CBF 14 ms, RRT 31 ms; full stack ≈ 97 ms per 150 ms cycle — real time,
  ~100× faster than the Shadow stack.
- Identified vs spec model: no effect on gestures.
