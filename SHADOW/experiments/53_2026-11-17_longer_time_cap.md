# Experiment 53: both controller stacks with a 20 s time cap

Date: 17 Nov 2026. Results: `result_summaries/2026-11-17_longer_time_cap_results.md` (when done). Follows exp 50 (gestures, 6 s) and
exp 51 (cube, 10 s). Requested by the user: "make the time 20 s and run both controller stacks again to see the full results".

## Why

The cap is in simulated time (the simulator pauses while the controller plans), so a longer cap gives both stacks more physical time,
not more compute. The stack reaches the cube goal later than CEM (quick check: 8.2 vs 6.35 s on the rendered case), so part of its
exp-51 deficit (36.7 vs 65.0%, seed 790000) may be time-outs rather than wrong behaviour. On gestures CEM (85.3%) might converge with more
time; the stack is already at 100%.

## Arms (same models, flags and seeds as exp 50/51; only `--seconds` changes)

| task | arm | episode | seeds × cases |
|---|---|---|---|
| cube | `cem:P:hyb_tg0:safe` (CEM-MPC, tendon hand model `exp51/g0/A/f4`) | 20 s | 790000, 791000, 792000 × 120 |
| cube | `stk:P:hyb_tg0:safe:rrt+ilqr+cbf` (same CEM + RRT + iLQR-MPC/TVLQR + CBF) | 20 s | same |
| gestures | `cem` (CEM-MPC, `exp49/ladder/A/f4`, gravity on) | 20 s | 610000, 611000, 612000 × 150 |
| gestures | `rrt_ilqr_cbf` | 20 s | same |

Seed 792000 had no stack arm in exp 51 (stopped). Success rules unchanged: cube = first time within 0.1 rad without a drop; gestures =
24-joint RMSE over the last 0.5 s < 2°.

## Analysis

- Success at 20 s per arm, paired McNemar stack vs CEM.
- Success-vs-time curve from the same runs (cube: `time_to_success` ≤ 5/10/15/20 s; gestures: joint RMSE averaged over the 0.5 s
  before 6/10/15/20 s from the per-step traces). The 10 s (cube) and 6 s (gesture) points must match exp 51/50 within run-to-run noise;
  that is the consistency check.
- Drops, penetration (> 1/3 mm steps), median time to success.

## Hypotheses (pre-registered)

| # | hypothesis |
|---|---|
| H1 | cube: the stack gains more from 10 → 20 s than CEM (stack time-outs are a real part of its deficit) |
| H2 | cube: CEM still beats the stack at 20 s (the deficit is mostly behaviour, not time) |
| H3 | gestures: CEM rises by ≥ 3 points at 20 s; the stack stays ≥ 99% |

## Cost and runs

Exp 51 at 10 s: CEM 175 s, stack ~1,260 s per 120-case seed; exp 50 at 6 s: CEM 520 s, stack 640 s per 150-task seed. At 20 s: cube
~50 min per seed, gestures ~65 min per seed, ~6 h in total on a shared GPU. Chain `logs/run_exp53.sh` (tmux `exp53`), outputs
`artifacts/exp53/{cube_s*,gestures_s*,analysis.json}`, physical GPU 5, one job at a time.
