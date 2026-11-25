# Experiment 53 results: both controller stacks with a 20 s time cap (stopped after one cube seed)

Plan: `experiments/53_2026-11-17_longer_time_cap.md`. Date: 17 Nov 2026. GPU 5 (shared with another user's job at ~98% utilisation,
which slowed everything ~10×). Stopped by the user at 13:20 after cube seed 790000 so that exp 54 could start; seeds 791000/792000 and the
gesture runs were not done (`artifacts/exp53/cube_s791000_partial_stopped` is incomplete, ignore it). Nothing committed.

## Cube, seed 790000 (120 cases, 20 s, success = first time within 0.1 rad without a drop)

Success by time, from `time_to_success` in the same runs:

| arm | 5 s | 10 s | 15 s | 20 s | drops | steps > 3 mm overlap | median time to success | wall time |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CEM-MPC `cem:P:hyb_tg0:safe` | 29.2 | 65.8 | 81.7 | **85.8** | 0.8% | 23.1% | 7.0 s | 62 min |
| CEM + RRT + iLQR + CBF `stk:P:hyb_tg0:safe:rrt+ilqr+cbf` | 25.8 | 42.5 | 58.3 | **71.7** | 1.7% | 1.4% | 7.6 s | 189 min |
| stack vs CEM, exact McNemar p | 0.39 | 3e-5 | 3e-5 | 0.005 | | | | |

- Consistency: at 10 s CEM 65.8% (exp 51: 65.0%), stack 42.5% (exp 51: 36.7%; settled starts differ slightly between processes).
- 10 → 20 s: CEM +20.0 points (24 cases gained, 0 lost), stack +29.2 (35 gained, 0 lost).

## Hypotheses

| # | hypothesis | outcome |
|---|---|---|
| H1 | the stack gains more from 10 → 20 s than CEM | **held** (+29.2 vs +20.0) |
| H2 | CEM still beats the stack at 20 s | **held** (85.8 vs 71.7%, p = 0.005) |
| H3 | gestures: CEM +3 at 20 s, stack ≥ 99% | not run |

## Reading

- The 10 s cap hid a lot of both controllers' ability: CEM at 20 s (85.8%) is at the level of exp 51's oracle CEM at 10 s (87%).
- Part of the stack's exp-51 deficit was time-outs (the gap shrinks from 23 to 14 points), but not all: at 20 s it still loses 14
  points, so its layers cost successes, consistent with exp 51's explanation (cube-blind corrections to CEM's commands).
- The stack's safety benefit holds at 20 s: > 3 mm overlap steps 23% → 1.4%.
- Single seed; the 95% intervals are CEM [78, 91], stack [63, 79].

## Files

`artifacts/exp53/cube_s790000/{results.json,traces_*.npz}`, `artifacts/exp53/analysis.json`, analysis `scripts/analyze/analyze_exp53.py`,
chain `logs/run_exp53.sh`, logs `logs/exp53_cube_s790000.log`.
