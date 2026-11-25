# Experiment 51 results: the exp-50 stack on the cube task (stopped)

Plan: `experiments/51_2026-11-14_full_stack_on_cube.md`. Date: 14 Nov 2026. Stopped by the user at 18:15 after 1.9 of 3 seeds
(seed 790000 complete; seed 791000 without the full-stack arm; seed 792000 and the iLQR follow-up arms not run). Nothing committed.

## Model (gate G1)

Gravity-off active self-exploration (ladder arm A: 3 random programs, then D-optimal rounds of 6/18/54), refit from scratch each round,
`artifacts/exp51/g0/A/f4` (label `tg0`), 25.4 valid minutes. Holdout equation loss vs exp 49's gravity-on ladder at the same rung:
f2 0.382 vs 0.126, f3 0.287 vs 0.094, f4 0.242 vs 0.139 (the gap narrows with data but stays ~1.7×; the loss scale depends on the data's
speed, so it is not a clean model-quality comparison).

## Cube results (120 cases per seed, 10 s, success < 0.1 rad)

| arm | s790000 | s791000 | pooled (240) | drops | steps > 1 mm pen. | > 3 mm |
|---|--:|--:|--:|--:|--:|--:|
| `ppo:P` | 35.8 | 46.7 | 41.3 | 1.3 | 65% | 28% |
| `mpc:P:oracle:safe` | 84.2 | 85.8 | 85.0 | 0 | 53% | 19% |
| `cem:P:oracle:safe` | 89.2 | 85.0 | 87.1 | 0.8 | 54% | 19% |
| `cem:P:hyb_res_phmro:safe` | 72.5 | 76.7 | 74.6 | 1.7 | 55% | 20% |
| `cem:P:hyb_tg0:safe` | 65.0 | 64.2 | 64.6 | 0 | 58% | 24% |
| `stk:P:hyb_tg0:safe:ilqr` | 53.3 | 48.3 | 50.8 | 4.2 | 54% | 22% |
| `stk:P:hyb_tg0:safe:cbf` | 62.5* | 55.8 | 59.2 | 0.8 | 31% | 5% |
| `stk:P:hyb_tg0:safe:rrt+ilqr+cbf` | 36.7* | – | – | 4.2 | 17% | 1.5% |

\* seed 790000's CBF and full-stack arms ran in a second process after two CUDA OOMs; its settled starts differ slightly from the first
run (median 0.0005 rad, max 0.05 rad on the same 120 draws), so those comparisons are approximately paired.

Paired (exact McNemar, 240 cases): `tg0` vs `res_phmro` 64.6 vs 74.6% (20/44, p = 0.004); iLQR vs `tg0` 50.8 vs 64.6% (21/54, p = 2e-4);
CBF vs `tg0` 59.2 vs 64.6% (25/38, p = 0.13). Rebuilt environment reproduces exp 39 (`res_phmro` 74.6 vs 70.3%, oracle CEM 87 vs 85%).

Stack internals (seed 790000): RRT replaced the chosen path on 602 of 4,080 case-decisions (~15%, 4 unsolved); CBF modified 81% of steps;
iLQR 67 replans × ~14 s per arm (20 min per seed).

## Hypotheses

| # | hypothesis | outcome (2 seeds) |
|---|---|---|
| H1 | `tg0` ≥ `res_phmro` − 5 | **failed** (−10.0, p = 0.004) |
| H2 | iLQR adds ≥ 3 | **failed**, it costs 13.8 (p = 2e-4) |
| H3 | CBF cuts > 1 mm steps ≥ 30% at ≤ 3 points | penetration −47% held; cost 5.4 points (n.s.), over the 3-point bound |
| H4 | RRT fires on < 5% of decisions | **failed** (~15%) |
| H5 | full stack ≥ `tg0` − 3 | **failed** (36.7 vs 65.0 on seed 790000) |

## Why the stack hurts on the cube

CEM scores command sequences against the simulated cube; every stack layer then changes the executed commands without re-scoring them.
iLQR's reference is the tendon model's free-motion response; the cube blocks the fingers, and TVLQR feedback pushes into it (seed 790000
traces: command–measured gap 5.1 vs 4.3° mean, p90 14.6 vs 11.1°; 7× more command reversals; tracking error 5.1° vs ~0.4° free-motion model
error; loss grows with goal difficulty). RRT substitutes slew-timed collision-free paths CEM never evaluated. The CBF fires constantly
because grasping presses fingers together. On gestures (exp 50) nothing touches the hand, so these corrections help; on the cube the
controller's model lacks the cube. The stack does cut penetration (> 3 mm: ~21% → 1.5% of steps).

## Next

Exp 52 (`experiments/52_2026-11-14_hybrid_replay_bottleneck.md`): is the hybrid's kinematic hand replay, not the hand model, the gap to the
oracle? Perfect-hand hybrid and compliant (soft) replay.

## Mid-run fixes and files

rsl_rl 2.0.x installed into `the Isaac environment` (missing from the freeze); `HandCSpace._signed_depth` chunked over rows (identical
values; OOM at 3,000–4,096 rows on the shared GPU); evaluator `stk:` arms, variants `ilqrc`/`ilqrnf` (never run); `ilqr.py` time-varying
reference, `U_init`, `fb_scale`. Outputs `artifacts/exp51/{g0,smoke,eval_s790000,eval_s790000_b,eval_s790000_merged,eval_s791000}`; chains
`logs/run_exp51{,_model,b}.sh`; logs `logs/exp51_*`.
