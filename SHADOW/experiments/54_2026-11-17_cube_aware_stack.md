# Experiment 54: a cube-aware stack on the soft hybrid

Date: 17 Nov 2026. Results: `result_summaries/2026-11-17_cube_aware_stack_results.md` (when done). Follows exp 51 (the exp-50 stack hurts
the cube: every layer changes CEM's commands with a hand model that has no cube, and nothing re-checks them), exp 52 (soft replay makes
CEM's branch hands feel the cube: ranking 0.66 → 0.94) and exp 53 (20 s cap; same seed and cases, used as the kinematic references).

## Idea

Keep the stack's layers but make each one either cube-aware or a suggestion that CEM tests in the cube branches, instead of a
post-processor that overrides CEM:

| layer | exp 51 (post-processor, kinematic hybrid) | exp 54 version |
|---|---|---|
| iLQR-MPC | re-optimises CEM's commands to track the free-motion prediction: −13.8 points | **dropped** as a post-processor |
| TVLQR (`tvlqr`) | bundled with iLQR, pushes into the cube | gains about CEM's own commands (no re-optimisation), tracking the **soft hybrid's predicted states**, i.e. where the fingers end up with the cube pushing back; corrects only real disturbances |
| RRT-Connect (`rrtp`) | replaced CEM's path on ~15% of decisions, untested on the cube | when CEM's chosen path collides, the RRT path replaces CEM's worst candidate and everything is re-scored in the branches; used only if it beats CEM's choice |
| CBF (`pen` + `cbf`) | fired on 81% of steps (grasping presses fingers together): −5.4 points | `pen`: predicted finger overlap (from the soft branches) is a CEM cost, 2 per mm beyond the start's own (`--pen-weight`); `cbf`: the final filter kept only as a backstop, tolerating 3 mm beyond the start (`--stack-cbf-tol-mm 3`, exp 51: 0.5) |

## Arms (seed 790000, 120 cases, 20 s, cost `safe`, 32 candidates, tendon hand model `tg0` = `exp51/g0/A/f4`)

| # | arm | question |
|---|---|---|
| 1 | `cem:P:hyb_soft_tg0:safe` | soft-replay CEM baseline (soft replay never run with `tg0`; exp 52 used `res_phmro`) |
| 2 | `stk:P:hyb_soft_tg0:safe:tvlqr` | does contact-aware TVLQR help or hurt? |
| 3 | `stk:P:hyb_soft_tg0:safe:rrtp` | does RRT as a proposer help? |
| 4 | `stk:P:hyb_soft_tg0:safe:pen` | overlap cost alone: penetration cut at what success cost? |
| 5 | `stk:P:hyb_soft_tg0:safe:pen+cbf` | plus the light final filter |
| 6 | `stk:P:hyb_soft_tg0:safe:rrtp+tvlqr+pen+cbf` | the cube-aware stack |

References on the same seed and cases from exp 53 (20 s, kinematic hybrid): `cem:P:hyb_tg0:safe` (85.8%) and the exp-51 full stack.
Each arm is compared with arm 1 (exact McNemar), plus success at 10 s from `time_to_success`, drops, steps with > 1 / > 3 mm overlap,
`rrtp_chosen`, and time per arm.

## Hypotheses (pre-registered)

| # | hypothesis |
|---|---|
| H1 | soft replay helps `tg0` too: arm 1 ≥ exp-53 kinematic CEM (85.8%) |
| H2 | contact-aware TVLQR costs ≤ 3 points vs arm 1 (exp 51 iLQR + TVLQR: −13.8) |
| H3 | RRT as a proposer costs ≤ 2 points (it can only win when CEM's branches score it higher) |
| H4 | `pen` cuts > 3 mm overlap steps by ≥ 50% vs arm 1 at ≤ 3 points |
| H5 | the cube-aware stack is within 3 points of arm 1 and keeps > 3 mm overlap below 5% of steps (exp 51 full stack: 36.7%, 1.5%) |

One seed (120 cases) detects ~10-point differences; a second seed (791000) follows if the arms are within a few points of each other.

## Code (`scripts/evaluate/evaluate_shadow_cube_mpc.py`, `shadow_dynamics/ilqr.py`)

New stack layers `tvlqr`, `rrtp`, `pen`; `stk` arms accept `hyb_soft_<tendon>`; flags `--pen-weight`, `--stack-cbf-tol-mm`; `ILQR.gains()`.
Defaults leave every earlier arm unchanged. Pre-change copies in the session scratchpad (`pre_exp54/`).

## Runs

Chain `logs/run_exp54.sh` (tmux `exp54`), queued behind exp 53 (starts when the `exp53` session ends), physical GPU 5. A smoke run
(1 case per bin, 2 s, all arms) must finish before the full run. Outputs `artifacts/exp54/{smoke,cube_s790000}`. Expected ~6–8 h on
the shared GPU (exp 53 CEM at 20 s took 62 min per arm).
