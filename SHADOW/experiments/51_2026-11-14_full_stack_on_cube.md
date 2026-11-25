# Experiment 51: the exp-50 planning-and-control stack on the cube task

Date: 14 Nov 2026. Results: `result_summaries/2026-11-14_full_stack_on_cube_results.md` (when done).

## Why

Exp 50 added RRT-Connect, iLQR-MPC + TVLQR and a CBF filter and reached 100% on gestures; it was never run on the cube. The user asked to
"run the full stack on the cube task". It is also exp 41's open question: does making the hand follow the planned joint trajectory
(feedback in contact) close part of the hybrid planner's ~15-point gap to the oracle?

## Mapping (decided, not asked)

The stack tracks a joint-space reference; the cube goal is an orientation. So the exp-39 CEM planner stays on top:

CEM (P-seeded, 3 × 32, hybrid forward model: hand model predicts joints, Isaac simulates the cube) → chosen candidate's **predicted joint
trajectory** (12 steps) = the reference → RRT layer → iLQR-MPC + TVLQR tracking → CBF filter → hand.

- **RRT layer:** the chosen command path (prev → 12 targets, plus midpoints) is checked for self-penetration beyond the start's or end's own
  (+0.5 mm, as exp 50). Where it penetrates, RRT-Connect plans from the current command to the candidate's last command; the path, timed at
  the 80 °/s slew, replaces both the reference and the chunk. The cube is not a C-obstacle (the fingers must touch it). Expected to fire rarely.
- **iLQR-MPC + TVLQR:** same tendon model as the planner (so CEM and iLQR share the model), time-varying reference (ilqr.py now accepts
  (E, H, 24)), replanned every 3 steps (twice per 6-step chunk), TVLQR between replans, 80 °/s slew box. Because the reference is the model's
  own response to the chosen commands, iLQR's open-loop answer ≈ those commands; what it adds is replanning from the measured state and
  feedback when the cube pushes the hand off its free-motion prediction.
- **CBF:** exactly exp 50's settings as run (n = 4, γ = 0.2, d_act = 20 mm; note the deck stated n = 2, γ = 0.3, 10 mm — the class defaults,
  not what ran), tol = the case's start penetration + 0.5 mm (no fixed goal pose exists).

## Hand model (gate G1)

Only the exp-49 Li-form `TendonHand` is a Markov step `f(x48, u20)` that iLQR/CBF can use. The exp-49 fits were made with **gravity on**;
the cube scene runs with hand gravity **off** (P was trained so; every baseline uses it). So: repeat exp 49's **active** self-exploration
(ladder arm A, the recipe of exp 50's A4 model) with `--no-gravity`: 3 slow random programs, then three rounds of D-optimal program design
with the current fitted model (6, 18, 54 programs), each followed by collection and a refit from scratch of the Li-form equation on all data
(`logs/run_exp51_model.sh`) → `artifacts/exp51/g0/A/f4`, label `tg0`. (Changed before any run, after the user asked whether exploration is
still closed loop; a one-shot random recipe was first planned.) The model is learned offline and frozen during the cube runs; it is not
updated online. Check: fit holdout per rung, and in the
`mpc:P:oracle` ranking diagnostic its cube-task hand error and Spearman vs the oracle next to exp 39's `res_phmro` (4.4°, 0.66).
Known: the model never saw contact or > 30 °/s (as all hand models in exps 39–44).

## Environment

The Isaac Sim venv (`.venv` → `surgical_robotics/simulation/.venv`) was deleted in a disk reorganisation. Rebuilt from
`environment-freeze.txt` at `the Isaac environment` (uv, Python 3.10, Isaac Sim 4.2.0.2, IsaacLab @ e00d625, no orbit-surgical);
scripts run with `SHADOW_PYTHON=$ISAAC_PYTHON`. GPU: **physical 2** (started on 4 per the user 14 Nov, moved to 2 at 15:34 by the user because 4 was shared with a 100%-load job; GPU 3 is occupied by
another user). Rerun drift check: P alone and `cem:P:hyb_res_phmro` should land near exp 39 (40.8%, 70.3%).

## Benchmark and arms

Cube benchmark of exps 19–44: seeds 790000/791000/792000 × 120 paired cases, 10 s, success < 0.1 rad, drop 240 mm, cost `safe`,
`--oracle-candidates 32`. Arms per seed:

| arm | planner | model | execution |
|---|---|---|---|
| `ppo:P` | – | – | P alone |
| `mpc:P:oracle:safe` | one-shot | Isaac | direct (ranking diagnostic for every hand model) |
| `cem:P:oracle:safe` | CEM | Isaac | direct (ceiling) |
| `cem:P:hyb_res_phmro:safe` | CEM | exp-39 eq+res+observer | direct (exp-39 reference) |
| `cem:P:hyb_tg0:safe` | CEM | gravity-off tendon | direct |
| `stk:P:hyb_tg0:safe:ilqr` | CEM | tendon | iLQR-MPC + TVLQR |
| `stk:P:hyb_tg0:safe:cbf` | CEM | tendon | CBF only |
| `stk:P:hyb_tg0:safe:rrt+ilqr+cbf` | CEM | tendon | **full stack** |

New per-arm metrics: fraction of alive steps with self-contact / self-penetration > 1 mm / > 3 mm of the measured hand (k-DOP, HandCSpace),
tracking error to the reference, RRT invocations, iLQR and CBF time.

## Hypotheses (pre-registered)

| # | Hypothesis | Prob. |
|---|---|---|
| H1 | `cem hyb_tg0` ≥ `cem hyb_res_phmro` − 5 points (the Markov tendon model without observer is good enough as the hybrid's hand) | 50% |
| H2 | iLQR tracking adds ≥ 3 points over `cem hyb_tg0` | 25% |
| H3 | CBF cuts steps with > 1 mm self-penetration by ≥ 30% at a success cost ≤ 3 points | 50% |
| H4 | the RRT layer fires on < 5% of decisions | 80% |
| H5 | full stack ≥ `cem hyb_tg0` − 3 points | 55% |

Exact McNemar on the 360 paired cases, Wilson 95% CIs.

## Gates and steps

1. G0: environment rebuilt; one-case Isaac start.
2. G1: gravity-off active exploration ladder + fits (`logs/run_exp51_model.sh`, ≈ 45–60 min).
3. G2 smoke: seed 790000, `--cases-per-bin 3 --seconds 3`, arms `cem:P:hyb_tg0:safe`, `stk:P:hyb_tg0:safe:rrt+ilqr+cbf`; no crash, finite
   commands, timing.
4. Chain `logs/run_exp51.sh` (tmux `exp51`), outputs `artifacts/exp51/eval_s{seed}`, analysis `scripts/analyze_exp39.py` pairs + penetration
   table. Budget: ≈ 40 min per seed (iLQR ≈ 67 replans × ~8 s per stack arm), ≈ 2 h.
5. Results file; nothing committed.

## Follow-up (added 14 Nov, after seed 790000)

Seed 790000: iLQR tracking cost 12 points (65.0 → 53.3%, 5% drops). Traces: iLQR commands sit further from the measured joints
(mean 5.1 vs 4.3°, p90 14.6 vs 11.1°), reverse 7× as often, track error 5.1° (free-motion model error ≈ 0.4°). Reading: the reference is
the model's free-motion response, the cube blocks the fingers, and TVLQR feedback pushes into the cube. The user asked for two arms after the
main chain (`logs/run_exp51b.sh`, tmux `exp51b`, seed 790000, `artifacts/exp51/followup_s790000`, paired with `eval_s790000`):
`stk:P:hyb_tg0:safe:ilqrc` (reference = the commanded joints, `cspace.full(fa_c)`) and `stk:P:hyb_tg0:safe:ilqrnf` (TVLQR gains × 0,
replanning from the measured state every 3 steps only). Note: with `ilqrc` the feedback still acts on the gap to the model's free-motion
nominal, so it may still squeeze; `ilqrnf` isolates the feedback.
Mid-run fixes: rsl_rl 2.0.x installed (not in the freeze); k-DOP depth chunked over rows (`HandCSpace._signed_depth`, identical values) after
two CUDA OOMs on the shared GPU 2; seed 790000's CBF and full-stack arms ran in `eval_s790000_b`, merged into `eval_s790000_merged`.
