# Shadow Hand: planning and control on a self-identified hand model

Simulated Shadow Hand (Isaac Sim 4.2 / Isaac Lab 1.2) doing two tasks:

1. **Gesture reaching.** Move from a start pose to one of 15 target gestures within 6 s. Success means the final 24-joint RMSE is below 2°.
2. **In-hand cube reorientation.** Turn a 6 cm cube to a goal orientation within 10 s. Success means the cube is within 0.1 rad (5.7°) of the goal.

Both tasks are solved with model-based control. The hand model is a tendon-structured dynamics model (Li et al., arXiv 2408.13044). It is identified from the hand's own exploration data, starting from nothing but the kinematics (experiments 49 and 51).

Two controller stacks are compared:

| Stack | What it does |
|---|---|
| **CEM-MPC** | Sampling-based MPC: 32 candidate command sequences × 3 refinement rounds, 0.6 s horizon, replanned every 0.15 s (gestures) or 0.3 s (cube). On the cube, a PPO policy proposes the candidate mean. The cube's response is simulated in Isaac branch environments while the hand follows the model's prediction. |
| **RRT + iLQR-MPC + CBF** | RRT-Connect plans a collision-free path in the 20-D joint-command space. iLQR-MPC (with TVLQR feedback every 50 ms) tracks it. A discrete control barrier function filter then minimally modifies each command to prevent finger–finger penetration. On the cube, these layers are added on top of the same CEM planner. |

Headline results (full numbers in `result_summaries/`):

| Task | CEM-MPC | RRT + iLQR-MPC + CBF | Source |
|---|---|---|---|
| Gestures (450 tasks) | 85.3% | **100%** | exp 50 |
| Cube (240 paired cases) | **64.6%** | 36.7% | exp 51 |

The demo runs only 4 gestures and 12 cube cases, so its tables illustrate the pipeline and will not reproduce these percentages (the quick check, for example, had CEM failing its one gesture and both arms solving all 3 cube cases).

On gestures the stack wins: RRT routes around finger collisions and iLQR tracks precisely. On the cube it loses, because every stack layer changes CEM's commands using a hand model that has no cube in it (exp 51 results, "why").

---

## Running the demo

Setup and every individual step are in [`../docs/installation.md`](../docs/installation.md) and [`../docs/shadow.md`](../docs/shadow.md).
Before the first run, do step 0 there: export `artifacts/viz_hand_model/` and download the start-pose dataset.

Isaac's RTX renderer could not create a device on our GPUs, so everything runs **headless** and videos are made afterwards. The
simulator records joint and cube trajectories; these are replayed with the hand meshes in a [viser](https://viser.studio) scene and
captured by headless Chromium into MP4.

```bash
export ISAAC_PYTHON=~/envs/isaac42/bin/python
QUICK=1 ./run_demo.sh                # setup check: 1 gesture + 3 cube cases + 1 cube video pair  (~75 min, measured)
./run_demo.sh                        # full demo: 4 gestures + 12 cube cases, all videos            (~4-5 h, estimated)
./run_demo.sh gestures               # gesture task only
./run_demo.sh cube                   # cube task only
NO_VIDEO=1 ./run_demo.sh             # numbers only
GPU=3 OUT=demo_outputs/mine ./run_demo.sh   # choose GPU and output folder
```

Each run writes to a new folder; existing output is never overwritten:

```
demo_outputs/<date_time>/
├── gestures/results.json, traces.npz        per-task metrics and recorded joint trajectories
├── cube/results.json, traces_<arm>.npz      per-case metrics and recorded hand + cube trajectories
├── videos/gestures/<gesture>_<controller>_<task>.mp4 (+ .gif)
├── videos/cube/case<NNN>_{CEM,STACK}.mp4 (+ .png poster, case<NNN>.json)
└── logs/                                    full stdout of every step
```

These timings were measured on the quick check, with one GPU and 2 CPUs:

| Stage | Time |
|---|---|
| gesture evaluation | 9 min |
| cube evaluation | 26 min (the stack arm is about 13× slower than CEM: 1450 s against 111 s for 3 cases) |
| rendering | about 8–10 min per video clip |

The full run adds 3 more gestures, 9 more cube cases and 10 more clips (14 in total), so most of its time goes to rendering. Use
`NO_VIDEO=1` for numbers only, or `RENDER_CASES="..."` to render fewer cube cases.

The gesture log also shows a `direct` row: an open-loop baseline that sends the target joints straight to the hand. It is printed
for reference only.

Pre-rendered GIFs and stills from the full experiments are in [`../docs/media/shadow/`](../docs/media/shadow).

---

## Folder layout

```
SHADOW/
├── README.md                this file
├── run_demo.sh              runs both controller stacks on both tasks and renders videos
├── run_python.sh            launcher for every Isaac script (ISAAC_PYTHON, cache dirs, optional CPU pinning)
├── cpu_limits.example.json  optional: copy to cpu_limits.json to pin all jobs to two CPUs
├── requirements-viz.txt     rendering environment (.venv_viz)
├── environment-freeze.txt   pip freeze of the Isaac environment
├── constraints.txt          version pins used to build it
├── shadow_dynamics/         the library: environment, hand models, planners, controllers
├── shadow_viz/              numpy forward kinematics + viser replay viewer (used for videos)
├── scripts/                 command-line entry points, grouped by purpose (below)
├── artifacts/               identified models and checkpoints the demo needs (about 40 MB)
├── experiments/             plans of experiments 49-54 (NN_YYYY-MM-DD_topic.md)
└── result_summaries/        their results (YYYY-MM-DD_topic_results.md)
```


### `shadow_dynamics/`: library

| File | Role |
|---|---|
| **Simulation** | |
| `env.py` | Isaac Lab Shadow Hand research environment (hand only or with the cube; gravity, tendon and physics overrides) |
| `validity.py` | Per-step physical validity in Isaac: self-contact and penetration classes |
| `cube_contacts.py` | Whole-hand cube contact sensors |
| `resources.py` | Selects one visible GPU and the CPU budget before Isaac/PyTorch import |
| **Kinematics and geometry** | |
| `kinematics_only.py` | Kinematics-only prior: joint tree, axes, limits, meshes; no dynamics (exp 49) |
| `hand_geometry.py` | Link-mesh convex hulls (k-DOPs), batched FK, finger–finger penetration on GPU |
| `hand_cspace.py` | The hand's 20-D configuration space for planning: collision checks, clearance |
| **Hand models** | |
| `tendon_hand_model.py` | **Tendon hand model** (Li et al. form): rigid-body terms from kinematics, plus tendon coupling and an elastic polynomial; fitted from self-exploration. Used by both stacks. |
| `parametric_hand.py` | "October" grey-box model: ~100 physical parameters (τ, K, armature, Coulomb, tendon, limit reaction) (exp 36) |
| `physics_prior.py` | Nominal per-finger physics prior the grey-box models start from |
| `flow_model.py` | Learned whole-horizon dynamics networks (deterministic / flow) and the neural residual on the grey-box model |
| `allegro_hand_model.py` | Allegro-hand variant of the grey-box model (imported by `flow_model.py`) |
| **Planning and control** | |
| `rrt_connect.py` | **RRT-Connect**, batched over many problems on GPU, with shortcut smoothing and a path follower |
| `ilqr.py` | **iLQR-MPC** with **TVLQR** feedback on the tendon model |
| `cbf.py` | **CBF safety filter**: discrete barrier h(q_{k+n}) ≥ (1−γ)^n h(q_k), solved as a small QP per step |
| `cube_mpc.py` | **CEM-MPC** for the cube: PPO-proposed candidates, hybrid model (hand model + Isaac cube), task cost |
| `cube_reward.py` | Cube task reward/cost shared by planner and PPO |
| `cube_terminator.py` | Learned drop predictor used in the safe cost |
| `imagination.py` | Learned-model environments; the evaluator imports its model wrappers |
| `inverse_proposer.py` | Inverse-dynamics action proposers (exp 30; imported by the evaluator) |
| **Data** | |
| `excitation.py`, `structured_excitation.py` | Motion programs for self-exploration (hand-only excitation) |
| `flow_data.py`, `cube_world_data.py` | Dataset loaders for hand and hand + cube recordings |
| `data/shadow_pose_library_v3_exp49.json` | The 15 target gestures (holdable under gravity) |
| `data/shadow_pose_library.json` | Reference poses and action names (default argument of several scripts) |

### `scripts/`: entry points

Run Isaac scripts with `./run_python.sh scripts/...` (needs `ISAAC_PYTHON`, see Environments). Run rendering and analysis scripts with `.venv_viz/bin/python scripts/...`. Commands for every step are in [`../docs/shadow.md`](../docs/shadow.md).

| Folder | Script | What it does |
|---|---|---|
| `evaluate/` | `evaluate_shadow_pose_mpc.py` | Gesture benchmark in Isaac. `--exp50-runs name\|model\|cem,ilqr,direct\|none,rrt\|cbf 0/1` selects the stacks. |
| | `evaluate_shadow_cube_mpc.py` | Cube benchmark in Isaac. Arms `cem:P:hyb_tg0:safe` (CEM-MPC) and `stk:P:hyb_tg0:safe:rrt+ilqr+cbf` (stack). |
| `identify/` | `collect_exp49.py` | Self-exploration: the hand runs motion programs and records what a real hand could measure |
| | `design_tendon_excitation.py` | Chooses the next exploration programs (D-optimal) with the current model |
| | `fit_tendon_hand.py` | Fits the tendon hand model from that data only |
| | `analyze_exp49_fit.py` | Scores fitted models against the simulator's true parameters |
| `render/` | `render_gesture_gifs.py` | Gesture traces → MP4 + GIF (target hand shown beside) |
| | `exp51_traces_to_viewer.py` | Cube traces → viewer format (hand-root frame) |
| | `render_cube_pair_clips.py` | One cube case under two controllers → MP4 + poster PNG |
| | `probe_hand_root_pose.py` | Records the hand-root pose used for re-framing (output in `artifacts/exp51/hand_root_pose.json`) |
| `analyze/` | `analyze_exp50.py` | Gesture tables over seeds (success, penetration, paired tests) |
| | `analyze_exp39.py` | Cube tables over seeds (Wilson CIs, McNemar pairs) |
| `train/` | `train_shadow_ppo_isaac.py` | Trains the PPO cube policy used as CEM's proposer (exp 19) |
| | `train_shadow_cube_terminator.py` | Trains the drop predictor |
| | `fit_parametric_hand.py`, `train_shadow_flow_v3.py` | Fit the October grey-box model and its neural residual (exp 36/37a) |
| | `export_shadow_viz_model.py` | Exports link meshes and kinematic tree from the Isaac USD (`artifacts/viz_hand_model`) |
| `tools/` | `cpu_budget.py` | Pins a process to the CPUs in `cpu_limits.json` before numerical libraries load |

`train/` documents where the provided checkpoints came from. The demo does not run it, and it was not re-run after the reorganisation.

### `artifacts/`: models and data in this repository

| Path | Content |
|---|---|
| `viz_hand_model/` | **not included**: link meshes and kinematic tree, exported from the Isaac Lab asset (docs/shadow.md, step 0) |
| `shadow_structured_v3/train_speeds/` | **not included**: start-pose dataset, from Hugging Face `robochad/shadow-hand-structured-v3` (step 0) |
| `exp49/ladder/A/f4/model.json` | tendon hand model, gravity on (gesture task): 4 rounds of D-optimal self-exploration, about 25 min of data |
| `exp49/min/f_m1/model.json` | the 1-minute model: round 1 + one designed program |
| `exp49/ladder/r1`, `exp49/min/r_m1` | the recorded exploration of those two rounds (`episodes.npz`), with `d_m1.npz` (the designed program) |
| `exp49/ladder/f1/model.json` | model after round 1 (the starting point of the designs) |
| `exp49/true_model.json` | the simulator's true parameters in the same format (reference only; never read by a fit) |
| `exp51/g0/A/f4/model.json` | tendon hand model, gravity off (cube task), same procedure |
| `ppo_cube_isaac_v1/policy.pt` | PPO cube policy P (CEM's proposal mean) |
| `cube_terminator_r6/best.pt` | drop predictor used in the safe cost |
| `cube_world_r34/A_all_s0/`, `exp34/E_A/term/` | learned cube world model and its terminator (loaded by the cube evaluator's `--models`) |
| `exp51/hand_root_pose.json` | hand-root pose for re-framing cube traces in the renderer |

---

## Environments

| Purpose | Python | Notes |
|---|---|---|
| Isaac (simulation, fitting, evaluation) | `$ISAAC_PYTHON` | Isaac Sim 4.2 / Isaac Lab 1.2, Python 3.10 ([installation](../docs/installation.md)) |
| Rendering | `.venv_viz/bin/python` | viser, playwright, imageio (`requirements-viz.txt`); Chromium in `.playwright/` |

Rules followed by every script:

- exactly one visible GPU per job (`CUDA_VISIBLE_DEVICES`, `--device cuda:0`);
- output folders are never overwritten.

## Reading order for the code

1. `run_demo.sh`: the two pipelines end to end.
2. `shadow_dynamics/tendon_hand_model.py`: the hand model both stacks plan with.
3. `shadow_dynamics/cube_mpc.py`: CEM-MPC.
4. `shadow_dynamics/rrt_connect.py` → `ilqr.py` → `cbf.py`: the planning and control stack.
5. `scripts/evaluate/evaluate_shadow_pose_mpc.py` (search `exp50`) and `evaluate_shadow_cube_mpc.py` (search `stk`): how the stacks
   are composed and scored.
6. `experiments/50_*`, `51_*`, `52_*` and the matching `result_summaries/`: design and results.

Some library modules and `scripts/train/` come from earlier experiments of the project (learned flow models, PPO in imagination, a
grey-box hand model). The evaluators import them, but the current pipeline does not use them. Their checkpoints are not included.

