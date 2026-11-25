# Shadow Hand: commands for every step

All commands run from `SHADOW/` with `ISAAC_PYTHON` set (see [installation](installation.md)). Isaac scripts go through
`./run_python.sh`. Pick the GPU with `CUDA_VISIBLE_DEVICES=<i>` and always pass `--device cuda:0`.

**Output folders are never overwritten.** Every script refuses to write into an existing folder, so give each run a new `--output`.

```
step 0  one-time setup            export hand meshes, download the start-pose dataset
step 1  self-exploration          collect -> fit -> design -> collect -> fit ...   (the hand learns its own model)
step 2  gesture benchmark         CEM-MPC vs RRT + iLQR-MPC/TVLQR + CBF, 15 gestures
step 3  cube benchmark            CEM-MPC vs CEM + stack, in-hand reorientation
step 4  analysis                  tables, paired tests
step 5  videos                    replay recorded traces with the hand meshes
demo    run_demo.sh               steps 2, 3 and 5 on a few tasks
```

---

## Step 0: one-time setup

**Hand meshes and kinematic tree.** These are exported from Isaac Lab's Shadow Hand asset and are not redistributed here:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_python.sh scripts/train/export_shadow_viz_model.py --headless --device cuda:0 --output artifacts/viz_hand_model
```

This writes:

| File | Content |
|---|---|
| `model.json` | joint tree, limits and names |
| `meshes.npz` | link meshes |
| `fk_check.npz` | simulator body poses, used to check the numpy forward kinematics |
| `episodes.npz` | a few test motions |

Every script reads this folder. The planners use it for forward kinematics and collision geometry, and the renderer uses it to draw
the hand.

**Start-pose dataset.** The gesture benchmark draws half of its start poses from recorded held-out exploration states:

```bash
pip install -U huggingface_hub     # gives the `hf` command (older versions: huggingface-cli download ...)
hf download robochad/shadow-hand-structured-v3 --repo-type dataset --include "train_speeds/*" \
   --local-dir artifacts/shadow_structured_v3
```

---

## Step 1: self-exploration and model identification (experiment 49)

The hand starts with nothing but its kinematics: link meshes, joint axes and limits. It then learns the parameters of a
tendon-driven dynamics model (the Li et al. form; `shadow_dynamics/tendon_hand_model.py`) from motions it chooses itself.

**Round 1: a few slow random programs.** Each program is a 20 s sequence of 20 joint targets, played open loop on the hand's own
joint PD drives at 20 Hz.

```bash
./run_python.sh scripts/identify/collect_exp49.py --headless --device cuda:0 \
    --random 3 --speeds 10 --program-seed 4920000 --output runs/exp49/r1
```

Before a program runs it is checked kinematically: the link meshes are moved along the commanded path. If two links would overlap by
more than 1 mm, the program is shrunk toward its start; if it still fails, it is dropped. While it runs, every control step gets a
validity label:

| Label | Meaning | What happens to the data |
|---|---|---|
| free | no links touching | used for the fit |
| self-contact | links touching | kept, but masked out of the fit |
| invalid | penetration or numerical failure | episode truncated |

**Fit.** Stage 1 fits the actuator gains K and D by least squares on the measured drive torque. Stage 2 minimises the equation-error
(inverse-dynamics) residual over the plant parameters with Adam, keeping the inertias physically consistent.

```bash
./run_python.sh scripts/identify/fit_tendon_hand.py --data runs/exp49/r1 --output runs/exp49/f1
```

**Design the next programs (D-optimal).** The equation residual is linear in about 388 plant parameters: τ = Y(q, q̇, q̈) π.
For each of 2048 candidate programs that pass the kinematic check, the current model predicts the motion and accumulates the
information F = Σ YᵀY. Candidates are then chosen greedily to maximise log det(F_collected + F_candidate).

```bash
./run_python.sh scripts/identify/design_tendon_excitation.py --model runs/exp49/f1/model.json --collected runs/exp49/r1 \
    --select 6 --first-program-seed 4922000 --output runs/exp49/A/d2          # writes d2.npz (programs) + d2.json (report)
./run_python.sh scripts/identify/collect_exp49.py --headless --device cuda:0 --programs runs/exp49/A/d2.npz --output runs/exp49/A/r2
./run_python.sh scripts/identify/fit_tendon_hand.py --data runs/exp49/r1 runs/exp49/A/r2 --output runs/exp49/A/f2
```

Repeat the design → collect → fit cycle with `--select 18` (seed `4923000`) and `--select 54` (seed `4924000`). Each fit is a fresh
fit on all data so far. The provided `artifacts/exp49/ladder/A/f4/model.json` is the result after those rounds (about 25 min of
data). For the random baseline, add `--random` to the design command.

**The 1-minute model.** One D-optimal program after round 1 is already enough for the gesture task: 89.3% of gestures, against
92.1% with the true parameters.

```bash
./run_python.sh scripts/identify/design_tendon_excitation.py --model artifacts/exp49/ladder/f1/model.json \
    --collected artifacts/exp49/ladder/r1 --select 1 --first-program-seed 4940100 --output runs/exp49/d_m1
./run_python.sh scripts/identify/collect_exp49.py --headless --device cuda:0 --programs runs/exp49/d_m1.npz --output runs/exp49/r_m1
./run_python.sh scripts/identify/fit_tendon_hand.py --data artifacts/exp49/ladder/r1 runs/exp49/r_m1 --output runs/exp49/f_m1
```

The recordings of round 1 and of the 1-minute program are included (`artifacts/exp49/ladder/r1`, `artifacts/exp49/min/r_m1`). You
can therefore run the fit alone, without the simulator. `fit_tendon_hand.py` only needs PyTorch and the hand model from step 0:

```bash
./run_python.sh scripts/identify/fit_tendon_hand.py --data artifacts/exp49/ladder/r1 artifacts/exp49/min/r_m1 --output runs/exp49/f_m1_refit
```

`artifacts/exp49/true_model.json` holds the simulator's true parameters in the same format, as a reference model ("TRUE"). The fits
never read it. `scripts/identify/analyze_exp49_fit.py` scores fits against the truth. It also needs a held-out gate-0 test
recording (`--test`, with `tendon_ls.json`), which is not included.

---

## Step 2: gesture benchmark (experiment 50)

**Task.** Move from a start pose to one of 15 target gestures within 6 s. Success means the final 24-joint RMSE is below 2°.
There are 150 tasks per seed (15 gestures × 10 starts), and the paper table uses seeds 610000, 611000 and 612000.

Controllers are chosen with `--exp50-runs "name|model|local controller|global planner|CBF"`:

| `--exp50-runs` entry | Stack |
|---|---|
| `cem\|A4\|cem\|none\|0` | CEM-MPC (128 candidates × 4 iterations, 0.6 s horizon) |
| `ilqr\|A4\|ilqr\|none\|0` | iLQR-MPC + TVLQR |
| `rrt_ilqr\|A4\|ilqr\|rrt\|0` | RRT-Connect path → iLQR-MPC + TVLQR |
| `cem_cbf\|A4\|cem\|none\|1` | CEM-MPC + CBF safety filter |
| `rrt_ilqr_cbf\|A4\|ilqr\|rrt\|1` | **full stack**: RRT-Connect → iLQR-MPC/TVLQR → CBF |
| `rrt_direct\|A4\|direct\|rrt\|0`, `rrt_cem\|A4\|cem\|rrt\|0` | ablations |

```bash
CUDA_VISIBLE_DEVICES=0 ./run_python.sh scripts/evaluate/evaluate_shadow_pose_mpc.py --headless --device cuda:0 --gravity \
    --poses shadow_dynamics/data/shadow_pose_library_v3_exp49.json --recorded-targets 0 --objectives joint --starts-per-target 10 \
    --models --tendon-models A4=artifacts/exp49/ladder/A/f4/model.json \
    --exp50-runs "cem|A4|cem|none|0" "rrt_direct|A4|direct|rrt|0" "rrt_cem|A4|cem|rrt|0" "ilqr|A4|ilqr|none|0" \
                 "rrt_ilqr|A4|ilqr|rrt|0" "cem_cbf|A4|cem|none|1" "rrt_ilqr_cbf|A4|ilqr|rrt|1" \
    --controllers direct --measure-validity --task-seed 610000 --seconds 6 --output runs/exp50/gestures_s610000
```

To test another hand model, replace `A4=...` with your own fit (for example `M1=runs/exp49/f_m1/model.json`) and use that label in
`--exp50-runs`. `--gestures ok_sign tripod` restricts the run to a few gestures.

The run writes:

| File | Content |
|---|---|
| `results.json` | per controller: success, joint and fingertip error, self-contact and penetration fractions, planning time, per-task metrics |
| `traces.npz` | `<controller>/q` (tasks × 121 steps × 24 joints), `targets`, errors (used by the renderer) |

CBF parameters are `--cbf-n 4 --cbf-gamma 0.2 --cbf-dact 0.02` (horizon, decay rate, activation distance in m). RRT replans when
progress stalls for `--stall-replan-s 1.0` s.

---

## Step 3: cube benchmark (experiment 51)

**Task.** Turn a 6 cm cube in the palm to a goal orientation within 10 s. Success means the cube ends within 0.1 rad of the goal.
Cases are drawn in three difficulty bins (0.4–1, 1–2 and 2–3.14 rad from the goal). The planner simulates the cube in Isaac branch
environments, while the hand in those branches follows the tendon model's prediction.

```bash
CUDA_VISIBLE_DEVICES=0 ./run_python.sh scripts/evaluate/evaluate_shadow_cube_mpc.py --headless --device cuda:0 \
    --policies P=artifacts/ppo_cube_isaac_v1/policy.pt --terminator artifacts/cube_terminator_r6/best.pt \
    --models A_all=artifacts/cube_world_r34/A_all_s0/best.pt,artifacts/exp34/E_A/term/best.pt \
    --tendon-hand-models tg0=artifacts/exp51/g0/A/f4/model.json --candidates 32 --oracle-candidates 32 \
    --cases-per-bin 40 --seconds 10 --seed 790000 --output runs/exp51/eval_s790000 \
    --arms cem:P:hyb_tg0:safe stk:P:hyb_tg0:safe:ilqr stk:P:hyb_tg0:safe:cbf stk:P:hyb_tg0:safe:rrt+ilqr+cbf
```

Arms are written `type:proposer:hand:cost[:layers]`:

| Arm | What it is |
|---|---|
| `cem:P:hyb_tg0:safe` | CEM-MPC. PPO policy P proposes the candidate mean; the hybrid model is the tendon hand plus the Isaac cube; `safe` is the drop-aware cost. |
| `stk:...:rrt+ilqr+cbf` | the same CEM, then RRT, iLQR/TVLQR and CBF layers on each command (`ilqr` or `cbf` alone are ablations) |
| `cem:P:oracle:safe` | upper bound: the branch environments run the true hand |

`artifacts/exp51/g0/A/f4/model.json` is the tendon model identified with gravity off, using the same step-1 procedure as the cube
task. The arguments of the other cube experiments (52–54: soft hand replay, a 20 s cap, the cube-aware stack) are listed in
`experiments/52_*` to `54_*`.

---

## Step 4: analysis

```bash
.venv_viz/bin/python scripts/analyze/analyze_exp50.py      # gesture tables over seeds, per-gesture success, paired McNemar tests
.venv_viz/bin/python scripts/analyze/analyze_exp39.py      # cube tables (Wilson CIs, McNemar pairs)
```

The analysis scripts read the `artifacts/exp50/...` and `artifacts/exp51/...` paths shown in their headers. Point them at your
`runs/` folders, or copy your runs there.

---

## Step 5: videos

The simulator records joint (and cube) trajectories; the renderer replays them with the hand meshes in viser, captured by headless
Chromium. Rendering takes about 5 s per frame, so a clip takes 5–10 minutes. Run these commands from `SHADOW/`:

```bash
export PLAYWRIGHT_BROWSERS_PATH=$PWD/.playwright
VIZ=".venv_viz/bin/python"

# gesture clips (closed-loop hand + green target hand), one per task:
$VIZ scripts/render/render_gesture_gifs.py --run runs/exp50/gestures_s610000 --controller cem --output runs/videos/gestures \
     --jobs cem:32 rrt_ilqr_cbf:32            # <controller>:<task index> pairs

# final-pose stills, goal as translucent ghost (the "CEM fails, stack corrects it" figure):
$VIZ scripts/render/render_final_pose_stills.py --run runs/exp50/gestures_s610000 --tasks 84 106 \
     --controllers cem rrt_ilqr_cbf --output runs/videos/stills

# gesture library sheet:
$VIZ scripts/render/render_gesture_gallery.py --run runs/exp50/gestures_s610000 --output runs/videos/library.png

# self-exploration GIF from a step-1 recording (measured joints, 120 Hz):
$VIZ scripts/render/render_exploration_gif.py --episodes artifacts/exp49/min/r_m1/episodes.npz --start 8.5 --seconds 8 \
     --output runs/videos/explore_m1

# cube clips: re-frame traces into the hand-root frame, then one case under two controllers:
$VIZ scripts/render/exp51_traces_to_viewer.py --pose artifacts/exp51/hand_root_pose.json --out runs/videos/cube_clips.npz \
     CEM=runs/exp51/eval_s790000/traces_cem__P__hyb_tg0__safe.npz STACK=runs/exp51/eval_s790000/traces_stk__P__hyb_tg0__safe__rrt+ilqr+cbf.npz
$VIZ scripts/render/render_cube_pair_clips.py --traces runs/videos/cube_clips.npz --case 87 --controllers CEM STACK \
     --labels "CEM-MPC" "CEM + RRT + iLQR + CBF" --output runs/videos/cube --stride 2
```

Each command writes an MP4 and a GIF, or a PNG for the stills. Use `--port` to run two renders at once.

---

## Demo: `run_demo.sh`

`run_demo.sh` runs steps 2, 3 and 5 on a small set of tasks with the provided models:

```bash
QUICK=1 ./run_demo.sh          # setup check: 1 gesture, 3 cube cases, 1 cube video pair  (~75 min on one GPU)
./run_demo.sh                  # 4 gestures + 12 cube cases + all videos                  (~4–5 h; mostly rendering)
./run_demo.sh gestures         # gesture task only
./run_demo.sh cube             # cube task only
NO_VIDEO=1 ./run_demo.sh       # numbers only
GPU=1 OUT=demo_outputs/mine ./run_demo.sh
```

The demo writes:

```
demo_outputs/<date_time>/
├── gestures/results.json, traces.npz
├── cube/results.json, traces_<arm>.npz
├── videos/gestures/<gesture>_<controller>_<task>.mp4 (+ .gif)
├── videos/cube/case<NNN>_{CEM,STACK}.mp4 (+ .png poster)
└── logs/
```

The demo uses few tasks, so its percentages are illustrative. The paper numbers come from the full step 2 and step 3 runs (450
gesture tasks, 240 paired cube cases).
