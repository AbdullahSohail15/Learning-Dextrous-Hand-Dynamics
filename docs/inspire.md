# Inspire RH56: commands for every step

The Inspire RH56 has six linear servos: thumb yaw, thumb bend, index, middle, ring and little. Through linkages they drive 12 joints.
This part replicates the Shadow pipeline for that hand in Isaac Sim, replacing the Shadow tendon model with a **servo–linkage model**.
The servos have no tendons and are not back-drivable. Each one is modelled as a speed- and force-limited position servo, plus the
reflected link inertia and gravity seen through the linkage (`inspire_dynamics/servo_model.py`).

All commands run from `INSPIRE/` with `ISAAC_PYTHON` set. `./run_python.sh` puts `../SHADOW` on `PYTHONPATH`, because the hand-agnostic
RRT-Connect, iLQR/TVLQR and CBF code is shared. Outputs are never overwritten, so use a new `--output` folder for every run.

```
step 0  setup              URDF, self-collision check, import smoke test, gesture library
step 1  self-exploration   closed-loop designed exploration + model fit  (exp 02, 04b)
step 2  gestures           direct / CEM / iLQR / RRT / CBF / full stack  (exp 03)
step 3  cube on the palm   CEM-MPC with the Isaac cube in the planner    (exp 04, 04b)
step 4  analysis
step 5  videos
demo    run_demo.sh
```

---

## Step 0: setup (experiment 01)

The asset is dex-urdf's `inspire_hand_right` (MIT licence, commit in `assets/inspire_hand/SOURCE_COMMIT.txt`). The prepared
simulation URDF is already in the repository. The USD file is created automatically the first time Isaac loads the hand.

```bash
python3 scripts/setup/prepare_inspire_urdf.py      # re-create assets/inspire_hand/inspire_hand_right.urdf (spec speed/force limits, mimic offsets)
../SHADOW/.venv_viz/bin/python scripts/setup/self_collision_pairs.py   # overlapping link pairs over the joint range -> artifacts/exp01/self_collision_pairs.json
CUDA_VISIBLE_DEVICES=0 ./run_python.sh scripts/setup/gate0_smoke.py --headless --coupling mimic --output runs/exp01/smoke_mimic
CUDA_VISIBLE_DEVICES=0 ./run_python.sh scripts/setup/demo_gesture_sequence.py --headless --device cuda:0 --output runs/exp01/gesture_sequence
```

What each command does:

- `prepare_inspire_urdf.py` regenerates the simulation URDF.
- `self_collision_pairs.py` lists link pairs that overlap. The dex-urdf collision meshes overlap the base and the thumb proximal link
  by 10.6 mm, so that pair is filtered in `inspire_dynamics/sim.py`.
- `gate0_smoke.py` checks the import, the PhysX mimic coupling, servo speeds and holding under gravity.
- `demo_gesture_sequence.py` sends hand-written gestures as direct servo commands. It shows where the thumb blocks a finger, which
  is the failure step 2 targets.

**Gesture library** (12 goals). Each goal is solved against the collision geometry, with at least 2 mm clearance for wraps and 1 mm
for pinches. The result is already in `assets/inspire_hand/gesture_library.json`.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:../SHADOW $ISAAC_PYTHON scripts/evaluate/build_gesture_library.py   # torch + numpy only
```

---

## Step 1: self-exploration and identification (experiments 02 and 04b)

**Gate.** First check that per-environment servo parameters reach PhysX. This step also measures the link inertia and gravity seen
through the linkage (`gate.json`, included).

```bash
CUDA_VISIBLE_DEVICES=0 ./run_python.sh scripts/identify/gate02.py --headless --device cuda:0 --output runs/exp02/gate
```

**Exploration.** One Isaac process runs, for each of 3 unknown hands, three arms side by side:

| Arm | How the next exploration chunk is chosen |
|---|---|
| CL | closed loop: designed from the current fit, then screened for self-collision |
| D0 | designed once from the spec model |
| R | random |

Every round adds 3 s chunks, up to a cumulative 15, 30, 60, 120 and 180 s. After each round, every arm is refitted on its data
(Adam, warm start). The CL arm rejects any candidate chunk whose predicted motion brings two links within 2 mm. This kept finger
contact at 0% while exploring, against 11–15% for random chunks.

```bash
CUDA_VISIBLE_DEVICES=0 ./run_python.sh scripts/identify/exp02_explore.py --headless --device cuda:0 \
    --gate artifacts/exp02/gate --output runs/exp02/explore          # writes explore.npz, fits.pt, report.json
```

The provided `artifacts/exp02/explore/fits.pt` comes from this run; every later step loads it. A 15 s exploration already predicts
free motion to 0.03° (spec model: 0.74°), and it recovers the servo speed limits to within 1%. Stiffness, damping, inertia and force
cap cannot be identified from free motion.

**Regularised refit (experiment 04b).** This is a MAP fit with the spec model as a log-space prior. It keeps the unidentifiable
parameters near their spec values instead of letting them drift (damping drifted to 0 in the plain fit). It needs PyTorch only, not
Isaac:

```bash
PYTHONPATH=.:../SHADOW $ISAAC_PYTHON scripts/identify/fit_regularised.py --explore artifacts/exp02/explore \
    --output runs/exp04b/fit --weight 1e-3
```

---

## Step 2: gesture benchmark (experiment 03)

The episode lasts 4 s at 20 Hz. Success means the final 6-servo RMSE is below 2°. All controllers use the model identified in
step 1 (arm CL, hand 0), except `full_N0`, which uses the spec model.

| Run name | Stack |
|---|---|
| `direct` | send the goal servo positions |
| `cem` | CEM-MPC |
| `ilqr` | iLQR-MPC + TVLQR |
| `rrt_direct`, `rrt_cem`, `rrt_ilqr` | RRT-Connect path followed by that controller |
| `cem_cbf` | CEM-MPC + CBF safety filter (no planner) |
| `full` | **RRT-Connect → iLQR-MPC/TVLQR → CBF**, plus the fallback (hold the servos, then replan with RRT) |
| `full_N0` | full stack with the spec model |

The `--task-set` option selects one of two task sets:

- `standard`: 12 gestures × 10 starts. It is saturated: direct commands already reach 99.7%.
- `crossing`: the 6 library pairs whose straight-line move collides, plus random starts whose straight line to a goal penetrates
  by more than 2 mm.

Both sets have 120 tasks per seed.

```bash
CUDA_VISIBLE_DEVICES=0 ./run_python.sh scripts/evaluate/exp03_gestures.py --headless --device cuda:0 --task-set crossing \
    --seeds 0,1,2 --runs direct,cem,ilqr,rrt_direct,rrt_cem,rrt_ilqr,cem_cbf,full,full_N0 \
    --fits artifacts/exp02/explore/fits.pt --output runs/exp03/crossing
```

The run writes:

| File | Content |
|---|---|
| `results.json` | per run and seed: success, finger contact, plan/CBF/RRT time, replans, per-task metrics, task list (`meta` = goal, start) |
| `traces.npz` | `<run>__<seed>`: tasks × 81 steps × 12 joints |

After the runs, the script also times every layer on one hand, on CPU and on GPU. Measured on one CPU core: iLQR 54 ms, CBF 14 ms,
RRT 31 ms. The full stack takes about 97 ms per 150 ms replanning cycle, so it runs in real time.

Stack parameters (`inspire_dynamics/control.py`):

| Layer | Parameters |
|---|---|
| CBF | γ = 0.3, n = 2, activation distance 10 mm |
| Fallback | when the re-predicted filtered command still breaks the barrier by more than 0.2 mm inside it, hold the servos (command = measured position), then let RRT replan |

---

## Step 3: cube on the palm (experiments 04 and 04b)

The hand is mounted palm up with a 4 cm, 30 g cube on its palm. The task is to turn the cube to a goal yaw (±10°, ±20° or ±30°)
within 10 s, without dropping it. Success means the yaw error is below 0.1 rad.

Every 3 control steps, CEM copies the live state into 24 branch environments and rolls each candidate out for 1.2 s, with the Isaac
cube in the loop. The arms differ in how the branch hand is simulated:

| Arm | Branch hand |
|---|---|
| `oracle` | true servo parameters |
| `param_CL` | identified servo parameters |
| `param_N0` | spec servo parameters |
| `param_CL_cbf` | identified parameters, with the CBF on the executed command |
| `param_CL_stack` | identified parameters, with CBF + TVLQR |
| `kin_CL` | identified-model trajectory written into the joints (kinematic replay) |

Feasibility probe (random servo programs and scripted curls; does the cube turn without falling?):

```bash
CUDA_VISIBLE_DEVICES=0 ./run_python.sh scripts/evaluate/exp04_probe.py --headless --device cuda:0 --output runs/exp04/probe
```

Main comparison (90 paired tasks per arm):

```bash
CUDA_VISIBLE_DEVICES=0 ./run_python.sh scripts/evaluate/exp04_cube.py --headless --device cuda:0 \
    --fits artifacts/exp02/explore/fits.pt --arms oracle,param_CL,param_N0,param_CL_stack,param_CL_cbf,kin_CL \
    --seeds 0,1,2 --tasks 30 --nb 24 --iters 2 --horizon 24 --goals-deg 10,-10,20,-20,30,-30 --output runs/exp04/cube_main
```

With the regularised model from step 1:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_python.sh scripts/evaluate/exp04_cube.py --headless --device cuda:0 \
    --fits artifacts/exp04b/fit/fits_reg.pt --gate artifacts/exp02/gate --arms param_REG,param_CL,param_N0 \
    --seeds 0,1,2 --tasks 30 --nb 24 --iters 2 --horizon 24 --goals-deg 10,-10,20,-20,30,-30 --output runs/exp04b/cube_reg
```

Isaac does not exit after a Python exception inside these scripts; it hangs. If a run stops printing, check the log for a
traceback and kill the process.

---

## Step 4: analysis

```bash
python3 scripts/analyze/analyze_exp02.py                                  # exploration table (reads artifacts/exp02/explore/report.json)
../SHADOW/.venv_viz/bin/python scripts/analyze/analyze_exp03.py runs/exp03/crossing   # per controller, per gesture, McNemar tests, failures
../SHADOW/.venv_viz/bin/python scripts/analyze/analyze_exp04.py runs/exp04/cube_main  # per arm, per goal size, paired tests
```

---

## Step 5: videos

Rendering works the same way as for Shadow: the recorded joints are replayed with the hand meshes in viser and captured by headless
Chromium. The Inspire meshes are included, so rendering works without Isaac. Run these commands from `INSPIRE/`:

```bash
export PLAYWRIGHT_BROWSERS_PATH=$PWD/../SHADOW/.playwright
VIZ="../SHADOW/.venv_viz/bin/python"

# any traces file with q (T, 12): included examples
$VIZ scripts/render/render_traces.py --traces artifacts/exp03/clips/call_full.npz --output runs/videos --name call_full \
     --title "Inspire RH56 · full stack · goal: call"
$VIZ scripts/render/render_traces.py --traces artifacts/exp02/clips/CL_hand0.npz --output runs/videos --name explore_CL \
     --title "closed-loop self-exploration"

# gesture benchmark -> clip of one task (goal shown as a translucent hand)
python3 scripts/render/exp03_clips.py runs/exp03/crossing --compare direct full 0          # list tasks where direct fails, full succeeds
python3 scripts/render/exp03_clips.py runs/exp03/crossing full 0 77 runs/clips/t77_full.npz
$VIZ scripts/render/render_traces.py --traces runs/clips/t77_full.npz --output runs/videos --name t77_full

# cube run -> clip (palm-up mount: render with --up 1 0 0)
python3 scripts/render/exp04_clips.py runs/exp04/cube_main param_CL 0 5 runs/clips/cube5.npz
$VIZ scripts/render/render_traces.py --traces runs/clips/cube5.npz --output runs/videos --name cube5 --up 1 0 0

# gesture library sheet
$VIZ scripts/render/render_gesture_gallery.py --output runs/videos/inspire_gestures.png
```

`render_traces.py` options:

| Option | Effect |
|---|---|
| `--poster --frame N` | write a single PNG |
| `--cam x y z tx ty tz` | camera position and target |
| `--every 2` | frame stride |
| `--ghost-offset` | where the goal hand is drawn |

---

## Demo: `run_demo.sh`

`run_demo.sh` runs the crossing gesture set with direct commands, CEM + CBF and the full stack. It then renders a task where direct
commands fail and the full stack succeeds:

```bash
./run_demo.sh                                   # 1 seed, 3 controllers, 2 clips
NO_VIDEO=1 ./run_demo.sh
SEEDS=0,1,2 RUNS=direct,cem,ilqr,rrt_direct,rrt_cem,rrt_ilqr,cem_cbf,full,full_N0 ./run_demo.sh   # full experiment-03 table
```

Outputs go to `demo_outputs/<date_time>/{gestures,clips,videos,logs}`.
