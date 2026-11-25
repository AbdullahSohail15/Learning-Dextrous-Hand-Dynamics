#!/usr/bin/env bash
# Shadow Hand demo: run both controller stacks headless in Isaac Sim and save rendered videos.
#
#   Stack 1  CEM-MPC                 sampling MPC, 32 candidates x 3 rounds, PPO proposer (cube) / tendon hand model
#   Stack 2  RRT + iLQR-MPC + CBF    RRT-Connect path -> iLQR-MPC with TVLQR feedback -> CBF self-collision filter
#
# Both stacks use the same identified tendon hand model (exp 49/51), learned only from the hand's own exploration.
#
# Usage (from anywhere; run inside tmux, it takes a while):
#   ./run_demo.sh                 # everything: gestures + cube, evaluation + videos
#   ./run_demo.sh gestures        # gesture reaching only (evaluation + videos)
#   ./run_demo.sh cube            # in-hand cube reorientation only (evaluation + videos)
#   QUICK=1 ./run_demo.sh         # smallest run (1 gesture, 3 cube cases, 1 cube video pair) to check the setup
#   NO_VIDEO=1 ./run_demo.sh      # numbers only, skip rendering
#
# Options (environment variables):
#   GPU=0                 physical GPU index (one visible GPU per job)
#   OUT=demo_outputs/...  output folder (default: demo_outputs/<date_time>; never overwritten)
#   GESTURES="..."        gestures to run (default: 4 gestures shown in the slides)
#   CUBE_CASES_PER_BIN=4  cube start cases per difficulty bin (3 bins: 0.4-1, 1-2, 2-3.14 rad from the goal)
#   RENDER_CASES="0 5"    cube cases to render (default: hardest case of each bin)
#   ISAAC_PYTHON=...      Python of your Isaac Sim 4.2 / Isaac Lab 1.2 install (required; SHADOW_PYTHON also accepted)
#   VIZ_PY=...            rendering Python with viser + playwright (default: .venv_viz/bin/python, see docs/installation.md)
#
# Rendering note: Isaac's RTX renderer cannot create a device on these GPUs, so physics runs headless and records joint and
# cube traces. Videos are made afterwards by replaying the traces with the hand meshes in a viser scene captured by headless
# Chromium (about 5 s per frame, so ~5-10 min per clip). Outputs: <OUT>/videos/{gestures,cube}/*.mp4 (+ .gif / .png poster).

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
MODE="${1:-all}"
QUICK="${QUICK:-0}"; NO_VIDEO="${NO_VIDEO:-0}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export ISAAC_PYTHON="${ISAAC_PYTHON:-${SHADOW_PYTHON:-}}"
[ -n "$ISAAC_PYTHON" ] || { echo "Set ISAAC_PYTHON (see docs/installation.md)"; exit 1; }
[ -f artifacts/viz_hand_model/model.json ] || { echo "artifacts/viz_hand_model is missing: run scripts/train/export_shadow_viz_model.py first (docs/shadow.md, step 0)"; exit 1; }
[ -f artifacts/shadow_structured_v3/train_speeds/manifest.json ] || { echo "artifacts/shadow_structured_v3/train_speeds is missing: download it from Hugging Face (docs/shadow.md, step 0)"; exit 1; }
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$ROOT/.playwright}"
VIZ_PY="${VIZ_PY:-$ROOT/.venv_viz/bin/python}"
OUT="${OUT:-demo_outputs/$(date +%Y%m%d_%H%M%S)}"
[ -e "$OUT/DONE" ] && { echo "$OUT already holds a finished demo; choose another OUT"; exit 1; }
mkdir -p "$OUT/logs" "$OUT/videos"

# ---- fixed inputs (trained models, all under artifacts/) -------------------------------------------------------------
GESTURE_LIBRARY=shadow_dynamics/data/shadow_pose_library_v3_exp49.json   # 15 target gestures
GESTURE_HAND_MODEL=artifacts/exp49/ladder/A/f4/model.json                # tendon hand model, gravity on (exp 49)
CUBE_HAND_MODEL=artifacts/exp51/g0/A/f4/model.json                       # tendon hand model, gravity off (exp 51)
PPO_PROPOSER=artifacts/ppo_cube_isaac_v1/policy.pt                       # PPO policy: CEM's proposal mean (cube)
TERMINATOR=artifacts/cube_terminator_r6/best.pt                          # drop predictor used by the safe cost
WORLD_MODEL=artifacts/cube_world_r34/A_all_s0/best.pt,artifacts/exp34/E_A/term/best.pt
HAND_ROOT_POSE=artifacts/exp51/hand_root_pose.json                       # Isaac hand root pose, for re-framing cube traces

if [ "$QUICK" = 1 ]; then GESTURES="${GESTURES:-thumb_middle_pinch}"; CUBE_CASES_PER_BIN="${CUBE_CASES_PER_BIN:-1}"
else GESTURES="${GESTURES:-thumb_middle_pinch ok_sign tripod thumbs_up}"; CUBE_CASES_PER_BIN="${CUBE_CASES_PER_BIN:-4}"; fi

VIZ() { python3 scripts/tools/cpu_budget.py "$VIZ_PY" "$@"; }   # rendering jobs under the same 2-CPU budget
step() { echo; echo "=== $1  ($(date +%T))"; }

# ---- 1. gesture reaching: 6 s per task, success = final 24-joint RMSE < 2 deg -----------------------------------------
gestures() {
  step "gestures: CEM-MPC vs RRT + iLQR-MPC + CBF  ($GESTURES)"
  # --exp50-runs: name | hand model label | local controller | global planner | CBF on/off
  ./run_python.sh scripts/evaluate/evaluate_shadow_pose_mpc.py --headless --device cuda:0 --gravity \
    --poses "$GESTURE_LIBRARY" --recorded-targets 0 --objectives joint --gestures $GESTURES --starts-per-target 1 \
    --models --tendon-models A4="$GESTURE_HAND_MODEL" \
    --exp50-runs "cem|A4|cem|none|0" "rrt_ilqr_cbf|A4|ilqr|rrt|1" --controllers direct --measure-validity \
    --task-seed 610000 --seconds 6 --output "$OUT/gestures" 2>&1 | tee "$OUT/logs/gestures.log" | grep -E "^RUN|^RRT|COMPLETE|Error|Traceback" || true
  grep -q POSE_MPC_COMPLETE "$OUT/logs/gestures.log" || { echo "gesture evaluation failed, see $OUT/logs/gestures.log"; exit 1; }
  python3 - "$OUT/gestures/results.json" <<'PY'
import json, sys
R = json.load(open(sys.argv[1]))["runs"]
print(f"{'controller':16s} {'success':>8s} {'joint RMSE (deg)':>17s} {'self-penetration':>17s} {'plan ms':>8s}")
for k in ["cem", "rrt_ilqr_cbf"]:
    s = R[k]; print(f"{k:16s} {s['success_joint']:8.2f} {s['joint_rmse_mean']:17.2f} {s['invalid_frac']:17.3f} {s['plan_ms']:8.0f}")
PY
  [ "$NO_VIDEO" = 1 ] && return
  step "gesture videos (both controllers, every task)"
  N=$(python3 -c "import json; print(len(json.load(open('$OUT/gestures/results.json'))['tasks']))")
  JOBS=""; for i in $(seq 0 $((N - 1))); do JOBS="$JOBS cem:$i rrt_ilqr_cbf:$i"; done
  VIZ scripts/render/render_gesture_gifs.py --run "$OUT/gestures" --controller cem --output "$OUT/videos/gestures" \
    --port 8131 --jobs $JOBS 2>&1 | tee "$OUT/logs/gesture_videos.log" | grep -E "SAVED|COMPLETE|Error|Traceback" || true
}

# ---- 2. in-hand cube reorientation: 10 s per case, success = cube within 0.1 rad (5.7 deg) of the goal ----------------
cube() {
  step "cube: CEM-MPC vs CEM + RRT + iLQR-MPC + CBF  ($CUBE_CASES_PER_BIN cases per bin)"
  # arms: cem:<proposer>:hyb_<hand model>:safe = CEM-MPC; the Isaac cube is simulated in the planner's branch envs,
  #       the hand inside them follows the tendon model's prediction.  stk:...:rrt+ilqr+cbf = the same CEM plus the stack.
  ./run_python.sh scripts/evaluate/evaluate_shadow_cube_mpc.py --headless --device cuda:0 \
    --policies P="$PPO_PROPOSER" --terminator "$TERMINATOR" --models A_all="$WORLD_MODEL" \
    --tendon-hand-models tg0="$CUBE_HAND_MODEL" --candidates 32 --oracle-candidates 32 \
    --cases-per-bin "$CUBE_CASES_PER_BIN" --seconds 10 --seed 790000 --output "$OUT/cube" \
    --arms cem:P:hyb_tg0:safe stk:P:hyb_tg0:safe:rrt+ilqr+cbf 2>&1 | tee "$OUT/logs/cube.log" | grep -E "^RUN|COMPLETE|Error|Traceback" || true
  grep -q CUBE_MPC_EVAL_COMPLETE "$OUT/logs/cube.log" || { echo "cube evaluation failed, see $OUT/logs/cube.log"; exit 1; }
  python3 - "$OUT/cube/results.json" <<'PY'
import json, sys
R = json.load(open(sys.argv[1]))["runs"]
print(f"{'arm':32s} {'success':>8s} {'drops':>6s} {'pen > 3 mm':>11s} {'median t (s)':>13s}")
for k in ["cem:P:hyb_tg0:safe", "stk:P:hyb_tg0:safe:rrt+ilqr+cbf"]:
    s = R[k]["summary"]; print(f"{k:32s} {s['success_rate']:8.2f} {s['drop_rate']:6.2f} {s['self_pen_3mm_frac']:11.3f} {s['median_time_to_success'] or float('nan'):13.2f}")
PY
  [ "$NO_VIDEO" = 1 ] && return
  step "cube videos"
  VIZ scripts/render/exp51_traces_to_viewer.py --pose "$HAND_ROOT_POSE" --out "$OUT/cube/clips.npz" \
    CEM="$OUT/cube/traces_cem__P__hyb_tg0__safe.npz" STACK="$OUT/cube/traces_stk__P__hyb_tg0__safe__rrt+ilqr+cbf.npz"
  if [ -z "${RENDER_CASES:-}" ]; then   # hardest start (largest initial error) in each difficulty bin
    RENDER_CASES=$(python3 - "$OUT/cube/results.json" "$QUICK" <<'PY'
import json, sys
R = json.load(open(sys.argv[1])); e = R["initial_error_rad"]; b = R["bins"]; pick = []
for lo, hi in zip(b[:-1], b[1:]):
    idx = [i for i, x in enumerate(e) if lo <= x < hi]
    if idx: pick.append(max(idx, key=lambda i: e[i]))
print(" ".join(map(str, pick[-1:] if sys.argv[2] == "1" else pick)))
PY
)
  fi
  for c in $RENDER_CASES; do
    VIZ scripts/render/render_cube_pair_clips.py --traces "$OUT/cube/clips.npz" --case "$c" --controllers CEM STACK \
      --labels "CEM-MPC + tendon hand model" "Same CEM + RRT + iLQR-MPC/TVLQR + CBF" \
      --output "$OUT/videos/cube" --port 8125 --stride 2 2>&1 | tee -a "$OUT/logs/cube_videos.log" | grep -E "SAVED|WROTE|Error|Traceback" || true
  done
}

case "$MODE" in
  gestures) gestures ;;
  cube)     cube ;;
  all)      gestures; cube ;;
  *) echo "usage: $0 [all|gestures|cube]"; exit 1 ;;
esac
touch "$OUT/DONE"
step "done: results in $OUT, videos in $OUT/videos"
find "$OUT/videos" -name "*.mp4" | sort
