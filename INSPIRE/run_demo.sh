#!/usr/bin/env bash
# Inspire RH56 demo: gesture transitions whose straight-line move would drive the thumb through a finger (experiment 03,
# "crossing" set), solved by direct servo commands, CEM + CBF (no planner) and the full stack (RRT + iLQR-MPC/TVLQR + CBF + fallback),
# all with the servo-linkage model the hand identified by self-exploration (artifacts/exp02/explore/fits.pt). Then renders one task
# where direct commands fail and the full stack succeeds.
#
# Usage (from anywhere):
#   ./run_demo.sh                    # 1 seed (120 tasks per controller, a few minutes on one GPU) + 2 rendered clips
#   NO_VIDEO=1 ./run_demo.sh         # numbers only
#   SEEDS=0,1,2 ./run_demo.sh        # the full benchmark of the paper table (3 seeds)
#
# Options (environment variables):
#   GPU=0                 physical GPU index (one visible GPU per job)
#   OUT=demo_outputs/...  output folder (default demo_outputs/<date_time>; never overwritten)
#   RUNS=direct,cem_cbf,full   controllers (all: direct,cem,ilqr,rrt_direct,rrt_cem,rrt_ilqr,cem_cbf,full,full_N0)
#   TASK_SET=crossing     or standard (12 gestures x 10 starts; saturated: direct already ~100%)
#   ISAAC_PYTHON=...      Python of your Isaac Sim 4.2 / Isaac Lab 1.2 install (required)
#   VIZ_PY=...            rendering Python (default ../SHADOW/.venv_viz/bin/python)
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
NO_VIDEO="${NO_VIDEO:-0}"; SEEDS="${SEEDS:-0}"; RUNS="${RUNS:-direct,cem_cbf,full}"; TASK_SET="${TASK_SET:-crossing}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$ROOT/../SHADOW/.playwright}"
VIZ_PY="${VIZ_PY:-$ROOT/../SHADOW/.venv_viz/bin/python}"
OUT="${OUT:-demo_outputs/$(date +%Y%m%d_%H%M%S)}"
[ -e "$OUT" ] && { echo "$OUT exists; choose another OUT"; exit 1; }
mkdir -p "$OUT/logs" "$OUT/videos"
step() { echo; echo "=== $1  ($(date +%T))"; }

step "gestures ($TASK_SET set, seeds $SEEDS): $RUNS"
./run_python.sh scripts/evaluate/exp03_gestures.py --headless --device cuda:0 --task-set "$TASK_SET" --seeds "$SEEDS" --runs "$RUNS" \
  --fits artifacts/exp02/explore/fits.pt --output "$OUT/gestures" 2>&1 | tee "$OUT/logs/gestures.log" | grep -E "^RUN|Error|Traceback" || true
[ -f "$OUT/gestures/results.json" ] || { echo "gesture run failed, see $OUT/logs/gestures.log"; exit 1; }
python3 - "$OUT/gestures/results.json" <<'PY'
import json, sys
R = json.load(open(sys.argv[1]))["runs"]; names = sorted({k.split("|")[0] for k in R}, key=list(dict.fromkeys(k.split("|")[0] for k in R)).index)
print(f"{'controller':12s} {'success':>8s} {'finger contact':>15s} {'plan ms':>8s} {'CBF ms':>7s} {'RRT replans':>12s}")
for n in names:
    rs = [v for k, v in R.items() if k.split("|")[0] == n]; m = lambda f: sum(r[f] for r in rs) / len(rs)
    print(f"{n:12s} {100 * m('success'):7.1f}% {100 * m('contact_frac'):14.1f}% {m('plan_ms_median'):8.0f} {m('cbf_ms_median'):7.0f} {sum(r['rrt_replans'] for r in rs):12d}")
PY
[ "$NO_VIDEO" = 1 ] && { touch "$OUT/DONE"; exit 0; }

step "videos: a task where direct commands fail and the full stack succeeds"
SEED0="${SEEDS%%,*}"
python3 scripts/render/exp03_clips.py "$OUT/gestures" --compare direct full "$SEED0" | tee "$OUT/logs/compare.txt"
TASK=$(awk '/^  task/ {print $2; exit}' "$OUT/logs/compare.txt")
[ -n "$TASK" ] || { echo "no task where direct fails and full succeeds in this run; nothing to render"; touch "$OUT/DONE"; exit 0; }
for c in direct full; do
  python3 scripts/render/exp03_clips.py "$OUT/gestures" "$c" "$SEED0" "$TASK" "$OUT/clips/task${TASK}_$c.npz"
  python3 ../SHADOW/scripts/tools/cpu_budget.py "$VIZ_PY" scripts/render/render_traces.py --traces "$OUT/clips/task${TASK}_$c.npz" \
    --output "$OUT/videos" --name "task${TASK}_$c" --title "Inspire RH56 · $c · task $TASK (goal: translucent hand)" 2>&1 \
    | tee -a "$OUT/logs/videos.log" | grep -E "SAVED|Error|Traceback" || true
done
touch "$OUT/DONE"
step "done: results in $OUT/gestures, videos in $OUT/videos"
ls "$OUT/videos"
