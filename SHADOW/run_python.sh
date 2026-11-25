#!/usr/bin/env bash
# Launcher for every Isaac Sim / Isaac Lab script in SHADOW.
#   ISAAC_PYTHON  Python of an Isaac Sim 4.2 + Isaac Lab 1.2 install (required; SHADOW_PYTHON is accepted as an alias)
# Accept the NVIDIA Omniverse EULA yourself before the first run (export OMNI_KIT_ACCEPT_EULA=YES once you have read it).
# If cpu_limits.json exists, the job is pinned to the CPUs listed there (scripts/tools/cpu_budget.py); otherwise it runs unpinned.
set -euo pipefail
SHADOW_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SHADOW_PROJECT_ROOT"
PY="${ISAAC_PYTHON:-${SHADOW_PYTHON:-}}"
[ -n "$PY" ] || { echo "Set ISAAC_PYTHON to the Python of your Isaac Sim 4.2 / Isaac Lab 1.2 install (see docs/installation.md)"; exit 1; }
CACHE="${TMPDIR:-/tmp}/${USER:-isaac}-isaaclab"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE/cache}" XDG_DATA_HOME="${XDG_DATA_HOME:-$CACHE/data}" \
       XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$CACHE/config}" MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE/mpl}" PYTHONUNBUFFERED=1
exec python3 scripts/tools/cpu_budget.py "$PY" "$@"
