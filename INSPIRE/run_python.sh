#!/usr/bin/env bash
# Launcher for every Isaac Sim / Isaac Lab script in INSPIRE. INSPIRE reuses SHADOW's hand-agnostic modules
# (RRT-Connect, iLQR/TVLQR, CBF, CPU budget), so ../SHADOW is put on PYTHONPATH.
#   ISAAC_PYTHON  Python of an Isaac Sim 4.2 + Isaac Lab 1.2 install (required; SHADOW_PYTHON is accepted as an alias)
# Accept the NVIDIA Omniverse EULA yourself before the first run (export OMNI_KIT_ACCEPT_EULA=YES once you have read it).
set -euo pipefail
INSPIRE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SHADOW_ROOT="$(cd -- "$INSPIRE_ROOT/../SHADOW" && pwd)"
cd "$INSPIRE_ROOT"
PY="${ISAAC_PYTHON:-${SHADOW_PYTHON:-}}"
[ -n "$PY" ] || { echo "Set ISAAC_PYTHON to the Python of your Isaac Sim 4.2 / Isaac Lab 1.2 install (see docs/installation.md)"; exit 1; }
CACHE="${TMPDIR:-/tmp}/${USER:-isaac}-isaaclab"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE/cache}" XDG_DATA_HOME="${XDG_DATA_HOME:-$CACHE/data}" \
       XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$CACHE/config}" MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE/mpl}" PYTHONUNBUFFERED=1
export PYTHONPATH="$INSPIRE_ROOT:$SHADOW_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$SHADOW_ROOT/scripts/tools/cpu_budget.py" "$PY" "$@"
