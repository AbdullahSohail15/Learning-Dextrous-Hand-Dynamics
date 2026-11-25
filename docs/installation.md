# Installation

Two Python environments are used:

| Environment | Used for | Contents |
|---|---|---|
| **Isaac** (`ISAAC_PYTHON`) | simulation, self-exploration, model fitting, benchmarks | Isaac Sim 4.2.0.2 + Isaac Lab 1.2 (commit `e00d625`), PyTorch 2.4.0, Python 3.10 |
| **Render** (`SHADOW/.venv_viz`) | replaying recorded traces into MP4/GIF, analysis scripts | viser, playwright + headless Chromium, imageio, trimesh, numpy |

Tested on Ubuntu 22.04 with NVIDIA A6000-class GPUs (driver ≥ 535). Every job uses one GPU.

## 1. Isaac Sim 4.2 + Isaac Lab 1.2

Follow the Isaac Lab 1.2 pip installation guide. A condensed version:

```bash
uv venv --python 3.10 ~/envs/isaac42 && source ~/envs/isaac42/bin/activate
uv pip install "isaacsim==4.2.0.2" "isaacsim-extscache-physics==4.2.0.2" "isaacsim-extscache-kit==4.2.0.2" \
    "isaacsim-extscache-kit-sdk==4.2.0.2" --extra-index-url https://pypi.nvidia.com
uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/isaac-sim/IsaacLab.git ~/envs/isaac42_src/IsaacLab
cd ~/envs/isaac42_src/IsaacLab && git checkout e00d62561a1b4ab19985bc15b40ac77c10bd5e33 && ./isaaclab.sh --install none
uv pip install -c /path/to/Learning-Dextrous-Hand-Dynamics/SHADOW/constraints.txt h5py rsl-rl-lib==2.0.2
```

`SHADOW/environment-freeze.txt` is the full `pip freeze` of the environment the results were produced with, and
`SHADOW/constraints.txt` holds the version pins used to build it.

Then point the launchers at it:

```bash
export ISAAC_PYTHON=~/envs/isaac42/bin/python
```

**NVIDIA licence.** Isaac Sim asks you to accept the NVIDIA Omniverse licence on its first start. Read it and accept it yourself:
interactively, or with `export OMNI_KIT_ACCEPT_EULA=YES`. The launchers in this repository do not accept it for you.

Every Isaac script runs headless, through `SHADOW/run_python.sh` or `INSPIRE/run_python.sh`. These scripts:

- set Isaac's cache directories;
- put the right folders on `PYTHONPATH`;
- start `$ISAAC_PYTHON`.

## 2. Rendering environment

Isaac's RTX renderer was not usable on our GPUs. The simulator therefore records joint (and cube) trajectories, and videos are made
afterwards: the trajectories are replayed with the hand meshes in a [viser](https://viser.studio) scene, and headless Chromium
captures each frame.

```bash
cd SHADOW
python3 -m venv .venv_viz
.venv_viz/bin/pip install -r requirements-viz.txt
PLAYWRIGHT_BROWSERS_PATH=$PWD/.playwright .venv_viz/bin/python -m playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=$PWD/.playwright .venv_viz/bin/python -m playwright install-deps chromium   # system libraries; needs sudo
```

The demo scripts use `SHADOW/.venv_viz` and `SHADOW/.playwright` by default. Override them with `VIZ_PY=...` and
`PLAYWRIGHT_BROWSERS_PATH=...`.

## 3. Optional: CPU pinning

On a shared machine you can confine every job, and every child process it starts, to two logical CPUs:

```bash
cp SHADOW/cpu_limits.example.json SHADOW/cpu_limits.json    # edit "logical_cpus"
```

If this file is present, `SHADOW/scripts/tools/cpu_budget.py` pins the process and sets library thread counts to 1. If it is absent,
the wrapper simply runs the program.

## 4. Data and assets that are not in git

| What | Needed by | How to get it |
|---|---|---|
| Shadow Hand link meshes and kinematic tree (`SHADOW/artifacts/viz_hand_model/`) | every Shadow script | Export it from the Isaac Lab Shadow Hand asset: [Shadow step 0](shadow.md#step-0-one-time-setup) |
| Shadow Hand exploration dataset v3 (`SHADOW/artifacts/shadow_structured_v3/train_speeds/`, 1.2 GB) | gesture benchmark (start poses, joint limits) | `hf download robochad/shadow-hand-structured-v3 --repo-type dataset --include "train_speeds/*" --local-dir SHADOW/artifacts/shadow_structured_v3` |
| Inspire hand USD (`INSPIRE/assets/inspire_hand/usd/`) | every Inspire Isaac script | Created automatically from the URDF on first use |

The Shadow Hand meshes come from NVIDIA's Isaac Lab asset and are not redistributed here. The Inspire meshes come from
[dex-urdf](https://github.com/dexsuite/dex-urdf) (MIT licence), commit `f5e7132`; see `INSPIRE/assets/inspire_hand/LICENSE.txt`.

## 5. Check the installation

```bash
cd INSPIRE && ./run_demo.sh            # Isaac + rendering, a few minutes
```

If you only want to check rendering, without Isaac:

```bash
cd INSPIRE
PLAYWRIGHT_BROWSERS_PATH=../SHADOW/.playwright ../SHADOW/.venv_viz/bin/python scripts/render/render_traces.py \
    --traces artifacts/exp03/clips/call_full.npz --output /tmp/inspire_check --name call_full --title "render check"
```
