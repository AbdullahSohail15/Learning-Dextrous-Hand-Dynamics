# INSPIRE: the Shadow pipeline on the Inspire RH56 hand (simulation)

Our lab's real hand is an Inspire RH56. Six linear servos (thumb yaw, thumb bend, index, middle, ring, little) drive 12 joints
through linkages. This folder replicates the Shadow experiments for that hand in Isaac Sim 4.2 / Isaac Lab 1.2, with a
servo–linkage model in place of the Shadow tendon model.

Commands for every step: [`../docs/inspire.md`](../docs/inspire.md). Quick demo: `./run_demo.sh` (needs `ISAAC_PYTHON`).

| Experiment | Replicates (Shadow) | Result |
|---|---|---|
| 01 setup, import, gate 0 | Allegro port, gate 0 of exp 49 | PhysX mimic linkages hold to 0.03° on GPU; dex-urdf base–thumb overlap filtered |
| 02 servo–linkage model from closed-loop self-exploration | 49 | 15 s → 0.03° free-motion prediction (spec 0.74°); 0% contact while exploring (random 11–15%) |
| 03 gestures: direct / CEM / iLQR / RRT / CBF / full stack | 50 | crossing set: direct 95.6% → full stack 100%; fallback 9/9 recovered; real time (≈ 97 ms per 150 ms cycle) |
| 04 cube on the palm, CEM-MPC with the Isaac cube in the planner | 51–54 | oracle 87.8%, spec 81.1%, identified 51.1% |
| 04b regularised (MAP) refit | – | cube 70.0% vs 46.7% (seed 0) |

## Layout

```
run_demo.sh                 crossing gestures with 3 controllers + rendered clips
run_python.sh               Isaac launcher (ISAAC_PYTHON; puts ../SHADOW on PYTHONPATH for RRT, iLQR, CBF, CPU budget)
assets/inspire_hand/        simulation URDF (+ meta: actuated joints, linkage law, spec limits), meshes, gesture library,
                            dex-urdf licence and source commit; usd/ is generated on first use
inspire_dynamics/
  sim.py                    Isaac Lab scene: N hands (+ cube), 6 servo drives, PhysX mimic linkages, 20 Hz control / 120 Hz physics
  servo_model.py            servo-linkage dynamics model (speed limit, force cap, reflected inertia, gravity) and its fit
  explore.py                exploration programs, D-optimal design, model-based collision screen
  control.py                CEM, iLQR/TVLQR, CBF and fallback for the 6-servo hand (RRT from SHADOW)
  torch_kin.py              batched GPU kinematics + collision polytopes (planning, CBF)
  urdf_kin.py               numpy URDF kinematics + collision hulls (renderer, checks)
scripts/
  setup/                    prepare_inspire_urdf, self_collision_pairs, gate0_smoke, demo_gesture_sequence
  identify/                 gate02, exp02_explore (self-exploration + fits), fit_regularised (exp 04b)
  evaluate/                 build_gesture_library, exp03_gestures, exp04_probe, exp04_cube, profile_steps
  analyze/                  analyze_exp02/03/04 (tables, paired tests)
  render/                   render_traces (replay to MP4/GIF), exp03_clips, exp04_clips, render_gesture_gallery
artifacts/
  exp01/self_collision_pairs.json
  exp02/explore/            fits.pt (identified models, used by everything downstream), explore.npz, report.json; gate/gate.json
  exp03/{bench,crossing}/results.json, exp04/cube_main/results.json, exp04b/{fit,cube_reg}/
  exp0*/clips/*.npz         short joint traces for rendering without Isaac
experiments/, result_summaries/   plan and results of every experiment
```

## Assumptions to replace with measurements from the real hand

We found no official RH56DFX data sheet, so the spec model uses reseller figures:

| Quantity | Value |
|---|---|
| finger speed | 260 °/s |
| thumb bend speed | 70 °/s |
| thumb yaw speed | assumed equal to thumb bend |
| tip force | 10 N finger, 15 N thumb |
| servo torque cap | tip force × lever: 0.70 N m finger, 0.90 N m thumb |
| reflected rotor inertia | 1e-4 kg m² |

Experiment 02 identifies these quantities rather than trusting them.

## Asset provenance

The Inspire model comes from [dex-urdf](https://github.com/dexsuite/dex-urdf) (MIT licence), commit `f5e7132`. The dex-urdf authors
derived it from Inspire's official STEP file (`assets/inspire_hand/LICENSE.txt`). We changed the URDF for Isaac
(`scripts/setup/prepare_inspire_urdf.py`): OBJ visual meshes, spec speed and force limits, and mimic offsets set to 0.
