# Experiment 01: Inspire RH56 in Isaac Sim — import, linkage, servo limits, self-collision (gate 0)

Date: 23–24 Nov 2026. Roadmap: `experiments/00_2026-11-24_inspire_replication_roadmap.md`. Replicates the setup part of SHADOW exp 46
(Allegro port) and the gate 0 of SHADOW exp 49. Results are below, in this file (as SHADOW exp 49's gate 0).

## Asset

- Source: dex-urdf (`github.com/dexsuite/dex-urdf`, commit in `assets/inspire_hand/SOURCE_COMMIT.txt`), `robots/hands/inspire_hand/
  inspire_hand_right.urdf`, derived by the dex-urdf authors from Inspire's official STEP file (their `LICENSE.txt` copied alongside).
- `scripts/setup/prepare_inspire_urdf.py` writes `assets/inspire_hand/inspire_hand_right.urdf` (+ `_meta.json`):
  - visual `.glb` → `.obj` (the Isaac 4.2 importer reads OBJ/STL/DAE);
  - spec limits replace dex-urdf's placeholders (0.5 rad/s, 1 N m);
  - mimic offsets set to 0 (see gate 0).
- Kinematics, inertias and collision meshes are dex-urdf's. Total link mass 0.19 kg (the real DFX is ~0.54 kg, but most of that is
  in the fixed palm, which does not matter with a fixed base).

## Simulation choices (`inspire_dynamics/sim.py`)

| item | choice | why |
|---|---|---|
| actuated joints | thumb yaw, thumb pitch (bend), index/middle/ring/pinky proximal — the 6 servos | Inspire hardware |
| servo | PhysX implicit position drive, K = 10 N m/rad, D = 0.07 N m s/rad, armature 1e-4 kg m² (reflected rotor, assumed) | stiff, critically damped; motion is set by the speed limit, like the real constant-speed servo |
| speed limit | fingers 260 °/s, thumb bend 70 °/s, thumb yaw 70 °/s | reseller / Galaxea spec; thumb yaw not published (assumed = bend). Set in both the URDF and `ImplicitActuatorCfg.velocity_limit`; before any speed sweep set only the actuator value and re-measure which one PhysX honours |
| torque cap (force threshold) | fingers 10 N × 0.07 m = 0.70 N m, thumb 15 N × 0.06 m = 0.90 N m | rated tip force × lever, an assumption |
| linkages | PhysX mimic joints from the URDF importer (no drive on the 6 linkage joints) | rigid coupling, as the real linkage |
| control / physics | 20 Hz control, 120 Hz physics (6 substeps), as SHADOW | comparability |
| gravity | on (hand gravity enabled) | as SHADOW exps 46–54 |
| self-collision | on, except `hand_base_link`–`thumb_proximal` | see below |
| mount | fixed base, URDF base frame at (0, 0, 0.5) m, fingers up (+z), palm facing +x | to be set per task (experiments 03/04) |

## Gate 0 (pre-registered checks) and results

Smoke program (`scripts/setup/gate0_smoke.py`, 23.5 s, 2 identical envs, GPU 5): each servo alone to 90% of its range and back, then a fist
(thumb yaw 50%, all others 90%) held 3 s, then open. Output `artifacts/exp01/smoke/{report.json,traces.npz}`, log `logs/exp01_smoke.log`.

| check | criterion | result |
|---|---|---|
| G1 linkage holds on the GPU pipeline | linkage error < 1° | **pass**: mean 0.03°, max 0.86° (thumb intermediate while the thumb presses on the index in the fist) |
| G2 servo speed = spec | 90% rise time ≈ range / speed | **pass**: fingers 0.30 s (1.32 rad at 4.54 rad/s = 0.29 s), thumb bend 0.45 s, thumb yaw 0.90 s |
| G3 holds under gravity | free-space hold error < 0.1° | **pass**: 0.01° on every free joint |
| G4 contact behaves | fist settles without explosion | **pass**: thumb stops on the index (yaw −1.3°, bend −3.2° short of target, index pushed +0.9°) |
| G5 determinism | two envs identical | **pass**: 1e-8° |

Problems found and fixed on the way (logs `logs/exp01_diag_*.log`, `artifacts/exp01/diag_*`):

1. **Mimic offset dropped.** The importer creates `PhysxMimicJointAPI` on all 6 linkage joints, but ignores the URDF offset: the finger
   intermediates sat a constant 2.6° (= 0.045 rad, dex-urdf's offset) off the URDF law. The sim URDF now uses offset 0, so the sim's
   linkage law is exactly q_link = multiplier × q_servo (dex-urdf's offset kept in the meta file).
2. **Thumb exploded with self-collision on.** At t = 0 thumb bend went to −25°, intermediate −96°, and thumb yaw jammed at 61°. Without
   self-collision the thumb was perfect (linkage 0.05°). `scripts/setup/self_collision_pairs.py` (pure numpy FK + convex hulls, 400 random
   poses, `artifacts/exp01/self_collision_pairs.json`) found the cause: `hand_base_link`–`thumb_proximal` overlap 10.6 mm at the open pose and
   in 92% of poses (the thumb's base mechanism sits inside the palm shell in the dex-urdf collision geometry). That pair is filtered
   (`UsdPhysics.FilteredPairsAPI`); every other overlapping pair is a real thumb–finger contact and stays (thumb vs index links: 7–10% of
   random poses, up to 9 mm).
3. **Driving the linkage joints as well** (the importer's documented mimic workaround) diverges to NaN against the PhysX mimic
   constraint. Removed; the mimic constraint alone is used.

## Preview of experiment 03: direct gesture sequence

`scripts/setup/demo_gesture_sequence.py` (draft gestures as fractions of each servo's range, 1.5 s each, direct commands; output
`artifacts/exp01/gesture_sequence`): open → point → peace → thumbs_up → rock → call → pinch → tripod → fist → pinch → open. Every gesture
reaches its servo targets within 0.03°, **except tripod → fist: the index stops 34° short**, blocked by the thumb that is still crossing
in front of it (the thumb moves at 70 °/s, the index at 260 °/s). The same fist commanded from the open pose (gate 0) ends at its goal (index +0.9°, thumb 1–3° short in contact), so the 34° is
path-dependent, not an unreachable goal: the path-dependent collision failure experiment 03 is built around.
The draft pinch/tripod targets do not yet make thumb–finger contact; experiment 03 settles them in Isaac.

## Rendering

Isaac's RTX renderer cannot create a device on these GPUs (as in SHADOW), so `scripts/render/render_traces.py` replays recorded joint
traces with the hand's visual meshes in viser, captured by headless Chromium (pure-numpy FK in `inspire_dynamics/urdf_kin.py`, palm-side
three-quarter view). Clips: `artifacts/exp01/videos/gate0_servo_program.{mp4,gif,png}` (smoke program) and
`artifacts/exp01/videos/gesture_sequence_direct.{mp4,gif,png}` (direct gesture sequence).

## Tooling

`.venv_tools` (uv; numpy, scipy, trimesh 4.5.3) for asset scripts; rendering uses SHADOW's `.venv_viz` (viser, playwright) unchanged.

## Next

Experiment 02 (servo–linkage model from self-exploration). The object task for experiment 04 needs the feasibility probe and the user's choice
(roadmap, issue 2).
