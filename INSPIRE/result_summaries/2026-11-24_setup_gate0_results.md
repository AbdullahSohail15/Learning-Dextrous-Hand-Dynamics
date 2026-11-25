# Experiment 01 results: Inspire RH56 in Isaac Sim, setup and gate 0 (23–24 Nov 2026)

Plan and details: `experiments/01_2026-11-24_setup_import_gate0.md`. Nothing committed (INSPIRE is not a git repository yet).

## Gate 0: all five checks pass

| check | result |
|---|---|
| linkage (PhysX mimic joints) on GPU | mean 0.03°, max 0.86° under thumb–index contact |
| servo speed = spec | fingers 0.30 s, thumb bend 0.45 s, thumb yaw 0.90 s to 90% of a 90%-range step (= range / spec speed) |
| hold under gravity | 0.01° |
| contact | fist settles; thumb stops on the index 1–3° short of its targets |
| determinism | 1e-8° between identical envs |

## Fixed on the way

1. The importer drops the URDF mimic offset (constant 2.6° on the finger intermediates) → sim URDF offset 0.
2. dex-urdf's thumb base hull lies 10.6 mm inside the palm (92% of poses) → the thumb exploded with self-collision on → that one pair is filtered.
3. Driving the linkage joints as well → NaN against the mimic constraint → removed.

## Direct-command gesture sequence (preview of experiment 03)

All 11 gestures reach their servo targets within 0.03°, except tripod → fist (index 34° short, blocked by the slower thumb crossing in front of it).
The same fist from the open pose ends at its goal (index +0.9°), so the failure is path-dependent.
Free gestures are trivial for direct servo commands on this hand; thumb-crossing transitions and contact force are the real problems.

## Videos

`artifacts/exp01/videos/gate0_servo_program.{mp4,gif,png}` (each servo alone, then a fist) and
`artifacts/exp01/videos/gesture_sequence_direct.{mp4,gif,png}` (the sequence above).
