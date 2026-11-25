# Experiment 49 results: Li-form tendon model from scratch, self-exploration only (Shadow Hand, gravity on)

Plan: `experiments/49_2026-11-04_tendon_model_from_scratch_self_identification.md`. Model `shadow_dynamics/tendon_hand_model.py` (structure of Li et al., arXiv 2408.13044: M(q)q̈ + N + G + τ_k + τ_d + τ_f = Cᵀ(f_t − f_f) + passive tendons), prior = kinematics only, data = self-exploration only. Isaac Lab Shadow Hand, gravity on, 20 Hz control, 120 Hz recording.

## Gate 0 and stage 1 (summary; details in the plan file)

- The model's rigid-body terms (M, G, N) match PhysX to about 1e-6. PhysX's mass matrix excludes armature.
- The simulator's fixed tendon is non-reciprocal and locks J1 when J0 is pinned at a limit. Option (a) is used: measured-tendon C fixed kinematically, separate force arms; pinned and self-contact windows are masked.
- With 9 random minutes, the model predicts free motion at 0.51° (true parameters 0.39°).

## Stage 2 ladder: designed (A) vs random (R) exploration, old gesture library v2

Rungs (requested minutes, cumulative): f1 = 1 min random, then 3 / 9 / 27 min. Gestures: 15 targets × 10 starts, success = final 24-joint RMSE < 2°. Two seeds (610000, 611000); the ladder was stopped before 612000 (user).

| controller | 610000 | 611000 |
|---|--:|--:|
| direct | 61 | 59 |
| TRUE | 70 | 70 |
| f1 (1 min) | 39 | 40 |
| A2 / A3 / A4 | 69 / 72 / 70 | 68 / 70 / 69 |
| R2 / R3 / R4 | 30 / 37 / 69 | 32 / 39 / 69 |

On the old library, designed exploration matches the true parameters from 3 minutes, while random exploration needs 27. The ceiling is 70% even with true parameters, because 4–5 targets cannot be held.

Model quality (`artifacts/exp49/ladder/analysis.json`):
- Gravity alone is about 100% wrong for A4/R4 (2× too large), but G + τ_k is within 19–34%. With a fixed base, the per-joint elastic polynomial absorbs gravity. This is an identifiability problem of the data, not of Li's equation. Fixes for the real hand: explore in 2–3 mounting orientations, or `--kel-reg` (added, not run).
- The mass-matrix error (134% without armature, 42–62% with it) is mostly the trade-off between armature and link inertia. Fingertip inertias come out about 7× too large and are barely observable.

## Minimum exploration for > 95% gestures (user request)

### Ceiling

`scripts/gesture_reachability_exp49.py` runs CEM over a constant command in Isaac itself (no model), from the neutral start (`artifacts/exp49/min/reach`). Five v2 targets cannot be held within 1°:
- rock 1.71°, pointing 1.92°, thumbs_up 1.95°, fist 2.28°: the curled finger's J2 settles about 6° past target and neighbouring J1/J2 about 3.6° past, because the tendon locks the finger when J0 hits its limit;
- thumb_middle_pinch 2.68°: the thumb and index finger collide.

They were replaced by the held poses: `shadow_dynamics/data/shadow_pose_library_v3_exp49.json`.

### New rungs

Below A2: r1 plus 1 (m1) or 3 (m3) D-optimal programs designed with f1.

Valid data (after truncation):

| rung | data | valid min |
|---|---|--:|
| f1 | r1 | 0.67 |
| m1 | r1 + 1 designed program | 1.0 |
| m3 | r1 + 3 programs | 1.67 |
| A2 | r1 + 5 programs | 2.3 |
| A3 | + 18 programs | 8.0 |
| A4 | + 53 programs | 24.9 |

### Gestures on v3

Seeds 610000–612000 (`artifacts/exp49/min/gestures_s*`, `scripts/analyze_exp49_min.py` → `summary.json`). thumb_middle_pinch is excluded (user): it is 0% for every controller, true parameters included, because its result depends on how the thumb and index finger tangle from each start.

| controller | all 15 | excl. thumb_middle_pinch | per seed | 95% CI |
|---|--:|--:|---|---|
| direct | 74.4 | 79.8 | 80.0 / 77.9 / 81.4 | 75.7–83.3 |
| TRUE | 86.0 | 92.1 | 93.6 / 90.7 / 92.1 | 89.2–94.4 |
| f1 (1 min) | 53.3 | 57.1 | | 52.4–61.8 |
| m1 (1.33 min) | 83.3 | 89.3 | 89.3 / 87.9 / 90.7 | 86.0–91.9 |
| m3 (2 min) | 81.1 | 86.9 | | 83.3–89.8 |
| A2 (3 min) | 83.3 | 89.3 | | 86.0–91.9 |
| A3 (9 min) | 88.9 | **95.2** | 96.4 / 92.9 / 96.4 | 92.8–96.9 |
| A4 (27 min) | 85.3 | 91.4 | | 88.4–93.7 |

Per gesture over 3 seeds, every model from m1 on equals TRUE except on thumbs_up:

| controller | thumbs_up success |
|---|--:|
| TRUE | 33 |
| m1 | 7 |
| m3 | 0 |
| A2 | 13 |
| A3 | 87 |
| A4 | 23 |
| direct | 0 |

The median thumbs_up error is 1.99–2.07° for every model except A3 (1.90°), right at the 2° threshold. A3's lead is therefore one borderline gesture, not better identification: it is significant (McNemar p = 0.007 vs TRUE, 0.002 vs A4) but not monotonic in data. The other residual misses (thumb_index_pinch, cylindrical_grasp, thumb_ring_pinch, ok_sign, about 10–13% each) are shared with TRUE: curled fingers catch on another finger from certain starts.

### Answer

- **The > 95% bar is not reached by the true-parameter model (92.1%).** It measures the targets, not the learned model.
- **The minimum self-exploration that reaches the true-parameter model's level is m1: 1 minute of valid data (2 collection runs: 3 slow random programs, then 1 designed program).** It matches TRUE on 13 of 14 gestures; m1 vs TRUE is McNemar 2 vs 14.
- **9 minutes (A3) gives 95.2%,** but only through the borderline thumbs_up gesture; 27 minutes (A4) gives 91.4%.
- **To make 95% meaningful,** thumbs_up's target would need to be held from every start pose, not just from neutral.

Not run (cancelled by the user): stage B closed-loop refinement (`logs/run_exp49_stageB.sh`, untested `--record-exp49`) and the `--kel-reg` refits (`logs/run_exp49_kelreg.sh`).
