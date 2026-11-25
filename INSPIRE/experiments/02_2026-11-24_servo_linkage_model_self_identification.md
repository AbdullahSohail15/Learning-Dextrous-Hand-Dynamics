# Experiment 02: the Inspire hand identifies a servo–linkage model of itself from self-exploration

Date: 24 Nov 2026 (plan; not started). Replicates SHADOW exp 49 (Li et al. tendon model from kinematics only + self-exploration) and the
identification lessons of exps 42–44 and 47. Setup and asset: experiment 01.

## Why

The real Inspire hand is the test target. As on Shadow (exp 49), the model must come from a **kinematics-only prior** (joint tree, axes,
limits, meshes, linkage ratios) plus a few minutes of the hand exploring by itself, and the code must transfer unchanged: sim and real differ
only in a **servo map config** (counts ↔ radians, raw force ↔ torque). The Shadow tendon structure does not apply: the Inspire hand has 6
geared linear servos, each driving a finger through a linkage.

## The model: servo + linkage (6 generalised coordinates)

Coordinates p ∈ R^6: the servo positions, expressed as the angle of the joint each servo drives (rad; thumb yaw, thumb bend, index, middle,
ring, little). The real hand reports them as counts 0–1000; `servo_map.json` holds count = o_i + s_i p_i (sim: identity).

**Linkage (kinematics).** All 12 joint angles q = c(p): the 6 driven joints q_a = p and the 6 linkage joints q_l = Σ_m k_{l,m} p_{s(l)}^m
(polynomial in the driving servo; sim truth is linear, k = 1.064 fingers, 1.334 / 0.667 thumb; the real linkage is a four-bar, so a cubic is
allowed). J_c(p) = ∂q/∂p (12 × 6, block-sparse). On the real hand k comes from the CAD (dex-urdf) and a camera check, never from joint
encoders (there are none).

**Plant (reduced Lagrangian through the linkage).**

    M̄(p) p̈ = u − ḡ(p) − F_c tanh(ṗ / 0.02) + τ_c
    M̄(p)    = J_cᵀ M(c(p)) J_c + diag(armature)          link inertia of every finger seen at its servo + reflected rotor inertia
    ḡ(p)    = J_cᵀ G(c(p)) = Y(p) π                       gravity, linear in the base inertial parameters π (mass × COM per link): identified
    u       = clip(K (a − p) − D ṗ, −u_max, u_max)         servo position loop with its force threshold (torque cap)
    |ṗ|     ≤ v_max(σ)                                     servo speed: on the real hand a per-move setting σ, so the model takes σ as an
                                                           input and identifies the map σ → rad/s (sim: one value until the speed sweep)

Implicit Euler at 120 Hz (6 substeps per 20 Hz command), as the SHADOW models. τ_c (contact) is the quasi-static load observer of
SHADOW exp 37a / 49, held over the planning horizon. Coriolis terms are kept (from the same M) but are negligible at these masses.

**Reduced servo model (B).** The real servo is a non-backdrivable lead screw, so a simpler law may be enough: ṗ = sat_{v_max}((a − p)/τ_s),
stalled while |load| > f_thr. Fitting both A (above) and B on the same data tells us whether the real hand needs any dynamics at all.

**Force channel.** The real hand reports a raw force per servo (0–1000, uncalibrated, arXiv 2603.08988). Model: f_raw = κ_i u_i + b_i. In
sim the "sensor" is Isaac's applied torque on the servo joint, scaled by an unknown κ_true and with noise, so κ must be identified from
self-contact presses during exploration (the thumb pressing on the index, fingers pressing on the closed thumb).

Parameters (per servo): K, D (or τ = D/K), armature, F_c, u_max, v_max(σ), π (gravity, shared), linkage k, κ, b. 6 × 8 + |π| ≈ 60.

**Expected identifiability (from gate 0).** Gravity is below the noise floor on this hand: the free hold error is 0.01° at K = 10 N m/rad,
i.e. the gravity torque is ~3e-3 N m. So π (and the inertial terms in M̄) are **not expected to be identified** from exploration, and that is
not a failure; it is what H4 / model B test. Likewise the servo reaches v_max in under a millisecond, so K and D show only in the last
~13° (one 20 Hz step) of each approach: τ will be weakly identified at 20 Hz. The parameters that matter and should be identified are
v_max(σ), u_max, κ and the linkage. Gate check 1 below is an implementation check with the true inertias, not an identification target.

## Data and fitting (as SHADOW exp 49)

- Self-exploration programs in the 6-D servo space (`scripts/identify/collect_exp02.py`): (R) random held targets with random speed settings σ,
  (D) D-optimal designed excitation (SHADOW `structured_excitation` logic, Inspire layout), (P) presses: thumb–finger and finger–thumb
  contacts held 0.5 s at random force thresholds (for u_max and κ). Self-collision checked in advance with `inspire_dynamics/urdf_kin.py` hulls;
  windows with unplanned contact are masked (exp 49 lesson).
- Measured channels per step: p (servo positions), ṗ by backward difference at 20 Hz (SHADOW exp 46 lesson: PhysX joint velocity chatters),
  raw force. Joint angles of the linkage joints are recorded for evaluation only (the real hand cannot measure them).
- Fit (`scripts/identify/fit_servo_linkage.py`): multi-step rollout loss over 0.6 s windows on p (and force), Adam, robust loss
  (SHADOW exp 44 fix), linear-scale armature/friction (exp 47). K is not fixed (SHADOW exp 46 lesson: fixing K made the gravity scale absorb the error).

## Gate 0 (must pass before any exploration)

1. M̄ and ḡ with the true URDF inertias match PhysX's articulation mass matrix and gravity forces projected through J_c (relative error < 1e-3).
2. With the true parameters (TRUE model), free-motion 0.6 s open-loop servo error median < 0.3°, and the linkage joints < 0.5°.
3. Exploration safety: < 5% of program time in unplanned self-contact.

## Design

Ladder of exploration time: 0.5, 1, 2, 5, 10 min per program type (R, D, R+P), 3 seeds; models A and B; evaluation on held-out programs
(fresh seeds) and on the experiment-03 gesture/transition benchmark. Reference models: TRUE (sim parameters), N0 (spec values only, no data).

## Hypotheses (pre-registered)

| # | hypothesis |
|---|---|
| H1 | TRUE model: free-motion 0.6 s open-loop error < 0.3° (gate 0) |
| H2 | 1 min of designed exploration (D) reaches ≤ 1.5 × TRUE's open-loop error; random (R) needs ≥ 3× longer (exp 49: designed 3 min vs random 27) |
| H3 | v_max identified within 5% (per speed setting); u_max and κ within 20% only when presses (P) are in the data; π and τ not identified (expected) |
| H4 | model B (first-order servo) is within 0.2° of model A in free motion — the real hand can start with B |
| H5 | N0 (spec values) is already within 2× TRUE in free motion; data matters for contact (u_max, κ) more than for free motion |

## Real-hand stage (not run here)

Same scripts with `servo_map.json` for the real hand (counts, force raw), the Inspire serial/Modbus driver in place of the Isaac env, a
camera check of the linkage ratios, and the same exploration programs at reduced speed for the first minute.

## What was run (24 Nov 2026) and deviations from the plan

- **Closed loop** = model in the exploration loop (the user's emphasis): each round designs its 3 s chunks with the model fitted so far
  (greedy D-optimal over 48 random candidates per chunk, on 0.6 s windows as the fit sees data) and screens them for self-collision
  with that model's predicted trajectory (every link pair ≥ 2 mm apart, `inspire_dynamics/torch_kin.py` polytopes), runs them on the
  hand, refits, and designs the next round. Rounds end at 15, 30, 60, 120, 180 s. Open-loop arms: **D0** (same designer, all chunks
  chosen up front from the spec model N0) and **R** (random chunks, no screen). Same fitter for all (400 Adam steps per round, warm start).
- **Unknown hands:** 3 hands = spec values perturbed (`sim.true_hand_params`: drive K, D ×0.6–1.6, armature ×0.5–2, finger speed ×0.7–1.0,
  thumb bend ×0.7–1.1, thumb yaw ×0.5–0.9, torque cap ×0.7–1.2), set per env at runtime. Hand 0 (thumb yaw 43.9 °/s vs the assumed 70)
  is the hand of experiments 03–04.
- Not run: the speed-setting sweep σ (one speed), presses (P) and the force channel κ, model B as a separate fit. Scripts:
  `scripts/identify/gate02.py`, `scripts/identify/exp02_explore.py`, `inspire_dynamics/{servo_model,explore}.py`, analysis
  `scripts/analyze/analyze_exp02.py`. Outputs `artifacts/exp02/{gate,explore}`, logs `logs/exp02_*.log`.

## Results

**Gate 0** (`artifacts/exp02/gate/gate.json`): per-env parameters reach PhysX exactly; a speed limit set at runtime is the one honoured
(measured/set 0.97–1.00, also at half speed). M̄ = J_cᵀ M J_c is diagonal (largest off-diagonal 0.05% of the diagonal); its diagonal varies
4–16% over the range for the fingers and thumb bend, 120% for thumb yaw (depends on the thumb pose). PhysX's mass matrix does not include
the drive armature (diagonal < armature), so m = M̄ᵢᵢ + armature. Gravity at the servos ≤ 0.9% of u_max. **G2/H1 pass:** TRUE model 0.035–0.045°
0.6 s open-loop error on held-out free motion (target < 0.3°); N0 0.62–0.69° (mostly thumb yaw).

**Exploration** (`artifacts/exp02/explore/report.json`; 0.6 s open-loop error on a held-out 60 s program per hand, mean of 3 hands):

| arm | 15 s | 30 s | 60 s | 120 s | 180 s | finger contact during exploration |
|---|---|---|---|---|---|---|
| CL (closed loop) | 0.030° | 0.035° | 0.035° | 0.033° | 0.033° | **0%** |
| D0 (designed once from N0) | 0.029° | 0.033° | 0.032° | 0.031° | 0.032° | 0–0.1% |
| R (random) | 0.031° | 0.034° | 0.033° | 0.031° | 0.031° | **11–15%** |
| TRUE parameters | 0.041° | | | | | |
| N0 (spec values) | 0.74° | | | | | |

Parameters (hand 0, CL): speed limits within 1% at 15 s (thumb yaw 43.9 °/s identified exactly; the spec says 70); u_max does not move
from the spec value (the cap is never reached in free motion); K, D, m stay 40–75% off with no effect on prediction (the servo reaches its
speed limit within one substep, so only v_max shows at 20 Hz). The safety screen rejected 73% of the candidate chunks (6343 / 8640).

| # | verdict |
|---|---|
| H1 | **pass** (0.035–0.045°) |
| H2 | **partly**: 15 s of any exploration already reaches TRUE-level error; designed exploration buys no accuracy over random here (free-motion identification is too easy on a speed-limited servo) |
| H3 | **v_max pass** (1%); u_max not identified without presses (expected, P not run); τ and π not identified (expected) |
| H4 | not run as a separate fit; the fitted model behaves as model B (speed-limited first-order servo) |
| H5 | **fail**: N0 is 18–25× TRUE, not 2× — the spec speeds are off by 10–50% (thumb yaw) and that is what free motion depends on |

**What closed loop buys on this hand is safety, not accuracy:** the model-screened programs never touched (0% vs 11–15% for random), at
equal identification quality. Exploration on the real hand should therefore start with a short screened burst (15 s) and refit.
