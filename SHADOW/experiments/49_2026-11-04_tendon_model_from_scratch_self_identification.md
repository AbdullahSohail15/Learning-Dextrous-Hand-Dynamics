# Experiment 49: the Shadow Hand identifies a tendon-structured model of itself from scratch (4 Nov 2026)

Results: `result_summaries/2026-11-04_tendon_model_from_scratch_results.md` (not started). Follows exps 36 and 42–44 (Shadow equation and self-exploration) and 46 (gravity on, Allegro). Status: **plan only; waiting for the user's go.**

## Why
The user will test the method on a real Shadow Hand. Two things must hold for the code to transfer unchanged:
1. **Same equation as the real hand.** Use the structure of Li, Chen, Ringwald, Pozo Fortunić, Ganguly, Haddadin, "Identification and validation of the dynamic model of a tendon-driven anthropomorphic finger", arXiv 2408.13044, eqs. (1)–(3). In simulation, some terms are zero (no actuation tendons), but the structure is the same and only the configuration changes between sim and real.
2. **No physical prior and no recorded data.** The model knows only the hand's kinematics. Every dynamic parameter is learned from the hand's own self-exploration, starting from nothing. This differs from exps 36–44, which started from the URDF export (masses, inertias, drive gains) and used the 46 h structured dataset for evaluation and baselines.

User decisions (4 Nov):
- **Prior knowledge:** kinematics only. Allowed: joint tree, joint axes and origins, joint limits, and link meshes for the self-collision check (the CAD a real-hand user has). Forbidden: masses, centres of mass, inertias, armature, drive gains, friction and tendon properties.
- **Gravity: on.** It matches the real hand and makes G(q) identifiable.
- **Real-hand sensing:** joint angles plus tendon forces (strain gauges). So the plant is driven by a measured force, as in the paper.
- **Process:** plan, then wait. Nothing runs before the user's go.

## The model: two layers

### Plant (Li et al. structure, configuration-driven)
\[ M(q;\pi)\,\ddot q + N(q,\dot q;\pi) + G(q;\pi) + h(q,\dot q) = C^\top(q)\big(f_t - f_f(f_t,\dot l)\big) + J_{\text{ext}}^\top f_{\text{ext}}, \qquad \dot l = C(q)\,\dot q \]
\[ h = \tau_k(q) + \tau_d(\dot q) + \tau_f(q,\dot q), \qquad M = M_{\text{rb}}(q;\pi) + \operatorname{diag}(\text{arm}) \]

- **M, N, G** come from a differentiable recursive Newton–Euler / composite-rigid-body implementation over the 24-joint tree. It uses the known kinematics and 10 inertial parameters per link (mass, first moment, rotational inertia).
  - The parameterisation is physically consistent (Cholesky factor of each link's pseudo-inertia), so every fitted set is a real rigid body.
  - M is the full matrix: J0 becomes a true degree of freedom with its own inertia, not a kinematic clip as in exp 36.
- **h** is joint-space.
  - Elasticity τ_k is a per-joint quadratic polynomial, as in the paper.
  - Damping τ_d is viscous.
  - Friction τ_f is Coulomb smoothed with tanh plus viscous.
- **Tendons** are defined by a configuration file: a list of tendons, each with the joints it crosses and its moment-arm form (constant or polynomial in q), and whether its force is measured or modelled.
  - **Simulation configuration:**
    - 20 signed "actuator tendons", one per driven joint. C is the identity on its own joint (moment arm learned, expected 1) and f_t is measured, as Isaac's drive torque.
    - 4 passive coupling tendons crossing J1 and J0 of FF, MF, RF and LF. Their C row is learned and their force is modelled, not measured: an elastic force at the tendon length limits with damping, mirroring PhysX's fixed tendon (limit stiffness 30, damping 0.1 in the asset; read back at gate 0).
  - **Real configuration (not run here):** 40 unsigned tendons (flexor and extensor per motor). On the real hand one tendon pair drives J1 and J0 together, so the coupling enters through C itself.
  - Tendon friction f_f(f_t, l̇) uses the paper's form (force-proportional Coulomb, smoothed in l̇). It is expected to be about 0 in simulation.
- **f_ext**: zero in free motion. In contact tasks it is absorbed by the residual observer of exp 37a, as before.

### Actuator: from command to tendon force
The controller sends joint targets a, while the plant needs f_t. A separate actuator model f_t = A(a, q, q̇) is identified against the measured f_t.
- **Simulation:** the PD drive with saturation, f_t = clip(K(a − q) − D q̇, ±u_max), per joint. Learned: K, D and u_max (true values: K 1 / 5, D 0.1 / 0.5, u_max 0.72–4.8 N m).
- **Real:** Shadow's tension/position controller. Same interface, possibly with a first-order lag; the form is chosen when real data exists.

With f_t measured, K and D are separately identifiable. Exps 36–47 could only identify τ = D/K from joint angles.

### Terms expected near zero in simulation
- f_f;
- τ_k;
- off-structure entries of C;
- link-level parameters that do not affect motion (only base parameters are identifiable).

They are fitted anyway, and their near-zero values are a check. Terms learned from nothing in simulation:
- all inertial parameters π, giving M, N and G;
- armature;
- τ_d and τ_f;
- the coupling tendons (C row, stiffness, limits);
- the actuator's K, D and u_max.

## Data and fitting
- **Rate:** states at physics rate (120 Hz, targets held 6 substeps; decimation 1 in the collector), commands at 20 Hz.
  - Joint velocity is the backward difference of q (exp 46: PhysX `joint_vel` is chatter under gravity). q̈ comes from a filtered second difference.
  - The real hand's EtherCAT loop runs at about 1 kHz, so this is realistic, not a simulator shortcut.
- **Fit, step 1 (initialisation):** linear least squares. With f_t measured, the plant is linear in π, armature, damping and friction coefficients and in the constant moment arms: \(Y(q,\dot q,\ddot q, f_t)\,\theta = C^\top f_t\). Only identifiable base parameters are solved for (QR on Y).
- **Fit, step 2 (refinement):** a differentiable rollout over 0.6 s windows from measured states with measured or modelled f_t. It uses the robust per-window RMS loss from exp 44. The physically consistent parameterisation is enforced here.
- **Actuator fit:** regress measured f_t on (a, q, q̇). It is separate from the plant, and the two are composed for prediction and control.
- **No pre-existing data.** No pre-existing recording is used for fitting or for design: not `shadow_structured_v3`, not exps 36–44. Evaluation uses fresh, fixed test sets generated once with gravity on.

## Self-exploration from scratch
- **Round 1 (no model):** random programs from the structured generator (`structured_excitation.py`) at low speed (≤ 10 °/s), checked only kinematically. That uses forward kinematics, the link meshes and joint limits, which are all allowed priors.
  - Low speed is the safety margin while the actuator and gravity load are unknown.
  - 3 programs (1 min free motion).
- **Rounds 2–4:**
  - **Arm A (D-optimal):** the Fisher information of the current regressor, \(F = Y^\top Y\) over the candidate's windows (sensitivities from the current fit for the nonlinear tendon terms). Greedy \(\max \log\det(F + \epsilon I)\), pool of 2,048 candidates, speeds up to 30 °/s once the actuator is identified.
  - **Arm R:** uniform from the same checked pool.
- **Budget ladder:** 1 / 3 / 9 / 27 min of free motion (rungs 1–4). Free motion only; there is no cube-grasp half in this experiment.
- **Runtime monitor:** as in the existing collector (cut at the first invalid state). With gravity on, the joint-limit and penetration thresholds are re-checked at gate 0.

## Gate 0: implementation and simulator checks (must pass before any exploration)
1. **Rigid-body terms match PhysX.**
   - Setup: plug the asset's true inertias into the new model; compare against the articulation view at 1,000 random poses and velocities.
   - PhysX calls: `get_mass_matrices()`, `get_generalized_gravity_forces()`, `get_coriolis_and_centrifugal_forces()`. These names were checked in the installed tensors API.
   - Pass: relative error < 1% for M and G, < 5% for N.
2. **Full model with true parameters predicts free motion well.**
   - Setup: true inertias, the asset's drive gains and tendon properties.
   - Pass: open-loop error at 0.6 s on a fresh gravity-on test set ≤ 0.25°. Exp 36's URDF equation reached 0.22° without gravity.
3. **"Measured" f_t is close to the real drive force.** Isaac Lab's `applied_torque` is its own estimate of the implicit drive torque, not PhysX's internal force. Compare it with the torque implied by the motion and the true model. Report the gap; if it is above 5% RMS, use the implied torque as the sensor model and note it.
4. **Fixed tendon.** Read back the coefficients, rest length, limits, limit stiffness and damping, and decide whether the coupling is near-rigid (stiff spring) or compliant.
5. **Gravity-on baselines.** Gravity on in the Shadow env (`env.py:177` sets `disable_gravity=True` today; add a flag).
   - Measure static sag.
   - Measure the direct-command gesture success (2° tolerance, exp 37a protocol, new task seed).
   - **Stop rule:** if direct commands already reach ≥ 90%, the gesture task cannot separate models. Then switch the control test to fast tracking (exp 47's protocol ported to the Shadow) and ask the user.

Gate 0 runs no exploration. It ends with a short report to the user before the ladder starts.

## Design (after gate 0)
Hand: stock Isaac Shadow, gravity on. Truth = the asset's parameters.

Models and arms, each fitted at every rung and in both exploration arms:
- **S-A, S-R:** the from-scratch Li-form model (plant + actuator), D-optimal and random exploration.
- **DIRECT:** send the gesture target as the command, no model.
- **URDF-REF:** the exp 36 equation with gravity added and URDF parameters. This uses the forbidden prior and is a labelled reference only.
- **TRUE:** the Li-form model with the asset's true parameters. Gate 0 validates it; it is the model ceiling.
- **ORACLE:** CEM with Isaac branches (exp 33), the planner ceiling.

Controller: the exp 37a / 44 CEM-MPC unchanged (128 candidates, 4 iterations, 8 elites, replan every 3 steps), with the residual observer on and off.

Metrics:
- **Parameter recovery**, in function space, because only base parameters are identifiable. Over 1,000 held-out poses and velocities, the relative error of:
  - M(q) (Frobenius);
  - G(q);
  - N(q, q̇);
  - the actuator K and D per joint;
  - the coupling-tendon gain c and limits.

  Report also the fitted f_f, τ_k and off-structure C entries (expected about 0).
- **Prediction:** open-loop joint RMS at 0.6 s and over 10 s rollouts on the fresh test set (mean of window RMS and median).
- **Control:** gesture success (450 tasks, paired McNemar tests), or tracking error if gate 0's stop rule triggers.

## Hypotheses (pre-registered; thresholds may be revised only at gate 0, before any ladder data exists)
- **H1 (identification):** at rung 4 (27 min), S-A recovers G(q) to a median relative error < 10%, M(q) < 15%, and the actuator K and D per joint to a median < 10%.
- **H2 (prediction):** at rung 3 (9 min), S-A's 0.6 s error is ≤ 1.5× TRUE's.
- **H3 (control):** at rung 3, S-A's gesture success is within 3 points of TRUE's and at least 15 points above DIRECT.
- **H4 (designed vs random):** S-A reaches H2's threshold at a smaller rung than S-R.
- **H5 (simulator zeros):** at rung 4 the fitted f_f, τ_k and off-structure C entries contribute less than 2% of the RMS joint torque on the test set.

## Code (to write)
- `shadow_dynamics/kinematics_only.py`: builds the allowed prior from `artifacts/viz_hand_model` (joint tree, axes, origins, limits, meshes) with every inertial field removed, plus a check that no mass or inertia leaks in.
- `shadow_dynamics/tendon_hand_model.py`: `TendonHand`, with differentiable RNEA/CRBA, physically consistent inertial parameters, the configurable tendon set, h, f_f and the actuator, and the same interface as `ParametricHand` (rollout, residual_load, values, to_prior_config). Tendon configurations live in `shadow_dynamics/data/tendons_sim.json` and later `tendons_real.json`.
- `scripts/gate49_physx_terms.py`: gate 0 items 1–4.
- `scripts/collect_shadow_structured.py`: add `--gravity`, `--record-hz 120` and recording of `applied_torque` as f_t (already recorded as `effort`).
- `scripts/fit_tendon_hand.py`: least-squares initialisation, then the robust rollout fit.
- `scripts/design_tendon_excitation.py`: regressor-based D-optimal design, with the round-1 no-model mode.
- `scripts/evaluate_shadow_pose_mpc.py`: accept a `TendonHand` model and `--gravity`.
- Chain `logs/run_exp49.sh` (own tmux session `exp49`, GPU 3); artifacts `artifacts/exp49/`; analysis `scripts/analyze_exp49.py`.

## Risks
- **Identifiability:** free motion with a palm-up hand may excite some inertial combinations weakly. They are scored in function space, and D-optimal design targets them.
- **The applied_torque sensor gap** (gate 0 item 3).
- **The PhysX fixed tendon** may not be exactly an elastic tendon (exp 36 needed a clip-and-reaction form). If gate 0 shows a large mismatch, the passive-tendon model is extended within the paper's structure (f_t of the passive tendon as a function of tendon length).
- **Gravity sag:** exp 46 found Isaac's static sag 1.1–1.5× larger than the equation's on the Allegro (substep chatter). Gate 0 item 2 measures this for the Shadow.

## Real-hand stage (not run here)
- Swap the tendon configuration to 40 measured tendons with polynomial C(q); tendon friction and τ_k become identifiable.
- Replace the actuator form with the controller's.
- Self-exploration, the safety check and the fitter are unchanged.

Everything except C(q)'s shape, f_f and τ_k is validated in simulation by this experiment.

## Gate 0 results (4 Nov 2026)
Code:
- `shadow_dynamics/kinematics_only.py` (allowed prior; refuses files with dynamic fields).
- `shadow_dynamics/tendon_hand_model.py` (`TendonHand`, Li form).
- `scripts/probe_shadow_tendons.py`.
- `scripts/gate49_physx_terms.py`.
- `env.make_cfg` `--gravity`, `ResearchShadowEnv.gravity_root()`.
- `evaluate_shadow_pose_mpc.py --gravity`.

Outputs: `artifacts/exp49/gate0/` (`tendon_probe.json`, `run3/{truth.json,gate0.json,recording.npz,item2_regimes.json,pinned_law.json}`, `gestures_g_s61{0,1,2}000`). Runs 1–2 were superseded by run 3 (bug fixes).

| item | result | pass? |
|---|---|---|
| 1. M, G, N vs PhysX (1,024 random states) | M 6e-7, G 3e-7, N 1.4e-6 median relative error (PhysX's mass matrix excludes armature; ours adds it on the diagonal) | yes |
| 2. free motion, true parameters, 0.6 s | see below | partly |
| 3. Isaac `applied_torque` vs implicit end-of-step drive force | 3.5% RMS gap (PhysX projected joint forces are total constraint forces, not comparable) | yes |
| 4. fixed tendon | see below: not PhysX's nominal law, not reciprocal | finding |
| 5. gravity on | sag mean 0.30°, max 5.0° (WRJ0). Direct-command gestures 61.0% (59.3/61.3/63.3 over seeds 610000–612000; hold 1.6%); stop rule (≥ 90%) not triggered | yes |

**Item 4: the PhysX fixed tendon.**
- Nominal settings: gearing −0.00805 (J1) and +0.00705 (J0), force coefficients 1 and 1, limits ±0.001, limit stiffness 30, damping 0.1, stiffness 0.
- Measured by inverse dynamics, with the exact rigid-body terms and the drive torque:
  - **J0 free:** a spring on the tendon length L = −0.00805 q1 + 0.00711 q0, with no slack band. Stiffness × J0 force arm ≈ 13–22, R² 0.94–0.97 per finger. The J1 side is below the drive-sensor noise.
  - **J0 pinned at its stop (about 30% of the time):** J1 torque = −570 L (R² 0.97, all fingers), which is −4.05 N·m/rad in joint terms. This is exp 36's κ (−4.47).
- The force split is not equal to the length split, so the tendon is not reciprocal.
- The Li form needs two simulator-only extensions to represent it: separate force arms (now supported by `force_arms` in the tendon config; it defaults to C, so the real-hand config is unchanged) and a stiffer pinned-regime term. With force arms ≠ moment arms, a large pinned-regime gain makes the implicit step unstable (the stiffness block is non-symmetric). The term needs explicit or semi-implicit treatment.

**Item 2: free motion with the true parameters plus the least-squares tendon spring** (2,630 contact-free 0.6 s windows; 30% of control steps had finger self-contact, 4% invalid penetration):

| regime | windows | RMS all joints | median window | RMS non-finger joints |
|---|--:|--:|--:|--:|
| J0, J1, J2 all > 1° inside their limits | 646 | 0.38° | 0.40° | 0.44° |
| some J0 pinned at a stop | 1,828 | 2.97° | 0.38° | 0.47° |
| J0 interior, some J1/J2 at a limit | 156 | 0.40° | 0.39° | 0.47° |
| with PhysX's nominal tendon law (all contact-free) | 2,630 | 3.26° | 2.30° | |

- **Interior regime:** 0.38° against the 0.25° target. The rest is a systematic sag bias on gravity-loaded joints (THJ3 +1.19°, finger J2 +0.2–0.36°, WRJ0 +0.18°): Isaac sags more than the exact equation, as exp 46 found on the Allegro.
- **Pinned regime:** the tail is the pinned-J0 lock.
- **Measured-force rollouts** (Isaac's drive torque fed open loop) diverge to about 42°. With finger inertia about 1e-6 kg m², the joint sits on a force balance and any torque error integrates. Measured f_t is for equation-error fitting, not open-loop prediction.
- **Joint limits:** PhysX lets J1 sit about 0.65° past its limit under tendon load; the model clamps hard.

**Decision needed (user):** how to represent the simulator's non-physical coupling.
- (a) Simulator-only extension (force arms plus a pinned-regime term) in the sim config, absent from the real config, and mask pinned-J0 windows in scoring.
- (b) Make the sim tendon reciprocal at spawn (USD gearing / forceCoefficient), if PhysX honours it; untested.
- (c) Keep exp 36's kinematic J0 plus κ as a labelled sim-only term.

Exploration does not start before this answer.

### Option (b) test: reciprocal tendon at spawn (4 Nov, user asked for it)
- **Mechanism:** `env.apply_tendon_override` rewrites the J1–J0 fixed-tendon USD attributes before PhysX parses the stage. The read-back confirms the new gearing and force coefficients.
- **Specs:** `artifacts/exp49/gate0/tendon_*.json`. Runs: `run4_recip`, `run5_reciprocal_k02`, `run5_nonrecip_k02`.

| tendon (gearing J1, J0 = −1.142, 1; L in rad, no band) | force coefficients | spring (N m/rad) | result |
|---|---|---|---|
| reciprocal | −1.142, 1 | 5 | NaN within the first steps |
| reciprocal | −1.142, 1 | 0.2 (the stock tendon's measured effective stiffness) | NaN within the first steps |
| control | 1, 1 (stock) | 0.2 | 67% of steps numerically invalid |

PhysX is stable only near the stock tendon settings. A reciprocal tendon at the stock gearing size (about 0.007) would be about 100× weaker in joint terms than the real hand's tight J1–J0 coupling. **Option (b) is dropped; option (a) is used**, as agreed with the user: the sim tendon config carries separate force arms and a pinned-regime term, and the real config omits both.

Completed (b) with stock-size gearings (runs `run6_reciprocal_stock`, `run7_reciprocal_stiff`; 10 s programs). The inverse-dynamics tendon torque was measured with the exact rigid-body terms; dτ/dL per finger:

| tendon | stable? | dτ_J0/dL, J0 free | dτ_J1/dL, J0 pinned | pinned-J0 samples |
|---|---|--:|--:|--:|
| stock (run 3) | yes | −28 | −570 | 52–93 |
| reciprocal, stock settings | yes | −0.4 | −0.2 to −1.3 | 14k–23k |
| reciprocal, limit stiffness 30 → 4000 | yes | −1.6 | −17 to −22 | 9k–13k |

- PhysX honours the reciprocal coefficients, and the J1 lock goes away.
- The coupling becomes 17–70× weaker than stock, and J0 falls into its stops for much of the time.
- Raising the limit stiffness 130× raises the felt stiffness only about 4×.
- No reciprocal setting reproduces the real hand's tight J1–J0 coupling. **(b) is dropped; (a) stands.**

## Stage 1: from-scratch pipeline (4 Nov, user: "go ahead with the pipeline under (a)")

### Code
- `shadow_dynamics/tendon_hand_model.py`:
  - inertial parameters with relative off-diagonals (well scaled);
  - simulator-only `force_arms`;
  - a simulator-only pinned-J0 term (`pin_joint` / `pin_target`; **off by default**).
- `scripts/collect_exp49.py`:
  - gravity on, 120 Hz q / qd / target / drive torque, contact labels per control step;
  - kinematic pre-check;
  - truncation at the first invalid state;
  - J0 starts at 0, so no coupling knowledge.
- `scripts/fit_tendon_hand.py`:
  - **stage 0:** C of the measured tendons is kinematic (Li ID.2; in simulation the actuator excursion is the joint angle, so C = 1, held fixed);
  - **stage 1:** actuator least squares (torque limit set only if saturation is seen);
  - **stage 2:** plant by equation error. Residuals are masked per joint at its limits, on J1/J0 of a finger with J0 pinned, and on self-contact steps;
  - **stage 3:** rollout refinement, optional and not used yet.
- `scripts/analyze_exp49_fit.py`: the truth scoring (evaluation only).
- Chain: `logs/run_exp49_check.sh`; artifacts: `artifacts/exp49/check`.

### Changes from the stage plan
- **The pinned-J0 term is not modelled.** Neither inverse-dynamics least squares (pinned windows 2.97 → 3.33°) nor a rollout fit (diverged) could use it. The lock occurs mostly with J1 pushed past its own limit (PhysX lets J1 sit about 0.65° beyond). Pinned-J0 residuals are masked in fitting and scored separately, as option (a) allows. In closed loop, the residual observer can absorb the lock as a J1 load.
- **Two bugs fixed:**
  - u_max was set to the largest torque seen, which is far below the limit in slow data;
  - the measured-tendon C drifted to 0.01–0.6. C and the whole plant share a scale, so force data cannot fix it; it is now fixed kinematically.

### Small check (fresh random programs, from scratch; scored on the gate-0 run 3 recording, 1,500 contact-free 0.6 s windows)

| model | data (valid / free min) | 0.6 s RMS, J0 interior | median window | 0.6 s RMS, J0 pinned | G rel. error | M rel. error | K, D error |
|---|---|--:|--:|--:|--:|--:|--:|
| TRUE (truth + gate-0 least-squares spring) | – | 0.39° | 0.40° | 3.00° | 0 | 0 | 0 |
| S9: 27 random programs, 18 passed the check, 1–30 °/s | 6.0 / 4.6 | **0.51°** | 0.46° | 3.01° | 18% | 86% | 0% |
| S1: 3 slow programs (≤ 10 °/s) | 1.0 / 0.8 | 18.5° | 18.3° | 21.6° | 62% | 268% | 0% |

- The from-scratch Li-form model predicts within 1.3× of the true-parameter model from 6 valid minutes of random self-exploration. Gravity is identified to 18%.
- The mass matrix is poorly identified (the finger inertias are tiny and quasi-static). Prediction does not need it.
- 1 slow minute is not enough to predict. That is acceptable for round 1, whose role is to seed the round-2 design.
- Actuator K and D are exact. Isaac's drive torque is an exact PD law; the real hand's controller will not be.

### Stage 2: the ladder (started 4 Nov 11:35; the user decided: no cube in exploration)

**New code:**
- `TendonHand.inertial_regressor` (linear rigid-body regressor; matches RNEA to 8e-8).
- `scripts/design_tendon_excitation.py`: regressor-based D-optimal design over about 388 linear parameters, starting from the information already collected; `--random` for the baseline.
- `evaluate_shadow_pose_mpc.py --tendon-models`.
- `analyze_exp49_fit.py --save-true`, which writes `artifacts/exp49/true_model.json`.

**Gesture smoke test** (seed 610000, 150 tasks, gravity on): direct 61%; CEM-MPC with TRUE 70.0% (1.90°); with the 9-minute from-scratch S9 68.7% (2.06°). About 8 minutes per controller per seed.

**Chain** `logs/run_exp49_ladder.sh`, tmux `exp49`, artifacts `artifacts/exp49/ladder`:
- Round 1 is shared: 3 slow random programs (1 dropped by the kinematic check).
- Arms A and R add 6, 18 and 54 programs each round from a shared pool.
- Every rung is a from-scratch fit on all data so far, scored against truth.
- Gestures for f1, A2–A4, R2–R4, TRUE and direct over 3 seeds.

Closed-loop refinement (stage B, planner data refit) is proposed as a later stage; the user has not decided yet.

### Stage B: closed-loop refinement (user asked for it, 4 Nov)
- **Starting model:** the ladder's A4 fit.
- **Each round b = 1..3:**
  - CEM-MPC gestures with the current model on training task seed 620000 + 1000(b−1). These seeds are separate from the evaluation seeds 610000–612000; the gesture library is the same, the starts differ.
  - The run is recorded at 120 Hz in the exploration format (`evaluate_shadow_pose_mpc.py --record-exp49`): 150 tasks × 6 s, 15 min per round.
  - A from-scratch fit on the A-arm exploration data plus all closed-loop data so far.
- **Same masks as exploration:** self-contact and pinned J0. So closed-loop data adds task-relevant poses and speeds, not the lock.
- **Scoring:** truth (`analysis.json`) and the evaluation gesture seeds, compared with A4 and TRUE from the ladder.
- **Chain:** `logs/run_exp49_stageB.sh` (tmux `exp49b`, waits for `EXP49_LADDER_COMPLETE`); artifacts `artifacts/exp49/stageB`.
- **Untested before launch:** the recording code. The GPU is busy with the ladder.

## Follow-up after stage B: gravity vs elastic-term identifiability (added 4 Nov, user request)

Finding (ladder analysis, checked on CPU): A4/R4 gravity G alone is ~100% wrong (|G| about 2× truth) but G + τ_k (static torque) is within 19–34%. With a fixed base, gravity depends only on q, and the per-joint elastic polynomial τ_k = k0 + k1 q + k2 q² absorbs much of it. More data does not fix it (S9 at 9 min had G 18%). The mass-matrix error without armature (134%) is mostly armature vs link-inertia trade-off (with armature 42–62%); the distal fingertip inertias are about 7× too large and barely observable under position drive.
1. `analyze_exp49_fit.py` now also reports `GK_rel_median` (G + τ_k) and `Marm_rel_median` (M including armature).
2. `fit_tendon_hand.py --kel-reg λ` (default 0, so earlier and stage B fits are unchanged): ridge on k_el / per-joint torque scale, training loss only. Refits of the final A and R data at λ ∈ {0.01, 0.1, 1}, re-analysis of all exp 49 fits, gestures on seeds 610000–612000. Chain `logs/run_exp49_kelreg.sh`, tmux `exp49c`, output `artifacts/exp49/kelreg`, ends with `EXP49_KELREG_COMPLETE`. It waits for stage B.
3. Real hand (planned): explore with the hand mounted in 2–3 orientations, so gravity changes direction while the springs don't.

## Change of plan (4 Nov, user): minimum exploration for > 95% gestures

The stage B and elastic-penalty chains were cancelled before they started (user). The `--kel-reg` option and the GK/Marm metrics stay in the code. Reason: on seed 610000 the true-parameter model reaches only 70%, and A4 also 70%. Fist, pointing, thumbs_up and thumb_middle_pinch fail for every controller, direct included, with the same joint errors (curled J2 about +6° past target, LFJ4 −4°, thumb_middle_pinch THJ3 −24°). So the ceiling is the targets, not the model.
1. Ceiling: `scripts/gesture_reachability_exp49.py` runs CEM over one constant command in Isaac itself (no model): ramp at 80 °/s from neutral, hold, 24-joint RMSE over the last 0.5 s. A gesture is holdable if it gets below 1°. Unholdable gestures are replaced by the pose the best command holds, in a new library `shadow_dynamics/data/shadow_pose_library_v3_exp49.json`; v2 is kept.
2. Fine rungs below A2: r1 plus 1 or 3 D-optimal programs designed with f1 (`m1`, `m3`; about 1.33 and 2 min requested).
3. Gestures on v3, seeds 610000–612000: direct, TRUE, f1, m1, m3, A2, A3, A4. The answer is the smallest rung with > 95% (mean of 3 seeds), provided TRUE is ≥ 95% on v3.

Chain `logs/run_exp49_min.sh`, tmux `exp49m`, output `artifacts/exp49/min`, ends with `EXP49_MIN_COMPLETE`. It waits for the ladder.
