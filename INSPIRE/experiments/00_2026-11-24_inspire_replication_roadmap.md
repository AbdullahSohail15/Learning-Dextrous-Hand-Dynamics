# Experiment 00: roadmap for replicating the SHADOW experiments on the Inspire RH56 hand (simulation)

Date: 24 Nov 2026. Requested by the user (23 Nov): the Shadow Hand is not available in the real world; the lab has an **Inspire hand**
(six linear servos). Replicate the relevant experiments for the Inspire hand in simulation, with a new mathematical model (it is not
tendon driven), save rendered videos, and write experiment files like SHADOW's.

## What changes from Shadow to Inspire

| | Shadow (SHADOW exps 49–54) | Inspire RH56 (this folder) |
|---|---|---|
| actuation | 20 actuated joints, tendons (Li et al. model, exp 49) | **6 linear servos** (thumb yaw, thumb bend, index, middle, ring, little), each a geared, non-backdrivable lead-screw drive with a speed setting and a force threshold |
| joints | 24 (4 passive distal via fixed tendons) | 12: 6 actuated + 6 driven through **linkages** (finger intermediate = 1.064 × proximal; thumb intermediate / distal = 1.334 / 0.667 × thumb bend) |
| what the real hand measures | joint angles + tendon forces (planned) | per servo: position (0–1000), raw force (0–1000, uncalibrated), no joint angles |
| hand model | tendon plant + actuator (exp 49) | **servo–linkage model**: 6 generalised coordinates, linkage Jacobian, rate- and force-limited servo (experiment 02) |
| collision model | visual hulls / capsule pre-check | dex-urdf convex hulls; one structural pair filtered (experiment 01) |
| sim asset | Isaac Lab stock Shadow | dex-urdf `inspire_hand_right` (CAD-derived), imported with PhysX mimic joints |

## Mapping of experiments

| INSPIRE | replicates SHADOW | question | status |
|---|---|---|---|
| 01 | exp 46 setup, exp 49 gate 0 | import, linkage, servo limits, self-collision, gravity hold, renderer | **done (gate 0 passed)** |
| 02 | exps 49, 42–44, 47 | servo–linkage model from kinematics only + self-exploration; how many minutes of exploration identify it | plan written |
| 03 | exps 27, 50 | gestures: direct vs CEM-MPC vs RRT-Connect + iLQR-MPC/TVLQR + CBF, on the identified model | plan written (task redesign needed, see below) |
| 04 | exps 39, 51–54 (+19 PPO proposer, terminator) | object task with the cube-aware stack (soft replay, overlap cost, TVLQR) | **task needs the user's choice** |

Not replicated (dropped as not relevant for the real Inspire hand): flow/det world models and their noise/partial-observability studies
(exps 20–26), physics-prior data efficiency on Shadow (26, 28), MuJoCo / Allegro ports (45–48), distillation (35).

## Known issues that shape the plans

1. **Gestures are too easy for direct servo commands.** The servos are stiff and the fingers weigh 4–5 g, so a direct command reaches any
   reachable target (gate 0: 0.01° tracking, 0.3 s for a finger). Exp 50's gain came from Shadow's weak, coupled, gravity-loaded joints. On
   Inspire the planning problem is **collisions between thumb and fingers on the way** (exp 50's RRT fixed the path-dependent
   thumb_middle_pinch) and **contact force at speed** (the real hand overshoots its force limit by up to +1618% at high speed, arXiv 2603.08988).
   Experiment 03 is therefore built around transitions between gestures (thumb crossing the fingers) and pinches with a force limit.
2. **In-hand cube reorientation is probably infeasible** with 6 coupled DoF (no finger abduction, one thumb bend for three thumb joints).
   Before planning 04, a feasibility probe (oracle CEM in Isaac, cube in the upturned palm) decides between: (a) cube rotation about one axis
   in the palm, (b) grasp-and-hold of objects of different size/mass against a perturbation, (c) force-limited pinch grasp of small objects.
3. Spec values are not calibrated (no official DFX sheet found; resellers: 260 °/s fingers, 70 °/s thumb bend, 10 N / 15 N tip force).
   They are assumptions in the sim URDF and are identified, not assumed, in experiment 02.

## Folder conventions (as SHADOW)

`experiments/NN_YYYY-MM-DD_topic.md` plans (results appended or in `result_summaries/`), `logs/run_expNN.sh` chains in tmux,
`artifacts/expNN/` outputs (never overwritten), `./run_python.sh` for Isaac (SHADOW is on `PYTHONPATH` for shared modules: RRT, iLQR, CBF,
cpu budget), one GPU job at a time (physical GPU 5 by default).
