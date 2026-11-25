# Method: the planning and control stack

This page describes the method behind both hands. File references point to the Shadow implementation; the Inspire version reuses
the same RRT, iLQR and CBF code with its own model.

## 1. The hand model, identified by the hand itself

**Shadow Hand.** The model is the tendon-driven structure of Li et al. (arXiv 2408.13044), in `shadow_dynamics/tendon_hand_model.py`:

$$
M(q)\,\ddot q + N(q,\dot q) + G(q) + \tau_k(q) + \tau_d(\dot q) + \tau_f(\dot q) = C^{\mathrm T}\big(f_t - f_f\big) + \tau_{\text{passive}}(q,\dot q)
$$

| Term | Meaning |
|---|---|
| $M, N, G$ | rigid-body inertia, Coriolis and gravity terms; their structure comes from the kinematics |
| $\tau_k = k_0 + k_1 q + k_2 q^2$ | joint elasticity |
| $\tau_d,\ \tau_f$ | damping and Coulomb friction |
| $C$ | tendon routing |
| $f_t$ | actuator force, with $f_t \approx K(a - q) - D\dot q$ |

**Prior knowledge** is only the kinematics: joint tree, axes, limits and link meshes. Everything else is fitted.

**Fit** (`scripts/identify/fit_tendon_hand.py`):

1. Least squares gives the actuator $K$ and $D$ from the measured drive torque.
2. The equation error $r_k = M\ddot q_k + N + G + \tau_k + \tau_d + \tau_f - C^{\mathrm T}(f_t - f_f)$ is scaled per joint and masked
   at contact, joint limits and pinned joints. It is minimised with Adam, with the inertias kept physically consistent (pseudo-inertia
   $J = UU^{\mathrm T}$).

Because $r$ is linear in the 388 plant parameters, $\tau = Y(q,\dot q,\ddot q)\,\pi$. The unconstrained optimum would be
$\pi^\star = (Y^{\mathrm T}Y)^{-1}Y^{\mathrm T}\tau$; Adam is used instead to keep the constraint.

**Inspire RH56.** Each servo is a position servo with a speed limit and a force cap, plus the reflected link inertia and gravity seen
through the linkage (`INSPIRE/inspire_dynamics/servo_model.py`). The model is fitted on 0.6 s rollouts.

## 2. Self-exploration that chooses its own motions (D-optimal design)

The hand needs exploration data that teaches it every unknown parameter. For every candidate program, the current model predicts the
motion, and the regressor rows $Y_k$ are accumulated into the information matrix:

$$
F(\text{program}) = \sum_k Y_k^{\mathrm T} Y_k, \qquad
\text{next programs} = \arg\max \ \log\det\!\big(F_{\text{collected}} + F_{\text{program}}\big)
$$

The selection is greedy, a few programs per round, from a pool of 2048 candidates (1062 pass the kinematic check).

| Programs | Time to reach the true-parameter level | Gestures with 1 min of data |
|---|---|---|
| designed | 3 min | 89.3% (true parameters: 92.1%) |
| random | 27 min | – |

**Safety while exploring.**

- *Shadow:* every program is checked kinematically before it runs. Where link meshes would overlap by more than 1 mm, the program is
  shrunk toward its start; if it still fails, it is dropped. Contact steps are labelled and masked out of the fit.
- *Inspire:* closed loop. Each candidate chunk is predicted with the current model and rejected if two links come within 2 mm. This
  gave 0% finger contact while exploring, against 11–15% for random programs.

## 3. CEM-MPC (the baseline stack)

CEM-MPC samples command sequences over a 0.6 s horizon, rolls them through the model, and refits a Gaussian to the elite samples:

$$
J = \frac{1}{H}\sum_{t=1}^{H}\frac{\lVert \hat q_t - r\rVert^2}{24} + 2\,\frac{\lVert \hat q_H - r\rVert^2}{24} + 10^{-3}\sum_t \lVert \Delta u_t\rVert^2
$$

| Setting | Gestures | Cube |
|---|---|---|
| candidates × iterations | 128 × 4 | 32 × 3 |
| elites | 8 | 8 |
| replan | every 0.15 s | every 0.3 s |
| proposal mean | Gaussian | PPO policy |
| cube in the planner | – | simulated in Isaac branch environments |

## 4. RRT-Connect + iLQR-MPC/TVLQR + CBF (the full stack)

**RRT-Connect** (`rrt_connect.py`) grows two trees in the 20-D command space, one from the start and one from the goal. Each step
samples a configuration, steers toward it by at most ε, and keeps the new node only if the link meshes are collision-free. When the
trees meet, the path is shortcut-smoothed. RRT is only invoked when the straight line collides, and it replans when progress stalls.

**iLQR-MPC** (`ilqr.py`) optimises the next 0.6 s around the RRT path, using the model's linearisation
$\delta x_{t+1} = A_t\,\delta x_t + B_t\,\delta u_t$. The backward Riccati pass gives a feedforward $k_t$ and gains $K_t$.

**TVLQR** applies those time-varying gains every 50 ms between re-solves:

$$
u_t = \bar u_t + k_t + K_t\,(x_t - \bar x_t)
$$

**CBF safety filter** (`cbf.py`). For every finger link pair, $h(q)$ is the signed distance minus a margin. The command is changed
as little as possible so that the predicted distance decays no faster than a geometric rate:

$$
u^\star = \arg\min_u \lVert u - u_{\text{nom}}\rVert^2 \quad \text{s.t.}\quad h(q_{k+n}) \ \ge\ (1-\gamma)^n\, h(q_k)
$$

The QP is solved with a dual projected gradient.

| Hand | γ | n | activation distance |
|---|---|---|---|
| Shadow | 0.2 | 4 | 20 mm |
| Inspire | 0.3 | 2 | 10 mm |

**Terminal state and fallback.** The normal terminal state is the goal gesture held with final joint error below 2°. The safe
terminal state is **HOLD**: every servo is commanded to its measured position. The escalation ladder is:

1. The CBF minimally changes the command.
2. Every 50 ms the filtered command is re-predicted. If a pair is still inside its barrier ($h<0$) and breaks the rule by more than
   0.2 mm, the hand goes to HOLD.
3. RRT replans from the measured pose and the hand resumes.
4. A non-finite command is replaced by the last safe command.

On the Inspire crossing set the full stack held 9 times and recovered every time. CBF without the replan held hundreds of times and
deadlocked (88.6% success).
