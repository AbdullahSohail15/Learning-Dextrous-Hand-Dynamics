# Experiment 04b: regularised (MAP) refit of the self-explored model, and the cube rerun

Date: 24 Nov 2026 (plan written before the cube rerun; fit result included). Follows experiment 04, whose failure analysis found that the
free-motion fit lets the parameters free motion cannot see (damping D, inertia m) drift to ~0, which broke the cube planner
(identified 51.1% vs spec values 81.1%).

## Fit (`scripts/identify/fit_regularised.py`, `artifacts/exp04b/fit`)

Same data as exp 02 arm CL (180 s closed-loop exploration, 3 hands), fresh fit from the spec model N0, 800 Adam steps, plus a Gaussian
prior centred on N0 in log space: σ = log 1.5 for K, D, m, u_max; log 3 for v_max (the data sets it); 5 × 1e-3 N m for gravity; weight 1e-3.

| model | 0.6 s free-motion error (3 hands) | median rel. error K / D / m / v_max / u_max |
|---|---|---|
| REG (MAP) | 0.025 / 0.024 / 0.038° | 19 / 24 / 15 / 1 / 11% |
| CL (exp 02, no prior) | 0.031 / 0.025 / 0.044° | 44 / 68 / 75 / 1 / 11% |
| TRUE | 0.035 / 0.029 / 0.058° | 0 |
| N0 (spec) | 0.62 / 0.72 / 0.88° | 18 / 22 / 18 / 28 / 11% |

Hand 0, REG: D = 0.030–0.064 on every servo (CL had 0 on three), v_max as CL (thumb yaw 43.9 °/s).

## Cube rerun (`artifacts/exp04b/cube_reg`)

Exp 04 setup unchanged (palm up, goals ±10/20/30°, H = 24, 24 candidates, 30 tasks × 3 seeds, same task seeds). Arms in one process:
param_REG, param_CL, param_N0 (paired within the run; oracle reference from exp 04 `cube_main`, 87.8%).

| # | hypothesis |
|---|---|
| H1 | param_REG > param_CL (paired McNemar p < 0.05) |
| H2 | param_REG ≥ param_N0 − 5 pts (keeping the identified speeds costs nothing once contact parameters are sane) |

## Result (stopped by the user after seed 0)

| arm (same run, seed 0, 30 tasks) | success | final |
|---|---|---|
| param_REG | **70.0** | 63.3 |
| param_CL | 46.7 | 43.3 |

Paired: REG only 10, CL only 3 (exact McNemar p = 0.09). Reference from exp 04 `cube_main` seed 0 (another process): oracle 90.0,
param_N0 83.3. By goal (REG): 10° 90%, 20° 60%, 30° 60% (exp 04 CL over 3 seeds: 80 / 47 / 27).
H1 direction supported, not significant at n = 30; H2 not yet met (70 vs 83). param_N0 did not run (stopped). Preliminary.
