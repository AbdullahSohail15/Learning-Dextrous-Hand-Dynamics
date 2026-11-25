# Experiment 54 results: a cube-aware stack on the soft hybrid

Plan: `experiments/54_2026-11-17_cube_aware_stack.md`. Date: 17 Nov 2026. GPU 5 (free during this run: 6–33 min per arm). Seed 790000,
120 cases, 20 s, cost `safe`, 32 candidates, tendon hand model `tg0`. Same seed and cases as exp 53 (settled starts differ by ≤ 0.05 rad
between processes, so cross-experiment comparisons are approximately paired). Nothing committed.

## Results

| arm | 20 s | 10 s | vs soft CEM (only arm / only CEM, p) | drops | > 1 mm steps | > 3 mm steps | median t | min |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 soft CEM `cem:P:hyb_soft_tg0:safe` | 91.7 | 78.3 | – | 5.8% | 52% | 18.0% | 4.8 s | 6 |
| 2 + TVLQR (contact-aware reference) | 94.2 | 81.7 | 5/2, 0.45 | 4.2% | 49% | 16.1% | 4.6 s | 7 |
| 3 + RRT as proposer | 90.0 | 74.2 | 6/8, 0.79 | 1.7% | 51% | 15.9% | 5.1 s | 14 |
| 4 + overlap cost | **95.0** | 82.5 | 9/5, 0.42 | 1.7% | 19% | **2.5%** | 5.0 s | 18 |
| 5 + overlap cost + light CBF | 93.3 | 70.0 | 7/5, 0.77 | 2.5% | 14% | 1.2% | 5.6 s | 22 |
| 6 all: RRT-proposer + TVLQR + overlap cost + light CBF | 93.3 | 69.2 | 7/5, 0.77 | 4.2% | 12% | **0.5%** | 6.3 s | 33 |
| exp 53 kinematic CEM | 85.8 | 65.8 | 3/10, 0.09 | 0.8% | 55% | 23.1% | 7.0 s | 62* |
| exp 53 kinematic exp-50 stack (RRT + iLQR + CBF) | 71.7 | 42.5 | 2/26, 3e-6 | 1.7% | 19% | 1.4% | 7.6 s | 189* |

\* contended GPU. Cube-aware stack (arm 6) vs the exp-50 stack: 93.3 vs 71.7% (30/4, p = 6e-6) at a lower > 3 mm rate (0.5 vs 1.4%).
Overlap cost (arm 4) vs kinematic CEM: 95.0 vs 85.8% (15/4, p = 0.02).

Internals: RRT was needed on 1,242 of 8,040 decisions (15%) but its path beat CEM's choice only 3 times; the light CBF still modified
~50% of steps (exp 51: 81%); TVLQR tracking error to the soft prediction 2.4° (exp 51 iLQR to the free-motion prediction: 5.1°).

## Hypotheses

| # | hypothesis | outcome |
|---|---|---|
| H1 | soft replay helps `tg0`: arm 1 ≥ kinematic CEM 85.8% | **held** (91.7%, +5.9, p = 0.09) |
| H2 | contact-aware TVLQR costs ≤ 3 points | **held**, it adds +2.5 (n.s.) instead of exp 51's −13.8 |
| H3 | RRT as a proposer costs ≤ 2 points | **held** (−1.7, n.s.); it is almost never chosen |
| H4 | overlap cost cuts > 3 mm steps ≥ 50% at ≤ 3 points | **held, strongly**: −86% (18.0 → 2.5%) at +3.3 points (n.s.) |
| H5 | cube-aware stack within 3 points of arm 1 and > 3 mm < 5% | **held** (93.3 vs 91.7%; 0.5%) |

## Reading

- Making each layer cube-aware removes the stack's success cost entirely: 71.7% → 93.3% for the full stack on the same cases, with the
  same safety (> 3 mm overlap 1.4% → 0.5%). The exp-51 losses came from cube-blind post-processing, not from the layers' ideas.
- The overlap cost inside CEM does most of the safety work and is the best single arm (95.0%, 2.5%). The final CBF adds a further halving
  of > 3 mm steps but slows success (10 s: 82.5 → 70.0%) because it still edits half the commands.
- TVLQR tracking the soft (contact-aware) prediction is harmless to mildly helpful; RRT adds nothing on the cube (CEM's own candidates
  already avoid the collisions it would route around) and doubles the time.
- Recommended cube stack from this seed: soft CEM + overlap cost (+ TVLQR); keep the light CBF only if the last 2 mm of overlap matter.
- Soft replay raised drops for plain CEM (5.8% vs 0.8% kinematic; exp 52 `res_phmro`: 0%), the overlap cost brings them back to 1.7%.
- Single seed: differences between arms 1–6 are all within noise (p ≥ 0.42); only the comparisons with the exp-53 arms are significant.
  A second seed (791000) on arms 1, 2, 4, 6 would separate them.

## Files

`artifacts/exp54/{smoke,cube_s790000,analysis_s790000.json}`, analysis `scripts/analyze/analyze_exp54.py`, chain `logs/run_exp54.sh`,
logs `logs/exp54_*.log`. Code: stack layers `tvlqr`, `rrtp`, `pen`, flags `--pen-weight`, `--stack-cbf-tol-mm`, `ILQR.gains()`.
Smoke bug fixed: `ILQR.gains` lacked `@torch.no_grad()` (trace save failed); trace saving now detaches.
