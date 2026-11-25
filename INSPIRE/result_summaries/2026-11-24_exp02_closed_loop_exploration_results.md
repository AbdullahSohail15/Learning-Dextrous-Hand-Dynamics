# Experiment 02 results: closed-loop self-exploration identifies the Inspire servo–linkage model (24 Nov 2026)

Plan and full results: `experiments/02_2026-11-24_servo_linkage_model_self_identification.md`.

- Model: per servo an implicit position loop with force cap and speed cap (7 parameters, 42 in total), linkage q_link = k p. With the
  true parameters it predicts Isaac to 0.04° over 0.6 s open loop.
- 3 "unknown" hands (spec values perturbed). **15 s of exploration** gives 0.03° (TRUE level) for every arm; the spec model is 0.74°.
  Speed limits are identified within 1% (hand 0 thumb yaw: 43.9 °/s, spec 70). u_max, K, D, m are not identifiable in free motion.
- **Closed loop** (design with the current model, screen for self-collision, run, refit): 0% finger contact during exploration vs
  11–15% for random programs, same accuracy. Designing for information did not beat random on accuracy here.
- Not run: speed-setting sweep, presses / force channel, separate model B fit.
