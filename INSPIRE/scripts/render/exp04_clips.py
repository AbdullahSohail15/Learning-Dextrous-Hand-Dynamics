"""Experiment 04: cut per-task traces (hand joints + cube pose in the hand root frame) out of artifacts/exp04/<run>/traces.npz for
render_traces.py. Palm-up mount: root at (0, 0, 0.5), rotated -90 deg about y; render with --up 1 0 0.
Usage: python3 scripts/render/exp04_clips.py artifacts/exp04/cube_param <arm> <seed> <task> <out.npz>"""
import sys
import numpy as np

run, arm, seed, task, outp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
z = np.load(f"{run}/traces.npz"); q = z[f"{arm}__{seed}__q"][task]; c = z[f"{arm}__{seed}__cube"][task]
qr = np.array([0.70710678, 0.0, -0.70710678, 0.0])                         # root orientation (wxyz)


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


w, x, y, zz = qr; R = np.array([[1 - 2 * (y * y + zz * zz), 2 * (x * y - zz * w), 2 * (x * zz + y * w)], [2 * (x * y + zz * w), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - x * w)],
                                [2 * (x * zz - y * w), 2 * (y * zz + x * w), 1 - 2 * (x * x + y * y)]])
p = (c[:, :3] - np.array([0, 0, 0.5])) @ R                                  # R^T (p - t)
qc = qmul(qr * np.array([1, -1, -1, -1]), c[:, 3:7])
np.savez(outp, q=q, joint_names=z["joint_names"], dt=float(z["dt"]), cube=np.concatenate([p, qc], 1), cube_size=0.04)
print(outp, q.shape)
