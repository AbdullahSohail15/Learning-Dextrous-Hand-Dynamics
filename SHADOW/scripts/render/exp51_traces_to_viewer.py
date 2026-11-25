"""Re-frame exp-51/52 MPC traces (state63 + goal quats, world axes) into the cube viewer's format (hand-root frame, metres), for clips.

  python scripts/render/exp51_traces_to_viewer.py --pose artifacts/exp51/hand_root_pose.json --out X.npz name=traces_a.npz name2=traces_b.npz
"""
import argparse, json, numpy as np
p = argparse.ArgumentParser(); p.add_argument("--pose", required=True); p.add_argument("--out", required=True); p.add_argument("arms", nargs="+"); a = p.parse_args()
P = json.loads(open(a.pose).read())
def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], -1)
def q2m(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1), np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1), np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)
def m2q(M):   # robust (Shepperd) rotation matrix -> wxyz
    out = np.zeros(M.shape[:-2] + (4,)); Mf = M.reshape(-1, 3, 3); of = out.reshape(-1, 4)
    for k, R in enumerate(Mf):
        t = np.trace(R)
        if t > 0: s = np.sqrt(t + 1) * 2; q = [0.25 * s, (R[2,1]-R[1,2]) / s, (R[0,2]-R[2,0]) / s, (R[1,0]-R[0,1]) / s]
        else:
            i = int(np.argmax(np.diag(R))); j, l = (i + 1) % 3, (i + 2) % 3; s = np.sqrt(1 + R[i,i] - R[j,j] - R[l,l]) * 2
            q = np.zeros(4); q[0] = (R[l,j]-R[j,l]) / s; q[1+i] = 0.25 * s; q[1+j] = (R[j,i]+R[i,j]) / s; q[1+l] = (R[l,i]+R[i,l]) / s
        of[k] = q
    return out
rq = np.array(P["root_quat_wxyz"]); rp = np.array(P["root_pos"]); Rr = q2m(rq); ic = rq * np.array([1, -1, -1, -1]); ih = np.array(P["in_hand"])
D = dict(cube_size=np.array(P["cube_size"]), control_hz=20, joint_names=np.array(P["joint_names"])); bins = None
for spec in a.arms:
    name, path = spec.split("=", 1); d = np.load(path); S = d["states"].astype(float)
    c0, c1 = S[..., 51:54], S[..., 54:57]; c0 = c0 / np.linalg.norm(c0, axis=-1, keepdims=True); c1 = c1 - (c0 * c1).sum(-1, keepdims=True) * c0; c1 /= np.linalg.norm(c1, axis=-1, keepdims=True)
    Rc = np.stack([c0, c1, np.cross(c0, c1)], -1)                       # world cube rotation
    Rg = q2m(d["goal_quat"].astype(float))
    err = np.arccos(np.clip((np.trace(np.swapaxes(Rc, -1, -2) @ Rg, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    pw = S[..., 48:51] / 1000 + ih                                      # env-frame position
    D[f"{name}/q"] = S[..., :24].astype(np.float32); D[f"{name}/cube_pos"] = ((pw - rp) @ Rr).astype(np.float32)
    D[f"{name}/cube_quat"] = qmul(ic, m2q(Rc)).astype(np.float32); D[f"{name}/goal_quat"] = qmul(ic, d["goal_quat"].astype(float)).astype(np.float32)
    D[f"{name}/rot_err"] = err.astype(np.float32); D[f"{name}/success_time"] = np.where(d["success"], d["time_to_success"], np.nan).astype(np.float32); D[f"{name}/dropped"] = d["dropped"]
    bins = d["bins"]; print(name, "success", d["success"].mean().round(3), "start err deg (median)", np.degrees(np.median(err[:, 0])).round(1))
D["bins"] = bins; np.savez_compressed(a.out, **D); print("WROTE", a.out)
