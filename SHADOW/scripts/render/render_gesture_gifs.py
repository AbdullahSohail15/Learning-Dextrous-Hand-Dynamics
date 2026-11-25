"""GIFs (and MP4s) of successful gesture-reaching rollouts (experiment 27), replayed from the joint traces that
evaluate_shadow_pose_mpc.py recorded in Isaac. Isaac's RTX renderer cannot initialise on these GPUs, so the recorded joints
are rendered with the hand meshes in a viser scene and captured in headless Chromium (as render_cube_trace_videos.py).
The left hand is the closed-loop Isaac motion; the green hand on the right is the target gesture.

For each gesture: a successful task (final joint RMSE < tolerance) of --controller, preferring recorded (non-neutral) starts with
the largest initial joint distance; if none, the first --fallback controller with a success. Sequence tasks: --sequences N renders
the N sequence tasks with the most segments reached.

  PLAYWRIGHT_BROWSERS_PATH=<dir> python3 scripts/tools/cpu_budget.py .venv_viz/bin/python scripts/render/render_gesture_gifs.py --run <eval dir> --controller mpc_res_det_f100:joint --output <dir>
"""
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", required=True); p.add_argument("--controller", required=True); p.add_argument("--fallback", nargs="*", default=[]); p.add_argument("--output", required=True)
p.add_argument("--port", type=int, default=8121); p.add_argument("--tol", type=float, default=2.0); p.add_argument("--sequences", type=int, default=0)
p.add_argument("--jobs", nargs="*", default=None, help="explicit ctrl:task_index pairs, rendered for the full episode whether or not they succeed (paired comparisons)");
p.add_argument("--only", nargs="*", default=None, help="render only these gestures"); p.add_argument("--gif-width", type=int, default=520); p.add_argument("--every", type=int, default=2, help="keep every n-th 20 Hz frame in the GIF")
a = p.parse_args()
import imageio.v2 as imageio, trimesh.transformations as tf, viser
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
from shadow_viz.kinematics import HandModel
out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
R = json.loads((Path(a.run) / "results.json").read_text()); d = np.load(Path(a.run) / "traces.npz")
tasks = R["tasks"]; hz = int(d["control_hz"]); seg = int(d["segment_steps"]); TS = d["targets"]; names = [t["target_name"] for t in tasks]
hm = HandModel("artifacts/viz_hand_model"); base = tf.quaternion_matrix((0.0, 0.0, -0.7071, 0.7071)); kp0, _ = hm.keypoints(np.zeros(24)); base[:3, 3] = -(base[:3, :3] @ kp0.mean(0))
OFF = np.array([-0.2, 0.0, 0.0])   # target hand beside the simulated one (it appears on the right in the palm-side view)
srv = viser.ViserServer(host="127.0.0.1", port=a.port); s = srv.scene; s.set_up_direction("+z")
srv.gui.configure_theme(control_layout="collapsed", dark_mode=False, show_logo=False, show_share_button=False)
act = {k: s.add_mesh_simple(f"/hand/{k}", v, f_, color=(214, 219, 226)) for k, (v, f_) in hm.meshes.items()}
tgt = {k: s.add_mesh_simple(f"/target/{k}", v, f_, color=(32, 168, 150), opacity=0.85) for k, (v, f_) in hm.meshes.items()}


def pose(handles, q, off):
    T = base @ hm.fk(q)
    for k, h in handles.items(): M = T[hm.body_names.index(k)]; h.position = M[:3, 3] + off; h.wxyz = tf.quaternion_from_matrix(M)


font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22); small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
LBL = dict(direct="direct command", cem="CEM-MPC", ilqr="iLQR-MPC", rrt_ilqr="RRT + iLQR-MPC", rrt_ilqr_cbf="RRT + iLQR-MPC + CBF")
def pick(ctrl, g):
    k = f"{ctrl}/joint_rmse"
    if k not in d.files: return None
    J = d[k]; cand = []
    for i, n in enumerate(names):
        if n != f"library:{g}" or J[i, -int(0.5 * hz):].mean() >= a.tol: continue
        cand.append((tasks[i]["start_name"] != "neutral", float(J[i, 0]), i))
    return max(cand)[2] if cand else None


jobs = []
forced = set()
if a.jobs:
    for spec in a.jobs:
        c, i = spec.rsplit(":", 1); i = int(i); jobs.append((names[i].split(":", 1)[1], c, i)); forced.add((c, i))
gestures = [g for g in dict.fromkeys(n.split(":", 1)[1] for n in names if n.startswith("library:")) if a.only is None or g in a.only]
for g in ([] if a.jobs else gestures):
    for c in [a.controller] + a.fallback:
        i = pick(c, g)
        if i is not None: jobs.append((g, c, i)); break
    else: print(f"NO_SUCCESS {g}", flush=True)
if a.sequences:
    J = d[f"{a.controller}/joint_rmse"]; nseg = TS.shape[1]
    reached = [sum(J[i, (j + 1) * seg - int(0.5 * hz) + 1:(j + 1) * seg + 1].mean() < a.tol for j in range(nseg)) for i in range(len(names))]
    for i in np.argsort(reached)[::-1][:a.sequences]: jobs.append((names[i].split(":", 1)[1].replace(">", " > "), a.controller, int(i)))
kept = []
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]); pg = b.new_page(viewport={"width": 1400, "height": 820}); pg.goto(f"http://127.0.0.1:{a.port}"); time.sleep(10)
    try: pg.get_by_text("Software WebGL rendering detected").locator("xpath=ancestor::div[1]").locator("button").first.click(timeout=3000)
    except Exception: pg.mouse.click(288, 81)
    for cl in srv.get_clients().values(): cl.camera.up_direction = (0, -1, 0); cl.camera.position = (-0.15, -0.03, 0.33); cl.camera.look_at = (-0.15, -0.03, 0.0)   # palm side, fingers up
    time.sleep(1)
    for g, c, i in jobs:
        Q = d[f"{c}/q"][i].astype(float); J = d[f"{c}/joint_rmse"][i]; T = len(Q) - 1
        hit = np.where(J < a.tol)[0]; end = T if (" > " in g or (c, i) in forced) else min(T, int(hit[0] + 1.5 * hz) if len(hit) else T)
        tag = g.replace(" > ", "-") + (f"_{c}_{i}" if (c, i) in forced else ""); frames = []
        for t in range(0, end + 1):
            sj = min(t // seg, TS.shape[1] - 1) if t else 0; pose(act, Q[t], 0); pose(tgt, TS[i, sj].astype(float), OFF); time.sleep(0.12)
            if t % a.every and t != end: continue
            im = Image.open(io.BytesIO(pg.screenshot(clip=dict(x=0, y=0, width=1060, height=820)))).convert("RGB"); dr = ImageDraw.Draw(im)
            dr.rectangle((0, 0, im.width, 70), fill=(18, 24, 33))
            dr.text((14, 8), f"{g.replace('_', ' ')}  ·  {LBL.get(c, c.replace('mpc_', 'MPC ').replace(':joint', ''))}  ·  closed loop in Isaac", font=small, fill=(255, 255, 255))
            ok = J[t] < a.tol
            dr.text((14, 36), f"t = {t / hz:4.2f} s   joint error {J[t]:5.1f} deg" + ("   REACHED" if ok else ""), font=font if ok else small, fill=(137, 220, 177) if ok else (230, 235, 240))
            dr.text((im.width - 330, im.height - 34), "left: Isaac (grey)   right: target (green)", font=small, fill=(60, 70, 80))
            frames.append(im)
        mp4 = out / f"{tag}.mp4"; w = imageio.get_writer(mp4, fps=hz // a.every, codec="libx264", quality=8, macro_block_size=1, ffmpeg_params=["-threads", "1", "-movflags", "+faststart"])
        for im in frames: w.append_data(np.asarray(im))
        w.close()
        small_frames = [im.resize((a.gif_width, int(im.height * a.gif_width / im.width)), Image.LANCZOS) for im in frames]
        small_frames[0].save(out / f"{tag}.gif", save_all=True, append_images=small_frames[1:], duration=int(1000 * a.every / hz), loop=0, optimize=True)
        kept.append(dict(gesture=g, controller=c, task=int(i), start=tasks[i]["start_name"], initial_joint_rmse=float(J[0]), final_joint_rmse=float(J[-int(0.5 * hz):].mean()), gif=f"{tag}.gif", mp4=f"{tag}.mp4"))
        print("SAVED", tag, c, i, flush=True)
    b.close()
(out / "gifs.json").write_text(json.dumps(dict(run=a.run, controller=a.controller, note="motion recorded in Isaac (closed loop); rendered by replay because Isaac's RTX renderer cannot initialise on these GPUs", items=kept), indent=1))
print("GIFS_COMPLETE", len(kept), flush=True); srv.stop()
