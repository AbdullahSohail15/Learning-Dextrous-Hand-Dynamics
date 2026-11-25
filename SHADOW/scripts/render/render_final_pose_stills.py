"""Palm-side stills of the final pose of exp-50 gesture tasks, per controller, with the goal gesture as a translucent green ghost on the
same hand (viser + headless Chromium, as render_gesture_gallery.py). For paired "controller A fails, B succeeds" figures.

  PLAYWRIGHT_BROWSERS_PATH=.playwright python3 scripts/tools/cpu_budget.py .venv_viz/bin/python scripts/render/render_final_pose_stills.py \
      --run artifacts/exp50/gestures_s610000 --tasks 84 106 --controllers cem rrt_ilqr_cbf --output artifacts/exp50/stills_palm
"""
import argparse, io, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--tasks", type=int, nargs="+", required=True)
p.add_argument("--controllers", nargs="+", required=True); p.add_argument("--output", required=True); p.add_argument("--port", type=int, default=8129); a = p.parse_args()
import trimesh.transformations as tf, viser
from PIL import Image
from playwright.sync_api import sync_playwright
from shadow_viz.kinematics import HandModel
out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
d = np.load(Path(a.run) / "traces.npz"); TS = d["targets"]
hm = HandModel("artifacts/viz_hand_model"); base = tf.quaternion_matrix((0.0, 0.0, -0.7071, 0.7071)); kp0, _ = hm.keypoints(np.zeros(24)); base[:3, 3] = -(base[:3, :3] @ kp0.mean(0))
srv = viser.ViserServer(host="127.0.0.1", port=a.port); s = srv.scene; s.set_up_direction("+z")
srv.gui.configure_theme(control_layout="collapsed", dark_mode=False, show_logo=False, show_share_button=False)
H = {k: s.add_mesh_simple(f"/hand/{k}", v, f_, color=(222, 128, 66)) for k, (v, f_) in hm.meshes.items()}
G = {k: s.add_mesh_simple(f"/goal/{k}", v, f_, color=(32, 168, 150), opacity=0.35) for k, (v, f_) in hm.meshes.items()}


def pose(handles, q):
    T = base @ hm.fk(q)
    for k, h in handles.items(): M = T[hm.body_names.index(k)]; h.position = M[:3, 3]; h.wxyz = tf.quaternion_from_matrix(M)


with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]); pg = b.new_page(viewport={"width": 1400, "height": 820}); pg.goto(f"http://127.0.0.1:{a.port}"); time.sleep(10)
    try: pg.get_by_text("Software WebGL rendering detected").locator("xpath=ancestor::div[1]").locator("button").first.click(timeout=3000)
    except Exception: pg.mouse.click(288, 81)
    for cl in srv.get_clients().values(): cl.camera.up_direction = (0, -1, 0); cl.camera.position = (0.0, -0.03, 0.30); cl.camera.look_at = (0.0, -0.03, 0.0)
    time.sleep(1)
    for i in a.tasks:
        pose(G, TS[i, 0].astype(float))
        for c in a.controllers:
            q = d[f"{c}/q"][i][-1].astype(float); pose(H, q); time.sleep(0.8)
            im = Image.open(io.BytesIO(pg.screenshot(clip=dict(x=200, y=40, width=660, height=740)))).convert("RGB")
            err = float(d[f"{c}/joint_rmse"][i][-1]); fn = out / f"task{i}_{c}.png"; im.save(fn); print("SAVED", fn, f"{err:.2f} deg")
    b.close()
srv.stop()
