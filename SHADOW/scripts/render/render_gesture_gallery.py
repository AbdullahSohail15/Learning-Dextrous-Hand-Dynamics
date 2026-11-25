"""One still per gesture of the exp-50 library (the 24-joint targets recorded in the benchmark), tiled into a labelled grid PNG.
Same viser + headless-Chromium rendering as render_gesture_gifs.py (palm side, fingers up).
  PLAYWRIGHT_BROWSERS_PATH=.playwright python3 scripts/tools/cpu_budget.py .venv_viz/bin/python scripts/render/render_gesture_gallery.py --run artifacts/exp50/gestures_s610000 --output <png>"""
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--output", required=True); p.add_argument("--port", type=int, default=8125)
p.add_argument("--cols", type=int, default=5); p.add_argument("--tile", type=int, default=380); a = p.parse_args()
import trimesh.transformations as tf, viser
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
from shadow_viz.kinematics import HandModel
R = json.loads((Path(a.run) / "results.json").read_text()); d = np.load(Path(a.run) / "traces.npz"); TS = d["targets"]
names = [t["target_name"] for t in R["tasks"]]; lib, seen = [], set()
for i, n in enumerate(names):
    if n.startswith("library:") and n not in seen: seen.add(n); lib.append((n[8:], TS[i, 0].astype(float)))
hm = HandModel("artifacts/viz_hand_model"); base = tf.quaternion_matrix((0.0, 0.0, -0.7071, 0.7071)); kp0, _ = hm.keypoints(np.zeros(24)); base[:3, 3] = -(base[:3, :3] @ kp0.mean(0))
srv = viser.ViserServer(host="127.0.0.1", port=a.port); s = srv.scene; s.set_up_direction("+z")
srv.gui.configure_theme(control_layout="collapsed", dark_mode=False, show_logo=False, show_share_button=False)
H = {k: s.add_mesh_simple(f"/hand/{k}", v, f_, color=(214, 219, 226)) for k, (v, f_) in hm.meshes.items()}
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26); tiles = []
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]); pg = b.new_page(viewport={"width": 1400, "height": 820}); pg.goto(f"http://127.0.0.1:{a.port}"); time.sleep(10)
    try: pg.get_by_text("Software WebGL rendering detected").locator("xpath=ancestor::div[1]").locator("button").first.click(timeout=3000)
    except Exception: pg.mouse.click(288, 81)
    for cl in srv.get_clients().values(): cl.camera.up_direction = (0, -1, 0); cl.camera.position = (0.0, -0.03, 0.30); cl.camera.look_at = (0.0, -0.03, 0.0)
    time.sleep(1)
    for g, q in lib:
        T = base @ hm.fk(q)
        for k, h in H.items(): M = T[hm.body_names.index(k)]; h.position = M[:3, 3]; h.wxyz = tf.quaternion_from_matrix(M)
        time.sleep(0.6); im = Image.open(io.BytesIO(pg.screenshot(clip=dict(x=200, y=40, width=660, height=740)))).convert("RGB")
        im = im.resize((a.tile, int(a.tile * 740 / 660))); dr = ImageDraw.Draw(im); dr.text((12, 10), g.replace("_", " "), font=font, fill=(22, 33, 44)); tiles.append(im)
    b.close()
srv.stop()
rows = (len(tiles) + a.cols - 1) // a.cols; W, Ht = tiles[0].size; grid = Image.new("RGB", (W * a.cols, Ht * rows), (255, 255, 255))
for i, t in enumerate(tiles): grid.paste(t, ((i % a.cols) * W, (i // a.cols) * Ht))
grid.save(a.output); print("SAVED", a.output, len(tiles))
