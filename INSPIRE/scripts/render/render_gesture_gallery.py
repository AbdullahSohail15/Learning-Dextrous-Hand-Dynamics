"""Experiment 03: one still per gesture of the Inspire library (assets/inspire_hand/gesture_library.json), tiled into a labelled grid.
Same viser + headless-Chromium rendering as render_traces.py (palm side, fingers up).
  PLAYWRIGHT_BROWSERS_PATH=../SHADOW/.playwright ../SHADOW/.venv_viz/bin/python scripts/render/render_gesture_gallery.py --output <png>"""
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--port", type=int, default=8126)
p.add_argument("--cols", type=int, default=6); p.add_argument("--tile", type=int, default=320)
p.add_argument("--cam", type=float, nargs=6, default=(0.20, -0.13, 0.19, 0.025, -0.005, 0.105)); a = p.parse_args()
import trimesh.transformations as tf, viser
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
from inspire_dynamics.urdf_kin import URDFKin
K = URDFKin(); lib = json.load(open(ROOT / "assets/inspire_hand/gesture_library.json")); act = lib["servo_order"]
srv = viser.ViserServer(host="127.0.0.1", port=a.port); s = srv.scene; s.set_up_direction("+z")
srv.gui.configure_theme(control_layout="collapsed", dark_mode=False, show_logo=False, show_share_button=False)
H = {}
for link, parts in K.visual_meshes().items():
    for j, m in enumerate(parts): H[(link, j)] = s.add_mesh_simple(f"/hand/{link}_{j}", np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32), color=(150, 158, 175) if link == "hand_base_link" else (214, 219, 226))
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24); tiles = []
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]); pg = b.new_page(viewport={"width": 1400, "height": 820}); pg.goto(f"http://127.0.0.1:{a.port}"); time.sleep(10)
    try: pg.get_by_text("Software WebGL rendering detected").locator("xpath=ancestor::div[1]").locator("button").first.click(timeout=3000)
    except Exception: pg.mouse.click(288, 81)
    for cl in srv.get_clients().values(): cl.camera.up_direction = (0, 0, 1); cl.camera.position = tuple(a.cam[:3]); cl.camera.look_at = tuple(a.cam[3:])
    time.sleep(1)
    for g, v in lib["gestures"].items():
        X = K.fk(dict(zip(act, v["a"])))
        for (link, j), h in H.items(): M = X[link]; h.position = M[:3, 3]; h.wxyz = tf.quaternion_from_matrix(M)
        time.sleep(0.6); im = Image.open(io.BytesIO(pg.screenshot(clip=dict(x=230, y=80, width=600, height=700)))).convert("RGB")
        im = im.resize((a.tile, int(a.tile * 700 / 600))); dr = ImageDraw.Draw(im); dr.text((10, 8), g.replace("_", " "), font=font, fill=(22, 33, 44)); tiles.append(im)
    b.close()
srv.stop()
rows = (len(tiles) + a.cols - 1) // a.cols; W, Ht = tiles[0].size; grid = Image.new("RGB", (W * a.cols, Ht * rows), (255, 255, 255))
for i, t in enumerate(tiles): grid.paste(t, ((i % a.cols) * W, (i // a.cols) * Ht))
grid.save(a.output); print("SAVED", a.output, len(tiles))
