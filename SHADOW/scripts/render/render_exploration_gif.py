"""GIF of the Shadow Hand's self-exploration (exp 49): the measured joint motion of one recorded exploration program
(collect_exp49.py episodes.npz, q_deg at 120 Hz), rendered with the hand meshes in viser + headless Chromium (as render_gesture_gallery.py).

  PLAYWRIGHT_BROWSERS_PATH=.playwright python3 scripts/tools/cpu_budget.py .venv_viz/bin/python scripts/render/render_exploration_gif.py \
      --episodes artifacts/exp49/min/r_m1/episodes.npz --start 8.5 --seconds 8 --output artifacts/exp49/gifs/explore_m1
"""
import argparse, io, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
p = argparse.ArgumentParser(); p.add_argument("--episodes", required=True); p.add_argument("--episode", type=int, default=0)
p.add_argument("--start", type=float, default=0.0); p.add_argument("--seconds", type=float, default=8.0); p.add_argument("--fps", type=int, default=10)
p.add_argument("--output", required=True); p.add_argument("--port", type=int, default=8127); p.add_argument("--gif-width", type=int, default=560)
p.add_argument("--label", default="self-exploration"); p.add_argument("--title", default="Shadow Hand self-exploration · designed program · measured joints (Isaac, gravity on)"); a = p.parse_args()
import imageio.v2 as imageio, trimesh.transformations as tf, viser
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
from shadow_viz.kinematics import HandModel
out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
Q = np.load(a.episodes)["q_deg"][a.episode].astype(float); HZ = 120
idx = list(range(int(a.start * HZ), min(len(Q), int((a.start + a.seconds) * HZ)), HZ // a.fps))
hm = HandModel("artifacts/viz_hand_model"); base = tf.quaternion_matrix((0.0, 0.0, -0.7071, 0.7071)); kp0, _ = hm.keypoints(np.zeros(24)); base[:3, 3] = -(base[:3, :3] @ kp0.mean(0))
srv = viser.ViserServer(host="127.0.0.1", port=a.port); s = srv.scene; s.set_up_direction("+z")
srv.gui.configure_theme(control_layout="collapsed", dark_mode=False, show_logo=False, show_share_button=False)
H = {k: s.add_mesh_simple(f"/hand/{k}", v, f_, color=(214, 219, 226)) for k, (v, f_) in hm.meshes.items()}
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18); big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
frames = []
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]); pg = b.new_page(viewport={"width": 1400, "height": 820}); pg.goto(f"http://127.0.0.1:{a.port}"); time.sleep(10)
    try: pg.get_by_text("Software WebGL rendering detected").locator("xpath=ancestor::div[1]").locator("button").first.click(timeout=3000)
    except Exception: pg.mouse.click(288, 81)
    for cl in srv.get_clients().values(): cl.camera.up_direction = (0, -1, 0); cl.camera.position = (0.0, -0.03, 0.34); cl.camera.look_at = (0.0, -0.03, 0.0)
    time.sleep(1)
    for k_ in idx:
        T = base @ hm.fk(Q[k_])
        for k, h in H.items(): M = T[hm.body_names.index(k)]; h.position = M[:3, 3]; h.wxyz = tf.quaternion_from_matrix(M)
        time.sleep(0.15); im = Image.open(io.BytesIO(pg.screenshot(clip=dict(x=200, y=0, width=760, height=820)))).convert("RGB")
        dr = ImageDraw.Draw(im); dr.rectangle((0, 0, im.width, 56), fill=(18, 24, 33)); dr.text((12, 6), a.label, font=big, fill=(240, 138, 75))
        dr.text((12, 32), f"t = {k_ / HZ:5.1f} s   (20 Hz commands, real time)", font=font, fill=(230, 235, 240)); frames.append(im)
    b.close()
srv.stop()
w = imageio.get_writer(out / "explore.mp4", fps=a.fps, codec="libx264", quality=8, macro_block_size=1, ffmpeg_params=["-threads", "1"])
for im in frames: w.append_data(np.asarray(im))
w.close()
sm = [im.resize((a.gif_width, int(im.height * a.gif_width / im.width)), Image.LANCZOS) for im in frames]
sm[0].save(out / "explore.gif", save_all=True, append_images=sm[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
frames[len(frames) // 2].save(out / "explore.png"); print("SAVED", out, len(frames), "frames")
