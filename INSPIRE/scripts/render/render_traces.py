"""Render joint traces recorded in Isaac (12 Inspire joints, optional cube pose) as MP4 + GIF. Isaac's RTX renderer cannot create a device on
these GPUs (as in SHADOW), so the recorded motion is replayed with the hand's visual meshes in a viser scene captured by headless Chromium.

  PLAYWRIGHT_BROWSERS_PATH=../SHADOW/.playwright ../SHADOW/.venv_viz/bin/python scripts/render/render_traces.py \
      --traces artifacts/exp01/smoke/traces.npz --output artifacts/exp01/videos --name smoke --title "..."

traces.npz: q (T, 12) joint angles in rad, joint_names (12,), dt; optional target (T, 6) (shown as text), cube (T, 7) pos + wxyz in the hand
root frame, ghost_q (T, 12) or (12,) for a translucent target hand beside the real one.
--poster: write a single PNG of frame --frame and exit (camera tuning).
"""
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--traces", required=True); p.add_argument("--output", required=True); p.add_argument("--name", required=True)
p.add_argument("--title", default=""); p.add_argument("--port", type=int, default=8141); p.add_argument("--every", type=int, default=2)
p.add_argument("--start", type=int, default=0); p.add_argument("--end", type=int, default=None)
p.add_argument("--cam", type=float, nargs=6, default=(0.20, -0.13, 0.19, 0.025, -0.005, 0.105), help="camera position xyz, look-at xyz (m, hand root frame)")
p.add_argument("--ghost-offset", type=float, nargs=3, default=(0.0, 0.16, 0.0)); p.add_argument("--poster", action="store_true"); p.add_argument("--frame", type=int, default=0)
p.add_argument("--gif-width", type=int, default=520); p.add_argument("--up", type=float, nargs=3, default=(0.0, 0.0, 1.0), help="camera up direction (hand root frame); palm-up mount: 1 0 0")
a = p.parse_args()
import imageio.v2 as imageio, trimesh.transformations as tf, viser
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
from inspire_dynamics.urdf_kin import URDFKin

out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
for f in [out / f"{a.name}.mp4", out / f"{a.name}.png"]: assert not f.exists(), f"{f} exists (never overwritten)"
d = np.load(a.traces); Q = d["q"].astype(float); names = [str(n) for n in d["joint_names"]]; dt = float(d["dt"]); T = len(Q)
end = T if a.end is None else min(a.end, T)
K = URDFKin(); V = K.visual_meshes()
ghost = d["ghost_q"].astype(float) if "ghost_q" in d.files else None
if ghost is not None and ghost.ndim == 1: ghost = np.repeat(ghost[None], T, 0)
srv = viser.ViserServer(host="127.0.0.1", port=a.port); s = srv.scene; s.set_up_direction(tuple(a.up))
srv.gui.configure_theme(control_layout="collapsed", dark_mode=False, show_logo=False, show_share_button=False)
COL = {"hand_base_link": (70, 74, 82)}


def add(prefix, color=None, opacity=1.0):
    H = {}
    for link, parts in V.items():
        for k, m in enumerate(parts):
            H[(link, k)] = s.add_mesh_simple(f"/{prefix}/{link}_{k}", np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32),
                                             color=color or COL.get(link, (196, 200, 208)), opacity=opacity)
    return H


hand = add("hand"); gh = add("ghost", (32, 168, 150), 0.6) if ghost is not None else None
cube = None
if "cube" in d.files:
    cube = s.add_box("/cube", dimensions=(float(d["cube_size"]),) * 3 if "cube_size" in d.files else (0.05,) * 3, color=(214, 120, 40))


def pose(H, q, off=(0, 0, 0)):
    X = K.fk(dict(zip(names, q)))
    for (link, k), h in H.items(): M = X[link]; h.position = M[:3, 3] + np.asarray(off); h.wxyz = tf.quaternion_from_matrix(M)


font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20); small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
frames = []
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]); pg = b.new_page(viewport={"width": 1400, "height": 820})
    pg.goto(f"http://127.0.0.1:{a.port}"); time.sleep(10)
    try: pg.get_by_text("Software WebGL rendering detected").locator("xpath=ancestor::div[1]").locator("button").first.click(timeout=3000)
    except Exception: pg.mouse.click(288, 81)
    for cl in srv.get_clients().values(): cl.camera.up_direction = tuple(a.up); cl.camera.position = tuple(a.cam[:3]); cl.camera.look_at = tuple(a.cam[3:])
    time.sleep(1)
    idx = [a.frame] if a.poster else [t for t in range(a.start, end) if (t - a.start) % a.every == 0 or t == end - 1]
    for t in idx:
        pose(hand, Q[t])
        if gh is not None: pose(gh, ghost[t], a.ghost_offset)
        if cube is not None: c = d["cube"][t]; cube.position = c[:3]; cube.wxyz = c[3:7]
        time.sleep(0.15)
        im = Image.open(io.BytesIO(pg.screenshot(clip=dict(x=0, y=0, width=1060, height=820)))).convert("RGB"); dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, im.width, 64), fill=(18, 24, 33))
        dr.text((14, 8), a.title or a.name, font=small, fill=(255, 255, 255))
        dr.text((14, 34), f"t = {t * dt:5.2f} s", font=font, fill=(230, 235, 240))
        if "target" in d.files:
            dr.text((200, 38), "servo targets (deg): " + " ".join(f"{v:5.1f}" for v in np.degrees(d["target"][t])), font=small, fill=(190, 200, 210))
        dr.text((14, im.height - 28), "Inspire RH56 (6 servos, 12 joints) · motion simulated in Isaac Sim 4.2 / PhysX, replayed with the hand meshes", font=small, fill=(60, 70, 80))
        frames.append(im)
    b.close()
srv.stop()
if a.poster: frames[0].save(out / f"{a.name}.png"); print("SAVED", out / f"{a.name}.png"); sys.exit()
fps = max(1, int(round(1 / (dt * a.every))))
w = imageio.get_writer(out / f"{a.name}.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=1, ffmpeg_params=["-threads", "1", "-movflags", "+faststart"])
for im in frames: w.append_data(np.asarray(im))
w.close()
sm = [im.resize((a.gif_width, int(im.height * a.gif_width / im.width)), Image.LANCZOS) for im in frames]
sm[0].save(out / f"{a.name}.gif", save_all=True, append_images=sm[1:], duration=int(1000 / fps), loop=0, optimize=True)
frames[len(frames) // 2].save(out / f"{a.name}.png")
print("SAVED", out / f"{a.name}.mp4", len(frames), "frames")
