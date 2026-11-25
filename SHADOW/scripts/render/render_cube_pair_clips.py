"""Exp-51 deck clips: one cube case under two controllers (e.g. CEM vs the RRT+iLQR+CBF stack), from exp51_traces_to_viewer.py output.

Each clip runs from the start to the success step (then holds that frame 1.5 s, before the next goal appears) or, if the controller never
solves, for --seconds. Goals continue after a success in the evaluator, so the recorded goal at the success step is already the next one;
the held frame uses the solved goal. Motion is Isaac's closed loop; rendering is replay in shadow_viz (Isaac's RTX renderer has no device here).
  PLAYWRIGHT_BROWSERS_PATH=<pw> python3 scripts/tools/cpu_budget.py .venv_viz/bin/python scripts/render/render_cube_pair_clips.py --traces X.npz --case 87 --controllers CEM STACK --labels "..." "..." --output DIR
"""
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shadow_viz.cube_viewer import CubeReplayViewer
p = argparse.ArgumentParser(); p.add_argument("--traces", required=True); p.add_argument("--case", type=int, required=True); p.add_argument("--controllers", nargs="+", required=True)
p.add_argument("--labels", nargs="+", required=True); p.add_argument("--output", required=True); p.add_argument("--seconds", type=float, default=10.0); p.add_argument("--port", type=int, default=8123); p.add_argument("--stride", type=int, default=1, help="render every n-th 20 Hz step (fps = 20 / n)")
a = p.parse_args()
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
d = dict(np.load(a.traces)); hz = float(d["control_hz"]); i = a.case
for c in a.controllers:   # freeze the solved goal at the success step
    s = d[f"{c}/success_time"][i]
    if np.isfinite(s):
        k = int(round(s * hz)); d[f"{c}/goal_quat"][i, k:] = d[f"{c}/goal_quat"][i, k - 1]
class NP(dict):
    files = property(lambda self: list(self.keys()))
v = CubeReplayViewer("artifacts/viz_hand_model", NP(d), port=a.port)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24); small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
kept = []
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]); pg = b.new_page(viewport={"width": 1400, "height": 820}); pg.goto(f"http://127.0.0.1:{a.port}"); time.sleep(10)
    try: pg.get_by_text("Software WebGL rendering detected").locator("xpath=ancestor::div[1]").locator("button").first.click(timeout=3000)
    except Exception: pg.mouse.click(288, 81)
    for cl in v.server.get_clients().values(): cl.camera.position = (0.17, -0.22, 0.17); cl.camera.look_at = (0.0, 0.0, 0.03)
    for c, lab in zip(a.controllers, a.labels):
        v.ui_ctrl.value = c; time.sleep(1); ep = next(e[0] for e in v.eps if e[1] == c and e[2] == i); v.ui_ep.value = ep; time.sleep(1)
        s = float(d[f"{c}/success_time"][i]); err = d[f"{c}/rot_err"][i]; drop = bool(d[f"{c}/dropped"][i])
        if np.isfinite(s): k = int(round(s * hz)); frames = list(range(0, k, a.stride)) + [k] * int(1.5 * hz / a.stride)
        else: frames = list(range(0, min(int(a.seconds * hz), err.shape[0] - 1) + 1, a.stride))
        Rc = v.d[f"{c}/cube_quat"][i]; Rg = v.d[f"{c}/goal_quat"][i]
        def e_at(t):   # rotation error vs the (frozen) goal
            w = abs(float(np.dot(Rc[t], Rg[t]))); return 2 * np.degrees(np.arccos(min(1.0, w)))
        name = f"case{i:03d}_{c}.mp4"; w = imageio.get_writer(out / name, fps=int(hz) // a.stride, codec="libx264", quality=8, macro_block_size=1, ffmpeg_params=["-threads", "1", "-movflags", "+faststart"], pixelformat="yuv420p")
        poster = None
        for n, t in enumerate(frames):
            v.ui_t.value = t; v.render(); time.sleep(0.15)
            im = Image.open(io.BytesIO(pg.screenshot(clip=dict(x=0, y=0, width=1060, height=820)))).convert("RGB"); dr = ImageDraw.Draw(im)
            dr.rectangle((0, 0, 1060, 70), fill=(22, 33, 44)); dr.text((16, 6), lab, font=small, fill=(242, 239, 232))
            solved = np.isfinite(s) and t >= int(round(s * hz))
            msg = f"t = {t / hz:4.1f} s   rotation error {e_at(t):5.1f}°" + (f"   SOLVED at {s:.1f} s" if solved else ("   not solved in 10 s" if (not np.isfinite(s) and n == len(frames) - 1) else ""))
            dr.text((16, 36), msg, font=font if solved else small, fill=(137, 220, 177) if solved else (240, 138, 75) if "not solved" in msg else (230, 235, 240))
            arr = np.asarray(im); w.append_data(arr)
            if n == min(len(frames) - 1, int(1.0 * hz) // a.stride): poster = im.copy()
        w.close(); poster.save(out / name.replace(".mp4", ".png"))
        kept.append(dict(case=i, controller=c, label=lab, success_time=None if not np.isfinite(s) else s, dropped=drop, start_err_deg=float(np.degrees(err[0])), final_err_deg=e_at(frames[-1]), file=name)); print("SAVED", name, kept[-1], flush=True)
    b.close()
(out / f"case{i:03d}.json").write_text(json.dumps(dict(traces=a.traces, clips=kept), indent=1)); print("PAIR_COMPLETE", flush=True); v.server.stop()
