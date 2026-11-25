"""Replay closed-loop cube-reorientation rollouts recorded in Isaac (evaluate_shadow_cube_ppo.py --save-traces).

The hand (real meshes, FK from the recorded joints), the cube (with a marked face so its orientation is readable),
a translucent goal cube (overlaid on the cube and also shown beside the hand, as Isaac's goal marker), and trails dragged
by the fingertips and the cube centre. Episode dropdown lists controller, initial-error band and outcome.
"""
from __future__ import annotations
import threading, time
import numpy as np
import trimesh, trimesh.transformations as tf
import viser
from .kinematics import HandModel

HAND = (214, 219, 226); CUBE = (240, 112, 84); GOAL = (32, 168, 150); TRAIL_TIPS = (120, 130, 145); TRAIL_CUBE = (240, 112, 84)
BASE_ROT = (0.0, 0.0, -0.7071, 0.7071)   # Isaac Lab Shadow Hand init_state.rot (wxyz)


def cube_mesh(size):
    """Box with a darker cap on its +z face so rotations are visible."""
    box = trimesh.creation.box(extents=size); cap = trimesh.creation.box(extents=(size[0] * 0.6, size[1] * 0.6, size[2] * 0.04)); cap.apply_translation((0, 0, size[2] / 2))
    mark = trimesh.creation.box(extents=(size[0] * 0.04, size[1] * 0.5, size[2] * 0.6)); mark.apply_translation((size[0] / 2, 0, 0))
    return box, cap, mark


class CubeReplayViewer:
    def __init__(self, model_dir, d, host="127.0.0.1", port=8080):
        self.hand = HandModel(model_dir); self.hz = float(d["control_hz"]); self.size = np.array(d["cube_size"], float)
        ctrls = sorted({k.split("/")[0] for k in d.files if "/" in k}); bins = d["bins"]; edges = ["0.4-1", "1-2", "2-pi"]
        self.eps = []   # (label, ctrl, index)
        for c in ctrls:
            succ = d[f"{c}/success_time"]; drop = d[f"{c}/dropped"]
            for i in range(len(succ)):
                out = f"success {succ[i]:.1f}s" if np.isfinite(succ[i]) else ("dropped" if drop[i] else "no success")
                self.eps.append((f"{c} | case {i:03d} | {edges[int(bins[i])]} rad | {out}", c, i, 0 if np.isfinite(succ[i]) else (2 if drop[i] else 1)))
        self.eps.sort(key=lambda e: (e[1], e[3], e[2])); self.eps = [e[:3] for e in self.eps]   # per controller: successes first, then no success, then drops
        self.d = {k: d[k] for k in d.files}
        base = tf.quaternion_matrix(BASE_ROT); kp0, _ = self.hand.keypoints(np.zeros(24)); self.base = base.copy(); self.base[:3, 3] = -(base[:3, :3] @ kp0.mean(0))
        self.server = viser.ViserServer(host=host, port=port); self.lock = threading.Lock(); self.playing = False; self.layers = {}
        self._scene(); self._gui(); self.render()

    def _scene(self):
        s = self.server
        s.gui.configure_theme(control_layout="collapsible", dark_mode=False, show_logo=False, show_share_button=False, brand_color=GOAL)
        s.scene.set_up_direction("+z")
        s.scene.add_grid("/grid", width=0.6, height=0.6, cell_size=0.02, section_size=0.1, cell_color=(226, 229, 234), section_color=(200, 205, 212), position=(0, 0, -0.16), shadow_opacity=0.08)
        self.meshes = {link: s.scene.add_mesh_simple(f"/hand/{link}", v, f, color=HAND) for link, (v, f) in self.hand.meshes.items()}
        box, cap, mark = cube_mesh(self.size)
        self.cube = [s.scene.add_mesh_simple("/cube/body", box.vertices, box.faces, color=CUBE), s.scene.add_mesh_simple("/cube/cap", cap.vertices, cap.faces, color=(150, 60, 40)), s.scene.add_mesh_simple("/cube/mark", mark.vertices, mark.faces, color=(250, 230, 120))]
        self.goal = [s.scene.add_mesh_simple("/goal/body", box.vertices, box.faces, color=GOAL, opacity=0.3), s.scene.add_mesh_simple("/goal/cap", cap.vertices, cap.faces, color=(10, 110, 95), opacity=0.5), s.scene.add_mesh_simple("/goal/mark", mark.vertices, mark.faces, color=(250, 230, 120), opacity=0.5)]
        self.goal_side = [s.scene.add_mesh_simple("/goal_side/body", box.vertices, box.faces, color=GOAL), s.scene.add_mesh_simple("/goal_side/cap", cap.vertices, cap.faces, color=(10, 110, 95)), s.scene.add_mesh_simple("/goal_side/mark", mark.vertices, mark.faces, color=(250, 230, 120))]

        @s.on_client_connect
        def _(client):
            client.camera.position = (0.20, -0.26, 0.20); client.camera.look_at = (0.0, 0.0, 0.0); client.camera.up_direction = (0, 0, 1)

    def _gui(self):
        g = self.server.gui
        g.add_markdown("**Shadow Hand · cube rollouts (Isaac, closed loop)**  \n<span style='color:rgb(240,112,84)'>■ cube</span> &nbsp; <span style='color:rgb(32,168,150)'>■ goal</span>")
        ctrls = sorted({c for _, c, _ in self.eps}); self.ui_ctrl = g.add_dropdown("Controller", ctrls, initial_value=next((c for c in ctrls if "res_det" in c and "r2c" in c), ctrls[0]))
        self.ui_ep = g.add_dropdown("Episode", [e[0] for e in self.eps if e[1] == self.ui_ctrl.value])
        T = self.d[f"{self.eps[0][1]}/q"].shape[1]
        self.ui_t = g.add_slider("Time step", 0, T - 1, 1, 0)
        self.ui_play = g.add_button("Play", icon=viser.Icon.PLAYER_PLAY); self.ui_speed = g.add_slider("Playback ×real time", 0.1, 2.0, 0.1, 1.0)
        with g.add_folder("Display"):
            self.ui_tail = g.add_slider("Trail length (steps)", 0, 80, 1, 30)
            self.ui_overlay = g.add_checkbox("Goal overlaid on cube", True); self.ui_side = g.add_checkbox("Goal beside the hand", True)
        self.ui_info = g.add_markdown("")
        def ctrl_changed(_):
            opts = [e[0] for e in self.eps if e[1] == self.ui_ctrl.value]; self.ui_ep.options = opts; self.ui_ep.value = opts[0]; self.ui_t.value = 0
        self.ui_ctrl.on_update(ctrl_changed)
        self.ui_ep.on_update(lambda _: setattr(self.ui_t, "value", 0) or self.render())
        for h in (self.ui_t, self.ui_tail, self.ui_overlay, self.ui_side): h.on_update(lambda _: self.render())
        self.ui_play.on_click(lambda _: self._toggle())

    def _toggle(self):
        self.playing = not self.playing; self.ui_play.label = "Pause" if self.playing else "Play"; self.ui_play.icon = viser.Icon.PLAYER_PAUSE if self.playing else viser.Icon.PLAYER_PLAY

    def _layer(self, name, segs, color, w):
        if name in self.layers: self.layers.pop(name).remove()
        if segs is not None and len(segs): self.layers[name] = self.server.scene.add_line_segments(name, segs, colors=color, thickness=w)

    def _pose(self, handles, p, q):
        M = self.base @ tf.quaternion_matrix(q); M[:3, 3] = self.base[:3, :3] @ p + self.base[:3, 3]
        for h in handles: h.position = M[:3, 3]; h.wxyz = tf.quaternion_from_matrix(M)

    def render(self):
        with self.lock:
            lab = self.ui_ep.value; _, c, i = next(e for e in self.eps if e[0] == lab); t = int(self.ui_t.value)
            q = self.d[f"{c}/q"][i].astype(float); cp = self.d[f"{c}/cube_pos"][i].astype(float); cq = self.d[f"{c}/cube_quat"][i].astype(float); gq = self.d[f"{c}/goal_quat"][i].astype(float); err = self.d[f"{c}/rot_err"][i]
            with self.server.atomic():
                T = self.base @ self.hand.fk(q[t])
                for link, h in self.meshes.items(): M = T[self.hand.body_names.index(link)]; h.position = M[:3, 3]; h.wxyz = tf.quaternion_from_matrix(M)
                self._pose(self.cube, cp[t], cq[t]); self._pose(self.goal, cp[t], gq[t]); self._pose(self.goal_side, np.array([0.11, 0.0, 0.40]), gq[t])   # beside the hand, at the cube's height along the hand
                for h in self.goal: h.visible = self.ui_overlay.value
                for h in self.goal_side: h.visible = self.ui_side.value
                L = int(self.ui_tail.value); t0 = max(0, t - L)
                if t - t0 >= 1:
                    tips = np.stack([self.hand.keypoints(q[k])[0][-5:] for k in range(t0, t + 1)]) @ self.base[:3, :3].T + self.base[:3, 3]
                    self._layer("/trails/tips", np.stack([tips[:-1], tips[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 3), TRAIL_TIPS, 0.0012)
                    cc = cp[t0:t + 1] @ self.base[:3, :3].T + self.base[:3, 3]; self._layer("/trails/cube", np.stack([cc[:-1], cc[1:]], 1), TRAIL_CUBE, 0.002)
                else:
                    self._layer("/trails/tips", None, None, 0); self._layer("/trails/cube", None, None, 0)
            dist = np.linalg.norm(self.d[f"{c}/cube_pos"][i][t] - self.d[f"{c}/cube_pos"][i][0]) * 1000
            self.ui_info.content = f"t = {t / self.hz:.2f} s · rotation error **{np.degrees(err[t]):.0f}°** (start {np.degrees(err[0]):.0f}°)  \n{lab}"

    def loop(self):
        while True:
            if self.playing:
                T = self.d[f"{self.eps[0][1]}/q"].shape[1]; self.ui_t.value = (self.ui_t.value + 1) % T; time.sleep(1.0 / (self.hz * float(self.ui_speed.value)))
            else: time.sleep(0.05)
