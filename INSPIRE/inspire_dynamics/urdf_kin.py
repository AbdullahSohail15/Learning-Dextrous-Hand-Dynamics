"""Pure-numpy URDF kinematics for the Inspire hand (no Isaac): joint tree, forward kinematics, collision geometry as convex hulls.
Used by the self-collision pair check (experiment 01), the trace renderer and later the hand model / CBF.
Requires trimesh only for loading meshes (collision_hulls / visual_meshes)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/inspire_hand/inspire_hand_right.urdf"


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def _T(xyz, rpy):
    T = np.eye(4); T[:3, :3] = _rpy(*rpy); T[:3, 3] = xyz; return T


def _origin(el):
    o = el.find("origin") if el is not None else None
    if o is None: return np.eye(4)
    return _T([float(v) for v in o.get("xyz", "0 0 0").split()], [float(v) for v in o.get("rpy", "0 0 0").split()])


def _axis_rot(axis, q):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(4); R[:3, :3] = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K; return R


class URDFKin:
    def __init__(self, path=URDF):
        self.path = Path(path); r = ET.parse(self.path).getroot()
        self.links = {l.get("name"): l for l in r.iter("link")}
        self.joints = []
        for j in r.iter("joint"):
            m = j.find("mimic"); lim = j.find("limit")
            self.joints.append(dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                                    T=_origin(j), axis=[float(v) for v in (j.find("axis").get("xyz") if j.find("axis") is not None else "0 0 1").split()],
                                    lo=float(lim.get("lower", 0)) if lim is not None else 0.0, hi=float(lim.get("upper", 0)) if lim is not None else 0.0,
                                    mimic=None if m is None else (m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0)))))
        children = {j["child"] for j in self.joints}
        self.root = next(n for n in self.links if n not in children)
        self.revolute = [j["name"] for j in self.joints if j["type"] == "revolute"]
        self.actuated = [j["name"] for j in self.joints if j["type"] == "revolute" and j["mimic"] is None]
        order, frontier = [], [self.root]          # topological order of joints
        while frontier:
            p = frontier.pop(0)
            for j in self.joints:
                if j["parent"] == p: order.append(j); frontier.append(j["child"])
        self.order = order

    def full_q(self, q):
        """q: dict name->angle for any subset; mimic joints filled from their source. Returns dict for all revolute joints."""
        out = {n: float(q.get(n, 0.0)) for n in self.revolute}
        for j in self.joints:
            if j["mimic"] and j["name"] not in q: s, k, b = j["mimic"]; out[j["name"]] = k * out[s] + b
        return out

    def fk(self, q, base=np.eye(4)):
        """q: dict joint->angle (actuated only is enough). Returns dict link -> 4x4 world transform."""
        q = self.full_q(q); X = {self.root: base}
        for j in self.order:
            T = X[j["parent"]] @ j["T"]
            if j["type"] == "revolute": T = T @ _axis_rot(j["axis"], q[j["name"]])
            X[j["child"]] = T
        return X

    def _geoms(self, kind):
        import trimesh
        out = {}
        for name, l in self.links.items():
            parts = []
            for g in l.findall(kind):
                T = _origin(g); geo = g.find("geometry"); m = None
                if geo.find("mesh") is not None: m = trimesh.load(self.path.parent / geo.find("mesh").get("filename"), force="mesh")
                elif geo.find("box") is not None: m = trimesh.creation.box([float(v) for v in geo.find("box").get("size").split()])
                elif geo.find("cylinder") is not None:
                    c = geo.find("cylinder"); m = trimesh.creation.cylinder(radius=float(c.get("radius")), height=float(c.get("length")), sections=24)
                elif geo.find("sphere") is not None: m = trimesh.creation.icosphere(radius=float(geo.find("sphere").get("radius")))
                if m is not None: m = m.copy(); m.apply_transform(T); parts.append(m)
            if parts: out[name] = parts
        return out

    def collision_hulls(self):
        """link -> list of convex hulls (PhysX uses the convex hull of each collision mesh when convex decomposition is off)."""
        return {k: [p.convex_hull for p in v] for k, v in self._geoms("collision").items()}

    def visual_meshes(self):
        return self._geoms("visual")


def hull_penetration(A, B):
    """Approximate penetration depth (m) between two convex hulls (trimesh, same frame): max over vertices of one inside the other of the
    distance to the nearest face plane. 0 if separated (vertex-based, misses pure edge-edge overlap)."""
    def inside(P, H):
        n, d = H.face_normals, (H.face_normals * H.triangles[:, 0]).sum(1)
        s = P @ n.T - d                      # signed distance to every face plane, < 0 inside
        ins = (s < 0).all(1)
        return float((-s[ins].max(1)).max()) if ins.any() else 0.0
    return max(inside(A.vertices, B), inside(B.vertices, A))
