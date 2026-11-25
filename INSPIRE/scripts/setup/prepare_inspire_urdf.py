"""Experiment 01 (setup): write the simulation URDF of the Inspire RH56 right hand from the dex-urdf source.

Changes from assets/inspire_hand/inspire_hand_right_dexurdf.urdf (dex-urdf, commit in SOURCE_COMMIT.txt):
  - visual meshes: .glb -> .obj (converted with trimesh, meshes/visual_obj/), because the Isaac Sim 4.2 URDF importer reads OBJ/STL/DAE only;
  - joint velocity / effort limits: dex-urdf has placeholders (0.5 rad/s, 1 N m). Replaced by spec values (ASSUMPTIONS, see
    experiments/01_*.md): fingers 260 deg/s, thumb bend 70 deg/s, thumb rotation 70 deg/s (not published, assumed = thumb bend);
    torque cap = rated tip force x lever (fingers 10 N x 0.07 m, thumb 15 N x 0.06 m). Mimic (linkage-driven) joints get the
    actuated joint's velocity x |multiplier|.
  - mimic offsets set to 0 (the importer's PhysX mimic joint ignores them; dex-urdf's finger offset is -0.045 rad).
Kinematics, inertias and collision meshes are unchanged.

Usage: python3 scripts/setup/prepare_inspire_urdf.py
"""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "assets/inspire_hand"
SPEC = {  # actuated joint: (velocity rad/s, effort N m)
    "thumb_proximal_yaw_joint": (math.radians(70), 15 * 0.06),
    "thumb_proximal_pitch_joint": (math.radians(70), 15 * 0.06),
    "index_proximal_joint": (math.radians(260), 10 * 0.07),
    "middle_proximal_joint": (math.radians(260), 10 * 0.07),
    "ring_proximal_joint": (math.radians(260), 10 * 0.07),
    "pinky_proximal_joint": (math.radians(260), 10 * 0.07),
}

tree = ET.parse(A / "inspire_hand_right_dexurdf.urdf"); r = tree.getroot()
for m in r.iter("mesh"):
    f = m.get("filename")
    if f.endswith(".glb"): m.set("filename", f.replace("meshes/visual/", "meshes/visual_obj/").replace(".glb", ".obj"))
coupling = {}
for j in r.iter("joint"):
    if j.get("type") != "revolute": continue
    n, lim, mim = j.get("name"), j.find("limit"), j.find("mimic")
    if mim is not None:
        src, k, off = mim.get("joint"), float(mim.get("multiplier")), float(mim.get("offset", 0))
        # the Isaac 4.2 importer's PhysX mimic joint drops the offset (gate 0: constant 2.6 deg on the finger intermediates), so the
        # simulation URDF uses offset 0 and the sim's linkage law is q_link = multiplier * q_act. dex-urdf's offset is kept in the meta.
        mim.set("offset", "0"); coupling[n] = dict(source=src, multiplier=k, offset=0.0, dexurdf_offset=off); v, e = SPEC[src]; v *= abs(k)
    else:
        v, e = SPEC[n]
    lim.set("velocity", f"{v:.4f}"); lim.set("effort", f"{e:.4f}")
missing = [m.get("filename") for m in r.iter("mesh") if not (A / m.get("filename")).exists()]
assert not missing, missing
tree.write(A / "inspire_hand_right.urdf", xml_declaration=True, encoding="utf-8")
json.dump(dict(actuated=list(SPEC), spec_limits={k: dict(velocity=v, effort=e) for k, (v, e) in SPEC.items()}, coupling=coupling),
          open(A / "inspire_hand_right_meta.json", "w"), indent=1)
print("WROTE", A / "inspire_hand_right.urdf"); print(json.dumps(coupling, indent=1))
