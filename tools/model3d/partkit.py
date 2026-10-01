"""Part kit for building named-part GLB models from primitives (on top of build_model.py from the pro-product-video
skill): material presets, shape helpers, swept tubes / springs / belts, and an indexed GLB writer whose nodes carry
bilingual labels, inspection notes and exploded-view offsets in their extras."""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_model as bm  # noqa: E402
from model3d import corner_normals, hex01, rot_xyz  # noqa: E402


# ---------------------------------------------------------------------------------------------- materials
BLACK = dict(color="#1E2023", roughness=0.6)
BLACK_GLOSS = dict(color="#16181A", roughness=0.35)
DARK = dict(color="#2C2F33", roughness=0.55)
RUBBER = dict(color="#141516", roughness=0.85)
ALU = dict(color="#B4B9BF", metallic=0.9, roughness=0.38)
CAST = dict(color="#9EA3A8", metallic=0.8, roughness=0.55)
STEEL = dict(color="#8C9197", metallic=1.0, roughness=0.3)
ZINC = dict(color="#C9CDD3", metallic=1.0, roughness=0.28)
LEAD = dict(color="#83878B", metallic=0.6, roughness=0.5)
COPPER = dict(color="#B87333", metallic=1.0, roughness=0.3)
BRASS = dict(color="#C8A24A", metallic=1.0, roughness=0.32)
CERAMIC = dict(color="#F0EEE8", roughness=0.25)
TANK = dict(color="#ECEAE2", roughness=0.3, alpha=0.42)
RED = dict(color="#C62828", roughness=0.45)
YELLOW = dict(color="#F2B705", roughness=0.45)
BLUE = dict(color="#1F5FAF", roughness=0.4)


def T(en: str, vi: str) -> dict:
    return {"en": en, "vi": vi}


def part(key, label, mat, shapes, explode=None, check=None):
    p = {"name": key, "label": label, "shapes": shapes, "explode": explode, "check": check}
    p.update({"color": mat["color"], "metallic": mat.get("metallic", 0.0), "roughness": mat.get("roughness", 0.5),
              "alpha": mat.get("alpha", 1.0)})
    return p


# ---------------------------------------------------------------------------------------------- shape helpers
def _v(p):
    return [float(x) for x in p]


def _seg(r, lo=16, hi=64, k=2.0):
    """Round-shape segment count from the radius (mm): small parts need fewer triangles on a phone GPU."""
    return int(max(lo, min(hi, round(r * k / 4) * 4)))


def cyl(r, h, pos=(0, 0, 0), axis="y", **k):
    k.setdefault("segments", _seg(max(r, k.get("r2") or 0)))
    return dict(type="cylinder", r=r, h=h, pos=_v(pos), axis=axis, **k)


def hexa(r, h, pos=(0, 0, 0), axis="y", **k):
    """Hex prism (r = corner radius)."""
    return dict(type="cylinder", r=r, h=h, pos=_v(pos), axis=axis, segments=6, **k)


def rb(size, radius=0.0, pos=(0, 0, 0), **k):
    if radius:
        k.setdefault("segments", 2 if radius <= 1.5 else 3 if radius <= 4 else 4 if radius <= 10 else 6)
    return dict(type="rbox" if radius else "box", size=_v(size), radius=radius, pos=_v(pos), **k)


def lathe(profile, pos=(0, 0, 0), axis="y", **k):
    k.setdefault("segments", _seg(max(p[0] for p in profile), lo=20, hi=64, k=1.6))
    return dict(type="lathe", profile=[_v(p) for p in profile], pos=_v(pos), axis=axis, **k)


def tor(R, r, pos=(0, 0, 0), axis="y", **k):
    k.setdefault("segments", _seg(R, lo=20, hi=64, k=1.6))
    k.setdefault("segments2", 10 if r < 2 else 14)
    return dict(type="torus", R=R, r=r, pos=_v(pos), axis=axis, **k)


def sph(r, pos=(0, 0, 0), **k):
    k.setdefault("segments", _seg(r, lo=12, hi=40))
    return dict(type="sphere", r=r, pos=_v(pos), **k)


def ext(h, pos=(0, 0, 0), axis="y", points=None, hull=None, holes=None, **k):
    k.setdefault("segments", 24)
    s = dict(type="extrude", h=h, pos=_v(pos), axis=axis, **k)
    if points is not None:
        s["points"] = [_v(p) for p in points]
    if hull is not None:
        s["hull_circles"] = [_v(c) for c in hull]
    if holes:
        s["holes"] = [_v(c) for c in holes]
    return s


def tube(points, r, sides=20, closed=False, smooth=True, **k):
    """Round tube swept along a Catmull-Rom spline through points (absolute coordinates). r: number or per-sample
    function f(t in 0..1) -> radius."""
    return dict(type="tube", points=[_v(p) for p in points], r=r, sides=sides, closed=closed, smooth=smooth, **k)


def helix(R, r, y0, y1, turns, pos=(0, 0, 0), axis="y", **k):
    n = int(turns * 24) + 1
    a = np.linspace(0, 2 * np.pi * turns, n)
    pts = np.stack([R * np.cos(a), np.linspace(y0, y1, n), R * np.sin(a)], 1)
    return dict(type="tube", points=pts.tolist(), r=r, sides=10, closed=False, smooth=False, pos=_v(pos),
                axis=axis, **k)


def around(n, radius, make, y=0.0, phase=0.0):
    """n copies of make(x, y, z, angle_deg) placed on a circle about the Y axis."""
    out = []
    for i in range(n):
        a = phase + 360.0 * i / n
        out.append(make(radius * math.cos(math.radians(a)), y, -radius * math.sin(math.radians(a)), a))
    return out


def around_z(n, radius, make, z=0.0, phase=0.0):
    """n copies of make(x, y, z, angle_deg) placed on a circle about the Z axis."""
    out = []
    for i in range(n):
        a = phase + 360.0 * i / n
        out.append(make(radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)), z, a))
    return out


def ridges(r_in, r_out, y0, y1, pitch):
    """Lathe profile points of a ribbed / threaded surface from y1 down to y0 (alternating radii)."""
    pts, y, k = [], y1, 0
    while y > y0 + 1e-6:
        pts.append([r_out if k % 2 == 0 else r_in, y])
        y -= pitch / 2
        k += 1
    pts.append([r_in, y0])
    return pts


def gear_points(r_root, r_tip, teeth, missing=(), seg=4):
    pts = []
    for i in range(teeth):
        a0 = 2 * math.pi * i / teeth
        step = 2 * math.pi / teeth
        if i in missing:
            for f in np.linspace(0, 1, seg * 2, endpoint=False):
                pts.append([r_root * math.cos(a0 + f * step), r_root * math.sin(a0 + f * step)])
            continue
        for f, rr in ((0.0, r_root), (0.18, r_tip), (0.5, r_tip), (0.68, r_root)):
            pts.append([rr * math.cos(a0 + f * step), rr * math.sin(a0 + f * step)])
    return pts


# ---------------------------------------------------------------------------------------------- custom soups
def catmull(points, n=10, closed=False):
    P = np.asarray(points, float)
    if len(P) < 3:
        return np.linspace(P[0], P[-1], max(2, n))
    if closed:
        Q = np.vstack([P[-1], P, P[0], P[1]])
    else:
        Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    segs = len(P) if closed else len(P) - 1
    for i in range(segs):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(P[-1])
    return np.array(out)


def sweep_soup(path, profile, closed=False, up=None, scale=None):
    """Sweep a closed 2D profile (k,2) along path (n,3) with parallel-transport frames; open ends are capped."""
    C = np.asarray(path, float)
    n = len(C)
    if closed:
        tg = np.roll(C, -1, 0) - np.roll(C, 1, 0)
    else:
        tg = np.gradient(C, axis=0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-12
    ref = np.array(up if up is not None else [0, 1, 0], float)
    if abs(ref @ tg[0]) > 0.9:
        ref = np.array([1.0, 0, 0])
    nrm = [np.cross(np.cross(tg[0], ref), tg[0])]
    nrm[0] /= np.linalg.norm(nrm[0])
    for i in range(1, n):
        v = nrm[-1] - (nrm[-1] @ tg[i]) * tg[i]
        nrm.append(v / (np.linalg.norm(v) + 1e-12))
    nrm = np.array(nrm)
    if closed:  # spread the transport twist over the loop so the seam closes
        b0 = np.cross(tg[0], nrm[0])
        v = nrm[-1] - (nrm[-1] @ tg[0]) * tg[0]
        twist = math.atan2(v @ b0, v @ nrm[0])
        for i in range(n):
            a = -twist * i / n
            b = np.cross(tg[i], nrm[i])
            nrm[i] = nrm[i] * math.cos(a) + b * math.sin(a)
    bin_ = np.cross(tg, nrm)
    prof = np.asarray(profile, float)
    k = len(prof)
    sc = np.ones(n) if scale is None else np.asarray(scale, float)
    R = (C[:, None, :] + sc[:, None, None] * (prof[None, :, 0:1] * nrm[:, None, :] + prof[None, :, 1:2] * bin_[:, None, :]))
    tris = []
    rows = n if closed else n - 1
    for i in range(rows):
        i2 = (i + 1) % n
        for j in range(k):
            j2 = (j + 1) % k
            a, b, c, d = R[i, j], R[i, j2], R[i2, j2], R[i2, j]
            tris += [[a, b, c], [a, c, d]]
    if not closed:
        for i, sgn in ((0, -1), (n - 1, 1)):
            cen = R[i].mean(0)
            for j in range(k):
                tris.append([cen, R[i, j], R[i, (j + 1) % k]])
    Tt = np.array(tris)
    area = np.linalg.norm(np.cross(Tt[:, 1] - Tt[:, 0], Tt[:, 2] - Tt[:, 0]), axis=1)
    return bm._fix_volume(Tt[area > 1e-10])


def tube_soup(s):
    pts = s["points"]
    closed = s.get("closed", False)
    C = catmull(pts, int(s.get("samples", 10)), closed) if s.get("smooth", True) else np.asarray(pts, float)
    sides = int(s.get("sides", 20))
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    circle = np.stack([np.cos(a), np.sin(a)], 1)
    r = s["r"]
    if callable(r):
        scale = np.array([r(t) for t in np.linspace(0, 1, len(C))])
    else:
        scale = np.full(len(C), float(r))
    return sweep_soup(C, circle, closed, scale=scale)


def belt_soup(s):
    """Poly-V belt around pulleys in the XY plane: s['pulleys'] = [[x, y, r, side]] in loop order (side +1: ribbed
    side on the pulley, -1: back of the belt on an idler); the belt centre line runs at r + thickness/2."""
    th, w = s["thickness"], s["width"]
    pul = [(np.array([x, y], float), r + th / 2, sd) for x, y, r, sd in s["pulleys"]]
    m = len(pul)
    lines = []
    for i in range(m):
        (c1, r1, s1), (c2, r2, s2) = pul[i], pul[(i + 1) % m]
        a, b = s1 * r1, s2 * r2
        D = c2 - c1
        L = np.linalg.norm(D)
        th_ = math.acos(max(-1.0, min(1.0, (a - b) / L)))
        phi = math.atan2(D[1], D[0])
        best = None
        for sg in (1, -1):
            nv = np.array([math.cos(phi + sg * th_), math.sin(phi + sg * th_)])
            t = np.array([-nv[1], nv[0]])  # travel direction = n rotated +90 (counter-clockwise loop)
            if t @ D > 0:
                best = nv
        lines.append((c1 + a * best, c2 + b * best, best))
    pts = []
    for i in range(m):
        c, r, sd = pul[i]
        _, p_in, n_in = lines[i - 1]
        p_out, _, n_out = lines[i]
        a0 = math.atan2(*(sd * n_in)[::-1])
        a1 = math.atan2(*(sd * n_out)[::-1])
        if sd > 0:
            while a1 < a0:
                a1 += 2 * math.pi
        else:
            while a1 > a0:
                a1 -= 2 * math.pi
        k = max(2, int(abs(a1 - a0) / math.radians(6)))
        for t in np.linspace(a0, a1, k):
            pts.append(c + r * np.array([math.cos(t), math.sin(t)]))
        q = np.linspace(p_out, lines[i][1], 6)[1:-1]
        pts += list(q)
    P3 = np.array([[p[0], p[1], 0.0] for p in pts])
    keep = np.r_[True, np.linalg.norm(np.diff(P3, axis=0), axis=1) > 1e-6]
    P3 = P3[keep]
    prof = np.array([[-th / 2, -w / 2], [th / 2, -w / 2], [th / 2, w / 2], [-th / 2, w / 2]])
    return sweep_soup(P3, prof, closed=True, up=[0, 0, 1])


CUSTOM = {"tube": tube_soup, "belt": belt_soup}


def shape_soup(s):
    if s["type"] not in CUSTOM:
        return bm.shape_soup(s)
    Tt = CUSTOM[s["type"]](s)
    Tt = Tt @ bm.AXIS[s.get("axis", "y")].T
    if s.get("rot"):
        Tt = Tt @ rot_xyz(s["rot"]).T
    return Tt + np.array(s.get("pos", [0, 0, 0]), float)


# ---------------------------------------------------------------------------------------------- GLB writer
def write_glb(path: Path, spec: dict) -> tuple[int, int]:
    """Indexed GLB (welded vertices), one node per part; returns (parts, triangles)."""
    js = {"asset": {"version": "2.0", "generator": "build_component_models.py (tools/model3d)",
                    "extras": {"pv_built": True}},
          "scene": 0, "scenes": [{"nodes": [], "extras": spec["extras"]}], "nodes": [], "meshes": [],
          "materials": [], "accessors": [], "bufferViews": [], "buffers": []}
    blob = bytearray()

    def add(arr, target, comp, typ, minmax=False):
        a = np.ascontiguousarray(arr)
        off = len(blob)
        blob.extend(a.tobytes())
        while len(blob) % 4:
            blob.append(0)
        js["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": a.nbytes, "target": target})
        acc = {"bufferView": len(js["bufferViews"]) - 1, "componentType": comp, "count": int(a.size // {"VEC3": 3, "SCALAR": 1}[typ]),
               "type": typ}
        if minmax:
            acc["min"] = a.reshape(-1, 3).min(0).tolist()
            acc["max"] = a.reshape(-1, 3).max(0).tolist()
        js["accessors"].append(acc)
        return len(js["accessors"]) - 1

    mats: dict = {}
    ntri = 0
    for p in spec["parts"]:
        groups: dict = {}
        for s in p["shapes"]:
            key = (s.get("color", p["color"]), float(s.get("metallic", p["metallic"])),
                   float(s.get("roughness", p["roughness"])), float(s.get("alpha", p["alpha"])))
            groups.setdefault(key, []).append(shape_soup(s))
        prims = []
        for (col, met, rough, alpha), soups in groups.items():
            S = np.concatenate(soups)
            ntri += len(S)
            N = corner_normals(S)
            V = S.reshape(-1, 3).astype(np.float32)
            Nn = N.reshape(-1, 3).astype(np.float32)
            keyarr = np.hstack([np.round(V, 3), np.round(Nn, 3)])
            _, first, inv = np.unique(keyarr, axis=0, return_index=True, return_inverse=True)
            inv = inv.ravel()
            V, Nn = V[first], Nn[first]
            idx = inv.astype(np.uint16 if len(V) < 65536 else np.uint32)
            mkey = (col, met, rough, alpha)
            if mkey not in mats:
                m = {"name": f"mat{len(mats)}", "pbrMetallicRoughness": {
                    "baseColorFactor": [float(c) for c in hex01(col)] + [alpha],
                    "metallicFactor": met, "roughnessFactor": rough}}
                if alpha < 1:
                    m["alphaMode"] = "BLEND"
                js["materials"].append(m)
                mats[mkey] = len(js["materials"]) - 1
            pa = add(V, 34962, 5126, "VEC3", True)
            na = add(Nn, 34962, 5126, "VEC3")
            ia = add(idx, 34963, 5123 if idx.dtype == np.uint16 else 5125, "SCALAR")
            prims.append({"attributes": {"POSITION": pa, "NORMAL": na}, "indices": ia, "material": mats[mkey], "mode": 4})
        js["meshes"].append({"name": p["name"], "primitives": prims})
        extras = {"label": p["label"]}
        if p.get("check"):
            extras["check"] = p["check"]
        if p.get("explode") is not None:
            extras["explode"] = [float(v) for v in p["explode"]]
        js["nodes"].append({"name": p["name"], "mesh": len(js["meshes"]) - 1, "extras": extras})
        js["scenes"][0]["nodes"].append(len(js["nodes"]) - 1)
    js["buffers"].append({"byteLength": len(blob)})
    jb = json.dumps(js, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    while len(jb) % 4:
        jb += b" "
    total = 12 + 8 + len(jb) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(jb), 0x4E4F534A))
        f.write(jb)
        f.write(struct.pack("<II", len(blob), 0x004E4942))
        f.write(bytes(blob))
    return len(spec["parts"]), ntri


# ---------------------------------------------------------------------------------------------- JSON spec (on-device build)
def resolve_shape(s: dict) -> dict:
    """JSON-ready copy of a shape: a tube radius given as a function becomes one radius per spline sample."""
    out = dict(s)
    if s["type"] == "tube" and callable(s["r"]):
        n = len(catmull(s["points"], int(s.get("samples", 10)), s.get("closed", False))) if s.get("smooth", True) else len(s["points"])
        out["r"] = [float(s["r"](t)) for t in np.linspace(0, 1, n)]
    return out


def spec_json(spec: dict) -> dict:
    """The model as primitives, for the app's on-device builder (ModelBuilder.kt): same shapes, same order."""
    parts = []
    for p in spec["parts"]:
        q = {k: p[k] for k in ("name", "label", "color", "metallic", "roughness", "alpha")}
        if p.get("check"):
            q["check"] = p["check"]
        if p.get("explode") is not None:
            q["explode"] = [float(v) for v in p["explode"]]
        q["shapes"] = [resolve_shape(s) for s in p["shapes"]]
        parts.append(q)
    return {"schema_version": 1, **spec["extras"], "parts": parts}


def part_stats(p: dict) -> dict:
    """Geometry fingerprint of one part (triangle soup before normals/welding) for the Kotlin parity test."""
    S = np.concatenate([shape_soup(s) for s in p["shapes"]])
    cr = np.cross(S[:, 1] - S[:, 0], S[:, 2] - S[:, 0])
    V = S.reshape(-1, 3)
    return {"triangles": int(len(S)), "area": float(0.5 * np.linalg.norm(cr, axis=1).sum()),
            "volume": float(np.einsum("ij,ij->i", S[:, 0], np.cross(S[:, 1], S[:, 2])).sum() / 6),
            "min": V.min(0).tolist(), "max": V.max(0).tolist()}
