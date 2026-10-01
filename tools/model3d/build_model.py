#!/usr/bin/env python3
"""Build a clean, named-part 3D model (GLB) from a JSON spec of primitives.

usage: python build_model.py model_spec.json out.glb [--preview preview.png]

Use when the user has no CAD/GLB file: compose the part from primitives that match the
reference photo / description. Every part becomes a named glTF node (so the video can
highlight, explode and label it) with a PBR material.

Spec (units are free, e.g. mm; Y is up):
{
  "parts": [
    {"name": "housing", "color": "#1E2023", "metallic": 0.0, "roughness": 0.55,
     "explode": [0, 0, -12],                       # optional exploded-view offset (same units)
     "shapes": [
       {"type": "rbox", "size": [20, 26, 18], "radius": 3, "pos": [0, 6, -13]},
       {"type": "cylinder", "r": 12, "h": 8, "axis": "z", "pos": [0, 0, 0], "segments": 72},
       {"type": "cylinder", "r": 6.2, "hole": 4.5, "h": 5, "axis": "z"},          # tube / bushing
       {"type": "cylinder", "r": 9, "r2": 7, "h": 10},                             # cone / frustum
       {"type": "lathe", "profile": [[0, 0], [8, 0], [8.5, 0.5], [8.5, 30], [0, 30]], "axis": "z"},
       {"type": "torus", "R": 9.6, "r": 1.3, "axis": "z", "pos": [0, 0, 8.5]},
       {"type": "sphere", "r": 3},
       {"type": "box", "size": [6, 1.2, 2.4], "rot": [0, 0, 15]},
       {"type": "extrude", "hull_circles": [[0, 0, 12], [27, 0, 9]], "holes": [[27, 0, 4.5]],
        "h": 4, "axis": "z", "pos": [0, 0, -2]}                                     # flange with bolt hole
     ]}
  ]
}
Shape keys: pos [x,y,z], rot [rx,ry,rz] degrees (applied after "axis"), scale [sx,sy,sz],
segments (round shapes), and per-shape "color"/"metallic"/"roughness" overrides.
  rbox:     rounded box — "radius" = fillet radius (real molded-fillet look). box = radius 0.
  cylinder: r, h, optional r2 (top radius), hole (inner radius -> tube), chamfer (edge chamfer).
  lathe:    profile [[r, y], ...] revolved about the axis; open ends are capped; "closed": true for a loop.
  torus:    R (ring radius), r (tube radius).   sphere: r.
  extrude:  "points" [[x, z], ...] polygon OR "hull_circles" [[x, z, r], ...] (convex hull of circles),
            optional "holes" [[x, z, r], ...]; thickness h along the axis (default y).
"""
import argparse, json, math, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model3d import rot_xyz, corner_normals, hex01  # noqa: E402


# ------------------------------------------------------------------ primitives -> triangle soup (m,3,3)
def _fix_volume(T):
    vol = np.einsum('ij,ij->i', T[:, 0], np.cross(T[:, 1], T[:, 2])).sum()
    return T[:, [0, 2, 1]] if vol < 0 else T


def revolve(profile, seg=64, closed=False):
    prof = [list(map(float, p)) for p in profile]
    if not closed:
        if prof[0][0] > 1e-9:
            prof.insert(0, [0.0, prof[0][1]])
        if prof[-1][0] > 1e-9:
            prof.append([0.0, prof[-1][1]])
    pr = np.array(prof)
    n = len(pr)
    th = np.linspace(0, 2 * np.pi, seg + 1)[:-1]
    V = np.zeros((n, seg, 3))
    V[:, :, 0] = pr[:, 0:1] * np.cos(th)[None]
    V[:, :, 1] = pr[:, 1:2]
    V[:, :, 2] = pr[:, 0:1] * np.sin(th)[None]
    tris = []
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        i2 = (i + 1) % n
        for j in range(seg):
            j2 = (j + 1) % seg
            a, b, c, d = V[i, j], V[i, j2], V[i2, j2], V[i2, j]
            tris.append([a, c, b]); tris.append([a, d, c])
    T = np.array(tris)
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    return _fix_volume(T[area > 1e-12])


def rbox(size, radius=0.0, seg=None):
    """Box with true rounded (filleted) edges. Grid is dense only inside the fillet bands."""
    h = np.array([s / 2 for s in size], float)
    r = float(min(radius, *h) if radius else 0)
    k = int(seg or 7)

    def samples(hh):
        if r <= 0:
            return np.array([-hh, hh])
        f = np.linspace(0, 1, k + 1)
        a = np.concatenate([-hh + r * f, (hh - r) + r * f])
        return np.unique(np.round(a, 12))
    S = [samples(v) for v in h]
    tris = []
    for ax in range(3):
        u, v = (ax + 1) % 3, (ax + 2) % 3
        gu, gv = S[u], S[v]
        for sgn in (-1, 1):
            U, Vv = np.meshgrid(gu, gv, indexing='ij')
            G = np.zeros(U.shape + (3,)); G[..., ax] = sgn * h[ax]; G[..., u] = U; G[..., v] = Vv
            a, b, c, d = G[:-1, :-1], G[1:, :-1], G[1:, 1:], G[:-1, 1:]
            tris.append(np.stack([a, b, c], -2).reshape(-1, 3, 3)); tris.append(np.stack([a, c, d], -2).reshape(-1, 3, 3))
    T = np.concatenate(tris)
    if r > 0:
        inner = h - r
        Q = np.clip(T, -inner, inner)
        dd = T - Q
        L = np.linalg.norm(dd, axis=2, keepdims=True)
        T = np.where(L > 1e-12, Q + dd / np.maximum(L, 1e-12) * r, T)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    c = T.mean(1)
    bad = (n * c).sum(1) < 0
    T[bad] = T[bad][:, [0, 2, 1]]
    return T[np.linalg.norm(n, axis=1) > 1e-12]


def cylinder(r, h, r2=None, hole=None, chamfer=0.0, seg=64):
    r2 = r if r2 is None else r2
    y0, y1 = -h / 2, h / 2
    c = float(chamfer or 0)
    if hole:
        prof = [[hole, y0], [r - c, y0], [r, y0 + c], [r2, y1 - c], [r2 - c, y1], [hole, y1]]
        prof = [p for k, p in enumerate(prof) if k == 0 or p != prof[k - 1]]
        return revolve(prof, seg, closed=True)
    prof = [[0, y0], [r - c, y0], [r, y0 + c], [r2, y1 - c], [r2 - c, y1], [0, y1]]
    prof = [p for k, p in enumerate(prof) if k == 0 or p != prof[k - 1]]
    return revolve(prof, seg)


def torus(R, r, seg=72, seg2=24):
    a = np.linspace(0, 2 * np.pi, seg2 + 1)[:-1]
    prof = [[R + r * math.cos(t), r * math.sin(t)] for t in a]
    return revolve(prof, seg, closed=True)


def sphere(r, seg=48):
    a = np.linspace(-np.pi / 2, np.pi / 2, seg // 2 + 1)
    return revolve([[r * math.cos(t), r * math.sin(t)] for t in a], seg)


def _hull(pts):
    pts = sorted(set(map(tuple, np.round(pts, 9))))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])        # CCW


def _signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * (x * np.roll(y, -1) - np.roll(x, -1) * y).sum()


def _earclip(P):
    n = len(P)
    idx = list(range(n))
    tris = []

    def inside(p, a, b, c):
        d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
        d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
        d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
        e = 1e-12
        return (d1 < -e and d2 < -e and d3 < -e) or (d1 > e and d2 > e and d3 > e)
    guard = 0
    while len(idx) > 3 and guard < 100000:
        guard += 1
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = P[i0], P[i1], P[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 1e-12:
                continue
            if any(inside(P[j], a, b, c) for j in idx if j not in (i0, i1, i2)
                   and not (np.allclose(P[j], a) or np.allclose(P[j], b) or np.allclose(P[j], c))):
                continue
            tris.append((i0, i1, i2)); idx.pop(k)
            break
        else:
            tris.append((idx[-1], idx[0], idx[1])); idx.pop(0)
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def extrude(h, points=None, hull_circles=None, holes=None, seg=48):
    if hull_circles:
        cp = []
        for x, z, r in hull_circles:
            a = np.linspace(0, 2 * np.pi, seg + 1)[:-1]
            cp += list(np.stack([x + r * np.cos(a), z + r * np.sin(a)], 1))
        outer = _hull(np.array(cp))
    else:
        outer = np.array(points, float)
    if _signed_area(outer) < 0:
        outer = outer[::-1]
    loops = [outer]
    hl = []
    for x, z, r in (holes or []):
        a = np.linspace(0, 2 * np.pi, seg + 1)[:-1][::-1]           # CW
        hl.append(np.stack([x + r * np.cos(a), z + r * np.sin(a)], 1))
    loops += hl
    # bridge holes into the outer loop, then ear-clip
    poly = outer.copy()
    for H in sorted(hl, key=lambda q: -q[:, 0].max()):
        j = int(np.argmax(H[:, 0]))
        i = int(np.argmin(np.linalg.norm(poly - H[j], axis=1)))
        Hr = np.concatenate([H[j:], H[:j + 1]])
        poly = np.concatenate([poly[:i + 1], Hr, poly[i:]])
    cap = _earclip(poly)
    y0, y1 = -h / 2, h / 2
    to3 = lambda p, y: [p[0], y, p[1]]  # noqa: E731
    tris = []
    for a, b, c in cap:
        tris.append([to3(poly[a], y1), to3(poly[b], y1), to3(poly[c], y1), [0, 1, 0]])
        tris.append([to3(poly[a], y0), to3(poly[b], y0), to3(poly[c], y0), [0, -1, 0]])
    for L in loops:
        for k in range(len(L)):
            p, q = L[k], L[(k + 1) % len(L)]
            d = q - p
            out = [d[1], 0, -d[0]]
            tris.append([to3(p, y0), to3(q, y0), to3(q, y1), out])
            tris.append([to3(p, y0), to3(q, y1), to3(p, y1), out])
    T = np.array([t[:3] for t in tris], float)
    tgt = np.array([t[3] for t in tris], float)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    bad = (n * tgt).sum(1) < 0
    T[bad] = T[bad][:, [0, 2, 1]]
    return T[np.linalg.norm(n, axis=1) > 1e-12]


AXIS = {'y': np.eye(3), 'x': rot_xyz([0, 0, -90]), 'z': rot_xyz([90, 0, 0])}


def shape_soup(s):
    t = s['type']
    seg = int(s.get('segments', 64))
    if t in ('rbox', 'box'):
        T = rbox(s['size'], s.get('radius', 0) if t == 'rbox' else 0, s.get('segments'))
    elif t == 'cylinder':
        T = cylinder(s['r'], s['h'], s.get('r2'), s.get('hole'), s.get('chamfer', 0), seg)
    elif t == 'lathe':
        T = revolve(s['profile'], seg, s.get('closed', False))
    elif t == 'torus':
        T = torus(s['R'], s['r'], seg, int(s.get('segments2', 24)))
    elif t == 'sphere':
        T = sphere(s['r'], seg)
    elif t == 'extrude':
        T = extrude(s['h'], s.get('points'), s.get('hull_circles'), s.get('holes'), seg)
    else:
        raise SystemExit(f'unknown shape type {t}')
    M = AXIS[s.get('axis', 'y')]
    if s.get('scale'):
        Sm = np.diag(s['scale']); T = T @ Sm.T
        if np.linalg.det(Sm) < 0:
            T = T[:, [0, 2, 1]]
    T = T @ M.T
    if s.get('rot'):
        T = T @ rot_xyz(s['rot']).T
    return T + np.array(s.get('pos', [0, 0, 0]), float)


# ------------------------------------------------------------------ GLB writer
def write_glb(path, parts):
    """parts: list of dict(name, explode, prims=[dict(T soup, color, metallic, roughness)])"""
    js = {'asset': {'version': '2.0', 'generator': 'product-video build_model.py', 'extras': {'pv_built': True}},
          'scene': 0, 'scenes': [{'nodes': []}], 'nodes': [], 'meshes': [], 'materials': [],
          'accessors': [], 'bufferViews': [], 'buffers': []}
    blob = bytearray()

    def add(arr, target, typ, minmax=False):
        a = np.ascontiguousarray(arr, np.float32)
        off = len(blob); blob.extend(a.tobytes())
        while len(blob) % 4:
            blob.append(0)
        js['bufferViews'].append({'buffer': 0, 'byteOffset': off, 'byteLength': a.nbytes, 'target': target})
        acc = {'bufferView': len(js['bufferViews']) - 1, 'componentType': 5126, 'count': len(a), 'type': typ}
        if minmax:
            acc['min'] = a.min(0).tolist(); acc['max'] = a.max(0).tolist()
        js['accessors'].append(acc)
        return len(js['accessors']) - 1
    mats = {}
    for p in parts:
        prims = []
        for pr in p['prims']:
            key = (tuple(np.round(pr['color'], 4)), pr['metallic'], pr['roughness'])
            if key not in mats:
                js['materials'].append({'name': f'{p["name"]}_mat{len(mats)}', 'pbrMetallicRoughness': {
                    'baseColorFactor': list(map(float, pr['color'])) + [1.0],
                    'metallicFactor': float(pr['metallic']), 'roughnessFactor': float(pr['roughness'])}})
                mats[key] = len(js['materials']) - 1
            T = pr['T']
            N = corner_normals(T)
            pa = add(T.reshape(-1, 3), 34962, 'VEC3', True)
            na = add(N.reshape(-1, 3), 34962, 'VEC3')
            prims.append({'attributes': {'POSITION': pa, 'NORMAL': na}, 'material': mats[key], 'mode': 4})
        js['meshes'].append({'name': p['name'], 'primitives': prims})
        node = {'name': p['name'], 'mesh': len(js['meshes']) - 1}
        if p.get('explode') is not None:
            node['extras'] = {'explode': [float(v) for v in p['explode']]}
        js['nodes'].append(node)
        js['scenes'][0]['nodes'].append(len(js['nodes']) - 1)
    js['buffers'].append({'byteLength': len(blob)})
    jb = json.dumps(js, separators=(',', ':')).encode()
    while len(jb) % 4:
        jb += b' '
    total = 12 + 8 + len(jb) + 8 + len(blob)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, total))
        f.write(struct.pack('<II', len(jb), 0x4E4F534A)); f.write(jb)
        f.write(struct.pack('<II', len(blob), 0x004E4942)); f.write(bytes(blob))


def build(spec):
    parts = []
    for p in spec['parts']:
        prims = {}
        for s in p['shapes']:
            col = s.get('color', p.get('color', '#8A8F96'))
            key = (col, float(s.get('metallic', p.get('metallic', 0.0))), float(s.get('roughness', p.get('roughness', 0.5))))
            prims.setdefault(key, []).append(shape_soup(s))
        parts.append({'name': p['name'], 'explode': p.get('explode'),
                      'prims': [{'T': np.concatenate(v), 'color': hex01(k[0]), 'metallic': k[1], 'roughness': k[2]}
                                for k, v in prims.items()]})
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spec'); ap.add_argument('out')
    ap.add_argument('--preview', help='also render a quick PNG preview')
    a = ap.parse_args()
    spec = json.load(open(a.spec, encoding='utf-8'))
    parts = build(spec)
    write_glb(a.out, parts)
    ntri = sum(len(pr['T']) for p in parts for pr in p['prims'])
    print(f'wrote {a.out}: {len(parts)} parts, {ntri} triangles')
    if a.preview:
        from inspect3d import views_sheet
        from model3d import Model3D
        views_sheet(Model3D(a.out, spec.get('view_opts', {})), a.preview, view=spec.get('view', [35, 20]))
        print('preview ->', a.preview)


if __name__ == '__main__':
    main()
