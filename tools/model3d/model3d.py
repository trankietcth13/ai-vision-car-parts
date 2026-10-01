"""3D model support for product-video / pro-product-video.

Pure numpy + OpenCV (no trimesh / OpenGL / numba needed), so it runs wherever the
video pipeline runs.

  * Loaders: .glb / .gltf (embedded or external .bin, base-colour textures -> vertex colours),
    .obj (+ .mtl Kd colours, o/g parts), .stl (binary / ascii).
  * Model3D: normalises the model (centre 0, bounding radius 1, Y up), crease-angle smooth
    normals, per-part materials, exploded-view offsets.
  * Software renderer: painter's-order triangle-ID buffer (cv2.fillConvexPoly) + vectorised
    deferred shading — studio key/fill/rim, world-fixed softbox reflections (highlights slide
    as the camera orbits), Fresnel, optional feature-line outlines, part highlight / dimming,
    distance-from-origin buffer for the reveal / pulse effects. Supersampled anti-aliasing.
  * Camera: orbit (yaw, pitch, zoom, target) + screen offset; keyframes for the video timeline.

Model coordinates used in storyboards ("point3d") are NORMALISED coordinates: model centred on
its bounding-box centre, scaled to bounding radius 1, Y up. Get them with inspect3d.py --pick.
"""
import base64, json, math, os, struct, urllib.parse
import numpy as np
import cv2

FOV = math.radians(30.0)
_TAN = math.tan(FOV / 2)
VIEW_R = 1.25            # half-height (model units) visible at the target when zoom == 1


# =============================================================== small math helpers
def rot_xyz(deg):
    rx, ry, rz = [math.radians(float(a)) for a in deg]
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def hex01(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def _norm(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2


# =============================================================== loaders
# each loader returns a list of raw parts: dict(name, V (n,3), F (m,3), C (n,3) 0-1 or None,
#                                                color (3,) 0-1 sRGB, metallic, roughness, explode (3,) or None)
_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}


def load_gltf(path):
    data = open(path, 'rb').read()
    js, binchunk = None, None
    if data[:4] == b'glTF':
        length = struct.unpack('<I', data[8:12])[0]
        off = 12
        while off < length:
            clen, ctype = struct.unpack('<II', data[off:off + 8])
            chunk = data[off + 8:off + 8 + clen]
            if ctype == 0x4E4F534A:
                js = json.loads(chunk.decode('utf-8'))
            elif ctype == 0x004E4942:
                binchunk = chunk
            off += 8 + clen
    else:
        js = json.loads(data.decode('utf-8'))
    for ext in js.get('extensionsRequired', []):
        if ext in ('KHR_draco_mesh_compression', 'EXT_meshopt_compression', 'KHR_mesh_quantization'):
            raise SystemExit(f'{os.path.basename(path)} uses {ext}; re-export the glTF/GLB without compression.')
    buffers = []
    for b in js.get('buffers', []):
        uri = b.get('uri')
        if uri is None:
            buffers.append(binchunk)
        elif uri.startswith('data:'):
            buffers.append(base64.b64decode(uri.split(',', 1)[1]))
        else:
            buffers.append(open(os.path.join(os.path.dirname(path), urllib.parse.unquote(uri)), 'rb').read())

    def view(vi):
        bv = js['bufferViews'][vi]
        o = bv.get('byteOffset', 0)
        return buffers[bv['buffer']][o:o + bv['byteLength']], bv.get('byteStride')

    def acc(ai):
        a = js['accessors'][ai]
        dt = np.dtype(_CT[a['componentType']]).newbyteorder('<'); n = _NC[a['type']]; cnt = a['count']
        if 'bufferView' not in a:
            return np.zeros((cnt, n), np.float64)
        raw, stride = view(a['bufferView']); off = a.get('byteOffset', 0)
        if stride and stride != dt.itemsize * n:
            arr = np.ndarray((cnt, n), dt, buffer=raw, offset=off, strides=(stride, dt.itemsize))
        else:
            arr = np.frombuffer(raw, dt, cnt * n, off).reshape(cnt, n)
        arr = np.array(arr)
        if a.get('normalized') and dt.kind in 'iu':
            arr = arr.astype(np.float64) / np.iinfo(dt).max
        return arr

    def image(ti):
        try:
            src = js['textures'][ti]['source']; im = js['images'][src]
            if 'bufferView' in im:
                raw, _ = view(im['bufferView'])
            elif im.get('uri', '').startswith('data:'):
                raw = base64.b64decode(im['uri'].split(',', 1)[1])
            else:
                raw = open(os.path.join(os.path.dirname(path), urllib.parse.unquote(im['uri'])), 'rb').read()
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            return None if img is None else img[..., ::-1].astype(np.float32) / 255
        except Exception:
            return None

    parts = []

    def walk(ni, parent):
        node = js['nodes'][ni]
        if 'matrix' in node:
            M = np.array(node['matrix'], float).reshape(4, 4).T
        else:
            M = np.eye(4)
            S = np.diag(list(node.get('scale', [1, 1, 1])) + [1])
            R = np.eye(4); R[:3, :3] = quat_mat(node.get('rotation', [0, 0, 0, 1]))
            T = np.eye(4); T[:3, 3] = node.get('translation', [0, 0, 0])
            M = T @ R @ S
        W = parent @ M
        if 'mesh' in node:
            mesh = js['meshes'][node['mesh']]
            name = node.get('name') or mesh.get('name') or f'part{len(parts) + 1}'
            ex = (node.get('extras') or {}).get('explode')
            for pr in mesh['primitives']:
                if pr.get('mode', 4) != 4:
                    continue
                V = acc(pr['attributes']['POSITION']).astype(np.float64)
                V = V @ W[:3, :3].T + W[:3, 3]
                F = acc(pr['indices']).reshape(-1, 3).astype(np.int64) if 'indices' in pr \
                    else np.arange(len(V)).reshape(-1, 3)
                if np.linalg.det(W[:3, :3]) < 0:
                    F = F[:, [0, 2, 1]]
                mat = js['materials'][pr['material']] if 'material' in pr else {}
                pbr = mat.get('pbrMetallicRoughness', {})
                bc = np.array(pbr.get('baseColorFactor', [0.8, 0.8, 0.8, 1]), np.float32)[:3]
                C = None
                if 'COLOR_0' in pr['attributes']:
                    C = acc(pr['attributes']['COLOR_0'])[:, :3].astype(np.float32) * bc
                tex = pbr.get('baseColorTexture')
                if tex is not None and 'TEXCOORD_0' in pr['attributes']:
                    img = image(tex['index'])
                    if img is not None:
                        uv = acc(pr['attributes']['TEXCOORD_0'])
                        h, w = img.shape[:2]
                        u = (np.mod(uv[:, 0], 1) * (w - 1)).astype(int); v = (np.mod(uv[:, 1], 1) * (h - 1)).astype(int)
                        C = img[v, u] * bc
                parts.append(dict(name=name, V=V, F=F, C=C, color=np.clip(bc, 0, 1),
                                  metallic=float(pbr.get('metallicFactor', 1.0 if pbr else 0.0)),
                                  roughness=float(pbr.get('roughnessFactor', 1.0 if pbr else 0.6)),
                                  explode=None if ex is None else np.array(ex, float),
                                  built=bool((js.get('asset', {}).get('extras') or {}).get('pv_built'))))
        for c in node.get('children', []):
            walk(c, W)

    sc = js.get('scenes', [{'nodes': list(range(len(js.get('nodes', []))))}])[js.get('scene', 0)]
    for ni in sc.get('nodes', []):
        walk(ni, np.eye(4))
    return parts


def load_obj(path):
    mats, cur_mat = {}, None
    vs, parts = [], {}
    order = []
    name = 'body'
    d = os.path.dirname(path)
    for line in open(path, encoding='utf-8', errors='ignore'):
        t = line.split()
        if not t:
            continue
        if t[0] == 'v':
            vs.append([float(x) for x in t[1:4]])
        elif t[0] in ('o', 'g') and len(t) > 1:
            name = ' '.join(t[1:])
        elif t[0] == 'usemtl':
            cur_mat = ' '.join(t[1:])
        elif t[0] == 'mtllib':
            p = os.path.join(d, ' '.join(t[1:]))
            if os.path.exists(p):
                m = None
                for l2 in open(p, encoding='utf-8', errors='ignore'):
                    s = l2.split()
                    if not s:
                        continue
                    if s[0] == 'newmtl':
                        m = ' '.join(s[1:]); mats[m] = {'Kd': [0.7, 0.7, 0.7], 'Pm': 0.0, 'Pr': 0.6}
                    elif m and s[0] == 'Kd':
                        mats[m]['Kd'] = [float(x) for x in s[1:4]]
                    elif m and s[0] == 'Pm':
                        mats[m]['Pm'] = float(s[1])
                    elif m and s[0] == 'Pr':
                        mats[m]['Pr'] = float(s[1])
                    elif m and s[0] == 'Ns':
                        mats[m]['Pr'] = float(np.clip(1 - math.sqrt(float(s[1]) / 1000), 0.05, 1))
        elif t[0] == 'f':
            idx = []
            for tok in t[1:]:
                i = int(tok.split('/')[0])
                idx.append(i - 1 if i > 0 else len(vs) + i)
            key = (name, cur_mat)
            if key not in parts:
                parts[key] = []; order.append(key)
            for k in range(1, len(idx) - 1):
                parts[key].append([idx[0], idx[k], idx[k + 1]])
    V = np.array(vs, float)
    out = []
    for key in order:
        F = np.array(parts[key], np.int64)
        used, inv = np.unique(F, return_inverse=True)
        m = mats.get(key[1], {'Kd': [0.7, 0.7, 0.7], 'Pm': 0.0, 'Pr': 0.6})
        out.append(dict(name=key[0], V=V[used], F=inv.reshape(-1, 3), C=None, color=np.array(m['Kd'], np.float32),
                        metallic=m['Pm'], roughness=m['Pr'], explode=None, built=False))
    return out


def load_stl(path):
    data = open(path, 'rb').read()
    if len(data) >= 84:
        n = struct.unpack('<I', data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]),
                                count=n, offset=84)
            V = rec['v'].reshape(-1, 3).astype(float)
            return [dict(name='body', V=V, F=np.arange(len(V)).reshape(-1, 3), C=None,
                         color=np.array([0.6, 0.62, 0.65], np.float32), metallic=0.0, roughness=0.5,
                         explode=None, built=False)]
    vs = [[float(x) for x in l.split()[1:4]] for l in data.decode('utf-8', 'ignore').splitlines()
          if l.strip().startswith('vertex')]
    V = np.array(vs, float)
    return [dict(name='body', V=V, F=np.arange(len(V)).reshape(-1, 3), C=None,
                 color=np.array([0.6, 0.62, 0.65], np.float32), metallic=0.0, roughness=0.5, explode=None, built=False)]


def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.glb', '.gltf'):
        return load_gltf(path)
    if ext == '.obj':
        return load_obj(path)
    if ext == '.stl':
        return load_stl(path)
    raise SystemExit(f'Unsupported 3D format {ext}. Use .glb/.gltf (preferred), .obj or .stl '
                     '(convert STEP/IGES/FBX/USDZ to GLB first, e.g. in Blender or a CAD exporter).')


# =============================================================== geometry processing
def corner_normals(P, crease_deg=35.0):
    """P: (m,3,3) triangle soup. Smooth normals per corner, not smoothing across edges sharper than crease."""
    m = len(P)
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    fu = _norm(fn)
    flat = P.reshape(-1, 3)
    scale = 1e5 / (np.abs(flat).max() + 1e-9)
    _, inv = np.unique(np.round(flat * scale).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    face = np.repeat(np.arange(m), 3)
    order = np.argsort(inv, kind='stable')
    vs, fs = inv[order], face[order]
    starts = np.flatnonzero(np.r_[True, vs[1:] != vs[:-1]])
    counts = np.diff(np.r_[starts, len(vs)])
    grp = np.repeat(np.arange(len(starts)), counts)
    cnt = counts[grp]
    if cnt.astype(np.int64).sum() > 40_000_000:      # pathological fans: fall back to flat shading
        return np.repeat(fu[:, None, :], 3, 1).astype(np.float32)
    ii = np.repeat(np.arange(len(vs)), cnt)
    off = np.arange(len(ii)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    jj = starts[grp[ii]] + off
    cf, nf = fs[ii], fs[jj]
    ok = (fu[cf] * fu[nf]).sum(1) > math.cos(math.radians(crease_deg))
    acc = np.zeros((len(vs), 3))
    for k in range(3):
        acc[:, k] = np.bincount(ii[ok], weights=fn[nf[ok], k], minlength=len(vs))
    out = np.zeros((m * 3, 3))
    out[order] = acc
    out = _norm(out.reshape(m, 3, 3))
    bad = np.linalg.norm(out, axis=2) < 0.5
    out[bad] = np.repeat(fu[:, None, :], 3, 1)[bad]
    return out.astype(np.float32)


def subdivide(arrs, maxlen, max_tris=400_000):
    """1->4 split of triangles whose longest edge > maxlen. arrs: dict of per-corner (m,3,k) and per-face (m,) arrays;
    'P' must be present."""
    for _ in range(6):
        P = arrs['P']
        if len(P) > max_tris:
            break
        e = np.max(np.stack([np.linalg.norm(P[:, 1] - P[:, 0], axis=1), np.linalg.norm(P[:, 2] - P[:, 1], axis=1),
                             np.linalg.norm(P[:, 0] - P[:, 2], axis=1)], 1), 1)
        big = e > maxlen
        if not big.any() or len(P) + 3 * big.sum() > max_tris:
            break
        new = {}
        for k, A in arrs.items():
            keep, S = A[~big], A[big]
            if A.ndim == 1:
                new[k] = np.concatenate([keep, np.repeat(S, 4)])
                continue
            a, b, c = S[:, 0], S[:, 1], S[:, 2]
            ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
            if k == 'N':
                ab, bc, ca = _norm(ab), _norm(bc), _norm(ca)
            T = np.stack([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                          np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)], 1).reshape(-1, 3, A.shape[2])
            new[k] = np.concatenate([keep, T.astype(A.dtype)])
        arrs = new
    return arrs


def srgb2lin(c):
    return np.power(np.clip(c, 0, 1), 2.2)


# =============================================================== model
def zbuffer(sx, sy, iz, idx, Ws, Hs):
    """Vectorised scan-line z-buffer. sx, sy, iz (=1/z): (F,3); idx: candidate faces.
    Every covered pixel (centre inside the triangle) becomes one fragment; the fragment with the
    largest 1/z (nearest) wins via a packed uint64 maximum. Returns (Hs, Ws) int32 face ids, -1 empty."""
    out = np.zeros(Ws * Hs, np.uint64)
    if len(idx) == 0:
        return np.full((Hs, Ws), -1, np.int32)
    X, Y, Z = sx[idx].astype(np.float64), sy[idx].astype(np.float64), iz[idx].astype(np.float64)
    ax, ay = X[:, 0], Y[:, 0]
    e1x, e1y, e2x, e2y = X[:, 1] - ax, Y[:, 1] - ay, X[:, 2] - ax, Y[:, 2] - ay
    den = e1x * e2y - e2x * e1y
    good = np.abs(den) > 1e-9
    den = np.where(good, den, 1.0)
    A1, B1, A2, B2 = e2y / den, -e2x / den, -e1y / den, e1x / den
    y0 = np.clip(np.ceil(Y.min(1) - 0.5), 0, Hs - 1).astype(np.int64)
    y1 = np.clip(np.floor(Y.max(1) - 0.5), -1, Hs - 1).astype(np.int64)
    nr = np.where(good, np.maximum(y1 - y0 + 1, 0), 0)
    R = int(nr.sum())
    if R == 0:
        return np.full((Hs, Ws), -1, np.int32)
    tr = np.repeat(np.arange(len(idx)), nr)
    yr = y0[tr] + (np.arange(R) - np.repeat(np.cumsum(nr) - nr, nr))
    dy = yr + 0.5 - ay[tr]
    lo = np.full(R, -np.inf); hi = np.full(R, np.inf)
    # constraints  a*dx + b >= 0  with dx = x + 0.5 - ax
    for a_, b_ in ((A1[tr], B1[tr] * dy), (A2[tr], B2[tr] * dy),
                   (-(A1[tr] + A2[tr]), 1 - (B1[tr] + B2[tr]) * dy)):
        b_ = b_ + 1e-7
        with np.errstate(divide='ignore', invalid='ignore'):
            r = -b_ / a_
        pos, neg = a_ > 1e-12, a_ < -1e-12
        lo = np.where(pos, np.maximum(lo, r), lo)
        hi = np.where(neg, np.minimum(hi, r), hi)
        dead = ~pos & ~neg & (b_ < 0)
        hi = np.where(dead, -np.inf, hi)
    xa = np.maximum(np.ceil(lo + ax[tr] - 0.5), 0)
    xb = np.minimum(np.floor(hi + ax[tr] - 0.5), Ws - 1)
    cnt = np.where(xb >= xa, xb - xa + 1, 0).astype(np.int64)
    N = int(cnt.sum())
    if N == 0:
        return np.full((Hs, Ws), -1, np.int32)
    xa = xa.astype(np.int64)
    dz1, dz2 = Z[:, 1] - Z[:, 0], Z[:, 2] - Z[:, 0]
    sl = A1 * dz1 + A2 * dz2                                    # d(1/z)/dx per triangle
    dxa = xa + 0.5 - ax[tr]
    z_row = Z[tr, 0] + (A1[tr] * dxa + B1[tr] * dy) * dz1[tr] + (A2[tr] * dxa + B2[tr] * dy) * dz2[tr]
    rid = np.repeat(np.arange(R), cnt)
    k = np.arange(N) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    px = xa[rid] + k
    zf = (z_row[rid] + sl[tr[rid]] * k).astype(np.float32)
    zf = np.maximum(zf, 1e-12)
    key = (zf.view(np.uint32).astype(np.uint64) << np.uint64(32)) | (idx[tr[rid]].astype(np.uint64) + np.uint64(1))
    np.maximum.at(out, yr[rid] * Ws + px, key)
    ids = (out & np.uint64(0xFFFFFFFF)).astype(np.int64) - 1
    return ids.astype(np.int32).reshape(Hs, Ws)


class Model3D:
    def __init__(self, path, opts=None):
        opts = dict(opts or {})
        self.path, self.opts = path, opts
        raw = load_any(path)
        if not raw:
            raise SystemExit(f'No triangle meshes found in {path}')
        ext = os.path.splitext(path)[1].lower()
        up = opts.get('up', 'z' if ext == '.stl' else 'y')
        R = np.eye(3)
        if up == 'z':
            R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
        if opts.get('rotate'):
            R = rot_xyz(opts['rotate']) @ R
        names, Ps, Ns, Cs, MET, RGH, PID, explode_raw = [], [], [], [], [], [], [], {}
        self.built = all(p.get('built') for p in raw)
        default_color = opts.get('color')
        for p in raw:
            if p['name'] not in names:
                names.append(p['name'])
            pid = names.index(p['name'])
            V = p['V'] @ R.T
            F = p['F']
            F = F[(F < len(V)).all(1)]
            Pp = V[F]
            vol = np.einsum('ij,ij->i', Pp[:, 0], np.cross(Pp[:, 1], Pp[:, 2])).sum()
            if vol < 0 and abs(vol) > 1e-9:          # inward-wound closed mesh -> flip
                Pp = Pp[:, [0, 2, 1]]
            if p['C'] is not None:
                C = p['C'][F]
            else:
                col = hex01(default_color) if (default_color and ext == '.stl') else p['color']
                C = np.broadcast_to(col, Pp.shape).copy()
            Ps.append(Pp); Ns.append(corner_normals(Pp, opts.get('crease', 35))); Cs.append(C)
            MET.append(np.full(len(Pp), p['metallic'], np.float32)); RGH.append(np.full(len(Pp), p['roughness'], np.float32))
            PID.append(np.full(len(Pp), pid, np.int32))
            if p.get('explode') is not None:
                explode_raw[pid] = R @ p['explode']
        P = np.concatenate(Ps)
        lo, hi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
        self.center = (lo + hi) / 2
        self.scale = 1.0 / (np.linalg.norm(P.reshape(-1, 3) - self.center, axis=1).max() + 1e-12)
        self.R = R
        P = (P - self.center) * self.scale
        arrs = dict(P=P.astype(np.float32), N=np.concatenate(Ns).astype(np.float32),
                    C=srgb2lin(np.concatenate(Cs)).astype(np.float32),
                    MET=np.concatenate(MET), RGH=np.concatenate(RGH), PID=np.concatenate(PID))
        arrs = subdivide(arrs, float(opts.get('max_edge', 0.12)))
        self.P, self.N, self.C = arrs['P'], arrs['N'], arrs['C']
        self.MET, self.RGH, self.PID = arrs['MET'], np.clip(arrs['RGH'], 0.04, 1), arrs['PID']
        self.names = names
        self.npart = len(names)
        # per-part centre & exploded-view offsets (normalised units)
        self.pcenter = np.zeros((self.npart, 3), np.float32)
        for i in range(self.npart):
            q = self.P[self.PID == i].reshape(-1, 3)
            self.pcenter[i] = (q.min(0) + q.max(0)) / 2 if len(q) else 0
        es = float(opts.get('explode_scale', 1.0))
        user = {k: np.array(v, float) for k, v in (opts.get('explode') or {}).items()}
        self.OFF = np.zeros((self.npart, 3), np.float32)
        for i, n in enumerate(names):
            if n in user:
                self.OFF[i] = R @ user[n] * self.scale * es
            elif i in explode_raw:
                self.OFF[i] = explode_raw[i] * self.scale * es
            else:
                d = self.pcenter[i]
                L = np.linalg.norm(d)
                self.OFF[i] = 0 if L < 0.12 else d / L * 0.45 * es
        self.cull = bool(opts.get('cull', self.built))
        self.v_all = self.P.reshape(-1, 3)

    # ---------------------------------------------------------------- points
    def part_index(self, name):
        if name is None:
            return -1
        if name in self.names:
            return self.names.index(name)
        low = [n.lower() for n in self.names]
        if name.lower() in low:
            return low.index(name.lower())
        raise SystemExit(f'3D part "{name}" not found. Parts: {", ".join(self.names)}')

    def point(self, spec):
        """Storyboard callout/label -> [x, y, z, part_index] in normalised model coords."""
        pid = self.part_index(spec.get('part'))
        if spec.get('point3d') is not None:
            p = np.array(spec['point3d'], float)
        elif pid >= 0:
            p = self.pcenter[pid]
        else:
            raise SystemExit(f'3D mode: give "point3d" (from inspect3d.py --pick) or "part" for {spec}')
        return [float(p[0]), float(p[1]), float(p[2]), int(pid)]

    # ---------------------------------------------------------------- camera
    @staticmethod
    def basis(cam):
        yaw, pitch = math.radians(cam['yaw']), math.radians(max(-80, min(80, cam['pitch'])))
        d = VIEW_R / max(cam['zoom'], 1e-3) / _TAN
        dirv = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
        T = np.array(cam.get('target', (0, 0, 0)), float)
        eye = T + d * dirv
        fwd = -dirv
        right = _norm(np.cross(fwd, [0, 1, 0]))
        upv = np.cross(right, fwd)
        return eye, right, upv, fwd

    def project(self, p, cam, W, H):
        eye, right, upv, fwd = self.basis(cam)
        q = np.array(p[:3], float)
        pid = int(p[3]) if len(p) > 3 else -1
        if pid >= 0:
            q = q + self.OFF[pid] * cam.get('explode', 0.0)
        v = q - eye
        z = v @ fwd
        if z <= 1e-3:
            return (-1e4, -1e4)
        f = (H / 2) / _TAN
        return (W / 2 + cam.get('ox', 0) * W + f * (v @ right) / z, H / 2 + cam.get('oy', 0) * H - f * (v @ upv) / z)

    # ---------------------------------------------------------------- render
    def _raster(self, cam, Ws, Hs):
        eye, right, upv, fwd = self.basis(cam)
        ex = float(cam.get('explode', 0.0))
        P = self.P + (self.OFF[self.PID][:, None, :] * ex if ex else 0)
        Q = P - eye.astype(np.float32)
        z = Q @ fwd.astype(np.float32)
        x = Q @ right.astype(np.float32)
        y = Q @ upv.astype(np.float32)
        f = (Hs / 2) / _TAN
        ok = (z > 0.02).all(1)
        zs = np.where(z > 0.02, z, 0.02)
        sx = Ws / 2 + cam.get('ox', 0) * Ws + f * x / zs
        sy = Hs / 2 + cam.get('oy', 0) * Hs - f * y / zs
        ok &= (sx.max(1) >= 0) & (sx.min(1) < Ws) & (sy.max(1) >= 0) & (sy.min(1) < Hs)
        area = (sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0])
        ok &= np.abs(area) > 1e-4
        if self.cull:
            ok &= area < 0          # counter-clockwise in world -> negative here because screen y points down
        idx = np.flatnonzero(ok)
        ids = zbuffer(sx, sy, 1.0 / zs, idx, Ws, Hs)
        return ids, P, sx, sy, z, (eye, right, upv, fwd)

    def ids(self, cam, W, H):
        return self._raster(cam, W, H)[0]

    def _interp(self, ids, P, sx, sy, z):
        ys, xs = np.nonzero(ids >= 0)
        t = ids[ys, xs]
        ax, ay = sx[t, 0], sy[t, 0]
        v0x, v0y = sx[t, 1] - ax, sy[t, 1] - ay
        v1x, v1y = sx[t, 2] - ax, sy[t, 2] - ay
        v2x, v2y = xs + 0.5 - ax, ys + 0.5 - ay
        den = v0x * v1y - v1x * v0y
        den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        l1 = (v2x * v1y - v1x * v2y) / den
        l2 = (v0x * v2y - v2x * v0y) / den
        L = np.clip(np.stack([1 - l1 - l2, l1, l2], 1), 0, 1)
        L = L / (L.sum(1, keepdims=True) + 1e-9)
        w = L / z[t]
        B = (w / w.sum(1, keepdims=True)).astype(np.float32)
        return ys, xs, t, B

    def pick(self, cam, W, H, x, y, search=10):
        ids, P, sx, sy, z, _ = self._raster(cam, W, H)
        hit = None
        for r in range(search + 1):
            y0, y1, x0, x1 = max(0, int(y) - r), min(H, int(y) + r + 1), max(0, int(x) - r), min(W, int(x) + r + 1)
            win = ids[y0:y1, x0:x1]
            if (win >= 0).any():
                yy, xx = np.argwhere(win >= 0)[0]
                hit = (y0 + yy, x0 + xx); break
        if hit is None:
            return None
        sub = np.full_like(ids, -1); sub[hit] = ids[hit]
        ys, xs, t, B = self._interp(sub, P, sx, sy, z)
        pos = (self.P[t] * B[:, :, None]).sum(1)[0]   # single pixel
        return [round(float(v), 4) for v in pos], self.names[int(self.PID[t[0]])]

    def render(self, cam, W, H, ss=1.5, style=None, alpha_only=False):
        """Returns dict(rgb (H,W,3) float32 BGR 0-255 (straight alpha), a (H,W) 0-1, dist (H,W) model units,
        pid (H,W) int16 -1=background)."""
        style = style or {}
        Ws, Hs = int(round(W * ss)), int(round(H * ss))
        ids, P, sx, sy, z, (eye, right, upv, fwd) = self._raster(cam, Ws, Hs)
        m = ids >= 0
        if alpha_only:
            a = m.astype(np.float32)
            return {'a': cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA) if ss != 1 else a}
        ys, xs, t, B = self._interp(ids, P, sx, sy, z)
        b0, b1, b2 = B[:, 0:1], B[:, 1:2], B[:, 2:3]
        lerp = lambda A: A[t, 0] * b0 + A[t, 1] * b1 + A[t, 2] * b2  # noqa: E731
        pos = lerp(P)
        n = _norm(lerp(self.N))
        base = lerp(self.C)
        met, rgh, pid = self.MET[t], self.RGH[t], self.PID[t]
        eye32 = eye.astype(np.float32)
        V = _norm(eye32 - pos)
        ndv = n[:, 0] * V[:, 0] + n[:, 1] * V[:, 1] + n[:, 2] * V[:, 2]
        flip = ndv < 0
        n[flip] *= -1; ndv = np.abs(ndv)
        f32 = lambda v: np.asarray(v, np.float32)
        Lk = f32(_norm(-0.45 * right + 0.65 * upv - 0.62 * fwd))
        Lf = f32(_norm(0.75 * right + 0.05 * upv - 0.65 * fwd))
        Lr = f32(_norm(0.15 * right + 0.55 * upv + 0.85 * fwd))
        nk = np.clip(n @ Lk, 0, 1); nf = np.clip(n @ Lf, 0, 1); nr = np.clip(n @ Lr, 0, 1)
        amb = 0.12 + 0.08 * n[:, 1]
        diff = base * (1 - met)[:, None] * (amb + 0.95 * nk + 0.32 * nf)[:, None]
        Rv = 2 * ndv[:, None] * n - V
        sharp = 2.0 + (1 - rgh) ** 2 * 160
        s1, s2 = f32(_norm([-0.55, 0.75, 0.55])), f32(_norm([0.85, 0.25, -0.35]))
        lg = lambda x: np.log(np.clip(x, 1e-6, 1))  # noqa: E731
        boost = (sharp / 60) ** 0.35
        env = (0.04 + 0.22 * np.clip(Rv[:, 1] * 0.5 + 0.5, 0, 1) ** 2
               + boost * (3.2 * np.exp(sharp * lg(Rv @ s1)) + 1.5 * np.exp(sharp * lg(Rv @ s2)))
               + 0.3 * np.exp(-(Rv[:, 1] / (0.06 + 0.25 * rgh)) ** 2))
        F0 = float(style.get('f0', 0.05)) * (1 - met)[:, None] + base * met[:, None]
        g5 = (1 - ndv) ** 2; g5 = g5 * g5 * (1 - ndv)
        fres = F0 + (1 - F0) * g5[:, None]
        Hk = _norm(Lk + V)
        shin = np.clip(2 / (rgh ** 4 + 1e-3), 4, 3000)
        spk = np.exp(shin * lg(n[:, 0] * Hk[:, 0] + n[:, 1] * Hk[:, 1] + n[:, 2] * Hk[:, 2])) * (shin + 8) / 25.0 * 0.6 * (nk > 0)
        col = diff + fres * (env + spk)[:, None]
        rimc = f32(style.get('rim', (0.55, 0.6, 0.7)))
        col += rimc[None, :] * (style.get('rim_k', 0.35) * (1 - ndv) ** 3 * (0.3 + nr))[:, None]
        hl = cam.get('hl') or []
        ha = float(cam.get('hl_a', 0.0))
        if hl and ha > 0:
            on = np.isin(pid, hl)
            acc = srgb2lin(f32(style.get('accent', (1.0, 0.65, 0.12))))
            e = (1 - ndv[on]) ** 3
            col[on] = col[on] * (1 + 0.6 * ha) + acc * (ha * 0.55 * e * e)[:, None] + acc * (ha * 0.004)
            off = ~on
            g = col[off].mean(1, keepdims=True)
            col[off] = (col[off] * (1 - 0.6 * ha) + g * 0.6 * ha) * (1 - 0.55 * ha)
        expo = float(style.get('exposure', 1.0))
        col = 1 - np.exp(-col * expo * 1.6)
        col = np.power(np.clip(col, 0, 1), 1 / 2.2)
        rgb = np.zeros((Hs, Ws, 3), np.float32)
        rgb[ys, xs] = col[:, ::-1] * 255                       # RGB -> BGR
        org = f32(cam.get('origin', (0, 0, 0)))
        dist = np.zeros((Hs, Ws), np.float32)
        dist[ys, xs] = np.linalg.norm(pos - org, axis=1)
        pidb = np.full((Hs, Ws), -1, np.int16); pidb[ys, xs] = pid
        a = m.astype(np.float32)
        if style.get('outline'):
            nb = np.zeros((Hs, Ws, 3), np.float32); nb[ys, xs] = n
            zb = np.zeros((Hs, Ws), np.float32); zb[ys, xs] = (pos - eye.astype(np.float32)) @ f32(fwd)
            e = np.zeros((Hs, Ws), bool)
            for dy, dx in ((0, 1), (1, 0)):
                p2 = np.roll(pidb, (-dy, -dx), (0, 1)); n2 = np.roll(nb, (-dy, -dx), (0, 1)); z2 = np.roll(zb, (-dy, -dx), (0, 1))
                both = m & np.roll(m, (-dy, -dx), (0, 1))
                e |= both & ((p2 != pidb) | ((nb * n2).sum(2) < 0.75) | (np.abs(z2 - zb) > 0.02 * zb))
            k = max(1, int(round(ss)))
            if k > 1:
                e = cv2.dilate(e.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & m
            lc = np.array(style.get('outline_color', (40, 36, 32)), np.float32)
            rgb[e] = rgb[e] * (1 - style.get('outline_k', 0.55)) + lc * style.get('outline_k', 0.55)
        if ss != 1:
            pre = cv2.resize(rgb * a[..., None], (W, H), interpolation=cv2.INTER_AREA)
            dd = cv2.resize(dist * a, (W, H), interpolation=cv2.INTER_AREA)
            a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
            rgb = pre / np.maximum(a, 1e-4)[..., None]
            dist = dd / np.maximum(a, 1e-4)
            pidb = cv2.resize(pidb, (W, H), interpolation=cv2.INTER_NEAREST)
        return {'rgb': rgb, 'a': a, 'dist': dist, 'pid': pidb}

    def dmax(self, origin):
        return float(np.linalg.norm(self.v_all - np.array(origin[:3], np.float32), axis=1).max()) + 0.05


_CACHE = {}


def get_model(m3d):
    key = (m3d['path'], json.dumps(m3d.get('opts', {}), sort_keys=True))
    if key not in _CACHE:
        _CACHE[key] = Model3D(m3d['path'], m3d.get('opts'))
    return _CACHE[key]


# =============================================================== storyboard / timeline glue
def model_opts(sb):
    m = sb['model']
    if isinstance(m, str):
        m = {'file': m}
    return m


def prepare(sb, sbp):
    """Resolve the model, convert every 3D point in the storyboard to [x,y,z,part]."""
    m = model_opts(sb)
    path = m['file'] if os.path.isabs(m['file']) else os.path.join(os.path.dirname(sbp), m['file'])
    opts = {k: v for k, v in m.items() if k != 'file'}
    sb['_m3d'] = {'path': path, 'opts': opts}
    M = get_model(sb['_m3d'])
    for lab in sb.get('overview', {}).get('labels', []):
        lab['_p'] = M.point(lab)
    for sc in sb['scenes']:
        for c in sc.get('callouts', []):
            c['_p'] = M.point(c)
    ro = sb.get('reveal_origin')
    if isinstance(ro, dict):
        sb['_reveal_origin'] = M.point(ro)
    elif isinstance(ro, str):
        sb['_reveal_origin'] = M.point({'part': ro})
    elif isinstance(ro, (list, tuple)) and len(ro) == 3:
        sb['_reveal_origin'] = [float(v) for v in ro] + [-1]
    else:
        first = next((c['_p'] for sc in sb['scenes'] for c in sc.get('callouts', [])), [0, 0, 0, -1])
        sb['_reveal_origin'] = first
    sb['_size'] = (1000, 1000)
    sb['_scale'] = 1.0
    return M


def _angdiff(a, b):
    return (a - b + 180) % 360 - 180


def auto_view(M, pts, hy, hp, explode=0.0):
    """Pick the orbit angle (near the hero view) from which the most callout points are visible."""
    if not pts:
        return hy, hp
    best = None
    W, H = 320, 200
    for dy in (0, -30, 30, -60, 60, -95, 95, 180):
        for dp in (0, 14, -14):
            c = {'yaw': hy + dy, 'pitch': max(-35, min(60, hp + dp)), 'zoom': 1.0, 'target': [0, 0, 0],
                 'ox': 0, 'oy': 0, 'explode': explode}
            ids = M.ids(c, W, H)
            vis = 0
            for p in pts:
                x, y = M.project(p, c, W, H)
                xi, yi = int(x), int(y)
                if 0 <= xi < W and 0 <= yi < H:
                    win = ids[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].ravel()
                    win = win[win >= 0]
                    if p[3] < 0 or (len(win) and (M.PID[win] == p[3]).any()):
                        vis += 1
            score = vis * 10 - abs(dy) / 30 - abs(dp) / 40
            if best is None or score > best[0]:
                best = (score, c['yaw'], c['pitch'])
    return best[1], best[2]


def build_cam3d(sb, tl, W, H):
    """Camera / explode keyframes + highlight windows for a 3D storyboard."""
    M = get_model(sb['_m3d'])
    o = sb['_m3d']['opts']
    hy, hp = [float(v) for v in o.get('view', [35, 20])]
    spin = float(o.get('intro_spin', 160))
    drift = float(o.get('drift', 6))
    org = sb['_reveal_origin'][:3]
    segs = tl['segments']
    kf, hls = [], []

    def K(t, yaw, pitch, zoom, target=(0, 0, 0), ox=0.0, oy=0.0, explode=0.0, unwrap=True):
        if unwrap and kf:                      # shortest way round from the previous keyframe
            yaw = kf[-1]['yaw'] + _angdiff(yaw, kf[-1]['yaw'])
        kf.append({'t': float(t), 'yaw': float(yaw), 'pitch': float(pitch), 'zoom': float(zoom),
                   'target': [float(v) for v in target], 'ox': float(ox), 'oy': float(oy), 'explode': float(explode)})

    scenes = sb['scenes']
    si = 0
    for g in segs:
        t0, t1 = g['t0'], g['t1']
        if g['kind'] == 'intro':
            K(0.0, hy - spin, hp, 0.78, ox=0.13, oy=0.11, unwrap=False)
            K(t1, hy - 6, hp, 0.84, ox=0.13, oy=0.11, unwrap=False)
        elif g['kind'] == 'overview':
            K(t0 + 1.3, hy, hp, 1.0)
            K(t1, hy + drift, hp, 1.02)
        elif g['kind'] == 'scene':
            sc = scenes[si]; si += 1
            ex = float(sc.get('explode', 0.0))
            pts = []
            for c in sc.get('callouts', []):
                p = np.array(c['_p'][:3], float)
                if c['_p'][3] >= 0:
                    p = p + M.OFF[c['_p'][3]] * ex
                pts.append(p)
            pts = np.array(pts) if pts else np.zeros((1, 3))
            if sc.get('view'):
                yaw, pitch = [float(v) for v in sc['view']]
            else:
                yaw, pitch = auto_view(M, [c['_p'] for c in sc.get('callouts', [])], hy, hp, ex)
            if sc.get('target') is not None:
                tgt = np.array(sc['target'][:3], float)
            else:
                tgt = (pts.min(0) + pts.max(0)) / 2
            if sc.get('zoom'):
                z = float(sc['zoom'])
            else:
                cam = {'yaw': yaw, 'pitch': pitch, 'zoom': 1, 'target': tgt}
                _, right, upv, _ = M.basis(cam)
                rel = pts - tgt
                exw = max(np.abs(rel @ right).max(), 0.05); eyh = max(np.abs(rel @ upv).max(), 0.05)
                z = min(VIEW_R * 0.34 / eyh, VIEW_R * (W / H) * 0.30 / exw)
                z = max(1.1, min(float(o.get('max_zoom', 2.3)), z))
            K(t0 + 1.3, yaw, pitch, z, tgt, explode=ex)
            K(t1, yaw + drift * 0.7, pitch, z * 1.04, tgt, explode=ex)
            hl = sc.get('highlight')
            if hl == 'auto':
                hl = sorted({M.names[c['_p'][3]] for c in sc.get('callouts', []) if c['_p'][3] >= 0})
            if hl:
                hls.append({'t0': t0, 't1': t1, 'parts': [M.part_index(n) for n in hl]})
        elif g['kind'] == 'outro':
            K(t0 + 1.4, hy, hp, 0.86, ox=0.2, oy=0.04)
            K(t1, hy + 28, hp, 0.88, ox=0.2, oy=0.04)
    tl['cam3d'] = kf
    tl['hl3d'] = hls
    tl['origin3d'] = [float(v) for v in org]
    tl['dmax3d'] = M.dmax(org)


def cam_at(tl, t):
    C = tl['cam3d']

    def mk(a, b, e):
        if b is None:
            c = dict(a)
        else:
            c = {'yaw': a['yaw'] + (b['yaw'] - a['yaw']) * e,
                 'pitch': a['pitch'] + (b['pitch'] - a['pitch']) * e,
                 'zoom': math.exp(math.log(a['zoom']) + (math.log(b['zoom']) - math.log(a['zoom'])) * e),
                 'target': [x + (y - x) * e for x, y in zip(a['target'], b['target'])],
                 'ox': a['ox'] + (b['ox'] - a['ox']) * e, 'oy': a['oy'] + (b['oy'] - a['oy']) * e,
                 'explode': a['explode'] + (b['explode'] - a['explode']) * e}
        hl, ha = [], 0.0
        for h in tl.get('hl3d', []):
            if h['t0'] - 0.2 <= t <= h['t1']:
                a_ = min(max((t - h['t0'] - 0.3) / 0.6, 0), 1) * min(max((h['t1'] - t) / 0.5, 0), 1)
                if a_ > ha:
                    hl, ha = h['parts'], a_
        c['hl'], c['hl_a'] = hl, ha
        c['origin'] = tl.get('origin3d', [0, 0, 0])
        return c

    if t <= C[0]['t']:
        return mk(C[0], None, 0)
    for a, b in zip(C, C[1:]):
        if t <= b['t']:
            e = ease((t - a['t']) / (b['t'] - a['t'])) if b['t'] > a['t'] else 1
            return mk(a, b, e)
    return mk(C[-1], None, 0)


def hero_cam(sb, zoom=1.0, ox=0.0, oy=0.0):
    o = sb['_m3d']['opts']
    hy, hp = o.get('view', [35, 20])
    return {'yaw': float(hy), 'pitch': float(hp), 'zoom': zoom, 'target': [0, 0, 0], 'ox': ox, 'oy': oy,
            'explode': 0.0, 'hl': [], 'hl_a': 0.0, 'origin': sb['_reveal_origin'][:3]}
