#!/usr/bin/env python3
"""Look at a 3D model and get exact 3D points for storyboard callouts.

  python inspect3d.py model.glb                      # parts list + views.png (6 labelled views)
  python inspect3d.py model.glb --grid 35 20         # view_35_20.png: one view with a pixel grid
  python inspect3d.py model.glb --pick 35 20 812,430 640,515
                                                     # pixel(s) on that grid view -> point3d + part
  python inspect3d.py --check storyboard.json        # points_check.png: every scene framed as in
                                                     # the video, with each callout marker drawn
Options: --opts '{"up":"z","rotate":[0,90,0]}' (same keys as storyboard "model"), --sb storyboard.json
(read the model + options from a storyboard), --out DIR. Grid/pick views are 1600x1000.
Points are NORMALISED model coordinates (centre 0, bounding radius 1, Y up) — paste them into
"point3d" together with the "part" name.
"""
import argparse, json, math, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model3d as m3  # noqa: E402

GW, GH = 1600, 1000
PART_COLS = [(66, 135, 245), (60, 180, 75), (230, 25, 75), (255, 165, 0), (145, 30, 180), (70, 240, 240),
             (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128)]


def cam(yaw, pitch, zoom=1.0, **k):
    c = {'yaw': float(yaw), 'pitch': float(pitch), 'zoom': zoom, 'target': [0, 0, 0], 'ox': 0, 'oy': 0,
         'explode': 0.0, 'hl': [], 'hl_a': 0.0}
    c.update(k)
    return c


def shade(M, c, W, H, bg=(245, 244, 242)):
    L = M.render(c, W, H, ss=1.5, style={'outline': True, 'rim_k': 0.25})
    a = L['a'][..., None]
    out = np.full((H, W, 3), bg, np.float32) * (1 - a) + L['rgb'] * a
    return out.astype(np.uint8), L


def label_parts(img, M, c, W, H, pidb=None):
    for i, n in enumerate(M.names):
        x, y = M.project(list(M.pcenter[i]) + [i], c, W, H)
        if not (0 <= x < W and 0 <= y < H):
            continue
        col = PART_COLS[i % len(PART_COLS)][::-1]
        cv2.circle(img, (int(x), int(y)), 5, col, -1)
        cv2.putText(img, n, (int(x) + 8, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, n, (int(x) + 8, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def views_sheet(M, out, view=(35, 20)):
    views = [('front 0,10', 0, 10), ('right 90,10', 90, 10), ('back 180,10', 180, 10),
             ('left -90,10', -90, 10), ('top 0,80', 0, 80), (f'hero {view[0]},{view[1]}', view[0], view[1])]
    W, H = 640, 400
    tiles = []
    for name, y, p in views:
        c = cam(y, p)
        img, _ = shade(M, c, W, H)
        label_parts(img, M, c, W, H)
        cv2.putText(img, f'yaw,pitch = {name}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
        tiles.append(img)
    sheet = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    cv2.imwrite(out, sheet)


def grid_view(M, yaw, pitch, out, zoom=1.0):
    c = cam(yaw, pitch, zoom)
    img, _ = shade(M, c, GW, GH)
    for x in range(50, GW, 50):
        major = x % 100 == 0
        cv2.line(img, (x, 0), (x, GH), (0, 0, 255) if major else (255, 140, 60), 1)
        if major:
            cv2.putText(img, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1, cv2.LINE_AA)
    for y in range(50, GH, 50):
        major = y % 100 == 0
        cv2.line(img, (0, y), (GW, y), (0, 0, 255) if major else (255, 140, 60), 1)
        if major:
            cv2.putText(img, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1, cv2.LINE_AA)
    label_parts(img, M, c, GW, GH)
    cv2.imwrite(out, img)


def check(sbp):
    sb = json.load(open(sbp, encoding='utf-8'))
    M = m3.prepare(sb, sbp)
    # fake timeline: same camera logic as the video
    segs, t = [{'kind': 'intro', 't0': 0, 't1': 5}], 5
    segs.append({'kind': 'overview', 't0': t, 't1': t + 6}); t += 6
    for i, sc in enumerate(sb['scenes']):
        segs.append({'kind': 'scene', 't0': t, 't1': t + 6, 'chapter': i + 1}); t += 6
    segs.append({'kind': 'outro', 't0': t, 't1': t + 6})
    tl = {'segments': segs}
    W, H = 960, 540
    m3.build_cam3d(sb, tl, W, H)
    tiles = []
    n = 0
    for g in segs:
        if g['kind'] not in ('overview', 'scene'):
            continue
        c = m3.cam_at(tl, g['t0'] + 1.5)
        img, _ = shade(M, c, W, H)
        if g['kind'] == 'overview':
            items = [(l['_p'], 'ov:' + l['text'], (200, 120, 0)) for l in sb.get('overview', {}).get('labels', [])]
            title = 'OVERVIEW'
        else:
            sc = sb['scenes'][g['chapter'] - 1]
            items = []
            for co in sc.get('callouts', []):
                n += 1
                items.append((co['_p'], f'{n:02d} {co.get("label", "")}', (0, 0, 230)))
            title = f'SCENE {g["chapter"]}: {sc.get("title", "")}  (view {c["yaw"]:.0f},{c["pitch"]:.0f} zoom {c["zoom"]:.2f}' \
                    f'{" explode " + str(sc.get("explode")) if sc.get("explode") else ""})'
        for p, name, col in items:
            x, y = M.project(p, c, W, H)
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 11, col, 2); cv2.circle(img, (x, y), 3, col, -1)
            cv2.putText(img, name, (x + 14, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, name, (x + 14, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    out = os.path.join(os.path.dirname(os.path.abspath(sbp)), 'points_check.png')
    cv2.imwrite(out, sheet)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', nargs='?')
    ap.add_argument('--opts', default='{}')
    ap.add_argument('--sb')
    ap.add_argument('--out', default='.')
    ap.add_argument('--grid', nargs=2, type=float, metavar=('YAW', 'PITCH'))
    ap.add_argument('--zoom', type=float, default=1.0)
    ap.add_argument('--pick', nargs='+')
    ap.add_argument('--check')
    a = ap.parse_args()
    if a.check:
        return check(a.check)
    opts = json.loads(a.opts)
    path = a.model
    if a.sb:
        sb = json.load(open(a.sb, encoding='utf-8'))
        mo = m3.model_opts(sb)
        path = path or os.path.join(os.path.dirname(os.path.abspath(a.sb)), mo['file'])
        opts = {**{k: v for k, v in mo.items() if k != 'file'}, **opts}
    if not path:
        ap.error('give a model file or --sb storyboard.json')
    M = m3.Model3D(path, opts)
    os.makedirs(a.out, exist_ok=True)
    if a.pick:
        yaw, pitch = float(a.pick[0]), float(a.pick[1])
        res = []
        for xy in a.pick[2:]:
            x, y = [float(v) for v in xy.split(',')]
            r = M.pick(cam(yaw, pitch, a.zoom), GW, GH, x, y)
            res.append({'px': [x, y], 'point3d': r[0] if r else None, 'part': r[1] if r else None})
        print(json.dumps(res, indent=1))
        return
    if a.grid:
        out = os.path.join(a.out, f'view_{a.grid[0]:g}_{a.grid[1]:g}.png')
        grid_view(M, a.grid[0], a.grid[1], out, a.zoom)
        print('wrote', out, f'({GW}x{GH}; pick with --pick {a.grid[0]:g} {a.grid[1]:g} x,y ...'
              + (f' --zoom {a.zoom:g}' if a.zoom != 1 else '') + ')')
        return
    ntri = len(M.P)
    print(f'model {os.path.basename(path)}: {M.npart} parts, {ntri} triangles (after subdivision), '
          f'scale {1 / M.scale:.4g} units = radius 1, cull={M.cull}')
    for i, n in enumerate(M.names):
        c = M.pcenter[i]
        print(f'  [{i}] {n:24s} centre [{c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}]  explode '
              f'[{M.OFF[i][0]:+.2f}, {M.OFF[i][1]:+.2f}, {M.OFF[i][2]:+.2f}]  tris {int((M.PID == i).sum())}')
    out = os.path.join(a.out, 'views.png')
    views_sheet(M, out, opts.get('view', [35, 20]))
    print('wrote', out)


if __name__ == '__main__':
    main()
