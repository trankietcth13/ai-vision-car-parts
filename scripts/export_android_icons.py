"""Export the app logo as the Android launcher icon (adaptive icon + legacy PNGs) straight into the app's resources.

The source (data/app_icons/logo_option_2_engine_diag.jpg) is a mock-up: a dark rounded tile with a vertical gradient
on a grey backdrop, the engine/camera logo in the middle. Using the whole picture as the icon would show an icon inside
an icon and the launcher mask would cut the logo. Instead:
  - the tile gradient is measured (per channel, linear in y) and becomes the adaptive BACKGROUND layer;
  - the logo is "un-blended" from that gradient (colour-to-alpha: the smallest alpha that explains each pixel as
    logo-over-tile), so the teal glow keeps its colour, and becomes the FOREGROUND layer, scaled so the engine shape
    stays inside the 66 dp safe zone of the 108 dp layer (circle, squircle and rounded-square masks keep it whole);
  - legacy icons (Android < 8) are the two layers composited, rounded square / circle.
Android 8+ uses mipmap-anydpi-v26/ic_launcher.xml, which takes precedence over the PNGs.

    python scripts/export_android_icons.py [--source data/app_icons/logo_option_2_engine_diag.jpg]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "apps" / "engine_bay_android" / "app" / "src" / "main" / "res"
DENSITY = {"mdpi": 1.0, "hdpi": 1.5, "xhdpi": 2.0, "xxhdpi": 3.0, "xxxhdpi": 4.0}

# Measured on logo_option_2 (1024 px): tile ~x 130-894 / y 133-895; logo incl. glow line ~x 185-840 / y 290-715.
LOGO_BOX = (175, 280, 850, 725)   # region that holds the logo (all inside the tile), alpha outside = 0
FEATHER = 14                      # px soft edge of that region
GRADIENT_ROWS = (300, 700)        # rows where the tile left of the logo is measured
GRADIENT_COLS = (145, 200)
NOISE = 0.035                     # alpha below this is tile texture / JPEG noise
LOGO_UNITS = 62.0                 # logo box width in the 108-unit adaptive layer (engine shape ~ inside 66 safe zone)


def tile_gradient(a: np.ndarray):
    """Per-channel linear fit of the tile colour over y (rows / columns beside the logo)."""
    ys = np.arange(*GRADIENT_ROWS)
    band = np.median(a[ys, GRADIENT_COLS[0]:GRADIENT_COLS[1]], axis=1)  # (n, 3)
    coef = [np.polyfit(ys, band[:, c], 1) for c in range(3)]
    return lambda y: np.stack([np.polyval(coef[c], y) for c in range(3)], -1).clip(0, 255)


def unblend(a: np.ndarray, grad) -> np.ndarray:
    """RGBA logo whose composite over the tile gradient reproduces the photo."""
    h, w, _ = a.shape
    B = grad(np.arange(h, dtype=float))[:, None, :].repeat(w, 1)
    alpha = np.max((a - B) / np.maximum(255 - B, 1e-6), axis=2)
    alpha = ((alpha - NOISE) / (1 - NOISE)).clip(0, 1)  # the tile's texture is not logo
    mask = Image.new("L", (w, h), 0)
    x0, y0, x1, y1 = LOGO_BOX
    ImageDraw.Draw(mask).rectangle((x0 + FEATHER, y0 + FEATHER, x1 - FEATHER, y1 - FEATHER), fill=255)
    mask = np.asarray(mask.filter(ImageFilter.GaussianBlur(FEATHER / 2)), float) / 255
    alpha = alpha * mask
    F = B + (a - B) / np.maximum(alpha, 1e-6)[..., None]
    out = np.dstack([F.clip(0, 255), alpha * 255])
    out[alpha < 1e-4] = 0
    return out.astype(np.uint8)


def layers(src: Path, size: int):
    """Foreground and background layers (size x size px = 108 dp) for one density."""
    a = np.asarray(Image.open(src).convert("RGB"), float)
    grad = tile_gradient(a)
    logo = Image.fromarray(unblend(a, grad), "RGBA")
    x0, y0, x1, y1 = LOGO_BOX
    # the 108-unit layer maps to this many source pixels, centred on the logo box
    src_px = (x1 - x0) * 108.0 / LOGO_UNITS
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (cx - src_px / 2, cy - src_px / 2, cx + src_px / 2, cy + src_px / 2)
    fg = logo.transform((size, size), Image.Transform.EXTENT, box, Image.Resampling.BICUBIC)
    # background: the same gradient, over the same source rows (so logo + background = the tile)
    ys = np.linspace(box[1], box[3], size)
    col = grad(ys)  # (size, 3)
    bg = Image.fromarray(np.repeat(col[:, None, :], size, 1).round().astype(np.uint8), "RGB")
    return fg, bg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=ROOT / "data" / "app_icons" / "logo_option_2_engine_diag.jpg")
    ap.add_argument("--res", type=Path, default=RES)
    args = ap.parse_args()
    for name, k in DENSITY.items():
        d = args.res / f"mipmap-{name}"
        d.mkdir(parents=True, exist_ok=True)
        fg, bg = layers(args.source, round(108 * k))
        fg.save(d / "ic_launcher_foreground.png", optimize=True)
        bg.save(d / "ic_launcher_background.png", optimize=True)
        # legacy 48 dp icon: the visible 72/108 centre of the composite, rounded square and circle
        full = bg.convert("RGBA")
        full.alpha_composite(fg)
        inset = round(18 * k)
        vis = full.crop((inset, inset, full.width - inset, full.height - inset)).resize((round(48 * k),) * 2, Image.Resampling.LANCZOS)
        s = vis.width
        for fname, shape in (("ic_launcher.png", "square"), ("ic_launcher_round.png", "circle")):
            m = Image.new("L", (s * 4, s * 4), 0)
            dr = ImageDraw.Draw(m)
            if shape == "circle":
                dr.ellipse((0, 0, s * 4 - 1, s * 4 - 1), fill=255)
            else:
                dr.rounded_rectangle((0, 0, s * 4 - 1, s * 4 - 1), radius=s * 4 * 0.18, fill=255)
            icon = vis.copy()
            icon.putalpha(m.resize((s, s), Image.Resampling.LANCZOS))
            icon.save(d / fname, optimize=True)
    adaptive = """<?xml version="1.0" encoding="utf-8"?>
<!-- Launcher icon (Android 8+). Layers written by scripts/export_android_icons.py from data/app_icons. -->
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@mipmap/ic_launcher_background" />
    <foreground android:drawable="@mipmap/ic_launcher_foreground" />
</adaptive-icon>
"""
    any26 = args.res / "mipmap-anydpi-v26"
    any26.mkdir(parents=True, exist_ok=True)
    (any26 / "ic_launcher.xml").write_text(adaptive, encoding="utf-8")
    (any26 / "ic_launcher_round.xml").write_text(adaptive, encoding="utf-8")
    # 512 px store / preview icon (not packaged)
    fg, bg = layers(args.source, 768)
    full = bg.convert("RGBA")
    full.alpha_composite(fg)
    out = ROOT / "data" / "app_icons" / "ic_launcher-playstore.png"
    full.crop((128, 128, 640, 640)).save(out)
    print(f"wrote launcher icons to {args.res.relative_to(ROOT)} (mipmap-*/ic_launcher*.png, mipmap-anydpi-v26) and {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
