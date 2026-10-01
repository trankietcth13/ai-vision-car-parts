"""Build the 3D component models of the Android app from the OBD2 knowledge base.

Every component of configs/diagnosis_knowledge.yaml (plus the detector class engine_cover) gets a 3D model made of
primitives (tools/model3d/build_model.py from the pro-product-video skill + tools/model3d/partkit.py):
  - one named part per real part, with a PBR material and an exploded-view offset (disassembly order),
  - per part: "label" {"en","vi"}, "check" {"en","vi"} (what to inspect on it, optional), "explode" [dx, dy, dz];
    per model: "component", "view" [yaw, pitch], "unit": "mm".
The models are generic illustrations, not the geometry of a specific vehicle: no real dimensions, pin-outs or
markings. Check texts are drafts (generic gasoline-ICE practice; "typical" values must be checked against OEM data).

The app builds the meshes ON THE DEVICE from the specs (ModelBuilder.kt, same algorithms), so it ships
  apps/engine_bay_android/app/src/main/assets/models/<component>.json      primitives + texts, a few KB each
  apps/engine_bay_android/app/src/test/resources/fixtures/model_stats.json  per-part geometry for the parity test
                                                       (+ fixtures/battery.glb for the GLB reader test)
--glb DIR also writes GLB files (other viewers, the skill's 3D video mode); --preview DIR renders 6-view sheets.

    python scripts/deployment/build_component_models.py [--only battery,spark_plug] [--glb output/models3d/glb --preview output/models3d]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "model3d"))
from partkit import *  # noqa: E402,F403  (materials, shape helpers, part(), T(), write_glb)

APP = ROOT / "apps" / "engine_bay_android" / "app"
OUT = APP / "src" / "main" / "assets" / "models"
FIXTURE = APP / "src" / "test" / "resources" / "fixtures" / "model_stats.json"  # Kotlin builder parity
KNOWLEDGE = ROOT / "configs" / "diagnosis_knowledge.yaml"
EXTRA_CLASSES = ["engine_cover"]  # detector classes with a model but no entry in the knowledge table


# ---------------------------------------------------------------------------------------------- the models
# Units mm, Y up, front = +Z. Each model function returns (parts, [yaw, pitch] hero view).
MODELS: dict = {}


def model(fn):
    MODELS[fn.__name__] = fn
    return fn


DARK_MARK = dict(color="#141516", metallic=0.0, roughness=0.6)
HOLE = "#0A0B0C"


def ribs_y(n, radius, y, size, phase=0.0):
    """Grip ribs around a cap (axis Y)."""
    return around(n, radius, lambda x, yy, z, a: rb(size, min(size) * 0.3, (x, y, z), rot=[0, a, 0]), phase=phase)


# ------------------------------------------------------------------------------------------ charging / starting
@model
def battery():
    plus = [rb([16, 1.2, 3.5], 0, (85, 174.6, 20), **RED), rb([3.5, 1.2, 16], 0, (85, 174.6, 20), **RED)]
    minus = [rb([16, 1.2, 3.5], 0, (-85, 174.6, 20), color="#1E3A8A")]
    parts = [
        part("case", T("Battery case", "Vỏ bình"), dict(color="#2A2D31", roughness=0.6),
             [rb([240, 160, 172], 6, (0, 80, 0)),
              rb([150, 62, 1.2], 0.5, (0, 88, 86.2), color="#E4E2DA", roughness=0.5),
              rb([150, 10, 1.4], 0.5, (0, 112, 86.4), color="#C62828", roughness=0.5)],
             check=T("Bulging sides, cracks or wet acid marks: replace the battery. The hold-down must keep it from moving.",
                     "Vỏ phồng, nứt hoặc có vết axit: thay ắc quy. Pát giữ phải giữ chặt bình không xê dịch.")),
        part("lid", T("Lid", "Nắp bình"), dict(color="#1B1D20", roughness=0.55),
             [rb([244, 16, 176], 5, (0, 166, 0)), rb([200, 3, 26], 1.2, (0, 175, -58))] + plus + minus),
        part("positive_post", T("Positive post (+)", "Cọc dương (+)"), LEAD,
             [cyl(9.5, 22, (85, 185, 52), r2=8.4, chamfer=0.6), tor(13, 2.2, (85, 175.5, 52), **RED)],
             check=T("White or green corrosion around the post: clean it. Resting voltage of a charged battery is typically about 12.6 V.",
                     "Cọc bị ăn mòn trắng hoặc xanh: vệ sinh sạch. Ắc quy đầy, để nghỉ, thường đo khoảng 12,6 V.")),
        part("negative_post", T("Negative post (−)", "Cọc âm (−)"), LEAD,
             [cyl(9.5, 22, (-85, 185, 52), r2=8.4, chamfer=0.6), tor(13, 2.2, (-85, 175.5, 52), color="#1E3A8A", roughness=0.45)],
             check=T("Disconnect the negative (−) cable first and reconnect it last.",
                     "Tháo cáp âm (−) trước, lắp lại sau cùng.")),
        part("indicator", T("Charge indicator (hydrometer eye)", "Mắt báo trạng thái"), dict(color="#111214", roughness=0.2),
             [cyl(10, 3, (0, 175.5, 30)), cyl(5.5, 1, (0, 177.2, 30), color="#2FBF4A", roughness=0.15)],
             check=T("If fitted: green = charged, dark = needs charging, clear or yellow = replace.",
                     "Nếu có: xanh = đủ điện, tối = cần sạc, trong hoặc vàng = thay bình.")),
    ]
    return parts, [35, 25]


@model
def battery_terminal():
    clamp_mat = dict(color="#A19D93", metallic=0.7, roughness=0.45)
    parts = [
        part("battery_lid", T("Battery lid", "Nắp ắc quy"), dict(color="#1B1D20", roughness=0.55),
             [rb([120, 12, 80], 4, (15, -6, 0))]),
        part("post", T("Battery post", "Cọc bình"), LEAD, [cyl(9.5, 22, (0, 11, 0), r2=8.4, chamfer=0.6)],
             check=T("Corrosion on the post. While cranking, the voltage drop from post to clamp is typically under 0.1 V.",
                     "Cọc bị ăn mòn. Khi đề, sụt áp từ cọc sang kẹp thường dưới 0,1 V.")),
        part("clamp", T("Terminal clamp", "Kẹp cọc"), clamp_mat,
             [cyl(15, 14, (0, 12, 0), hole=9.4), rb([30, 14, 7], 2, (24, 12, 7.5)), rb([30, 14, 7], 2, (24, 12, -7.5))],
             explode=[0, 45, 0],
             check=T("The clamp must not turn by hand. White or green crust: disconnect (negative first), clean post and clamp, re-tighten.",
                     "Vặn tay kẹp không được xoay. Có lớp ăn mòn trắng/xanh: tháo cáp (âm trước), vệ sinh cọc và kẹp, siết lại.")),
        part("bolt", T("Clamp bolt", "Bu lông kẹp"), STEEL,
             [cyl(3.2, 30, (28, 12, 0), axis="z"), hexa(6.5, 5, (28, 12, 14.5), axis="z")], explode=[0, 45, 40]),
        part("nut", T("Nut", "Đai ốc"), STEEL, [hexa(6.5, 5, (28, 12, -14.5), axis="z")], explode=[0, 45, -40]),
        part("cable_lug", T("Cable lug", "Đầu cốt cáp"), dict(color="#B9B2A0", metallic=0.8, roughness=0.4),
             [rb([26, 10, 16], 3, (50, 12, 0))], explode=[0, 45, 0]),
        part("cable", T("Battery cable", "Cáp ắc quy"), dict(color="#B71C1C", roughness=0.5),
             [tube([(60, 12, 0), (85, 12, 0), (115, 2, 15), (150, -25, 40)], 8.5)], explode=[0, 45, 0],
             check=T("Swollen or cracked insulation, green corrosion creeping under it, crushed cable.",
                     "Vỏ cáp phồng, nứt, ăn mòn xanh lan vào trong vỏ, cáp bị dập.")),
    ]
    return parts, [40, 30]


@model
def alternator():
    fins = around_z(24, 66, lambda x, y, z, a: rb([8, 3.2, 14], 1, (x, y, -4), rot=[0, 0, a]))
    vents = around_z(10, 50, lambda x, y, z, a: rb([16, 5, 1.4], 1.2, (x, y, 38.6), rot=[0, 0, a], color=HOLE), phase=18)
    groove = [[0, 40], [33, 40]] + [[33 if k % 2 == 0 else 30, 41 + 1.6 * k] for k in range(11)] + [[33, 59], [14, 60], [0, 60]]
    blades = around_z(10, 30, lambda x, y, z, a: rb([22, 2.2, 6], 0.6, (x, y, 37.5), rot=[0, 0, a]))
    parts = [
        part("stator", T("Stator", "Stato"), dict(color="#5A5E63", metallic=0.8, roughness=0.45),
             [cyl(64, 16, (0, 0, -4), axis="z")] + fins),
        part("front_housing", T("Front housing", "Vỏ trước"), ALU,
             [cyl(64, 34, (0, 0, 21), axis="z", chamfer=4),
              ext(24, (0, 0, 16), axis="z", hull=[[50, -36, 13], [80, -64, 13]], holes=[[80, -64, 5.5]])] + vents),
        part("rear_housing", T("Rear housing", "Vỏ sau"), ALU,
             [cyl(62, 36, (0, 0, -30), axis="z", chamfer=4),
              ext(24, (0, 0, -28), axis="z", hull=[[-50, 36, 13], [-80, 64, 13]], holes=[[-80, 64, 5.5]])]),
        part("rear_cover", T("Rear cover (regulator)", "Nắp sau (bộ tiết chế)"), BLACK,
             [cyl(50, 10, (0, 0, -53), axis="z", chamfer=2)]),
        part("cooling_fan", T("Cooling fan", "Cánh quạt làm mát"), STEEL, [cyl(26, 3, (0, 0, 39.5), axis="z")] + blades),
        part("pulley", T("Pulley", "Puli"), STEEL, [lathe(groove, axis="z"), hexa(11, 8, (0, 0, 64), axis="z")],
             explode=[0, 0, 55],
             check=T("Belt alignment and wear. Growling, wobble or a rough-turning pulley = bearing or one-way clutch.",
                     "Dây curoa thẳng hàng, không mòn. Kêu ù, đảo hoặc quay nặng = hỏng bạc đạn hoặc puli một chiều.")),
        part("b_terminal", T("Output terminal (B+)", "Cọc ra (B+)"), BRASS,
             [cyl(3.6, 22, (28, 20, -64), axis="z"), hexa(7, 5, (28, 20, -63), axis="z", **STEEL),
              rb([18, 16, 12], 4, (28, 20, -72), **RED)], explode=[0, 0, -40],
             check=T("Clean and tight. With the engine running the charging voltage is typically 13.8–14.8 V.",
                     "Sạch và chặt. Khi nổ máy, điện áp sạc thường 13,8–14,8 V.")),
        part("connector", T("Regulator connector", "Giắc bộ tiết chế"), BLACK,
             [rb([26, 18, 16], 3, (-28, 22, -66)), rb([10, 4, 12], 1, (-28, 33, -66))], explode=[0, 0, -40],
             check=T("Locked and free of corrosion. Battery light on with the engine running: test the charging circuit.",
                     "Giắc cài chặt, không ăn mòn. Đèn ắc quy sáng khi nổ máy: kiểm tra mạch sạc.")),
    ]
    return parts, [60, 20]


@model
def starter_motor():
    parts = [
        part("motor", T("Motor", "Mô-tơ"), dict(color="#2E3135", metallic=0.5, roughness=0.45),
             [cyl(42, 120, (-40, 0, 0), axis="x", chamfer=3),
              lathe([[0, -112], [36, -112], [40, -106], [40, -100], [0, -100]], axis="x")]),
        part("nose_housing", T("Drive housing", "Vỏ đầu khởi động"), CAST,
             [lathe([[0, 20], [42, 20], [42, 30], [34, 60], [28, 90], [0, 90]], axis="x"),
              ext(10, (25, 0, 0), axis="x", hull=[[0, 0, 46], [0, 62, 12], [0, -62, 12]], holes=[[0, 62, 6], [0, -62, 6]])]),
        part("solenoid", T("Solenoid", "Rơ-le đề (solenoid)"), ZINC,
             [cyl(23, 80, (10, 64, 0), axis="x", chamfer=2), cyl(20, 10, (-33, 64, 0), axis="x", **BLACK)],
             check=T("A click but no crank: solenoid contacts or low voltage. No click: control circuit (S terminal, relay, park/neutral switch).",
                     "Có tiếng tách nhưng không quay: tiếp điểm solenoid hoặc điện yếu. Không có tiếng: mạch điều khiển (cọc S, rơ-le, công tắc P/N).")),
        part("main_terminal", T("Battery terminal (B+)", "Cọc nguồn (B+)"), BRASS,
             [cyl(4, 18, (-44, 72, 0), axis="x"), hexa(7, 6, (-46, 72, 0), axis="x", **STEEL),
              tube([(-50, 72, 0), (-80, 72, 0), (-120, 95, 20), (-160, 115, 40)], 8.5, **RED)],
             check=T("Tight and clean. Voltage drop on the main cable while cranking is typically under 0.5 V.",
                     "Chặt và sạch. Sụt áp trên cáp chính khi đề thường dưới 0,5 V.")),
        part("s_terminal", T("Start signal terminal (S)", "Cọc tín hiệu đề (S)"), BLACK,
             [rb([16, 12, 14], 2, (-44, 50, 0)), tube([(-52, 50, 0), (-80, 48, 10), (-110, 60, 30)], 3, color="#D4A017")]),
        part("pinion", T("Pinion gear", "Bánh răng đề"), STEEL,
             [ext(20, (100, 0, 0), axis="x", points=gear_points(13, 17, 11)), cyl(8, 30, (95, 0, 0), axis="x")],
             explode=[40, 0, 0],
             check=T("Worn or chipped teeth cause grinding; check the flywheel ring gear too.",
                     "Răng mòn hoặc mẻ gây tiếng rít khi đề; kiểm tra cả vành răng bánh đà.")),
    ]
    return parts, [-50, 25]


@model
def serpentine_belt():
    pul = [(0, -130, 75, 1), (175, -60, 55, 1), (155, 150, 30, 1), (60, 105, 30, -1), (-40, 175, 50, 1), (-150, 50, 45, 1)]

    def pulley(x, y, r, grooved=True, w=24, **mat):
        if grooved:
            prof = [[0, -w / 2 - 2], [r - 3, -w / 2 - 2], [r, -w / 2]] + \
                   [[r if k % 2 == 0 else r - 2.2, -w / 2 + 2.4 * k] for k in range(1, 10)] + [[r, w / 2], [r - 3, w / 2 + 2], [0, w / 2 + 2]]
        else:
            prof = [[0, -w / 2 - 2], [r - 1, -w / 2 - 2], [r + 2, -w / 2], [r, -w / 2 + 1], [r, w / 2 - 1], [r + 2, w / 2], [r - 1, w / 2 + 2], [0, w / 2 + 2]]
        return [lathe(prof, (x, y, 0), axis="z", **mat), cyl(r * 0.35, w + 8, (x, y, 2), axis="z", **CAST),
                hexa(max(7, r * 0.18), 6, (x, y, w / 2 + 6), axis="z", **STEEL)]

    pivot = (115, 55)
    ang = math.degrees(math.atan2(105 - pivot[1], 60 - pivot[0]))
    arm_c = ((60 + pivot[0]) / 2, (105 + pivot[1]) / 2)
    arm_len = math.hypot(60 - pivot[0], 105 - pivot[1])
    parts = [
        part("belt", T("Serpentine belt", "Dây curoa tổng"), dict(color="#1A1B1D", roughness=0.75),
             [dict(type="belt", pulleys=[list(p) for p in pul], thickness=5, width=22)], explode=[0, 0, 60],
             check=T("Cracks across the ribs, missing chunks, glazing, frayed edges, squeal at start-up: replace the belt.",
                     "Nứt ngang gân, mất miếng, bóng mặt, sờn mép, kêu rít khi nổ máy: thay dây curoa.")),
        part("crank_pulley", T("Crankshaft pulley", "Puli trục khuỷu"), dict(color="#5F6368", metallic=0.8, roughness=0.4),
             pulley(0, -130, 75)),
        part("ac_pulley", T("A/C compressor pulley", "Puli lốc điều hoà"), STEEL, pulley(175, -60, 55)),
        part("alternator_pulley", T("Alternator pulley", "Puli máy phát"), STEEL, pulley(155, 150, 30)),
        part("tensioner", T("Belt tensioner", "Bộ tăng đơ dây curoa"), ZINC,
             pulley(60, 105, 30, grooved=False) +
             [rb([arm_len + 20, 18, 10], 5, (arm_c[0], arm_c[1], -14), rot=[0, 0, ang], **CAST),
              cyl(16, 26, (pivot[0], pivot[1], -14), axis="z", **CAST)],
             check=T("The indicator must sit inside its range; the arm must not shake or rattle with the engine running.",
                     "Kim chỉ báo phải nằm trong vùng cho phép; cần tăng đơ không rung, không kêu khi nổ máy.")),
        part("ps_pulley", T("Power steering pump pulley", "Puli bơm trợ lực"), STEEL, pulley(-40, 175, 50)),
        part("water_pump_pulley", T("Water pump pulley", "Puli bơm nước"), STEEL, pulley(-150, 50, 45),
             check=T("Wobble, noise or coolant weeping from the pump weep hole = water pump bearing or seal.",
                     "Puli đảo, kêu hoặc rỉ nước ở lỗ thoát của bơm = hỏng bạc đạn hoặc phớt bơm nước.")),
    ]
    return parts, [20, 15]


@model
def fuse_relay_box():
    fuse_cols = ["#C62828", "#1E63C6", "#F2B705", "#E8E6DF", "#2E9E4F", "#C62828", "#F2B705"]
    fuses = []
    for r, z in enumerate((-48, -28, -8)):
        for c in range(7):
            x = -100 + 13 * c
            fuses.append(rb([11, 12, 4], 1, (x, 97, z), color=fuse_cols[(c + r) % len(fuse_cols)], roughness=0.3))
    maxi = [rb([22, 14, 22], 2, (x, 97, 40), color=col, roughness=0.3)
            for x, col in zip((-95, -68, -41, -14), ("#E25A9B", "#2E9E4F", "#F2B705", "#1E63C6"))]
    relays = [rb([30, 28, 30], 3, (x, 104, z)) for x in (32, 72) for z in (-35, 5)]
    parts = [
        part("base", T("Box base", "Đế hộp"), BLACK,
             [rb([230, 90, 150], 6, (0, 45, 0)), rb([212, 2, 132], 1, (0, 90.5, 0), color="#33363A")]),
        part("lid", T("Lid", "Nắp hộp"), BLACK,
             [rb([236, 34, 156], 10, (0, 107, 0)), rb([8, 26, 16], 2, (119, 95, 0)), rb([8, 26, 16], 2, (-119, 95, 0))],
             explode=[0, 110, 0],
             check=T("Lid seal and clips closed. Water or green corrosion inside causes intermittent faults.",
                     "Gioăng nắp và khoá cài kín. Nước hoặc ăn mòn xanh bên trong gây lỗi chập chờn.")),
        part("mini_fuses", T("Blade fuses", "Cầu chì cài"), dict(color="#C62828", roughness=0.3), fuses,
             check=T("Look for a broken element. Replace only with the same rating (same colour); a fuse that blows again = short circuit.",
                     "Kiểm tra dây chảy bị đứt. Chỉ thay cầu chì cùng định mức (cùng màu); thay xong lại cháy = có chập mạch.")),
        part("maxi_fuses", T("Main fuses (fusible links)", "Cầu chì tổng"), dict(color="#E25A9B", roughness=0.3), maxi,
             check=T("A blown main fuse kills a whole group of circuits (fan, ABS, main power). Find the cause before replacing.",
                     "Cầu chì tổng cháy làm mất cả nhóm mạch (quạt, ABS, nguồn chính). Tìm nguyên nhân trước khi thay.")),
        part("relays", T("Relays", "Rơ-le"), BLACK_GLOSS, relays,
             check=T("A relay should click when switched. Swap with an identical relay to test (fan, fuel pump, main relay).",
                     "Rơ-le phải kêu tách khi đóng. Thử bằng cách đổi với rơ-le cùng loại (quạt, bơm xăng, rơ-le chính).")),
        part("main_feed", T("Main power feed", "Đầu cấp nguồn chính"), BRASS,
             [cyl(4, 14, (95, 97, 55)), hexa(7, 5, (95, 102, 55), **STEEL),
              tube([(95, 106, 55), (110, 106, 62), (135, 70, 72), (175, 40, 82)], 8, **RED)]),
    ]
    return parts, [25, 40]


@model
def ecu_module():
    fins = [rb([3.5, 14, 140], 1, (x, 42, -6)) for x in range(-96, 97, 16)]
    tabs = [rb([24, 4, 22], 2, (sx * 122, 3, sz * 55)) for sx in (-1, 1) for sz in (-1, 1)]
    holes = [cyl(4.5, 4.4, (sx * 126, 3, sz * 55), color=HOLE) for sx in (-1, 1) for sz in (-1, 1)]
    parts = [
        part("housing", T("ECU housing", "Vỏ hộp ECU"), ALU,
             [rb([220, 36, 160], 5, (0, 18, 0)), rb([1, 24, 70], 0.3, (110.4, 18, -20), color="#E8E6DF", metallic=0, roughness=0.6)]
             + fins + tabs + holes,
             check=T("Water marks or corrosion on the housing = water intrusion: check seals, drains and the scuttle area.",
                     "Vết nước hoặc ăn mòn trên vỏ = vào nước: kiểm tra gioăng, lỗ thoát nước và khu vực hốc gió.")),
        part("connector_a", T("Connector A", "Giắc A"), BLACK,
             [rb([70, 30, 30], 4, (-50, 20, 94)), rb([72, 5, 34], 2, (-50, 38, 92), **DARK),
              tube([(-50, 20, 108), (-50, 20, 140), (-70, 0, 190), (-110, -20, 230)], 12, color="#202225")],
             explode=[0, 0, 60],
             check=T("Unlock and inspect: green corrosion, water, bent or pushed-back pins. Reseat and lock.",
                     "Mở khoá và kiểm tra: ăn mòn xanh, nước, chân giắc cong hoặc tụt. Cắm lại và khoá chặt.")),
        part("connector_b", T("Connector B", "Giắc B"), BLACK,
             [rb([60, 30, 30], 4, (40, 20, 94)), rb([62, 5, 34], 2, (40, 38, 92), **DARK),
              tube([(40, 20, 108), (40, 20, 140), (30, 0, 190), (10, -20, 240)], 11, color="#202225")],
             explode=[0, 0, 60],
             check=T("Communication codes (U0100): check the ECU power and ground fuses and the ground points first.",
                     "Mã lỗi mất giao tiếp (U0100): kiểm tra cầu chì nguồn, mass ECU và các điểm tiếp mass trước.")),
    ]
    return parts, [30, 30]


# ------------------------------------------------------------------------------------------ cooling
@model
def coolant_reservoir():
    parts = [
        part("tank", T("Expansion tank", "Bình giãn nở"), TANK,
             [rb([170, 150, 110], 18, (0, 75, 0)), cyl(22, 18, (30, 158, 0)), cyl(4, 14, (-50, 157, 0)),
              cyl(9, 26, (-95, 22, 0), axis="x")],
             check=T("Level between MIN and MAX with the engine cold. Dried crust or wet spots = leak.",
                     "Mức nước nằm giữa MIN và MAX khi máy nguội. Có cặn khô hoặc vết ướt = rò rỉ.")),
        part("coolant", T("Coolant", "Nước làm mát"), dict(color="#E35C94", roughness=0.15),
             [rb([160, 92, 100], 14, (0, 48, 0))],
             check=T("Colour should be clear. Oil film, rust or a brown sludge = contamination (head gasket, cooler).",
                     "Nước phải trong. Váng dầu, gỉ hoặc cặn nâu = nhiễm bẩn (gioăng mặt máy, két làm mát dầu).")),
        part("level_marks", T("MIN / MAX marks", "Vạch MIN / MAX"), DARK_MARK,
             [rb([34, 2, 1.2], 0, (-40, 40, 55.4)), rb([34, 2, 1.2], 0, (-40, 95, 55.4)), rb([2, 57, 1.2], 0, (-57, 67.5, 55.4))]),
        part("cap", T("Pressure cap", "Nắp áp suất"), BLACK,
             [cyl(28, 22, (30, 178, 0), chamfer=2)] + [dict(s, pos=[s["pos"][0] + 30, s["pos"][1], s["pos"][2]])
                                                     for s in ribs_y(12, 28, 178, [5, 18, 5])],
             explode=[0, 60, 0],
             check=T("Open only when the engine is cold: hot coolant under pressure can spray and burn. Check the cap seal.",
                     "Chỉ mở khi máy nguội: nước nóng có áp suất có thể phun ra gây bỏng. Kiểm tra gioăng nắp.")),
        part("hose", T("Outlet hose", "Ống ra"), RUBBER,
             [tube([(-104, 22, 0), (-130, 22, 0), (-165, 0, 20), (-190, -30, 40)], 11), tor(12.8, 2, (-115, 22, 0), axis="x", **ZINC)],
             check=T("Cracks, soft spots, a seeping clamp.", "Nứt, mềm nhũn, rỉ nước ở cổ dê.")),
    ]
    return parts, [30, 25]


@model
def radiator_cap():
    top = lathe([[0, 40], [22, 40], [18, 44.5], [0, 45.5]])
    parts = [
        part("filler_neck", T("Radiator filler neck", "Cổ châm két nước"), ALU,
             [cyl(23, 30, (0, 0, 0), hole=20), tor(23, 2, (0, 15, 0))], explode=[0, -45, 0],
             check=T("Dents or nicks on the sealing seat let pressure escape.", "Mặt tựa gioăng bị móp hoặc sứt làm xì áp suất.")),
        part("valve", T("Vacuum valve and plate", "Van chân không và đĩa van"), STEEL,
             [cyl(15, 2, (0, 9, 0)), cyl(3.5, 6, (0, 5, 0), **BRASS)], explode=[0, 22, 0]),
        part("pressure_seal", T("Pressure seal", "Gioăng áp suất"), RUBBER, [tor(16, 2.6, (0, 12.5, 0))], explode=[0, 30, 0],
             check=T("A cracked or hardened seal loses pressure: coolant loss and overheating.",
                     "Gioăng nứt hoặc chai cứng làm mất áp suất: hao nước làm mát và quá nhiệt.")),
        part("spring", T("Pressure spring", "Lò xo áp suất"), STEEL, [helix(9, 1.1, 14, 28, 4)], explode=[0, 48, 0],
             check=T("Pressure-test the cap; the rated pressure is marked on it (typically about 0.9–1.1 bar).",
                     "Thử áp suất nắp; áp suất định mức ghi trên nắp (thường khoảng 0,9–1,1 bar).")),
        part("cap", T("Cap", "Nắp"), dict(color="#3A3D42", metallic=0.7, roughness=0.4),
             [cyl(26, 12, (0, 34, 0), chamfer=1.5), top, rb([30, 5, 9], 2, (0, 47, 0))] + ribs_y(24, 26, 34, [4, 10, 3]),
             explode=[0, 75, 0],
             check=T("Never open it with the engine hot: hot coolant can spray out and burn.",
                     "Không mở khi máy còn nóng: nước làm mát nóng có thể phun ra gây bỏng.")),
    ]
    return parts, [30, 22]


@model
def radiator_hose():
    path = [(-170, 20, 0), (-125, 20, 0), (-75, 0, 25), (-10, -25, 45), (55, -5, 30), (110, 35, 5), (160, 40, 0)]

    def clamp(x, y):
        return [cyl(21.5, 12, (x, y, 0), axis="x", hole=19.2), rb([12, 10, 14], 2, (x, y + 23, 0)),
                cyl(3, 18, (x, y + 23, 0), axis="z")]

    parts = [
        part("hose", T("Radiator hose", "Ống két nước"), dict(color="#16171A", roughness=0.8),
             [tube(path, 19, sides=28, samples=12)],
             check=T("Squeeze it with the engine cold: hard, cracked, soft or swollen = replace. Never open the system hot.",
                     "Bóp thử khi máy nguội: cứng, nứt, mềm nhũn hoặc phồng = thay. Không mở hệ thống khi còn nóng.")),
        part("radiator_outlet", T("Radiator outlet", "Cổ ra két nước"), ALU, [cyl(16, 60, (-180, 20, 0), axis="x")]),
        part("engine_outlet", T("Engine outlet", "Cổ ra động cơ"), CAST, [cyl(16, 60, (170, 40, 0), axis="x")]),
        part("clamp_radiator", T("Hose clamp (radiator side)", "Cổ dê (phía két nước)"), ZINC, clamp(-158, 20),
             explode=[55, 0, 0],
             check=T("Tight, with no dried coolant crust around it.", "Siết chặt, không có cặn nước làm mát khô quanh cổ dê.")),
        part("clamp_engine", T("Hose clamp (engine side)", "Cổ dê (phía động cơ)"), ZINC, clamp(148, 40), explode=[-50, 0, 0],
             check=T("Tight, with no seepage.", "Siết chặt, không rỉ nước.")),
    ]
    return parts, [20, 25]


@model
def thermostat_housing():
    therm = [cyl(30, 2, (0, -1, 0), hole=21, **BRASS), cyl(20, 2, (0, -6, 0), **BRASS), cyl(9, 22, (0, -17, 0), **COPPER),
             helix(15, 1.4, -27, -7, 3.5, **STEEL), sph(2.2, (24, -1, 0), **BRASS),
             rb([3, 26, 6], 0.5, (19, -14, 0), **BRASS), rb([3, 26, 6], 0.5, (-19, -14, 0), **BRASS),
             rb([44, 2.4, 6], 0.5, (0, -27, 0), **BRASS)]
    parts = [
        part("housing", T("Thermostat housing", "Vỏ van hằng nhiệt"), CAST,
             [cyl(36, 26, (0, 17, 0)), ext(8, (0, 4, 0), hull=[[-46, 0, 9], [46, 0, 9], [0, 0, 36]], holes=[[-46, 0, 4.5], [46, 0, 4.5]]),
              tube([(0, 26, 0), (0, 40, 0), (25, 55, 0), (75, 60, 0)], 16), tor(16, 2.5, (62, 59.6, 0), axis="x")],
             check=T("Plastic housings crack. Look for seepage and dried crust at the joint and the outlet.",
                     "Vỏ nhựa dễ nứt. Tìm vết rỉ và cặn khô ở mặt ghép và cổ ra.")),
        part("bolts", T("Housing bolts", "Bu lông vỏ"), STEEL,
             [hexa(7, 6, (-46, 11, 0)), hexa(7, 6, (46, 11, 0))], explode=[0, 45, 0]),
        part("seal", T("Seal ring", "Gioăng làm kín"), RUBBER, [tor(31, 2.2, (0, 1.5, 0))], explode=[0, -30, 0],
             check=T("Always fit a new seal; a leaking joint lets air in and coolant out.",
                     "Luôn thay gioăng mới; mặt ghép rò làm lọt khí và mất nước.")),
        part("thermostat", T("Thermostat", "Van hằng nhiệt"), BRASS, therm, explode=[0, -75, 0],
             check=T("Stuck open: slow warm-up, low gauge, P0128. Stuck closed: overheating. Test it in hot water.",
                     "Kẹt mở: lâu ấm máy, kim nhiệt thấp, mã P0128. Kẹt đóng: quá nhiệt. Thử bằng cách ngâm nước nóng.")),
    ]
    return parts, [30, 20]


@model
def radiator_cooling_fan():
    def blade_poly():
        rs = np.linspace(46, 182, 12)
        le = [[r, 0.0022 * (r - 46) ** 2 + (24 + 16 * (r - 46) / 136)] for r in rs]
        te = [[r, 0.0022 * (r - 46) ** 2 - (24 + 16 * (r - 46) / 136)] for r in rs[::-1]]
        return le + te

    blades = [ext(3, (0, 0, 22), axis="z", points=blade_poly(), rot=[28, 0, 360 / 7 * i]) for i in range(7)]
    struts = around_z(4, 121, lambda x, y, z, a: rb([142, 10, 8], 2, (x, y, -10), rot=[0, 0, a]), phase=45)
    c = 195
    parts = [
        part("shroud", T("Fan shroud", "Khung quạt"), BLACK,
             [ext(16, (0, 0, -8), axis="z", hull=[[-c, -c, 25], [c, -c, 25], [-c, c, 25], [c, c, 25]], holes=[[0, 0, 192]]),
              cyl(196, 44, (0, 0, 14), axis="z", hole=191)] + struts),
        part("motor", T("Fan motor", "Mô-tơ quạt"), DARK, [cyl(50, 56, (0, 0, -36), axis="z", chamfer=4)],
             check=T("The fan must run when the engine is hot or the A/C is on. If not: fuse, relay, connector, then the motor.",
                     "Quạt phải chạy khi máy nóng hoặc bật điều hoà. Nếu không: kiểm tra cầu chì, rơ-le, giắc, rồi đến mô-tơ.")),
        part("blades", T("Fan blades", "Cánh quạt"), BLACK_GLOSS, [cyl(46, 26, (0, 0, 20), axis="z", chamfer=3)] + blades,
             explode=[0, 0, 70],
             check=T("Ignition off: the blades must spin freely by hand, no cracked or missing blade. The fan can start by itself when hot.",
                     "Tắt khoá điện: quay tay cánh phải nhẹ, không nứt hoặc gãy cánh. Quạt có thể tự chạy khi máy nóng.")),
        part("connector", T("Fan connector", "Giắc quạt"), BLACK,
             [rb([26, 20, 22], 3, (70, -60, -58)), tube([(70, -60, -70), (70, -60, -100), (120, -90, -120), (170, -110, -130)], 4)],
             check=T("Corroded or melted pins: high current flows here.", "Chân giắc ăn mòn hoặc chảy nhựa: dòng điện qua đây lớn.")),
    ]
    return parts, [25, 20]


# ------------------------------------------------------------------------------------------ fluids
@model
def brake_fluid_reservoir():
    parts = [
        part("master_cylinder", T("Brake master cylinder", "Xi-lanh phanh chính (tổng phanh)"), CAST,
             [cyl(24, 150, (0, 0, -10), axis="z"), cyl(20, 10, (0, 0, 69), axis="z"),
              ext(14, (0, 0, -85), axis="z", hull=[[-46, 0, 13], [46, 0, 13], [0, 0, 30]], holes=[[-46, 0, 6], [46, 0, 6]]),
              cyl(8, 10, (0, 28, -30), **RUBBER), cyl(8, 10, (0, 28, 30), **RUBBER)]),
        part("brake_lines", T("Brake line fittings", "Đầu nối ống dầu phanh"), STEEL,
             [hexa(7, 10, (29, 0, -20), axis="x"), hexa(7, 10, (29, 0, 30), axis="x"),
              tube([(34, 0, -20), (55, 0, -20), (75, -20, -40), (95, -35, -80)], 2.4),
              tube([(34, 0, 30), (60, 0, 30), (80, -25, 20), (100, -40, -10)], 2.4)],
             check=T("Wetness at a fitting or line = brake fluid leak: do not drive until repaired.",
                     "Ướt ở đầu nối hoặc đường ống = rò dầu phanh: không chạy xe cho đến khi sửa xong.")),
        part("reservoir", T("Fluid reservoir", "Bình chứa dầu phanh"), TANK,
             [rb([130, 70, 70], 10, (0, 68, 0)), cyl(18, 12, (-20, 108, 0))],
             check=T("Level between MIN and MAX. It drops slowly as the pads wear; a sudden drop = leak.",
                     "Mức dầu nằm giữa MIN và MAX. Mức giảm dần khi má phanh mòn; giảm đột ngột = rò rỉ.")),
        part("brake_fluid", T("Brake fluid", "Dầu phanh"), dict(color="#D9A12B", roughness=0.15),
             [rb([120, 40, 60], 8, (0, 52, 0))],
             check=T("Dark or brown fluid is old and has absorbed water: replace it (often every 2 years). It damages paint.",
                     "Dầu sẫm hoặc nâu là dầu cũ, đã ngậm nước: thay dầu (thường 2 năm). Dầu phanh làm hỏng sơn.")),
        part("level_marks", T("MIN / MAX marks", "Vạch MIN / MAX"), DARK_MARK,
             [rb([30, 2, 1.2], 0, (35, 82, 35.6)), rb([30, 2, 1.2], 0, (35, 50, 35.6))]),
        part("cap", T("Cap", "Nắp bình"), BLACK,
             [cyl(22, 16, (-20, 121, 0), chamfer=1.5)] + [dict(s, pos=[s["pos"][0] - 20, s["pos"][1], s["pos"][2]])
                                                       for s in ribs_y(12, 22, 121, [4, 12, 4])],
             explode=[0, 50, 0],
             check=T("Seal intact and vent clear. Wipe spilled fluid at once.", "Gioăng nắp còn tốt, lỗ thông hơi thông. Lau sạch dầu tràn ngay.")),
    ]
    return parts, [30, 25]


@model
def washer_fluid_reservoir():
    parts = [
        part("tank", T("Washer tank", "Bình nước rửa kính"), TANK,
             [rb([200, 220, 120], 18, (0, 110, 0)), tube([(60, 205, 0), (66, 250, 0), (80, 300, 0)], 19)],
             check=T("Cracks at the bottom or the pump grommet cause a slow leak.", "Nứt ở đáy hoặc ở gioăng bơm gây rò chậm.")),
        part("washer_fluid", T("Washer fluid", "Nước rửa kính"), dict(color="#3B82F6", roughness=0.15),
             [rb([188, 130, 108], 14, (0, 72, 0))],
             check=T("Top up with washer fluid; plain water freezes and leaves deposits.",
                     "Châm nước rửa kính chuyên dụng; nước thường dễ đóng cặn (và đóng băng ở xứ lạnh).")),
        part("cap", T("Filler cap", "Nắp châm"), BLUE,
             [cyl(24, 18, (80, 310, 0), chamfer=2)] + [dict(s, pos=[s["pos"][0] + 80, s["pos"][1], s["pos"][2]])
                                                     for s in ribs_y(12, 24, 310, [4, 12, 4])],
             explode=[0, 45, 0]),
        part("pump", T("Washer pump", "Bơm nước rửa kính"), BLACK,
             [cyl(15, 70, (-55, 15, 75)), rb([22, 16, 18], 3, (-55, 58, 75)), cyl(3.5, 16, (-55, -28, 75))],
             explode=[0, 0, 45],
             check=T("Humming but no spray = blocked hose or nozzle. Silence = fuse, connector, switch or pump.",
                     "Nghe tiếng bơm nhưng không phun = nghẹt ống hoặc vòi. Không có tiếng = kiểm tra cầu chì, giắc, công tắc hoặc bơm.")),
        part("grommet", T("Pump grommet", "Gioăng bơm"), RUBBER, [tor(10, 3, (-55, 0, 61), axis="z")], explode=[0, 0, 25]),
        part("hose", T("Washer hose", "Ống nước rửa kính"), RUBBER,
             [tube([(-55, -36, 75), (-55, -60, 75), (-20, -90, 95), (30, -100, 115)], 4)], explode=[0, 0, 45]),
    ]
    return parts, [30, 20]


@model
def power_steering_reservoir():
    parts = [
        part("tank", T("Reservoir", "Bình chứa"), TANK,
             [cyl(45, 100, (0, 50, 0), chamfer=4), cyl(7, 30, (0, -10, 0)), cyl(6, 30, (55, 70, 0), axis="x"),
              rb([12, 70, 40], 3, (-52, 40, 0), **STEEL)],
             check=T("Wet hoses or clamps = leak. Whining when turning = low fluid or air in the system.",
                     "Ống hoặc cổ dê bị ướt = rò rỉ. Kêu rít khi đánh lái = thiếu dầu hoặc lọt khí.")),
        part("fluid", T("Power steering fluid", "Dầu trợ lực lái"), dict(color="#B3261E", roughness=0.15),
             [cyl(41, 62, (0, 33, 0))],
             check=T("Foamy or dark fluid = air or worn fluid. Use only the fluid type specified for the vehicle.",
                     "Dầu có bọt hoặc sẫm màu = lọt khí hoặc dầu đã cũ. Chỉ dùng đúng loại dầu nhà sản xuất quy định.")),
        part("level_marks", T("HOT / COLD marks", "Vạch HOT / COLD"), DARK_MARK,
             [rb([16, 2, 1.2], 0, (0, 75, 45.3)), rb([16, 2, 1.2], 0, (0, 55, 45.3))]),
        part("cap", T("Cap with dipstick", "Nắp có que thăm"), BLACK,
             [cyl(32, 18, (0, 109, 0), chamfer=2), rb([10, 50, 2], 0.5, (0, 75, 0), **STEEL)] + ribs_y(16, 32, 109, [4, 14, 4]),
             explode=[0, 80, 0],
             check=T("Read the level on the cap dipstick, engine off: HOT range when warm, COLD range when cold.",
                     "Đọc mức dầu trên que thăm ở nắp khi tắt máy: vùng HOT khi máy nóng, vùng COLD khi máy nguội.")),
        part("hoses", T("Supply and return hoses", "Ống cấp và ống hồi"), RUBBER,
             [tube([(0, -25, 0), (0, -45, 0), (-20, -70, 20), (-60, -90, 40)], 9),
              tube([(70, 70, 0), (95, 70, 0), (120, 50, 20), (150, 30, 40)], 8),
              tor(10, 1.8, (0, -20, 0), **ZINC), tor(9, 1.8, (78, 70, 0), axis="x", **ZINC)]),
    ]
    return parts, [30, 20]


# ------------------------------------------------------------------------------------------ lubrication
@model
def oil_filler_cap():
    can = [[-11, 2], [5, 2], [13, -5], [15, -4.5], [7, -7], [-6, -7], [-6, -9], [-9, -9], [-9, -7], [-11, -7]]
    parts = [
        part("valve_cover", T("Valve cover (filler neck)", "Nắp quy lát (cổ châm dầu)"), dict(color="#3A3D42", metallic=0.5, roughness=0.5),
             [rb([160, 14, 110], 6, (0, -7, 0)), cyl(25, 14, (0, 4, 0), hole=19)]),
        part("gasket", T("Cap gasket", "Gioăng nắp"), dict(color="#D35400", roughness=0.7), [tor(21.5, 2.3, (0, 12.5, 0))],
             explode=[0, 40, 0],
             check=T("A cracked or missing gasket leaks oil and lets unmetered air in (lean running, whistling).",
                     "Gioăng nứt hoặc mất gây rò dầu và lọt khí không qua cảm biến (hỗn hợp nghèo, tiếng rít).")),
        part("cap", T("Oil filler cap", "Nắp châm dầu"), YELLOW,
             [cyl(30, 12, (0, 21, 0), chamfer=1.5), rb([50, 14, 12], 5, (0, 32, 0), **BLACK),
              rb([8, 4, 6], 1, (19, 13, 0), **BLACK), rb([8, 4, 6], 1, (-19, 13, 0), **BLACK),
              ext(1.2, (0, 27.6, 18), points=can, **BLACK), sph(1.6, (16, 27.6, 22), **BLACK)],
             explode=[0, 70, 0],
             check=T("Milky, mayonnaise-like deposit under the cap = short trips or coolant in the oil. The cap must be closed and seated.",
                     "Cặn trắng đục như mayonnaise dưới nắp = xe chạy quãng ngắn hoặc nước lẫn vào dầu. Nắp phải đóng kín.")),
    ]
    return parts, [25, 35]


@model
def oil_dipstick():
    hatch = [rb([9.5, 0.9, 2], 0, (0, y, 0), rot=[0, 0, 45], **DARK_MARK) for y in (-30, -37, -44, -51)]
    up = [0, 200, 0]
    parts = [
        part("guide_tube", T("Dipstick tube", "Ống dẫn que thăm"), STEEL, [cyl(7.5, 150, (0, 45, 0), hole=5.5)]),
        part("handle", T("Handle", "Tay cầm"), YELLOW,
             [tor(20, 4.5, (0, 160, 0), axis="z"), cyl(9, 14, (0, 127, 0)), tor(7.6, 1.8, (0, 121, 0), **RUBBER)],
             explode=up,
             check=T("Push it fully home after reading; a loose dipstick can leak oil vapour.",
                     "Cắm lại hết cỡ sau khi đọc; que thăm cắm lỏng có thể làm hơi dầu thoát ra.")),
        part("blade", T("Dipstick blade", "Thân que thăm"), STEEL, [rb([5, 150, 1.2], 0, (0, 45, 0))], explode=up),
        part("level_marks", T("Level marks (MIN–MAX)", "Vạch mức dầu (MIN–MAX)"), STEEL,
             [rb([7, 40, 1.6], 0.4, (0, -40, 0)), rb([8.2, 1.6, 2.2], 0, (0, -24, 0), **DARK_MARK),
              rb([8.2, 1.6, 2.2], 0, (0, -56, 0), **DARK_MARK)] + hatch,
             explode=up,
             check=T("Car level, engine warm and off for ~5 min: pull, wipe, re-insert fully, read. Oil must be between MIN and MAX. Black and gritty, milky or fuel-smelling oil = investigate.",
                     "Xe đỗ phẳng, máy ấm và tắt ~5 phút: rút ra, lau, cắm lại hết cỡ, rút ra đọc. Dầu phải nằm giữa MIN và MAX. Dầu đen có sạn, trắng đục hoặc mùi xăng = cần kiểm tra thêm.")),
    ]
    return parts, [30, 10]


@model
def oil_filter():
    flutes = around(16, 38.2, lambda x, y, z, a: rb([2.4, 18, 4], 0.8, (x, 14, z), rot=[0, a, 0]))
    drains = around(6, 22, lambda x, y, z, a: cyl(3.2, 0.6, (x, 102.2, z), color=HOLE))
    down = [0, -70, 0]
    parts = [
        part("adapter", T("Filter mount (engine)", "Đế lọc (trên động cơ)"), CAST,
             [cyl(44, 14, (0, 118, 0), chamfer=2), cyl(8, 20, (0, 108, 0), **STEEL)],
             check=T("Wipe clean before fitting; check for leaks after the first start.",
                     "Lau sạch trước khi lắp; kiểm tra rò rỉ sau lần nổ máy đầu tiên.")),
        part("canister", T("Filter canister", "Vỏ lọc dầu"), dict(color="#1F5FAF", metallic=0.4, roughness=0.35),
             [lathe([[0, 0], [30, 0], [36, 3], [38, 8], [38, 96], [37, 99], [0, 99]])] + flutes,
             explode=down,
             check=T("Dents or oil wetness at the seal. Tighten by hand (about 3/4 turn after the gasket touches).",
                     "Vỏ móp hoặc ướt dầu ở mép gioăng. Siết bằng tay (khoảng 3/4 vòng sau khi gioăng chạm đế).")),
        part("base_plate", T("Base plate (threaded)", "Đế ren"), STEEL,
             [cyl(37, 3, (0, 100.5, 0)), cyl(11, 10, (0, 104, 0), hole=8.5)] + drains, explode=down),
        part("gasket", T("Filter gasket", "Gioăng lọc dầu"), RUBBER, [tor(31, 2.5, (0, 103, 0))], explode=[0, -35, 0],
             check=T("The old gasket must not stay stuck on the engine (double gasket = big leak). Oil the new gasket.",
                     "Gioăng cũ không được dính lại trên đế (hai lớp gioăng = rò dầu nặng). Bôi dầu lên gioăng mới.")),
    ]
    return parts, [30, 15]


# ------------------------------------------------------------------------------------------ air intake & metering
@model
def air_filter_box():
    pleats = [rb([4, 8, 190], 1.5, (x, 143, 0), color="#D88E25") for x in range(-140, 141, 10)]
    clips = [(x, z) for x in (-100, 100) for z in (-113, 113)]
    parts = [
        part("lower_housing", T("Lower housing", "Vỏ dưới"), BLACK,
             [rb([320, 110, 220], 14, (0, 55, 0)), rb([110, 60, 80], 20, (-200, 60, 0))],
             check=T("Leaves, debris or water in the box; the inlet snorkel must be clear.",
                     "Lá cây, rác hoặc nước trong hộp; miệng hút gió phải thông thoáng.")),
        part("filter_element", T("Air filter element", "Lõi lọc gió"), dict(color="#E9A23B", roughness=0.8),
             [rb([296, 40, 196], 4, (0, 120, 0)), rb([306, 6, 206], 3, (0, 104, 0), **RUBBER)] + pleats,
             explode=[0, 65, 0],
             check=T("Hold it up to the light: clogged, oily or wet = replace. A dirty filter reduces power.",
                     "Soi lõi lọc ra ánh sáng: nghẹt, dính dầu hoặc ướt = thay. Lọc bẩn làm giảm công suất.")),
        part("upper_lid", T("Lid with outlet", "Nắp hộp và cổ ra"), BLACK,
             [rb([326, 70, 226], 16, (0, 150, 0)), cyl(42, 50, (190, 160, 0), axis="x"), tor(42, 3, (214, 160, 0), axis="x")],
             explode=[0, 130, 0]),
        part("clips", T("Lid clips", "Khoá cài nắp"), ZINC,
             [rb([22, 40, 5], 2, (x, 112, z)) for x, z in clips],
             check=T("All clips closed and the lid seated: an open clip lets unmetered air in (MAF codes).",
                     "Tất cả khoá cài đóng, nắp khít: khoá hở làm lọt khí không qua cảm biến (mã lỗi MAF).")),
    ]
    return parts, [30, 30]


@model
def air_intake_duct():
    path = [(-210, 0, 0), (-150, 0, 0), (-90, 10, 15), (-20, 50, 35), (50, 85, 20), (120, 100, 0), (200, 100, 0)]
    bellows = lambda t: 36 + (3.5 * math.sin((t - 0.32) * 2 * math.pi * 30) if 0.32 < t < 0.68 else 0.0)  # noqa: E731

    def clamp(x, y):
        return [cyl(39.5, 14, (x, y, 0), axis="x", hole=37), rb([14, 12, 16], 2, (x, y + 44, 0)), cyl(3, 22, (x, y + 44, 0), axis="z")]

    parts = [
        part("duct", T("Intake duct (bellows)", "Ống hút gió (đoạn xếp)"), dict(color="#1A1B1E", roughness=0.8),
             [tube(path, bellows, sides=32, samples=40)],
             check=T("Flex the folds: cracks let unmetered air in (lean codes P0171 / P0174, rough idle).",
                     "Uốn thử các nếp gấp: vết nứt làm lọt khí không qua cảm biến (mã P0171 / P0174, không tải rung).")),
        part("clamp_filter", T("Clamp (filter side)", "Cổ dê (phía lọc gió)"), ZINC, clamp(-195, 0), explode=[40, 0, 0],
             check=T("Tight, duct fully seated on the outlet.", "Siết chặt, ống cắm hết vào cổ ra.")),
        part("clamp_throttle", T("Clamp (throttle side)", "Cổ dê (phía cổ họng ga)"), ZINC, clamp(185, 100), explode=[-40, 0, 0],
             check=T("Tight, duct fully seated on the throttle body.", "Siết chặt, ống cắm hết vào cổ họng ga.")),
        part("breather_port", T("Breather (PCV) port", "Cổ thông hơi (PCV)"), dict(color="#1A1B1E", roughness=0.8),
             [cyl(7, 34, (-150, 0, 40), axis="z")],
             check=T("The breather hose must be attached and not cracked.", "Ống thông hơi phải được cắm và không nứt.")),
    ]
    return parts, [25, 30]


@model
def maf_sensor():
    up = [0, 60, 0]
    parts = [
        part("housing", T("Sensor housing (tube)", "Thân cảm biến (đoạn ống)"), dict(color="#26292D", roughness=0.55),
             [cyl(38, 90, (0, 0, 0), axis="x", hole=33), cyl(43, 8, (-45, 0, 0), axis="x", hole=33),
              cyl(43, 8, (45, 0, 0), axis="x", hole=33), rb([60, 10, 30], 3, (0, 39, 0))],
             check=T("Cracks or loose clamps downstream of the sensor let unmetered air in.",
                     "Nứt hoặc cổ dê lỏng phía sau cảm biến làm lọt khí không qua cảm biến.")),
        part("sensor_body", T("Plug-in sensor", "Cụm cảm biến cắm"), BLACK,
             [rb([64, 4, 30], 2, (0, 44, 0)), rb([44, 26, 30], 4, (0, 57, 0))], explode=up),
        part("sensing_element", T("Sensing element (hot film)", "Phần tử đo (màng nhiệt)"), dict(color="#3A3D42", roughness=0.5),
             [rb([20, 30, 6], 2, (0, 25, 0)), rb([6, 8, 1.5], 0.3, (0, 16, 3.5), color="#D9D3C3", metallic=0.6)],
             explode=up,
             check=T("Never touch it; clean only with MAF cleaner spray. Dirt causes lean running, hesitation, P0101.",
                     "Không chạm tay; chỉ vệ sinh bằng dung dịch xịt chuyên dụng cho MAF. Bẩn gây hỗn hợp nghèo, hụt ga, mã P0101.")),
        part("screws", T("Screws", "Vít"), STEEL, [cyl(3.5, 4, (25, 47, 0)), cyl(3.5, 4, (-25, 47, 0))], explode=[0, 90, 0]),
        part("connector", T("Connector", "Giắc cắm"), BLACK, [rb([30, 22, 26], 3, (0, 81, 0)), rb([12, 4, 14], 1, (0, 94, 0))],
             explode=[0, 110, 0],
             check=T("Corroded or loose pins; compare the MAF reading (g/s) with the expected value at idle.",
                     "Chân giắc ăn mòn hoặc lỏng; so sánh giá trị MAF (g/s) với giá trị chuẩn ở không tải.")),
    ]
    return parts, [30, 25]


@model
def throttle_body():
    sq = [[-48, -48, 10], [48, -48, 10], [-48, 48, 10], [48, 48, 10]]
    bolt_holes = [[x, y, 4.5] for x, y, _ in sq]
    parts = [
        part("body", T("Throttle body housing", "Thân cổ họng ga"), ALU,
             [cyl(46, 60, (0, 0, 0), axis="z", hole=34),
              ext(10, (0, 0, -30), axis="z", hull=sq, holes=[[0, 0, 34]] + bolt_holes),
              cyl(20, 80, (10, -62, 0), axis="x"), rb([70, 26, 40], 4, (10, -50, 0))]),
        part("throttle_plate", T("Throttle plate and shaft", "Bướm ga và trục"), BRASS,
             [cyl(33.5, 1.6, (0, 0, 0), axis="z", rot=[15, 0, 0]), cyl(3, 96, (0, 0, 0), axis="x", **STEEL)],
             check=T("A black carbon ring around the plate causes rough or high idle and stalling. Clean with the ignition off (electronic throttles can snap shut).",
                     "Vòng muội đen quanh bướm ga gây không tải rung hoặc cao, chết máy. Vệ sinh khi đã tắt khoá điện (bướm ga điện tử có thể đóng sập).")),
        part("motor_cover", T("Motor and sensor cover", "Nắp mô-tơ và cảm biến"), BLACK,
             [rb([24, 120, 70], 10, (60, -25, 0))]),
        part("connector", T("Connector (motor + TPS)", "Giắc (mô-tơ + cảm biến vị trí bướm ga)"), BLACK,
             [rb([20, 20, 28], 3, (82, -75, 0)), rb([6, 10, 12], 1, (93, -75, 0))], explode=[40, 0, 0],
             check=T("Corrosion or loose pins cause throttle position codes (P0121–P0123, P2135).",
                     "Giắc ăn mòn hoặc lỏng chân gây mã lỗi vị trí bướm ga (P0121–P0123, P2135).")),
        part("gasket", T("Flange gasket", "Gioăng mặt bích"), dict(color="#3A3A3A", roughness=0.8),
             [ext(1.5, (0, 0, -36), axis="z", hull=sq, holes=[[0, 0, 34]] + bolt_holes)], explode=[0, 0, -40],
             check=T("A leaking gasket lets unmetered air in (high idle, lean codes).", "Gioăng rò làm lọt khí (không tải cao, mã hỗn hợp nghèo).")),
        part("bolts", T("Mounting bolts", "Bu lông bắt"), STEEL, [hexa(6, 5, (x, -y, -22.5), axis="z") for x, y, _ in sq],
             explode=[0, 0, 30]),
    ]
    return parts, [35, 20]


@model
def intake_manifold():
    xs = (-150, -50, 50, 150)
    runners = [tube([(x, 0, 6), (x, 0, 40), (x * 0.95, 50, 110), (x * 0.9, 120, 140), (x * 0.9, 160, 150)], 24, sides=24) for x in xs]
    parts = [
        part("plenum", T("Plenum", "Buồng khí nạp"), BLACK, [rb([420, 100, 120], 40, (0, 175, 150))],
             check=T("Cracks in the plastic, loose bolts.", "Nứt vỏ nhựa, bu lông lỏng.")),
        part("runners", T("Intake runners", "Ống nạp nhánh"), BLACK, runners),
        part("throttle_inlet", T("Throttle body inlet", "Cổ lắp cổ họng ga"), BLACK,
             [cyl(42, 40, (-230, 175, 150), axis="x"), tor(42, 3, (-248, 175, 150), axis="x")]),
        part("head_flange", T("Cylinder-head flange", "Mặt bích bắt nắp máy"), CAST,
             [rb([440, 70, 12], 4, (0, 0, 0))] + [hexa(7, 6, (x, y, 9), axis="z", **STEEL) for x in (-200, -100, 0, 100, 200) for y in (-25, 25)]),
        part("gaskets", T("Port gaskets", "Gioăng cổ hút"), dict(color="#2E8B57", roughness=0.7),
             [tor(24, 3, (x, 0, -9), axis="z") for x in xs], explode=[0, 0, -40],
             check=T("Hissing at idle or lean codes: smoke-test, or carefully spray-test with the engine running.",
                     "Có tiếng xì ở không tải hoặc mã hỗn hợp nghèo: thử khói, hoặc xịt thử cẩn thận khi máy chạy.")),
        part("vacuum_ports", T("Vacuum ports", "Cổ chân không"), BLACK,
             [cyl(5, 22, (60, 234, 150)), cyl(5, 22, (110, 234, 150)), cyl(8, 26, (160, 236, 150))],
             check=T("Every vacuum hose attached, not cracked or collapsed (brake booster, PCV, EVAP).",
                     "Mọi ống chân không phải được cắm, không nứt, không bẹp (trợ lực phanh, PCV, EVAP).")),
    ]
    return parts, [25, 30]


@model
def map_sensor():
    parts = [
        part("manifold", T("Intake manifold (mounting face)", "Cổ hút (mặt lắp)"), CAST, [rb([90, 6, 60], 3, (0, -14, 0))]),
        part("body", T("Sensor body", "Thân cảm biến"), BLACK,
             [rb([42, 22, 32], 4, (0, 11, 0)), cyl(4, 16, (0, -6, 0)),
              ext(4, (0, 2, 0), hull=[[0, 0, 12], [32, 0, 8]], holes=[[32, 0, 3.2]])],
             check=T("Key on, engine off: the MAP reading should be close to barometric pressure. A blocked port gives wrong readings.",
                     "Bật khoá, chưa nổ máy: giá trị MAP phải gần bằng áp suất khí quyển. Cổ đo bị nghẹt cho số đo sai.")),
        part("o_ring", T("O-ring", "Gioăng O"), RUBBER, [tor(4.3, 1.1, (0, -4, 0))], explode=[0, -20, 0],
             check=T("A torn O-ring leaks vacuum.", "Gioăng O rách làm rò chân không.")),
        part("screw", T("Screw", "Vít"), STEEL, [hexa(5, 3, (32, 5.5, 0)), cyl(2.5, 12, (32, -2, 0))], explode=[0, 30, 0]),
        part("connector", T("Connector", "Giắc cắm"), BLACK, [rb([18, 18, 24], 3, (-28, 12, 0))],
             check=T("Usually 3 wires: 5 V reference, signal, ground (typical). Check for corrosion.",
                     "Thường 3 dây: nguồn 5 V, tín hiệu, mass (điển hình). Kiểm tra ăn mòn.")),
    ]
    return parts, [35, 30]


@model
def pcv_valve():
    parts = [
        part("valve_cover", T("Valve cover", "Nắp quy lát"), dict(color="#3A3D42", metallic=0.5, roughness=0.5),
             [rb([140, 12, 100], 6, (0, -6, 0))]),
        part("grommet", T("Grommet", "Gioăng cao su"), RUBBER, [cyl(13, 10, (0, 4, 0), chamfer=1.5)], explode=[0, 20, 0],
             check=T("A hard or cracked grommet leaks oil and air.", "Gioăng chai cứng hoặc nứt gây rò dầu và lọt khí.")),
        part("valve", T("PCV valve", "Van PCV"), DARK,
             [lathe([[0, 0], [8, 0], [8, 18], [10, 20], [10, 26], [6, 28], [0, 28]], (0, 6, 0)), hexa(10.5, 5, (0, 30, 0)),
              lathe([[0, 34], [5, 34], [6, 40], [4.5, 41], [4.5, 50], [0, 50]])],
             explode=[0, 45, 0],
             check=T("Shake it with the engine off: a rattle means it moves freely. A stuck valve causes oil consumption or sludge.",
                     "Lắc van khi tắt máy: nghe lạch cạch là van còn hoạt động. Van kẹt gây hao dầu hoặc đóng cặn.")),
        part("hose", T("PCV hose", "Ống PCV"), RUBBER,
             [tube([(0, 48, 0), (0, 62, 0), (30, 85, 15), (90, 95, 40)], 8.5), tor(9.5, 1.6, (0, 46, 0), **ZINC)],
             explode=[0, 80, 0],
             check=T("A collapsed, cracked or soft hose causes whistling, a vacuum leak and lean codes (P0171).",
                     "Ống bẹp, nứt hoặc mềm gây tiếng huýt, rò chân không và mã hỗn hợp nghèo (P0171).")),
    ]
    return parts, [30, 30]


# ------------------------------------------------------------------------------------------ ignition / fuel / sensors
@model
def ignition_coil():
    boot = lathe([[11, -93]] + ridges(10.6, 12, -116, -96, 4) + [[10.5, -138], [9, -142], [0, -142]])
    parts = [
        part("head", T("Coil head", "Đầu bô-bin"), BLACK_GLOSS,
             [rb([56, 26, 40], 5, (0, 0, 0)), ext(5, (0, -8, 0), hull=[[-34, 0, 9], [-22, 0, 12]], holes=[[-34, 0, 4]])]),
        part("connector", T("Connector", "Giắc cắm"), DARK, [rb([22, 22, 26], 3, (38, 2, 0)), rb([10, 4, 14], 1, (40, 15, 0))],
             explode=[40, 0, 0],
             check=T("Locked and free of corrosion. Swap the coil with another cylinder: if the misfire follows it, the coil is faulty.",
                     "Giắc cài chặt, không ăn mòn. Đổi bô-bin sang xi-lanh khác: lỗi bỏ máy đi theo bô-bin thì bô-bin hỏng.")),
        part("bolt", T("Mounting bolt", "Bu lông bắt"), STEEL, [hexa(6.5, 5, (-34, -3, 0)), cyl(3.2, 20, (-34, -15, 0))],
             explode=[0, 40, 0]),
        part("body", T("Coil body", "Thân bô-bin"), BLACK_GLOSS, [cyl(11, 80, (0, -53, 0)), tor(12.5, 2, (0, -24, 0), **RUBBER)],
             check=T("Cracks or burn marks on the body. Switch the engine off before removing: ignition high voltage.",
                     "Thân nứt hoặc có vết cháy. Tắt máy trước khi tháo: điện cao áp đánh lửa.")),
        part("boot", T("Rubber boot", "Chụp cao su"), dict(color="#3A3E44", roughness=0.8), [boot], explode=[0, -40, 0],
             check=T("Cracks, thin black carbon tracks or oil in the spark plug well cause misfires.",
                     "Nứt, vệt muội đen mảnh hoặc dầu trong lỗ bugi gây bỏ máy.")),
        part("spring", T("Contact spring", "Lò xo tiếp điểm"), STEEL, [helix(3.2, 0.7, -150, -141, 4)], explode=[0, -60, 0]),
    ]
    return parts, [35, 15]


@model
def spark_plug():
    insul = [[0, 90], [4.8, 90]] + ridges(5.6, 6.4, 60, 88, 3) + [[7.2, 57], [7.2, 54], [0, 54]]
    thread = [[0, 39]] + ridges(6.3, 7.0, 21, 39, 1.3) + [[0, 21]]
    parts = [
        part("terminal", T("Terminal nut", "Đầu cực"), STEEL,
             [lathe([[0, 100], [3.8, 100], [4.2, 98], [4.2, 94], [2.6, 93], [2.6, 90], [0, 90]])]),
        part("insulator", T("Ceramic insulator", "Sứ cách điện"), CERAMIC, [lathe(insul)],
             check=T("Cracks or thin carbon tracks (flashover) cause misfires.", "Sứ nứt hoặc có vệt muội mảnh (phóng điện bề mặt) gây bỏ máy.")),
        part("shell", T("Shell and hex", "Thân và lục giác"), STEEL,
             [hexa(9.6, 12, (0, 47, 0)), lathe([[0, 53], [8.5, 53], [7.5, 56], [0, 56]]), cyl(7.4, 1.6, (0, 40.2, 0))]),
        part("gasket", T("Sealing washer", "Vòng đệm làm kín"), COPPER, [tor(8, 1.2, (0, 39.4, 0))]),
        part("thread", T("Thread", "Ren"), ZINC, [lathe(thread)],
             check=T("Thread it in by hand first and torque to spec: an aluminium head strips easily.",
                     "Vặn bằng tay trước, siết đúng lực: nắp máy nhôm dễ toét ren.")),
        part("insulator_tip", T("Insulator nose", "Đầu sứ"), CERAMIC, [lathe([[0, 21], [3, 21], [2.8, 15.5], [0, 15.5]])],
             check=T("Colour tells the story: light tan = normal, black soot = rich, oily = oil burning, white or blistered = too hot.",
                     "Màu đầu sứ: nâu nhạt = bình thường, muội đen = hỗn hợp giàu, ướt dầu = ăn dầu, trắng hoặc rộp = quá nóng.")),
        part("electrodes", T("Electrodes", "Điện cực"), dict(color="#9A9EA3", metallic=1.0, roughness=0.35),
             [cyl(1.1, 4, (0, 14, 0)), tube([(6.2, 21, 0), (6.2, 13, 0), (5, 11.2, 0), (1.5, 10.8, 0)], 1.1, sides=8, smooth=False)],
             check=T("Rounded, worn electrodes or a wide gap cause misfires and hard starting. Set the gap to the OEM spec.",
                     "Điện cực mòn tròn hoặc khe hở lớn gây bỏ máy, khó nổ. Chỉnh khe hở theo thông số nhà sản xuất.")),
    ]
    return parts, [30, 15]


@model
def fuel_injector():
    parts = [
        part("inlet", T("Fuel inlet and filter", "Cổ vào và lưới lọc"), STEEL,
             [cyl(6, 12, (0, 64, 0)), cyl(4.5, 1, (0, 70.5, 0), color=HOLE)]),
        part("o_ring_top", T("Upper O-ring", "Gioăng O trên"), dict(color="#5B3A1E", roughness=0.6), [tor(6.6, 1.6, (0, 62, 0))],
             explode=[0, 25, 0],
             check=T("Fuel smell or wetness at the O-rings = replace them. The fuel line is under pressure: relieve it first.",
                     "Mùi xăng hoặc ướt ở gioăng O = thay gioăng. Đường xăng có áp suất: xả áp trước khi tháo.")),
        part("body", T("Injector body", "Thân kim phun"), BLACK_GLOSS,
             [lathe([[0, 24], [8, 24], [9.5, 27], [9.5, 52], [7, 56], [6.5, 58], [0, 58]])]),
        part("connector", T("Connector", "Giắc cắm"), BLACK, [rb([16, 16, 14], 2, (13, 46, 0)), rb([6, 3, 10], 1, (17, 55, 0))],
             check=T("Coil resistance is typically 11–18 Ω for high-impedance injectors; listen for the click with a stethoscope.",
                     "Điện trở cuộn dây kim phun trở kháng cao thường 11–18 Ω; dùng ống nghe kiểm tra tiếng tách.")),
        part("metal_body", T("Valve body", "Thân van"), STEEL,
             [lathe([[0, 8], [6.5, 8], [8, 10], [8, 23], [0, 23]]),
              lathe([[0, -4], [3, -4], [5, -1], [5.2, 1], [5.2, 8], [0, 8]])]),
        part("o_ring_bottom", T("Lower O-ring", "Gioăng O dưới"), RUBBER, [tor(6.4, 1.6, (0, 4.5, 0))], explode=[0, -25, 0],
             check=T("A leaking lower O-ring lets air in (lean) and can leak fuel.", "Gioăng O dưới rò gây lọt khí (hỗn hợp nghèo) và có thể rò xăng.")),
        part("nozzle", T("Spray nozzle", "Đầu phun"), dict(color="#5A5E63", metallic=0.9, roughness=0.35),
             [cyl(3.2, 0.8, (0, -4.3, 0))] + around(4, 1.6, lambda x, y, z, a: cyl(0.5, 0.4, (x, -4.8, z), color=HOLE)),
             check=T("Clogged or dripping nozzles cause misfire or rich/lean running: balance or flow test.",
                     "Đầu phun nghẹt hoặc nhỏ giọt gây bỏ máy hoặc hỗn hợp giàu/nghèo: kiểm tra cân bằng hoặc lưu lượng.")),
    ]
    return parts, [35, 15]


@model
def cam_crank_sensor():
    parts = [
        part("body", T("Sensor body", "Thân cảm biến"), BLACK,
             [cyl(9, 36, (0, -18, 0)), lathe([[0, -40], [7.5, -40], [9, -37], [0, -37]]),
              ext(5, (0, 2.5, 0), hull=[[0, 0, 13], [26, 0, 8]], holes=[[26, 0, 3.4]])],
             check=T("Metal debris on the magnetic tip; the air gap to the wheel must be within spec.",
                     "Mạt kim loại bám ở đầu từ; khe hở đến bánh răng phải đúng thông số.")),
        part("o_ring", T("O-ring", "Gioăng O"), RUBBER, [tor(9.3, 1.4, (0, -6, 0))], explode=[0, -15, 0],
             check=T("Oil leak at the sensor seal.", "Rò dầu ở gioăng cảm biến.")),
        part("bolt", T("Mounting bolt", "Bu lông bắt"), STEEL, [hexa(6.5, 6, (26, 8, 0)), cyl(3, 16, (26, -3, 0))], explode=[0, 30, 0]),
        part("connector", T("Connector", "Giắc cắm"), BLACK, [rb([20, 24, 18], 3, (-3, 17, 0)), rb([8, 4, 12], 1, (-3, 30, 0))],
             check=T("Oil inside the connector, corroded pins. Check the signal with a scope while cranking.",
                     "Dầu lọt vào giắc, chân giắc ăn mòn. Kiểm tra tín hiệu bằng máy hiện sóng khi đề.")),
        part("tone_wheel", T("Trigger wheel (reluctor)", "Bánh răng tín hiệu"), STEEL,
             [ext(10, (0, -113.5, 0), axis="z", points=gear_points(66, 72, 36, missing=(9,))), cyl(30, 16, (0, -113.5, 0), axis="z")],
             check=T("Damaged or missing teeth (the gap is intentional) give wrong position signals.",
                     "Răng hỏng hoặc mất răng (khoảng trống là có chủ ý) làm sai tín hiệu vị trí.")),
    ]
    return parts, [30, 15]


# ------------------------------------------------------------------------------------------ emissions
@model
def evap_purge_valve():
    parts = [
        part("valve_body", T("Purge valve (solenoid)", "Van xả (van điện từ)"), BLACK_GLOSS, [cyl(17, 44, (0, 0, 0), chamfer=2)],
             check=T("It must be closed with the engine off: blow through the canister side – no air should pass.",
                     "Van phải đóng khi tắt máy: thổi vào phía bình than – không được có khí đi qua.")),
        part("connector", T("Connector", "Giắc cắm"), DARK, [rb([18, 16, 20], 3, (0, 30, 0))], explode=[0, 30, 0]),
        part("ports", T("Hose ports", "Cổ nối ống"), BLACK,
             [cyl(4.2, 30, (-28, -10, 0), axis="x"), tor(4.6, 0.9, (-38, -10, 0), axis="x"),
              cyl(4.2, 30, (28, -10, 0), axis="x"), tor(4.6, 0.9, (38, -10, 0), axis="x")]),
        part("bracket", T("Bracket", "Pát bắt"), STEEL, [ext(3, (0, -22, 0), hull=[[0, 0, 14], [0, 30, 8]], holes=[[0, 30, 3.5]])]),
        part("hose_canister", T("Hose to the charcoal canister", "Ống từ bình than hoạt tính"), RUBBER,
             [tube([(-43, -10, 0), (-60, -10, 0), (-80, -20, 20), (-100, -40, 30)], 6.5)], explode=[-30, 0, 0],
             check=T("Cracked or disconnected hoses cause EVAP leak codes.", "Ống nứt hoặc tuột gây mã lỗi rò EVAP.")),
        part("hose_intake", T("Hose to the intake (vacuum)", "Ống về đường nạp (chân không)"), RUBBER,
             [tube([(43, -10, 0), (60, -10, 0), (85, 0, -20), (105, 10, -40)], 6.5)], explode=[30, 0, 0],
             check=T("A valve stuck open feeds fuel vapour at idle: rough idle and hard starting after refuelling.",
                     "Van kẹt mở làm hơi xăng vào khi không tải: không tải rung, khó nổ máy sau khi đổ xăng.")),
    ]
    return parts, [30, 25]


@model
def gas_cap():
    up = [0, 60, 0]
    parts = [
        part("filler_neck", T("Filler neck", "Cổ bình xăng"), STEEL, [cyl(30, 34, (0, -20, 0), hole=26), tor(30, 2, (0, -3, 0))],
             check=T("Dirt or rust on the sealing lip prevents a tight seal.", "Bụi hoặc gỉ trên mép làm kín khiến nắp không kín.")),
        part("seal", T("Cap seal", "Gioăng nắp"), RUBBER, [tor(26.5, 2.6, (0, -1, 0))], explode=[0, 30, 0],
             check=T("A cracked or flattened seal triggers EVAP leak codes (P0455, P0456, P0457).",
                     "Gioăng nứt hoặc bẹp gây mã lỗi rò EVAP (P0455, P0456, P0457).")),
        part("cap", T("Fuel cap", "Nắp bình xăng"), DARK,
             [cyl(34, 16, (0, 8, 0), chamfer=2), rb([60, 16, 12], 5, (0, 22, 0), **BLACK),
              lathe([[0, -1]] + ridges(22, 24, -22, -1, 3) + [[0, -22]], **BLACK)],
             explode=up,
             check=T("Tighten until it clicks. After fixing a loose cap the code clears only after several drive cycles.",
                     "Vặn đến khi nghe tiếng tách. Sau khi siết lại nắp, mã lỗi chỉ tự xoá sau vài chu kỳ vận hành.")),
        part("tether", T("Tether", "Dây giữ nắp"), BLACK,
             [tube([(33, 6, 0), (48, 0, 0), (55, -25, 10), (45, -50, 15), (30, -60, 5)], 2.2)], explode=up),
    ]
    return parts, [30, 30]


@model
def egr_valve():
    flange = dict(hull=[[-38, 0, 10], [38, 0, 10], [0, 0, 30]])
    parts = [
        part("actuator", T("Actuator", "Bộ chấp hành"), dict(color="#2E3135", metallic=0.6, roughness=0.45),
             [lathe([[0, 18], [34, 18], [34, 48], [28, 58], [0, 60]])]),
        part("connector", T("Connector", "Giắc cắm"), BLACK, [rb([22, 20, 20], 3, (34, 50, 0))], explode=[30, 0, 0]),
        part("base", T("Valve base", "Đế van"), CAST,
             [ext(12, (0, 12, 0), holes=[[-38, 0, 5], [38, 0, 5]], **flange), cyl(14, 20, (0, -2, 0), hole=10)]),
        part("gasket", T("Gasket", "Gioăng"), dict(color="#4A4A4A", metallic=0.5, roughness=0.5),
             [ext(1.5, (0, 5, 0), holes=[[-38, 0, 5], [38, 0, 5], [0, 0, 11]], **flange)], explode=[0, -30, 0],
             check=T("Exhaust leaks at the flange; fit a new gasket whenever the valve is removed.",
                     "Rò khí xả ở mặt bích; thay gioăng mới mỗi lần tháo van.")),
        part("poppet", T("Valve stem and poppet", "Ty van và đĩa van"), dict(color="#3B2F2A", metallic=0.3, roughness=0.8),
             [cyl(3, 26, (0, -6, 0)), lathe([[0, -24], [12, -22], [13, -20], [4, -16], [0, -16]])], explode=[0, -55, 0],
             check=T("Carbon build-up holds the valve open: rough idle and stalling. Clean or replace.",
                     "Muội than làm van kẹt mở: không tải rung, chết máy. Vệ sinh hoặc thay.")),
        part("bolts", T("Bolts", "Bu lông"), STEEL, [hexa(7, 6, (-38, 21, 0)), hexa(7, 6, (38, 21, 0))], explode=[0, 40, 0]),
    ]
    return parts, [30, 20]


@model
def o2_sensor():
    slots = around(6, 7.05, lambda x, y, z, a: rb([1.2, 7, 3], 0.3, (x, 3, z), rot=[0, a, 0], color=HOLE))
    parts = [
        part("shield", T("Protective shield (sensing tip)", "Ống bảo vệ (đầu đo)"), STEEL, [cyl(7, 18, (0, 3, 0))] + slots,
             check=T("Tip deposits: white = coolant or silicone, black soot = rich mixture, oily = oil burning.",
                     "Cặn ở đầu đo: trắng = nước làm mát hoặc silicon, muội đen = hỗn hợp giàu, ướt dầu = ăn dầu.")),
        part("thread", T("Thread", "Ren"), ZINC, [lathe([[0, 23]] + ridges(8.4, 9, 12, 23, 1.5) + [[0, 12]])],
             check=T("New sensors usually have anti-seize already applied. The exhaust is hot: let it cool first.",
                     "Cảm biến mới thường đã bôi sẵn chất chống kẹt ren. Ống xả nóng: chờ nguội trước khi tháo.")),
        part("washer", T("Sealing washer", "Vòng đệm"), ZINC, [tor(10, 1.3, (0, 24.6, 0))]),
        part("hex", T("Hex body", "Thân lục giác"), STEEL,
             [hexa(12.7, 10, (0, 31, 0)), lathe([[0, 36], [8, 36], [8, 44], [6, 46], [6, 56], [4.5, 58], [0, 58]])]),
        part("wire", T("Sensor wire", "Dây cảm biến"), dict(color="#2B2D30", roughness=0.6),
             [cyl(4.8, 6, (0, 59, 0), **RUBBER), tube([(0, 60, 0), (0, 80, 0), (10, 105, 0), (40, 125, 0), (75, 130, 0)], 2.6)],
             check=T("Melted on the exhaust or chafed wire: heater circuit codes (P0135 etc.).",
                     "Dây chảy do chạm ống xả hoặc bị cọ sờn: mã lỗi mạch sấy (P0135...).")),
        part("connector", T("Connector", "Giắc cắm"), BLACK, [rb([22, 14, 16], 3, (88, 130, 0))], explode=[30, 0, 0]),
    ]
    return parts, [35, 15]


# ------------------------------------------------------------------------------------------ not in the knowledge table
@model
def engine_cover():
    ribs = [rb([14, 6, 220], 3, (x, 58, 0)) for x in range(-160, 161, 40)]
    parts = [
        part("cover", T("Engine cover", "Nắp che động cơ"), dict(color="#1E2023", roughness=0.5),
             [rb([560, 40, 380], 40, (0, 20, 0)), rb([380, 24, 240], 16, (0, 44, 0), color="#2A2D31")] + ribs,
             explode=[0, 80, 0],
             check=T("Lift it off to look underneath: oil leaks, rodent nests, chewed wires, loose connectors.",
                     "Nhấc nắp để kiểm tra bên dưới: rò dầu, tổ chuột, dây điện bị cắn, giắc lỏng.")),
        part("oil_cap_opening", T("Oil filler cap opening", "Lỗ nắp châm dầu"), YELLOW,
             [cyl(36, 10, (200, 41, 110), hole=28, **BLACK), cyl(27, 14, (200, 44, 110))],
             check=T("The filler cap stays on the engine; remove it only to top up oil.", "Nắp châm dầu nằm trên động cơ; chỉ mở khi châm dầu.")),
        part("grommets", T("Mounting grommets", "Chân cao su bắt nắp"), RUBBER,
             [cyl(14, 16, (sx * 220, -8, sz * 140)) for sx in (-1, 1) for sz in (-1, 1)], explode=[0, 80, 0],
             check=T("Pull straight up at the grommets; do not pry the edges (old plastic is brittle).",
                     "Kéo thẳng lên tại vị trí chân cao su; không bẩy ở mép (nhựa cũ giòn, dễ gãy).")),
    ]
    return parts, [30, 35]



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated component keys")
    ap.add_argument("--out", type=Path, default=OUT, help="where the JSON specs go (the app builds the meshes)")
    ap.add_argument("--glb", type=Path, help="also write <component>.glb here (other viewers, the 3D video skill)")
    ap.add_argument("--preview", type=Path, help="write a 6-view PNG per model into this folder (needs --glb)")
    args = ap.parse_args()
    knowledge = yaml.safe_load(KNOWLEDGE.read_text(encoding="utf-8"))
    keys = list(knowledge["components"]) + EXTRA_CLASSES
    if args.only:
        keys = [k for k in args.only.split(",") if k]
    missing = [k for k in keys if k not in MODELS]
    if missing:
        raise SystemExit(f"no model for: {missing}")
    args.out.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURE
    stats = json.loads(fixture_path.read_text(encoding="utf-8")) if fixture_path.is_file() else {}
    for key in keys:
        parts, view = MODELS[key]()
        names = [p["name"] for p in parts]
        if len(set(names)) != len(names):
            raise SystemExit(f"{key}: duplicate part names")
        for p in parts:
            for t in [p["label"]] + ([p["check"]] if p.get("check") else []):
                if not (t.get("en") and t.get("vi")):
                    raise SystemExit(f"{key}.{p['name']}: text missing a language")
        spec = {"parts": parts, "extras": {"component": key, "view": view, "unit": "mm",
                                           "note": "generic illustration, not a specific vehicle's part"}}
        path = args.out / f"{key}.json"
        path.write_text(json.dumps(spec_json(spec), ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        stats[key] = {p["name"]: part_stats(p) for p in parts}
        if key == "battery":  # GLB reader test of the app (ComponentModelsTest.glbReaderMatchesPython)
            FIXTURE.parent.mkdir(parents=True, exist_ok=True)
            write_glb(FIXTURE.parent / "battery.glb", spec)
        ntri = sum(v["triangles"] for v in stats[key].values())
        print(f"{key:26s} {len(parts):2d} parts {ntri:6d} triangles  spec {path.stat().st_size / 1024:5.1f} KB")
        if args.glb:
            args.glb.mkdir(parents=True, exist_ok=True)
            glb = args.glb / f"{key}.glb"
            write_glb(glb, spec)
            if args.preview:
                from inspect3d import views_sheet
                from model3d import Model3D
                args.preview.mkdir(parents=True, exist_ok=True)
                views_sheet(Model3D(str(glb)), str(args.preview / f"{key}.png"), view=view)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(json.dumps(stats, indent=0) + "\n", encoding="utf-8")
    print(f"wrote {fixture_path.relative_to(ROOT)} ({len(stats)} models)")


if __name__ == "__main__":
    main()
