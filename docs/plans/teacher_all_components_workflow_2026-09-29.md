# Workflow: teacher phát hiện mọi linh kiện nhìn thấy, phân theo hệ thống (2026-09-29)

**Mục tiêu:** teacher (chạy server) nhận diện **mọi linh kiện nhìn thấy được** trong ảnh khoang máy, gom theo hệ thống. Student học lại từ teacher để deploy.

**Nguồn ý tưởng** (xem `ref_llm_approaches_review_2026-09-29.md`):
- **[C]** Claude: VLM trả lời "là gì", hình học trả lời "ở đâu"; lớp dễ nhầm (confusers); chia tầng tin cậy; cờ `is_region`; prior vị trí; active learning.
- **[G]** ChatGPT: taxonomy phân tầng; class chung trước, gọi tên chi tiết bằng truy hồi embedding; chia ô; đồng thuận nhiều tín hiệu; hard negative; skeleton; chỉ số CCR và provenance.
- **[Ge]** Gemini: phân rã theo cụm chức năng; tên song ngữ + mô tả + cảnh báo cho từng linh kiện; viewer JSON → HTML.
- **[P]** Những gì dự án đã chứng minh:
  - KD (+1..3 điểm) và trung bình checkpoint (+3..6 điểm);
  - split theo xe;
  - SAM2 vẽ mask từ box;
  - skill review chuyên gia;
  - ngưỡng theo class từ CV;
  - chia ô khi inference (recall vật nhỏ 0,39 → 0,58).

**Định nghĩa "mọi linh kiện":** mọi vật **nhìn thấy được** thuộc taxonomy v2 (mục 1). Bộ phận bị che không tính là "phát hiện", mà ghi vào `expected_not_visible` theo loại động cơ (skill automotive-expert). Nếu không định nghĩa như vậy thì không đo được.

## 0. Hiện trạng dữ liệu (đếm 2026-09-29)

- Nhãn review theo ontology 36 class (530 ảnh review gốc + 382 ảnh hybrid):
  - class train hiện tại có từ vài chục đến vài trăm instance;
  - `exhaust_manifold_heat_shield` có **68**, đủ đưa vào train ngay;
  - `power_steering_reservoir` 24, `hood_latch_mechanism` 24, `brake_booster` 17, `intercooler_piping` 14, `strut_tower_brace` 12;
  - còn lại dưới 10.
- Class ngoài ontology 36 (van EVAP, MAP, O2, EGR, thermostat, fuel rail, kim phun, bó dây, giắc, ống chân không…): **0 nhãn**.
- Hệ quả: nút thắt là **tạo nhãn mới có kiểm soát chất lượng**, không phải kiến trúc model.

## 1. Taxonomy v2: 3 tầng

```
Hệ thống (system, 13)  ─►  Nhóm hình dạng (group)  ─►  Class chi tiết (fine)
```

| Hệ thống | Class chi tiết (tier A = train ngay; B = train khi đủ nhãn) | Class chung (tier C: bắt mọi thứ còn lại của hệ) |
|---|---|---|
| Điện & khởi động | A: battery, battery_terminal, fuse_relay_box, ecu_module, alternator · B: wiring_harness, starter_motor, jump_start_post | `other_electrical_module` |
| Đánh lửa | A: ignition_coil · B: coil_pack, spark_plug_wire | – |
| Nạp khí | A: air_filter_box, air_intake_duct, maf_sensor, throttle_body, intake_manifold · B: map_sensor, intake_resonator | `other_intake_part` |
| Tăng áp | B: turbocharger, intercooler_piping | – |
| Nhiên liệu & EVAP | B: fuel_rail, fuel_injector, evap_purge_valve, fuel_supply_line | `other_fuel_part` |
| Khí xả & khí thải | A: exhaust_manifold_heat_shield (68 nhãn) · B: o2_sensor, egr_valve, exhaust_manifold | – |
| Làm mát | A: coolant_reservoir, radiator_cap, radiator_hose · B: thermostat_housing, heater_hose, radiator, radiator_cooling_fan | – |
| Bôi trơn | A: oil_filler_cap, oil_dipstick, oil_filter · B: valve_cover, pcv_valve / breather_hose | – |
| Truyền động phụ | B: serpentine_belt, ac_compressor, belt_tensioner | – |
| Phanh | A: brake_fluid_reservoir · B: brake_booster, abs_modulator_unit, brake_master_cylinder | – |
| Lái | B: power_steering_reservoir | – |
| Rửa kính | A: washer_fluid_reservoir | – |
| Vỏ & tham chiếu | A: engine_cover · B: strut_tower, hood_latch_mechanism, strut_tower_brace | – |
| (Dụng cụ) | A: multimeter_diagnostic_tool | – |

**Class chung theo hình dạng** (tier C, học được vì gộp nhiều loại lại):

| Class chung | Bao gồm |
|---|---|
| `other_reservoir` | bình PS, bình dầu côn, bình nước làm mát inverter… |
| `other_sensor_actuator` | cảm biến hoặc van nhỏ có giắc, chưa có class riêng |
| `other_hose_line` | ống, đường dẫn chưa có class riêng |
| `other_cap_plug` | nắp, nút chưa có class riêng |
| `other_module_box` | hộp, module chưa có class riêng |
| `other_pulley_device` | puly, thiết bị trên dây curoa chưa có class riêng |
| `electrical_connector` | giắc điện |

Quy tắc:
- Class chi tiết được "thăng hạng" từ tier B lên A khi có ≥ 60 instance review, đến từ ≥ 8 xe. Trước đó, instance của nó được **train dưới tên class chung** của nhóm hình dạng.
- Class tier A chỉ bị hạ xuống khi còn dưới 40 instance (có trễ, để không dao động giữa hai tier). Ngưỡng nằm trong `configs/taxonomy_v2.yaml` (`promotion`).
- Tên chi tiết vẫn lưu trong nhãn (`fine_name`), nên không mất thông tin.
- Hệ thống lấy từ bảng ánh xạ `fine → system`. Với class chung, hệ thống xác định ở bước gọi tên (mục 4).
- Mỗi class có thẻ tri thức **[Ge]**: tên VI/EN, cue nhận dạng, lớp dễ nhầm và cách phân biệt **[C]**, vị trí thường gặp. Nguồn: `component_catalog.md` của skill automotive-expert.
- Chọn class tier B theo giá trị chẩn đoán: tần suất DTC trong dữ liệu Innova, không theo độ dễ nhìn.

## 2. Workflow tổng thể

```
 W0 Taxonomy v2 + thẻ tri thức ──────────────────────────────────────────────┐
                                                                              │
 W1 GÁN NHÃN MỞ RỘNG (cho 1.081 ảnh đã có + xe mới)                            │
   ảnh gốc 6000×4000                                                          │
    ├─ a. Bối cảnh: YMME (metadata Request / chữ trên nắp máy) → loại động cơ  │
    │     → danh sách "nên có" + "bị che"                          [C][skill]  │
    ├─ b. Đề xuất vùng: teacher hiện tại + SAM2 "segment everything"            │
    │     trên ô 3×3 ở độ phân giải gốc                            [G][P]      │
    ├─ c. Gọi tên: VLM (DeepSeek, thinking OFF) xem ảnh có đánh số vùng         │
    │     (Set-of-Mark), bắt buộc ≥ 1 cue + phân biệt confusers      [C]        │
    ├─ d. Truy hồi DINOv2 trên crop → tên gần nhất trong gallery   [G]         │
    ├─ e. Điểm tin cậy = f(VLM, teacher, retrieval, chất lượng mask, prior vị trí)│
    │     f học bằng hồi quy logistic trên verdict review cũ       [C][G][P]   │
    ├─ f. Chia tầng: chắc → duyệt nhanh; cao/ước đoán → skill review chuyên gia │
    │     is_region → không train                                  [C]         │
    └─ g. Chỉ xuất nhãn approved + provenance (nguồn, model, người duyệt) [C][G]│
                                                                              │
 W2 BUILD DATASET v10 (versioned; split theo xe; gộp nhóm trùng)     [P]       │
   + tile 800 px cắt từ ảnh 1600/gốc cho vật nhỏ (E3)               [G][P]     │
                                                                              │
 W3 TRAIN TEACHER (DGX): yolo11l-seg, hoặc yolo26l-seg nếu thắng A/B           │
   640, 100 epoch, save_period 5, avg5; cân bằng class (repeat-factor, cls_pw) │
   2 seed; chọn theo CV                                              [P]       │
                                                                              │
 W4 "TEACHER SYSTEM" khi inference (server, offline)                           │
   ảnh ─► teacher (ảnh toàn cảnh + 2×2 ô) ─► gộp WBF/NMS ─► SAM2 vẽ lại mask   │
        ─► truy hồi DINOv2 chỉ để đặt tên class chung other_* ─► luật ngữ cảnh   │
          (prior vị trí, số lượng tối đa, quan hệ cha-con) ─► ngưỡng theo class │
        ─► JSON gom theo hệ thống ─► viewer HTML              [G][C][Ge][P]    │
                                                                              │
 W5 VÒNG CẢI TIẾN                                                              │
   • active learning: xếp ảnh chưa nhãn theo bất đồng VLM ↔ teacher system  [C][G]│
   • hard negative: bu-lông ↔ nút nhựa, giắc ↔ tai kẹp, ống ↔ vỏ bó dây     [G]  │
   • ưu tiên xe/dòng chưa có (mục tiêu B)                                        │
   └──────────────► quay lại W1 ◄────────────────────────────────────────────┘

 W6 STUDENT (KD): học tier A + class chung từ teacher; FGD theo kích thước (E1); 2 seed; avg5
 W7 ĐÁNH GIÁ: theo class, theo hệ thống, theo cỡ vật, CCR, tỉ lệ người phải sửa
```

## 3. Chi tiết từng bước

### W1. Gán nhãn mở rộng: khối việc lớn nhất

| Bước | Cách làm | Dùng lại trong repo | Kiểm tra trước khi dùng |
|---|---|---|---|
| a. Bối cảnh | xác định YMME và loại động cơ; skill automotive-expert suy ra danh sách "nên có" và "bị che" | metadata Request_ID, skill `automotive-expert` | – |
| b. Đề xuất vùng | teacher hiện tại (21 class) + SAM2 automatic mask generator trên ô 3×3 cắt từ ảnh gốc; lọc mask quá nhỏ, quá lớn, trùng | `refine_masks_sam2.py`, `inspect_image.py tiles` | recall vùng trên 60 ảnh bake-off: bao nhiêu % vật GT có ít nhất 1 mask IoU ≥ 0,5 |
| c. Gọi tên (Set-of-Mark) | ảnh có vẽ số lên từng vùng → VLM trả `{mark_id, fine_name, cues, confusers_ruled_out}` | `hybrid_qwen_deepseek.py` (thêm chế độ SoM) | **so với cách hiện tại** (VLM tự trả box) trên bộ bake-off 60 ảnh: F1 phải cao hơn 0,450 |
| d. Truy hồi | DINOv2 embedding của crop (theo mask) → kNN trên gallery từ nhãn đã review | mới: `build_gallery.py`, `retrieve.py` | top-1 accuracy trên crop test, nhất là 3 loại bình chứa |
| e. Điểm tin cậy | features: điểm VLM, điểm teacher, khoảng cách kNN, tỉ lệ mask/box, IoU với prior vị trí, có cue hay không | verdict cũ ở `data/engine_bay_review*/verdicts` làm nhãn đúng/sai | hiệu chỉnh: precision của tầng "chắc" ≥ 95% trên tập giữ lại |
| f. Duyệt | tầng "chắc": duyệt nhanh theo lô; "cao"/"ước đoán": subagent review theo lô | skill `engine-bay-label-review`, `apply_review.py` | tỉ lệ người phải sửa theo từng vòng |
| g. Xuất | chỉ `approved`; lưu `fine_name`, `system`, `provenance` | `build_full_dataset.py` (thêm cột) | không rò ảnh val/test sang train |

**Ước lượng công:**
- Mỗi ảnh khoang máy đầy đủ có khoảng 15–30 vật thuộc taxonomy v2 (hiện trung bình khoảng 3 nhãn/ảnh).
- Làm theo lô 100 ảnh, đo tỉ lệ sửa, rồi mới mở rộng.
- Vật nhỏ và ảnh cận là nơi VLM yếu nhất (DeepSeek precision 0,23 ở Phase 2), nên dự kiến cần review người nhiều nhất ở đây.

### W3. Teacher

- Giữ nguyên công thức đã chứng minh; chỉ đổi dữ liệu (taxonomy v2) để so sánh sạch với teacher p5_reg.
- Class chung giúp model **không bỏ sót** (bắt vật nhìn thấy dù chưa biết tên chi tiết). Class chi tiết giữ độ chính xác cho các class đã có dữ liệu.
- Tile 800 px (E3) và sửa FGD (E1) là hai thí nghiệm độc lập; xếp riêng trong hàng DGX.

### W4. Teacher system: đầu ra gom theo hệ thống

```json
{
  "image": "...", "vehicle_hint": "Toyota 2ZR-FE (from cover text)",
  "systems": {
    "cooling": [
      {"fine_name": "coolant_reservoir", "name_vi": "bình nước làm mát phụ", "class_src": "detector",
       "score": 0.91, "tier": "chac", "box": [..], "mask": "RLE", "cues": ["MIN/MAX", "pink fluid"]},
      {"fine_name": "thermostat_housing", "class_src": "other_hose_line→retrieval", "score": 0.62, "tier": "cao"}
    ],
    "electrical": [ ... ]
  },
  "expected_not_visible": ["spark_plug", "knock_sensor"],
  "needs_review": [ ... ]
}
```

- Viewer HTML đọc JSON này **[Ge][C]**: lọc theo hệ thống, màu theo tầng, hover hiện thẻ tri thức. Viewer không tự chứa nhãn.
- Đặt trong `apps/engine_bay_web` như một endpoint `POST /api/analyze` riêng cho server.

### W7. Đánh giá và gate

| Chỉ số | Đo trên | Gate đợt đầu |
|---|---|---|
| mask mAP50-95 các class tier A cũ | 125 ảnh test (3 xe) | **không thấp hơn** teacher p5_reg (0,354) quá 1 điểm, tức thêm class không làm hại class cũ |
| recall theo hệ thống (IoU ≥ 0,5, tính cả class chung) | test + bộ xe lạ (Jeep, ảnh ngoài đã review) | tăng so với teacher hiện tại; báo cáo riêng từng hệ |
| CCR: linh kiện nhìn thấy được nhận đúng / tổng linh kiện nhìn thấy | 30 ảnh test có nhãn taxonomy v2 đầy đủ | đặt mốc sau đợt đầu (chưa có baseline) |
| độ chính xác gọi tên class chung (retrieval + VLM) | crop test | top-1 ≥ 0,8 cho các class đưa lên UI |
| recall theo cỡ vật (< 32, 32–96, > 96 px), `save_json=True` | test | – |
| tỉ lệ người phải sửa ở W1 | log review | giảm qua từng vòng |

Quy tắc chung: 2 seed cho student, chọn model theo val/CV, không chọn theo test.

## 4. Lộ trình theo đợt

| Đợt | Việc | GPU | Kết quả |
|---|---|---|---|
| 0 (ngay) | Chốt taxonomy v2 + thẻ tri thức; đưa `exhaust_manifold_heat_shield` lên tier A; gom 36 → nhóm chung | không | configs/taxonomy_v2.yaml |
| 1 | Thử W1b–c (SoM) và W1d (DINOv2) trên bộ bake-off 60 ảnh; học trọng số điểm tin cậy từ verdict cũ | CPU + API | quyết định dùng SoM hay box trực tiếp; gallery |
| 2 | Gán nhãn mở rộng lô 100 ảnh (5+ xe) → đo tỉ lệ sửa → mở rộng dần tới 1.081 ảnh | API + review | dataset v10 |
| 3 | Teacher v10 (2 seed) + teacher system W4; A/B YOLO11l và YOLO26l | DGX | báo cáo theo hệ thống |
| 4 | Student KD v10 + E1; active learning trên ảnh xe mới | DGX | student deploy + vòng W5 |

## 5. Kết quả đã đo trên branch `distill_v2`

| Việc | Kết quả | Quyết định |
|---|---|---|
| Taxonomy v2 (`configs/taxonomy_v2.yaml`, `scripts/data_pipeline/taxonomy.py`) | 63 linh kiện, 14 hệ thống (13 + dụng cụ), **27 class train** (21 tier A + 6 class chung); test `tests/test_taxonomy_v2.py` | dùng |
| Thống kê (`taxonomy_stats.py`, `docs/reports/taxonomy_v2_stats.md`) | 912 ảnh review có sẵn trên máy local: hệ Nhiên liệu & EVAP **0 nhãn**; `other_sensor_actuator`, `other_cap_plug` 0 nhãn; không class tier A nào dưới ngưỡng giữ 40 | W1 là điều kiện bắt buộc |
| Prior vị trí (`position_priors.py`) | vùng p10–p90 rộng gần hết ảnh với hầu hết class; rõ nhất: bình dầu phanh ở nửa trên (vách máy), bình nước rửa kính và nắp két nước ở nửa dưới | chỉ dùng làm feature, không lọc cứng |
| Truy hồi DINOv2-small (`scripts/evaluation/retrieval_probe.py`, `docs/reports/retrieval_probe_v2.md`) | top-1 trên crop của xe khác 0,60–0,63. Teacher p5_reg khi đã khớp vật thì đúng class **88,8%**; gộp với truy hồi không cải thiện (0,879–0,886). Teacher bỏ sót 36% vật (IoU ≥ 0,5, conf 0,1) | **không** dùng truy hồi để sửa class của teacher; chỉ dùng đặt tên `other_*` và làm feature W1e. Nút thắt của teacher là **recall**, không phải nhầm class |
| W1e mô hình tin cậy nhãn (`scripts/data_pipeline/label_confidence.py`, `docs/reports/label_confidence_v2.md`) | 4.627 box máy đã review, GroupKFold theo xe: **AUROC 0,844** (gradient boosting) so với 0,617 khi chỉ dùng confidence của VLM. Feature mạnh nhất: chênh lệch embedding DINOv2 (giống crop đúng − giống crop sai). Nhóm điểm ≥ 0,9: 22% box, precision 0,941. Muốn precision ≥ 0,95 thì chỉ còn 14% box | dùng để **chia tầng duyệt**. Chưa được tự nhận nhãn khi không có người duyệt (precision < 0,95 ở mức phủ có ích) |
| W4 teacher system (`scripts/inference/teacher_system.py`, `docs/reports/teacher_system_eval.md`) | 125 ảnh test, teacher p5_reg, conf 0,25. Ảnh toàn cảnh: R 0,582 / P 0,632. + 2×2 ô: R **0,621** / P 0,552. + gộp khác class (IoU ≥ 0,85) + giới hạn số lượng: R 0,614 / P 0,581. Theo hệ thống: bôi trơn +7, làm mát +6, nạp khí +4, điện +3 điểm recall | chế độ có ô dùng cho **tạo nhãn** (cần recall, có người duyệt). Với người dùng cuối, F1 gần như không đổi (0,606 → 0,597); cần ngưỡng theo class từ CV. Xuất JSON gom theo hệ thống + viewer HTML |
| W2 build v2 (`build_full_dataset.py --taxonomy --extra-from`, `vehicle_split.py`) | chạy thử local: 918 ảnh, 28 xe, 25/27 class có nhãn; lấy lại **68/68** instance tấm chắn nhiệt mà bản build v1 đã bỏ; bản chia theo xe ra đúng 125 ảnh test / 3 xe như tham chiếu | dùng |
| Chain DGX (`scripts/operations/v2_teacher_chain.sh`) | teacher yolo11l-seg trên 27 class, chia theo xe, xếp sau phase 3; gate: 20 class v1 không kém p5_reg quá 1 điểm | **chưa chạy**: DGX không SSH được lúc 20:30; cần đồng bộ `v2code` trước |

## 6. Không làm (đã cân nhắc)

- D-FINE (cả ba nguồn đề xuất): chỉ ra box, không có segmentation, không hợp KD hiện tại.
- Model toàn cảnh ở 1536–4096 px: đã đo, vật lớn bị giảm recall.
- Trọng số tin cậy tự đặt / tự động nhận ở ≥ 0,9: dùng trọng số học từ verdict và hiệu chỉnh precision.
- Gán bu-lông, kẹp trên toàn ảnh: để cho tầng crop theo tác vụ (L3) trong `diagnosis_guidance_architecture_2026-09-29.md`.
- Keypoint model, GNN: hình học từ mask + luật trước.
