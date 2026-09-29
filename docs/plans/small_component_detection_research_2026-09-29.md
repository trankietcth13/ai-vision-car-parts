# Nghiên cứu: phát hiện component nhỏ trong khoang máy (2026-09-29)

Phạm vi: các class nhỏ của model engine-bay (teacher yolo11l-seg, student yolo11n-seg KD @640).
Mọi số đo dưới đây chạy trên **CPU local** (không dùng GPU local, không đụng DGX), tập test 125 ảnh / 3 xe.

## 1. Tóm tắt

1. **Ảnh gốc 6000×4000 nhưng model chỉ thấy 640 px** (gốc → 1600 khi lưu → 640 khi train). Mỗi chi tiết mất ~9,4 lần độ phân giải theo chiều dài. Que thăm dầu có cạnh trung vị 29 px ở 640, trong khi ở ảnh gốc là ~270 px.
2. **Nút thắt lớn nhất là KD chứ không phải teacher.** Ở cùng 640, teacher bắt được 69% vật < 32 px, student chỉ 39%. Khoảng cách này là 30 điểm với vật nhỏ, nhưng chỉ 7–10 điểm với vật lớn hơn.
3. FGD trong code KD của mình dùng **mask foreground nhị phân**, chuẩn hoá theo tổng diện tích foreground ([feature_loss.py:170-173](../src/distillation/feature_loss.py#L170-L173)). Hệ quả: hộp lọc gió và nắp máy chiếm gần hết tín hiệu chưng cất. FGD gốc thì trọng số từng box theo 1/diện tích.
4. **Tăng độ phân giải chỉ lúc inference thì không đủ.** Chạy student ở 960 nâng recall vật nhỏ nhưng vật lớn giảm; ở 1280 tổng recall còn tụt. Model phải được **train** ở scale đó, hoặc train trên crop/tile.
5. **Tiled inference (kiểu SAHI) có lợi ngay, không cần train lại:** student recall 0,475 → 0,543; vật < 32 px 0,39 → 0,58. Đổi lại precision giảm 0,59 → 0,53 và chậm ~5,7 lần.
6. **Lưu ý quan trọng:** các class yếu nhất hiện nay đều là **vật lớn**: coolant_reservoir (0,008), ecu_module (0,13), battery (0,11), radiator_hose, intake_manifold. Làm tốt vật nhỏ sẽ **không** sửa được những class này. Chúng là vấn đề nhãn/dữ liệu, cần xử lý riêng.

## 2. Số liệu đo

### 2.1 Kích thước vật theo class (cạnh √diện tích box, quy về input 640; 1.193 ảnh v5s)

| class | n | trung vị px | p10 | < 32 px | < 48 px | AP mask student (p5t_s0_avg5) | AP teacher |
|---|---|---|---|---|---|---|---|
| oil_dipstick | 156 | **29** | 16 | **59%** | 81% | 0,433 | 0,500 |
| radiator_cap | 79 | 47 | 26 | 20% | 53% | 0,525 | 0,557 |
| oil_filler_cap | 212 | 52 | 27 | 20% | 41% | 0,315 | **0,482** |
| washer_fluid_reservoir* | 61 | 53 | 34 | 10% | 38% | 0,570 | 0,604 |
| brake_fluid_reservoir | 129 | 61 | 37 | 3% | 29% | 0,527 | 0,413 |
| battery_terminal | 292 | 62 | 26 | 18% | 39% | 0,325 | 0,360 |
| ignition_coil | 375 | 71 | 31 | 12% | 34% | 0,292 | **0,385** |
| maf_sensor | 93 | 95 | 42 | 2% | 18% | 0,197 | 0,285 |
| (tất cả class) | 3.273 | 116 | – | 8% | 19% | – | – |

\* Bình nước rửa kính thường chỉ lộ phần nắp nên box nhỏ.
Thêm: 24% que thăm dầu có cạnh ngắn < 16 px, tức mỏng hơn 1 ô của feature map P3.

### 2.2 Recall theo kích thước (box IoU ≥ 0,5, conf 0,25, 125 ảnh test)

| model | chế độ | recall | precision | < 32 (n=36) | 32–64 (102) | 64–128 (97) | ≥ 128 (216) | CPU s/ảnh |
|---|---|---|---|---|---|---|---|---|
| student kd_n_p5t_s0_avg5 | 640 | 0,475 | 0,591 | 0,39 | 0,47 | 0,46 | 0,50 | 0,06 |
| student | 960 | 0,483 | 0,589 | 0,53 | 0,49 | 0,49 | 0,47 | 0,10 |
| student | 1280 | 0,392 | 0,571 | 0,53 | 0,43 | 0,41 | **0,34** | 0,16 |
| student | tile 2×2 + ảnh toàn cảnh @640 | **0,543** | 0,527 | **0,58** | 0,56 | 0,51 | 0,55 | 0,34 |
| teacher p5_reg_avg5 | 640 | 0,579 | 0,643 | **0,69** | 0,57 | 0,56 | 0,57 | 0,91 |
| teacher | 960 | 0,561 | 0,537 | 0,72 | 0,54 | 0,51 | 0,57 | 0,77 |
| teacher | 1280 | 0,517 | 0,530 | 0,61 | 0,50 | 0,53 | 0,50 | 1,65 |
| teacher | tile 2×2 + ảnh toàn cảnh @640 | 0,610 | 0,585 | 0,72 | 0,63 | 0,58 | 0,60 | 2,17 |

Recall theo từng class nhỏ (student 640 → student tile → teacher 640):

| class | student 640 | student tile | teacher 640 |
|---|---|---|---|
| oil_dipstick | 11/21 | 15/21 | 16/21 |
| oil_filler_cap | 10/21 | 13/21 | 16/21 |
| battery_terminal | 31/56 | 35/56 | 37/56 |
| ignition_coil | 24/58 | 26/58 | 31/58 |

Hạn chế của phép đo:
- Chỉ có 36 vật < 32 px, nên sai số khoảng ±8 điểm.
- Nhãn test lấy từ bản v5s (cùng ảnh test, nhãn có thể lệch nhẹ so với v8).
- Tile ghép bằng NMS box, chưa ghép mask.
- Script: scratchpad `probe_small.py`. Nên đưa vào repo nếu dùng tiếp (xem E0).

## 3. Chẩn đoán

| nguyên nhân | bằng chứng | đòn bẩy |
|---|---|---|
| **A. KD không truyền kiến thức vật nhỏ** | teacher − student: 30 điểm ở < 32 px, 7–10 điểm ở bucket khác. oil_filler_cap AP 0,48 → 0,31 | FGD scale mask (E1) |
| **B. Mất độ phân giải** | gốc 6000 → 640. Tile giúp +19 điểm recall cho vật < 32 px của student | train ở imgsz cao hơn (E2), train trên tile/crop (E3), tiled inference (E4), crop 2 tầng (E5) |
| **C. Mask thô và đo sai** | proto 160×160 @640: que thăm dầu ~7×7 ô. Validator của Ultralytics so mask ở 1/4 độ phân giải khi không bật `save_json` (đã kiểm trong 8.4.75: `segment/val.py:74,136`) | đo lại bằng `save_json=True` (E0), P2/proto 320 (E7), YOLO26 Proto26 (E8) |
| **D. Scale augmentation làm vật nhỏ nhỏ thêm** | `scale 0.7` (p5_reg) có thể thu que thăm dầu 29 px xuống ~9 px | `scale=(0.8,1.5)` dạng tuple (8.4.75 đã hỗ trợ) |

Một điều quan trọng cho E3: `load_image` của Ultralytics resize ảnh về cạnh dài = `imgsz` **trước khi** chạy mosaic và scale. Vì vậy augmentation phóng to không tạo thêm chi tiết. Muốn có chi tiết thật thì phải tăng `imgsz`, hoặc cắt tile/crop offline từ ảnh 1600 hay ảnh gốc.

## 4. Thử nghiệm đề xuất (theo thứ tự ưu tiên)

Quy tắc chung:
- Chạy trên DGX, mỗi lần một job.
- So với `kd_n_p5t_s0_avg5` (test mask mAP50-95 0,302) và vẫn dùng 2 seed + avg5.
- Thêm gate riêng cho vật nhỏ: mask AP (đo bằng `save_json`) của 4 class oil_dipstick, oil_filler_cap, battery_terminal, ignition_coil, cộng recall bucket < 32 px.

**E0. Sửa cách đo.** Chi phí gần bằng 0.
- `qa_test.py` gọi val với `save_json=True` để có mask ở độ phân giải thật.
- Thêm bảng recall theo bucket kích thước (đưa `probe_small.py` vào `qa_test.py` hoặc `scripts/data_pipeline/`).
- Chưa có E0 thì các cải thiện về mask nhỏ sẽ không nhìn thấy.

**E1. FGD scale-aware mask cho KD.** Khoảng 20 dòng code, **không tốn latency**. Nhắm đúng nguyên nhân A.
- Trong `build_box_masks` ([feature_loss.py:243](../src/distillation/feature_loss.py#L243)), đổi mask nhị phân thành trọng số `1/(h_box·w_box)` theo ô, như FGD gốc (Yang et al., CVPR 2022). Chồng lấn thì lấy max.
- Loss fg chuẩn hoá theo số box thay vì theo tổng diện tích.
- Bảo đảm mỗi box có **ít nhất 1 ô** ở mỗi level: lấy ô chứa tâm box. Hiện tại box nhỏ hơn 1 ô P5 có thể nhận 0 ô.
- Tuỳ chọn: tăng trọng số cls-KD cho anchor dương của GT nhỏ.
- Kỳ vọng: thu hẹp phần lớn khoảng 30 điểm teacher/student ở vật < 32 px.

**E2. Train ở imgsz 960 (student, sau đó teacher).** Ảnh 1600 hiện có là đủ, không cần xuất lại.
- Chạy inference với rect letterbox: ảnh 3:2 ở 960 chỉ là 960×640, rẻ hơn 33% so với ảnh vuông.
- Student khoảng 2,25 lần FLOPs; latency GB10 ước tính ~5–7 ms (chưa đo).
- Lưu ý: teacher v4 train ở 1024 từng thua 640 trên test. Lúc đó đang overfit nặng với 19 xe, dữ liệu nay đã khác, nhưng **phải đo lại chứ không giả định**. Cần so cả vật lớn: kết quả probe cho thấy vật lớn có thể giảm.

**E3. Fine-tune trên tile (slice-aided fine-tuning, SAHI paper).**
- Cắt offline crop 800×800 từ ảnh 1600 (zoom 2 lần), overlap 20%, giữ polygon bị cắt nếu còn ≥ 50% diện tích.
- Trộn với ảnh toàn cảnh theo tỉ lệ ~1:1, train ở 640. Inference 640 thường không đổi latency.
- Trên VisDrone, SAHI ghi nhận AP50 29,4 → 34,7 chỉ nhờ sliced inference, và → 43,5 khi fine-tune thêm trên tile.
- Nên chạy song song ý tưởng với E2 (mỗi lần 1 job) để chọn một trong hai.

**E4. Chế độ "chính xác cao" bằng tiled inference.** Không cần train lại.
- Đã đo: student +6,8 điểm recall (vật nhỏ +19), precision −6.
- Dùng cho:
  - web UI (tuỳ chọn, ~20–25 ms trên GPU);
  - **tạo pseudo-label bằng teacher** (teacher tile: recall 0,61).
- Nếu dùng thư viện `sahi`: `AutoDetectionModel(model_type="ultralytics")` hỗ trợ seg. Với chi tiết nhỏ nằm sát nhau (2 cọc bình), dùng `postprocess_type="NMS"` chứ đừng dùng GREEDYNMM, vì kiểu merge đó hợp nhất mask.
- Cần lọc thêm FP bằng ngưỡng theo class (lấy từ CV).

**E5. Crop 2 tầng theo component cha, dùng ảnh gốc 6000×4000.**
- battery → crop → tìm battery_terminal.
- Vùng nắp máy/nắp dàn cò → oil_filler_cap.
- oil_dipstick không có "cha" rõ ràng: dùng một class vùng tổng hợp kiểu CZDet (Cascaded Zoom-in Detector), hoặc tile.
- Chi phí khoảng +3 ms cho mỗi crop. Ở ảnh gốc que thăm dầu ~270 px, nên đây là đòn bẩy mạnh nhất cho độ chính xác nếu chấp nhận tăng latency.
- Model crop có thể là chính student nếu E3 đã đưa crop vào dữ liệu train.

**E6. Copy-paste vật nhỏ bằng mask SAM2 và oversample theo class.**
- Dán que thăm dầu, nắp dầu, cọc bình lên các vùng khoang máy hợp lý, đúng scale, 2–4 lần mỗi ảnh (Kisantal 2019: APs tăng tương đối +7–10%).
- Thử nhanh trước bằng `cls_pw=0.5` (8.4.75 có sẵn) và `scale=(0.8,1.5)`.

**E7. Head P2 cho segmentation.**
- Copy `yolo26-p2.yaml` (8.4.75 có sẵn, chỉ detect) và đổi head cuối thành `Segment`. Proto sẽ lấy từ level stride 4, thành **320×320**.
- Khoảng +60–80% FLOPs với bản nano.
- Chỉ làm nếu sau E1–E3 que thăm dầu vẫn yếu. Với KD của mình, chỉ khớp feature P3–P5 với teacher.

**E8. Chuyển sang YOLO26n-seg / l-seg.** Cùng framework, Proto26 đa tỉ lệ; COCO mask ghi nhận cao hơn YOLO11. Đổi lớn, nên để sau.

**Ưu tiên thấp:** NWD / WIoU / Inner-IoU (box loss của Ultralytics hard-code CIoU, phải tự viết), SPD-Conv, BiFPN, attention.
- Vật của mình đa số > 16 px, và TAL assigner 8.4.75 đã tự nới box nhỏ hơn stride lên bằng stride khi gán anchor (`utils/tal.py:306-309`).
- Không bỏ P5: battery, nắp máy, hộp lọc gió cần P5.

### Hàng đợi DGX đề xuất

1. E0 + E1: KD student với FGD scale mask, 2 seed, avg5.
2. E2 (student 960) và E3 (tile fine-tune): mỗi cái 1 job, chọn cái tốt hơn.
3. E4 dùng ngay cho pseudo-label và web UI.
4. E5 / E7 tuỳ kết quả.

## 5. Nguồn

- SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection. https://arxiv.org/abs/2202.06934 · Hướng dẫn Ultralytics: https://docs.ultralytics.com/guides/sahi-tiled-inference
- FGD: Focal and Global Knowledge Distillation for Detectors (CVPR 2022). https://arxiv.org/abs/2111.11837 · https://github.com/yzd-v/FGD
- ScaleKD (CVPR 2023): https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.html
- Multi-Scale Aligned Distillation for Low-Resolution Detection: https://arxiv.org/abs/2109.06875
- Augmentation for small object detection (Kisantal 2019): https://arxiv.org/abs/1902.07296
- CZDet, Cascaded Zoom-in Detector: https://arxiv.org/abs/2303.08747 · ClusDet: https://arxiv.org/abs/1904.08008
- NWD: https://arxiv.org/abs/2110.13389 · WIoU: https://arxiv.org/abs/2301.10051 · Inner-IoU: https://arxiv.org/abs/2311.02877 · SPD-Conv: https://arxiv.org/abs/2208.03641
- YOLO26: https://docs.ultralytics.com/models/yolo26 · RF-DETR-Seg: https://arxiv.org/abs/2511.09554 · DEIMv2: https://arxiv.org/abs/2509.20787
- Nguồn 2026 do agent khảo sát tìm được, **chưa tự kiểm chứng**: arXiv 2608.23636 (YOLO11-seg 960 cho vật nhỏ), 2609.30395 (CSCWD, chưng cất P2→P3), 2605.24831 (YOLO26 trên VisDrone), 2503.04452 (FDM-YOLO, P2 +2,2 mAP).
