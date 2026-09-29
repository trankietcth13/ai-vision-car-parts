# Lộ trình nâng độ chính xác nhận diện linh kiện khoang máy — 28/09/2026

Tài liệu tổng hợp kết quả 7 vòng huấn luyện (v1–v7), chẩn đoán nguyên nhân độ chính xác còn thấp, và kế hoạch chi tiết cho các bước tiếp theo, bắt đầu từ nhãn v8.

**Kết luận chính:**

- Pipeline distillation hoạt động đúng. KD student luôn hơn baseline 3–5 điểm mask mAP50, và đạt khoảng 95% điểm của teacher trong khi nhỏ hơn 10 lần.
- Độ chính xác trên **xe mới** vẫn thấp: mask mAP50 0,44 và mAP50-95 0,25. Nguyên nhân là **dữ liệu** (chỉ 19 xe để train, nhãn chưa sạch), không phải thuật toán.
- Nhãn v8 đã sẵn sàng và nên chạy ngay. Để đạt yêu cầu "detect chính xác trên `dataset`", cần hoàn thiện nhãn cho toàn bộ 1.307 ảnh rồi train model cuối trên cả 28 xe.

---

## 1. Hai mục tiêu cần phân biệt

| Mục tiêu | Ý nghĩa | Cách đo | Hiện trạng (KD v7) | Mức đạt được |
|---|---|---|---|---|
| **A. Chính xác trên `dataset`** (1.307 ảnh, 28 xe tại `E:\Research\GEN AI\AI Vision\Distillation\dataset`) | Model dùng cho chính các xe và ảnh này, ví dụ để gán nhãn, tra cứu, kiểm tra | 5-fold cross-validation theo ảnh, sau khi mọi ảnh đã có nhãn chuẩn | Trên xe đã học: mask mAP50 **0,94**, mAP50-95 **0,72** | mask mAP50 ≥ 0,90 sau khi hoàn thiện nhãn |
| **B. Chính xác trên xe mới** cùng kiểu chụp | Model chạy trên xe khách hàng chưa từng thấy | Tập test gồm 3 xe giữ riêng (Request_ID_13, 23, 29) | mask mAP50 **0,44**, mAP50-95 **0,25** | Cần thêm nhiều xe mới. Tinh chỉnh chỉ tăng 1–3 điểm |

Người dùng đã xác nhận yêu cầu: **"cần detect được tập dataset từ `dataset`, detect chính xác component"**. Vì vậy mục tiêu A là ưu tiên số 1, mục tiêu B là định hướng dài hạn.

---

## 2. Hiện trạng

### 2.1 Dữ liệu và nhãn

| Hạng mục | Số lượng |
|---|---|
| Ảnh trong `dataset` | 1.307 ảnh, 28 Request_ID (xe). 18 ảnh lỗi (JSON Qwen hỏng hoặc file ảnh hỏng) |
| Chia tập theo xe | train 19 xe, val 6 xe (125 ảnh), test 3 xe (125 ảnh) |
| Danh sách lớp huấn luyện | 20 lớp (`configs/engine_bay_train_classes.yaml`). `oil_filter` đã bỏ từ v6 vì chỉ có 9 mẫu |
| Nhãn val và test | 100% đã review bằng skill chuyên gia và được Codex kiểm chứng |
| Nhãn train đạt mức review | 713 / 1.059 ảnh (**67%**): 324 do skill chuyên gia duyệt, 200 do Codex sửa, 382 là nhãn hybrid đã review (v8) |
| Nhãn train còn là pseudo-label | khoảng **346 ảnh** |
| Dữ liệu ngoài (v7) | 175 ảnh khoang máy và linh kiện (Wikimedia, Toyota Corolla), chỉ đưa vào train. Nguồn ghi tại `data/external_candidates/ATTRIBUTION.md` |

### 2.2 Kết quả qua các vòng

Tất cả số liệu dưới đây là mask mAP50-95. Các vòng từ v6 trở đi dùng cùng tập val (6 xe) và tập test (3 xe).

| Vòng | Thay đổi chính | Teacher (test) | KD student (test) | Baseline (test) |
|---|---|---|---|---|
| v4 | 324 ảnh train đã review, pseudo-label từ teacher v2 | 0,266 | 0,220 | 0,198 |
| v5 | Codex sửa 95 nhãn (teacher dừng sớm ở epoch 34) | 0,219 | 0,193 | 0,165 |
| v6 | imgsz 640, copy_paste/mixup, repeat-factor sampling, pseudo-label lại bằng teacher v4, bỏ `oil_filter` | 0,259 | **0,245** (2 seed) | 0,218 (2 seed) |
| v7 | v6 + 175 ảnh ngoài | 0,258 | **0,249** (2 seed) | — |
| v8 | Nhãn hybrid đã review thay pseudo-label trên 358 ảnh | *chưa chạy* | *dự kiến 0,26–0,28* | — |

**Kiểm thử 100 ảnh test** (KD v7 seed 0, ngưỡng tin cậy 0,25):

- 47/100 ảnh đạt yêu cầu.
- Tìm được 158 trong 373 linh kiện: precision 0,57, recall 0,42.
- Độ trễ trung vị 10 ms trên DGX.

### 2.3 Theo từng lớp (KD v7, mask AP50 trên test)

| Nhóm | Lớp |
|---|---|
| Tốt (≥ 0,5) | multimeter 0,79 · brake_fluid_reservoir 0,70 · radiator_cap 0,63 · washer_fluid_reservoir 0,58 · battery_terminal 0,56 · ignition_coil 0,53 |
| Trung bình (0,35–0,5) | engine_cover, air_filter_box, oil_dipstick, intake_manifold, throttle_body, oil_filler_cap, air_intake_duct, battery, fuse_relay_box, alternator, maf_sensor |
| Yếu (< 0,25) | ecu_module 0,20 · radiator_hose 0,02 · coolant_reservoir 0,01 |

Riêng coolant_reservoir, radiator_hose, washer_fluid_reservoir và ecu_module chỉ có 3–7 mẫu trong tập test, nên điểm của các lớp này dao động rất mạnh.

---

## 3. Chẩn đoán

1. **Model học thuộc từng xe.** Trên ảnh train đã review, KD v7 đạt mAP50-95 0,72, nhưng trên xe mới chỉ đạt 0,25, tức khoảng cách **0,47**. Ngay cả teacher lớn gấp 10 lần cũng dừng ở khoảng 0,26 trên test. Nguyên nhân là 19 xe × ~48 ảnh mỗi xe, các ảnh của cùng một xe rất giống nhau, nên độ đa dạng thực tế chỉ tương đương khoảng 19 mẫu.
2. **Model thiếu tự tin do nhãn thiếu.** Khoảng 30% linh kiện thật trong train không có nhãn, vì recall của nhãn máy chỉ khoảng 0,70. Model vì vậy học rằng các linh kiện đó là nền. Ví dụ, trên một ảnh Honda rõ nét, model chỉ cho dipstick, duct và oil cap điểm tin cậy 0,14–0,30. Hạ ngưỡng từ 0,35 xuống 0,25 giúp số ảnh đạt tăng từ 43 lên 47.
3. **Nhầm lẫn giữa các lớp giống nhau:**
   - Bình dầu phanh bị gán thành bình nước làm mát.
   - Cổ hút bị gán thành engine cover.
   - Nắp bình nước làm mát bị gán thành radiator cap.
   - Alternator bị gán nhầm thành ignition coil.
4. **Đo lường còn nhiễu.** Tập test chỉ có 3 xe, nên chênh lệch ±1–2 điểm có thể chỉ là ngẫu nhiên. Tập val (6 xe) được dùng để chọn checkpoint nên cho số lạc quan hơn thực tế.
5. **Những gì đã thử mà không giúp đáng kể:**
   - Train teacher ở imgsz 1024. Ở 640 teacher còn đạt test cao hơn.
   - Tăng model từ m lên l: chỉ +0,4 điểm.
   - Augmentation mạnh hơn: không thu hẹp được khoảng cách train–test.
   - 175 ảnh ngoài: chỉ +0,4 điểm trên test.

---

## 4. Kế hoạch chi tiết

### Giai đoạn 1 — Chạy v8 (nhãn hybrid đã review) · khoảng 3,5 giờ trên DGX · sẵn sàng

**Mục đích:** đo tác động của việc làm sạch nhãn trên các xe đã có, giữ nguyên mọi yếu tố khác.

**Các bước:**

1. Đẩy `data/engine_bay_reviewed_hybrid/` (nhãn, `EXCLUDED.txt`, ảnh) và `scripts/data_pipeline/build_v8.py` lên DGX.
2. Dựng dataset:
   ```bash
   python scripts/data_pipeline/build_v8.py --base data/engine_bay_train_v7 --hybrid data/engine_bay_reviewed_hybrid --out data/engine_bay_train_v8
   ```
3. Train teacher yolo11l-seg, rồi KD yolo11n-seg với 2 seed, rồi QA so với v7 và 100 test case. Hyperparameter giữ **nguyên như v7** (`configs/kd_hyperparams_v6.yaml`, teacher imgsz 640 batch 16, 100 epoch, cos_lr, copy_paste 0.3, mixup 0.1, degrees 5).

**Những gì v8 thay đổi:**

- 358 ảnh được thay nhãn, số đối tượng tăng từ 929 lên 1.007.
- brake_fluid_reservoir tăng từ 14 lên 36, coolant_reservoir giảm từ 25 xuống 14.
- intake_manifold tăng từ 27 lên 64, engine_cover giảm từ 39 xuống 27.
- oil_dipstick tăng từ 24 lên 59.
- 96 ảnh trùng với phần Codex đã sửa dùng nhãn hybrid, vì bộ này được review đầy đủ.

**Kỳ vọng:**

- Test: mask mAP50-95 khoảng 0,26–0,28, mask mAP50 khoảng 0,46–0,49, số ảnh đạt khoảng 50–55/100.
- Các lớp brake, coolant, intake_manifold, engine_cover, dipstick có thể tăng 5–15 điểm AP mỗi lớp.

**Điều kiện đạt:** trung bình 2 seed của KD v8 ≥ KD v7 + 1 điểm mask mAP50-95. Nếu đạt, v8 trở thành model tham chiếu.

**Phân công:** chỉ một phiên (Claude hoặc Codex) được chạy trên DGX, để tránh trùng việc và tranh chấp bộ nhớ với vLLM.

### Giai đoạn 2 — Hoàn thiện nhãn cho toàn bộ `dataset` (mục tiêu A) · 1–2 ngày

**Mục đích:** mọi ảnh trong `dataset` đều có nhãn chuẩn. Đây là điều kiện tiên quyết để detect chính xác và để đo chính xác.

**Các bước:**

1. Xác định khoảng **346 ảnh train** chưa có nhãn đạt mức review, cùng 18 ảnh lỗi. Ảnh lỗi thì gán nhãn lại hoặc loại hẳn nếu file hỏng.
2. Sinh nhãn ban đầu bằng model tốt nhất (teacher v8) trên DGX. Nhãn này tốt hơn nhiều so với nhãn Qwen cũ.
3. Review bằng skill chuyên gia (`.claude/skills/engine-bay-label-review`), khoảng 12 lô × 30 ảnh, tập trung bổ sung linh kiện bị sót.
4. Codex kiểm chứng theo `CODEX_QA_HANDOFF.md`, sau đó chạy `apply_codex_verdicts.py`.
5. Chạy QA cho nhãn: lấy ngẫu nhiên 50 ảnh để người xem lại. Mục tiêu precision và recall của nhãn đều ≥ 0,9.

**Chi phí:** khoảng 5–8 triệu token cho subagent, và khoảng 1 giờ DGX cho SAM2 cùng pseudo-label.

**Đầu ra:** `data/engine_bay_full_reviewed/`, gồm 1.289 ảnh có nhãn chuẩn, dùng 20 lớp.

### Giai đoạn 3 — Train model cuối cho mục tiêu A · khoảng 1 ngày DGX

1. **Đo lường:** 5-fold cross-validation **theo ảnh** trên toàn bộ `dataset`. Mỗi fold train teacher và KD student, báo cáo trung bình ± độ lệch chuẩn.
2. **Model cuối:** train trên **cả 28 xe**, tức toàn bộ ảnh có nhãn, với cấu hình tốt nhất ở giai đoạn 1.
3. **Đánh giá:** chạy `test_cases.py` trên toàn bộ `dataset`, xuất `gallery.html` để xem từng ảnh.
4. **Kỳ vọng:** mask mAP50 ≥ 0,90, mAP50-95 ≥ 0,70, và ≥ 90% số ảnh đạt yêu cầu.
5. **Hiệu chỉnh:** tìm ngưỡng tin cậy riêng cho từng lớp bằng cross-validation. Lớp mà model hay thiếu tự tin (duct, dipstick) dùng ngưỡng thấp. Lớp hay báo nhầm (battery, coil) dùng ngưỡng cao.

**Lưu ý:** model cuối của giai đoạn này **không còn tập test theo xe**. Con số mục tiêu A đo độ chính xác trên `dataset`, không đo khả năng tổng quát sang xe mới. Nếu vẫn cần theo dõi mục tiêu B, giữ lại 3 xe test ở một bản song song.

### Giai đoạn 4 — Mở rộng sang xe mới (mục tiêu B) · phụ thuộc dữ liệu

1. **Thu thập:** ưu tiên **Request_ID nội bộ của xe mới**, cùng kiểu chụp như dữ liệu hiện có. Mục tiêu 40–60 xe, gấp 2–3 lần hiện tại. Mỗi xe khoảng 30 ảnh là đủ. Một xe mới có giá trị hơn nhiều so với thêm 50 ảnh của cùng một xe.
2. **Hướng dẫn chụp:**
   - Chụp toàn cảnh khoang máy từ trên xuống và chéo trái, chéo phải.
   - Chụp cận các linh kiện chính.
   - Chụp nhiều điều kiện ánh sáng.
   - Không chụp ngoại thất.
3. **Mở rộng tập test** lên ≥ 6 xe (và val ≥ 6 xe), để chênh lệch 2 điểm trở lên có ý nghĩa thống kê.
4. **Nguồn ngoài:** chỉ lấy ảnh khoang máy hoặc linh kiện, tuyệt đối không dùng ảnh ngoại thất hay nội thất. Bộ Roboflow "Engine Bay Parts" (khoảng 2.300 ảnh, MIT, cần `ROBOFLOW_API_KEY`) là nguồn tiềm năng tiếp theo. Bài học từ v7: ảnh ngoài khác kiểu chụp nên chỉ giúp rất ít.
5. **Chỉ số theo dõi:** khoảng cách mAP train–test (hiện 0,47, mục tiêu < 0,30) và mask mAP50-95 của student trên test (mục tiêu ≥ 0,30).

### Giai đoạn 5 — Tinh chỉnh cách train · chạy song song khi DGX rảnh

Kỳ vọng mỗi hạng mục tăng khoảng 1–3 điểm, và cần đo từng thay đổi riêng lẻ:

- Freeze backbone của teacher trong 10 epoch đầu (`freeze=10`).
- Đặt cố định `optimizer=SGD, lr0=0.01` cho teacher. `auto` đã vô tình chuyển sang AdamW ở v6 và v7.
- Tăng `weight_decay` lên 0,001 và `scale` lên 0,7, thêm perspective và blur.
- Dùng trung bình trọng số của các checkpoint tốt thay vì một checkpoint "best", vì mAP trên val dao động ±4 điểm.
- Nếu thiết bị triển khai là máy chủ, thử student yolo11s-seg.

### Giai đoạn 6 — Triển khai · khoảng 0,5 ngày

1. Xuất student tốt nhất sang ONNX, rồi TensorRT FP16 cho thiết bị đích: `scripts/deployment/export_onnx.py`, `scripts/deployment/build_tensorrt_engine.py`.
2. Đo độ trễ chuẩn: khởi động (warm-up) trước, chạy lặp ≥ 200 lần, đo riêng model và toàn bộ pipeline.
3. Áp dụng ngưỡng tin cậy riêng cho từng lớp (từ giai đoạn 3).
4. Đổi model mặc định của web UI (`web_ui/app.py`) sang student khoang máy.
5. **Cách dùng khuyến nghị theo mức chính xác hiện có:**
   - Trên các xe trong `dataset`: tự động (sau giai đoạn 3).
   - Trên xe mới: chỉ dùng như gợi ý để kỹ thuật viên xác nhận, cho đến khi đạt mục tiêu B.

---

## 5. Lịch dự kiến

| Thời điểm | Giai đoạn | Kết quả |
|---|---|---|
| 28/09 | 1 — v8 | Báo cáo QA v8 so với v7 |
| 29–30/09 | 2 — Hoàn thiện nhãn | Toàn bộ `dataset` có nhãn chuẩn, kèm báo cáo chất lượng nhãn |
| 30/09–01/10 | 3 — Model cuối (mục tiêu A) | Model cuối, số liệu 5-fold CV, gallery toàn bộ `dataset` |
| 01/10 | 6 — Triển khai | ONNX/TensorRT, web UI, số đo độ trễ |
| Tùy dữ liệu | 4 — Xe mới (mục tiêu B) | Phụ thuộc số Request_ID mới thu được |
| Song song | 5 — Tinh chỉnh | Các thí nghiệm riêng lẻ khi DGX rảnh |

---

## 6. Quyết định cần chốt

1. **Ưu tiên mục tiêu A hay B.** Theo yêu cầu gần nhất là A. Cần xác nhận có giữ 3 xe test để theo dõi B song song hay không.
2. **Phiên nào chạy v8 trên DGX** (Claude hay Codex), để tránh chạy trùng.
3. **Thiết bị triển khai** (Jetson/tablet hay máy chủ). Câu trả lời quyết định dùng student n@640 hay s@1024.
4. **Có thu thập thêm Request_ID xe mới được không**, và khoảng bao nhiêu xe.
5. **Có dùng dữ liệu Roboflow không.** Cần API key và kiểm tra giấy phép nếu dùng cho mục đích thương mại.

---

## 7. Rủi ro

| Rủi ro | Ảnh hưởng | Giảm thiểu |
|---|---|---|
| DGX dùng chung bộ nhớ với vLLM Qwen của Codex | Train có thể bị OOM hoặc làm sập vLLM | Mỗi lúc chỉ chạy 1 job, theo dõi bộ nhớ trống (cảnh báo khi < 3 GB) |
| Tập test nhỏ (3 xe) | Kết luận sai từ chênh lệch 1–2 điểm | Báo cáo trung bình 2 seed, mở rộng test lên ≥ 6 xe |
| Rò rỉ dữ liệu giữa train và test | Điểm cao ảo | Tách tập theo xe. Không bao giờ đưa ảnh test vào train. Chọn checkpoint chỉ theo val |
| Nhãn hybrid còn lỗi | Trần độ chính xác bị giới hạn | QA nhãn giai đoạn 2 với mục tiêu precision và recall của nhãn ≥ 0,9 |
| Mục tiêu A cho số liệu rất cao | Dễ hiểu nhầm là model đã tổng quát tốt | Luôn báo song song số liệu trên xe mới (mục tiêu B) |

---

## 8. Phụ lục — công cụ và quy ước

**Quy ước trên DGX** (`admin@dgx-host`, NVIDIA GB10, 121 GB bộ nhớ dùng chung):

- Chỉ train trên DGX, không dùng GPU máy local.
- Môi trường Python: `~/distillation_workspace/.venv` (torch 2.14 cu130, ultralytics 8.4.75).
- Trước khi train phải `export CPATH=$HOME/pyheaders/usr/include/python3.12:$HOME/pyheaders/usr/include`. Triton cần header Python để JIT, và các header này đã được giải nén không cần sudo.
- Ultralytics tự thêm `runs/segment/` vào trước `--project` tương đối, nên luôn dùng đường dẫn tuyệt đối. `runs/train_kd` là symlink.

**Script chính:**

| Việc | Script |
|---|---|
| Train, trạng thái, kéo kết quả trên DGX | `dgx_train.py` (`check`, `push`, `teacher`, `kd`, `status`, `resume`, `stop`, `pull-qa`), `check_training.py --watch 60` |
| Distillation | `train_kd.py`, `src/distillation/ultralytics_kd.py`, `configs/kd_hyperparams_v6.yaml` |
| Tạo mask | `scripts/data_pipeline/refine_masks_sam2.py` |
| Review nhãn | `.claude/skills/engine-bay-label-review/SKILL.md`, `scripts/data_pipeline/build_review_packets.py`, `scripts/data_pipeline/apply_review.py` |
| Dựng dataset | `build_training_dataset.py`, `pseudo_label_merge.py`, `resplit_val.py`, `build_v6.py`, `build_v7.py`, `build_v8.py` |
| Dữ liệu ngoài | `curate_external.py`, `external_to_seg.py`, `data/external_candidates/ATTRIBUTION.md` |
| QA | `qa_test.py`, `test_cases.py`, `mine_errors.py`, `merge_errors.py`, `apply_codex_verdicts.py`, `CODEX_QA_HANDOFF.md` |

**Kết quả đã có:** `qa_results/qa_results_v7/QA_REPORT.md`, `qa_results/test_cases_kd_v7_s0/gallery.html`, `qa_results/test_cases_kd_v6_s0_conf025/`.
