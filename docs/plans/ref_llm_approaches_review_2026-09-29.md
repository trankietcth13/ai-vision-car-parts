# Đánh giá cách làm tham khảo từ Claude, ChatGPT, Gemini (2026-09-29)

Nguồn: `data/ref/`. Cả ba đều phân rã **một ảnh** khoang máy Jeep 2.0L Turbo, kèm tài liệu mô tả pipeline.

| Nguồn | Demo | Tài liệu |
|---|---|---|
| Claude | `engine_bay_parts.html`: 27 box, 3 mức tin cậy | `engine_bay_autolabel_spec.md`: pipeline VLM + SAM2 để gán nhãn tự động |
| ChatGPT | `engine_bay_component_breakdown.html`: 34 **điểm**, không có box | `Engine_Bay_Auto_Labeling_Specification_v1.0.docx`: pipeline phân tầng L1–L6, scene graph |
| Gemini | `interactive_engine_bay_inspector.html`: 12 box, không nhúng ảnh (khung 1024×683) | `Engine_Bay_Decomposition_Algorithm_Specs.doc.docx`: chủ yếu là UI (pin CSS) + Grounding DINO + SAM |

## 1. Chất lượng demo (tôi vẽ nhãn của cả ba lên ảnh để kiểm tra)

**Claude, 27 box**
- Đúng tốt với vật có đặc điểm rõ: nắp cọc dương màu đỏ, cảm biến O2, khớp nối chốt đỏ, bình trắng bên trái, nắp có nhãn vàng.
- Box thô với vật dài: ống, bó dây. Hộp cầu chì lệch trái.
- Tự ghi nhận sai số ±30–60 px; 11/27 nhãn tự đánh dấu là "ước đoán".

**ChatGPT, 34 điểm**
- Điểm chỉ dùng được làm prompt cho SAM, không train detector trực tiếp được.
- Khoảng 3/4 điểm đặt hợp lý. Có vài điểm sai rõ: "harness", "clip" và "connector" rơi vào lỗ trên nắp máy; "white hose clips" lệch khỏi các kẹp trắng.
- Nhãn "High" gán cho cả những điểm sai.

**Gemini, 12 box**
- Vị trí đại khái đúng.
- Confidence 0,88–0,98 **không đến từ model nào**, nên không dùng được.
- Tài liệu hứa "< 25 ms CPU cho YOLO26s-seg". Số đo thật: YOLO26n-seg ONNX CPU 53 ms theo Ultralytics, bản s 118 ms.

**Ba model mâu thuẫn nhau về danh tính:**
- Bình trắng bên trái: nước làm mát intercooler (Claude) / "bình dung dịch" (ChatGPT) / bình nước rửa kính (Gemini).
- Ống đen chạy ngang trước nắp máy: bó dây (Claude, ChatGPT) / ống thông hơi PCV (Gemini).
- Nhãn LLM từ một ảnh **không phải ground truth**. Cần YMME hoặc tài liệu sửa chữa và chuyên gia duyệt.

**So với model của mình (teacher_full, conf 0,25, CPU):**
- Chỉ ra 3 detection: engine_cover 0,91 và air_intake_duct 0,88 (mask rất tốt), cộng một air_filter_box 0,28 nhận nhầm.
- Bỏ sót nắp cọc dương, hộp cầu chì, hai bình chứa. Jeep không có trong 28 xe train, đúng vấn đề tổng quát hoá sang xe mới.
- Kết luận: **VLM mạnh về "đây là gì" trên xe lạ; model của mình và SAM2 mạnh về "nằm chính xác ở đâu".** Hai bên bổ sung cho nhau.

## 2. Nên áp dụng

| # | Ý tưởng | Nguồn | Nút thắt giải quyết | Cách làm trong repo | Chi phí |
|---|---|---|---|---|---|
| 1 | **`confusers` + bắt nêu đặc điểm phân biệt** trước khi chốt tên cho các lớp dễ nhầm | Claude | coolant_reservoir AP 0,008–0,045; nhầm 3 loại bình | thêm vào prompt Qwen/DeepSeek và skill review: màu nắp, ký hiệu, vị trí so với vách/két nước | thấp |
| 2 | **Class chung + truy hồi embedding (DINOv2/CLIP) để gọi tên chi tiết** | ChatGPT | nhầm class; mở rộng class mà không phải train lại detector | gallery từ khoảng 3.300 chi tiết đã review → kNN trên crop; thử trước trên 3 loại bình và mobin/bugi | thấp |
| 3 | **Gộp nhiều tín hiệu thành độ tin cậy, chia tầng (tier)** | Claude, ChatGPT | pseudo-label bẩn lọt vào train | **Không dùng trọng số tự đặt.** Học trọng số bằng hồi quy logistic trên verdict review đã có (khoảng 2.300 detection có kết luận đúng/sai). Tín hiệu: VLM, teacher của mình, chất lượng mask SAM, prior vị trí | trung bình |
| 4 | **Prior vị trí theo class** (`expected_zone`) | Claude | báo nhầm; luật ngữ cảnh | tự tính heatmap vị trí từng class từ 1.081 ảnh; theo YMME thì để sau khi có tài liệu sửa chữa | thấp |
| 5 | **Active learning bằng độ bất đồng VLM ↔ model** | Claude, ChatGPT | độ đa dạng xe (mục tiêu B) | mở rộng `mine_errors.py` sang kho ảnh chưa nhãn; ưu tiên dòng xe chưa có | thấp |
| 6 | **Cờ `is_region`** cho bộ phận bị che (ắc quy dưới tấm ốp), mặc định không train | Claude | battery AP 0,11: có thể do nhãn "khu vực" không nhất quán | kiểm tra nhãn battery hiện có, thống nhất quy định | thấp |
| 7 | **Centerline/skeleton cho ống và bó dây** | ChatGPT | vẽ hướng dẫn (đầu ống, đường đi) | xử lý sau trên mask (tầng L4) | thấp |
| 8 | **Taxonomy phân tầng** (`parent_class`: Fastener / Retainer / Electrical / Fluid-Air / Mounting / Major) | ChatGPT | mở rộng class có trật tự | dùng khi thêm class chẩn đoán và chi tiết nhỏ | thấp |
| 9 | **Chỉ số vận hành:** CCR (tỉ lệ bao phủ linh kiện), tỉ lệ người phải sửa, provenance từng nhãn | ChatGPT | đo hiệu quả gán nhãn | thêm vào `apply_review` / báo cáo | thấp |
| 10 | **Viewer HTML đọc JSON** (pin, lọc theo nhóm, màu theo tier) | cả ba | giao diện hướng dẫn L5 | `engine_bay_web` xuất JSON chuẩn; viewer chỉ hiển thị, không tự chứa nhãn | thấp |

## 3. Không nên áp dụng (và lý do)

| Đề xuất | Của | Lý do |
|---|---|---|
| D-FINE làm teacher | cả ba | chỉ phát hiện box, không có segmentation; đầu box không hợp với YOLO26 và KD hiện tại |
| Model toàn cảnh ở 1536–4096 px | ChatGPT, bản pipeline trước | đo được: tăng scale khi không train ở đó thì vật lớn giảm recall; mình train 640 tốt hơn 1024 |
| Tự nhận nhãn khi confidence ≥ 0,90, trọng số tự đặt | ChatGPT, Claude | chưa hiệu chỉnh; VLM tự tin cao cả khi sai (demo ChatGPT "High" cho điểm sai) |
| Grounding DINO / OWLv2 làm detector chính cho linh kiện ô tô | Gemini, Claude (kiểm tra chéo) | chưa đo trên dữ liệu của mình; teacher chuyên ngành của mình có lẽ là tín hiệu kiểm tra chéo tốt hơn. Nếu thử thì đo precision trên 125 ảnh test trước |
| Gán bu-lông/kẹp trên toàn ảnh, 40–80 class ngay | ChatGPT | chi phí nhãn rất lớn; VLM kém trên ảnh cận (DeepSeek precision 0,23) |
| Keypoint model, GNN ngay từ đầu | ChatGPT | hình học từ mask + luật đủ cho giai đoạn đầu (chính tài liệu ChatGPT cũng khuyên "rule-based cho POC") |
| Mục tiêu POC: recall linh kiện chính > 95%, connector > 90% | ChatGPT | model hiện tại khoảng 0,30 mask mAP50-95 trên xe mới; đặt mục tiêu theo số đo thực |
| Lưới tọa độ 100 px phủ lên ảnh cho VLM | Claude | Qwen3-VL đã grounding tọa độ trực tiếp; lưới có thể giúp hoặc hại, phải so trên tập bake-off 60 ảnh trước khi dùng |

## 4. Thay đổi đề xuất cho lộ trình

- Bổ sung vào **Giai đoạn A** (thử rẻ, không cần GPU):
  - #1 confusers trong prompt và skill review;
  - #2 truy hồi DINOv2 cho các bình chứa;
  - #4 prior vị trí làm luật ngữ cảnh.
  - Đo trên 125 ảnh test như mọi thí nghiệm khác.
- Bổ sung vào **pipeline gán nhãn cho xe mới** (mục tiêu B):
  - VLM đưa ra tên và cue quan sát được;
  - teacher của mình + SAM2 đưa ra vị trí và mask;
  - #3 độ tin cậy học từ verdict cũ → tier;
  - #5 chọn ảnh cần duyệt theo độ bất đồng.
- Giữ nguyên các quyết định ở `diagnosis_guidance_architecture_2026-09-29.md`: không D-FINE, không GNN sớm, crop theo tác vụ, quan hệ lấy từ tri thức sửa chữa.
