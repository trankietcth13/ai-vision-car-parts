# Quy Trình Gán Nhãn & Tích Hợp Huấn Luyện (Annotation & Model Integration)

Tài liệu này xác định cách thức chuyển đổi các hình ảnh khoang máy thô sau khi cào và lọc thành **tập dữ liệu phân đoạn thực thể (Instance Segmentation Dataset)** sẵn sàng nạp vào các vòng huấn luyện tiếp theo (**v9 / v10**) của dự án.

---

## 1. Chiến Lược Bổ Khuyết 20 Lớp Linh Kiện (Component Balancing)

Theo bảng phân tích hiệu năng tại [`../engine_bay_accuracy_roadmap_2026-09-28.md`](../engine_bay_accuracy_roadmap_2026-09-28.md#23-theo-từng-lớp-kd-v7-mask-ap50-trên-test), mô hình v7 đang gặp tình trạng mất cân bằng nghiêm trọng giữa các lớp linh kiện:

| Nhóm hiệu năng | Các lớp hiện tại | Vấn đề cốt lõi | Giải pháp từ dữ liệu cào mới |
| :--- | :--- | :--- | :--- |
| **Yếu (< 0,25 AP)** | `coolant_reservoir` (0,01)<br/>`radiator_hose` (0,02)<br/>`ecu_module` (0,20) | Chỉ có 3–7 mẫu trong tập test; hình dạng ống và hộp ECU biến thiên quá lớn giữa các hãng. | Ưu tiên chọn ảnh khoang máy xe Đức/Mỹ (lộ rõ ECU cạnh bình ắc quy) và xe Nhật (lộ rõ ống dẫn két nước trên/dưới). |
| **Trung bình (0,35–0,5 AP)** | `oil_dipstick`<br/>`intake_manifold`<br/>`alternator`<br/>`engine_cover`<br/>`fuse_relay_box` | Thường bị nhầm lẫn với nhau (vd: cổ hút nhầm thành nắp máy, máy phát nhầm thành mobin đánh lửa). | Tận dụng ảnh góc chéo và cận cảnh từ Bring a Trailer để học cấu trúc không gian 3D. |
| **Tốt (≥ 0,5 AP)** | `battery`, `brake_fluid_reservoir`<br/>`washer_fluid_reservoir`<br/>`radiator_cap` | Màu sắc và vị trí tương đối ổn định (nắp vàng, nắp xanh, cọc bình kim loại). | Duy trì tần suất xuất hiện tự nhiên, không oversample quá mức. |

---

## 2. Quy Trình Gán Nhãn Bán Tự Động (Semi-Automated Labeling Flow)

Tận dụng hạ tầng có sẵn trong `scripts/data_pipeline/`:

```mermaid
flowchart LR
    A["Ảnh Khoang Máy Đã Lọc<br/>(H:\\AI_Datasets\\01)"] ──▶ B["Teacher v8 / Qwen-VL<br/>(BBox Detection)"]
    B ──▶ C["SAM 2 Refiner<br/>(refine_masks_sam2.py)"]
    C ──▶ D["YOLO11-seg Mask Labels<br/>(0.0 - 1.0 polygon coords)"]
    D ──▶ E["Kiểm Định Chất Lượng (QA)<br/>(apply_codex_verdicts.py)"]
    E ──▶ F["Tập Huấn Luyện v9<br/>(data/engine_bay_train_v9)"]
```

### Bước 1: Sinh Hộp Giới Hạn Sơ Bộ (Prompt Generator)
- Chạy checkpoint **Teacher tốt nhất (v8)** trên máy chủ DGX với ngưỡng tin cậy thấp (`conf=0.20`) để tối đa hóa độ phủ (recall) của các linh kiện nhỏ (dipstick, radiator cap, hose).
- Đối với các lớp hiếm (`ecu_module`, `radiator_hose`), kết hợp chạy VLM Grounding (`qwen_grounding_annotator.py`).

### Bước 2: Sinh Mặt Nạ Đa Giác Bằng SAM 2 (Segment Anything 2)
- Hộp giới hạn từ Bước 1 làm gợi ý (bounding box prompt) cho SAM 2 để sinh mặt nạ phân đoạn pixel chính xác tuyệt đối.
- Script tích hợp:
  ```bash
  python scripts/data_pipeline/refine_masks_sam2.py \
      --images H:/AI_Datasets/01 \
      --boxes data/crawled_pseudo_boxes/ \
      --output data/engine_bay_crawled_seg/
  ```

### Bước 3: Kiểm Chứng Ngẫu Nhiên (QA Sampling)
- Rút mẫu ngẫu nhiên **10% số ảnh** đưa qua công cụ trực quan hóa [`scripts/data_pipeline/qc_visualizer.py`](../../scripts/data_pipeline/qc_visualizer.py) để chuyên gia duyệt độ chính xác của nhãn trước khi gộp vào tập train chính.

---

## 3. Nguyên Tắc Phân Chia Tập Dữ Liệu (Split Isolation Rules)

Để bảo đảm tính khách quan tuyệt đối của hệ thống đo lường:

> ⚠️ **QUY TẮC BẤT DI BẤT DỊCH**:
> 1. **100% dữ liệu cào từ Internet chỉ được đưa vào tập TRAIN** (không bao giờ đưa vào tập Validation hoặc Test).
> 2. Tập **Test (3 xe)** và tập **Val (6 xe)** hiện tại được giữ nguyên vẹn để làm thước đo chuẩn so sánh tiến bộ qua các phiên bản (v6 $\rightarrow$ v7 $\rightarrow$ v8 $\rightarrow$ v9).
> 3. Tuyệt đối không để xảy ra rò rỉ danh tính xe (vehicle-identity leakage). Ảnh cùng một xe phải luôn nằm trọn vẹn trong một tập duy nhất.

---

## 4. Kế Hoạch Đóng Gói Dataset v9 (Build v9 Specification)

Tạo script `scripts/data_pipeline/build_v9.py` kế thừa từ `build_v8.py`:

```bash
python scripts/data_pipeline/build_v9.py \
    --base data/engine_bay_train_v8 \
    --crawled data/engine_bay_crawled_seg \
    --out data/engine_bay_train_v9 \
    --max-external 3000 \
    --balance-classes
```

### Tiêu Chuẩn Nghiệm Thu Vòng v9:
1. **Số lượng đối tượng linh kiện**: Tăng từ ~1.000 lên **> 5.000 instances** trong tập train.
2. **Độ đa dạng xe**: Đạt ít nhất **100 mẫu xe độc lập**.
3. **Mục tiêu điểm số trên test**:
   - Mask mAP50-95 của Student đạt **≥ 0,35** (tăng ít nhất 10 điểm so với v7 = 0,249).
   - Mask mAP50 đạt **≥ 0,55** (tăng hơn 10 điểm so với v7 = 0,44).
   - Tỷ lệ ảnh test đạt yêu cầu tăng từ **47/100** lên **≥ 70/100 ảnh**.
