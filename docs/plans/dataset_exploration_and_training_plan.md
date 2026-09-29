# KẾ HOẠCH TOÀN DIỆN: KHÁM PHÁ DATASET & HUẤN LUYỆN DISTILLATION TRÊN DGX
## HỆ THỐNG NHẬN DIỆN & PHÂN ĐOẠN LINH KIỆN KHOANG ĐỘNG CƠ (ENGINE BAY COMPONENT DETECTION & SEGMENTATION)

---

## 1. KẾT QUẢ KHÁM PHÁ TẬP DỮ LIỆU (DATASET EXPLORATION)

Thư mục phân tích: `E:\Research\GEN AI\AI Vision\Distillation\dataset\`

### 1.1. Thống kê Định lượng (Quantitative Statistics)
- **Tổng số lượng ảnh**: **1,307 ảnh** (1,304 `.jpg` + 3 `.jpeg`).
- **Tổng dung lượng**: **10.43 GB** (trung bình **8.17 MB/ảnh**; dải từ 1.09 MB đến 14.91 MB).
- **Cấu trúc phân bổ**: 28 thư mục chẩn đoán thực tế (`Request_ID_01` $\rightarrow$ `Request_ID_58`).
  - Mỗi Request tương ứng với một phiên kiểm tra/chẩn đoán một phương tiện cụ thể (từ 18 đến 82 ảnh/request).
- **Độ phân giải gốc (Native Resolution)**: **$6000 \times 4000$ pixels** (24 Megapixels, chuẩn máy ảnh DSLR kiểm định kỹ thuật).
- **Không gian màu**: 100% RGB, 3 kênh, độ sâu màu 8-bit.
- **Hiện trạng nhãn**: Dữ liệu ảnh thô (chưa có nhãn thủ công đi kèm).

### 1.2. Phân tích Ngữ cảnh Hình ảnh qua DGX Qwen3-VL-30B
Qua lấy mẫu trực quan đa chiều và suy luận bằng mô hình thị giác ngôn ngữ trên DGX, tập dữ liệu có các đặc tính chuyên sâu:
1. **Miền dữ liệu (Domain)**: Toàn bộ là **Khoang động cơ xe hơi (Engine Bay / Under-hood Inspection)**, không phải ngoại thất xe.
2. **Đa dạng góc chụp**:
   - *Góc toàn cảnh (Overview Bay)*: Chụp từ trên xuống, thấy tổng thể máy, nắp cabo, bình nước phụ, hộp cầu chì, bình ắc quy.
   - *Góc cận cảnh Macro (Close-up Inspection)*: Tập trung cực nét vào cụm họng ga (throttle body), cổ hút (intake manifold), cảm biến lưu lượng khí nạp (MAF sensor), cuộn đánh lửa (ignition coils), cọc bình ắc quy.
   - *Góc chẩn đoán thực nghiệm*: Chứa các công cụ đo kiểm trực tiếp (đồng hồ VOM / multimeter đang kẹp que đo hiển thị điện áp $14.2\text{V}$, găng tay kỹ thuật viên đang chỉ điểm chi tiết, mã barcode OEM).
3. **Thách thức thị giác máy tính**:
   - Độ phân giải rất lớn ($6000 \times 4000$) $\rightarrow$ Cần chiến lược nén đa tỷ lệ / SAHI (Slicing Aided Hyper Inference) tránh mất chi tiết cảm biến nhỏ hoặc gây tràn VRAM GPU.
   - Mức độ che khuất (occlusion) và chằng chịt của đường ống cao su, giắc điện, bó dây (wiring harnesses).

---

## 2. BẢNG ĐẶC TẢ LINH KIỆN MỤC TIÊU (20 CLASSES ONTOLOGY)

Dựa trên cấu trúc ảnh thực tế và sự đồng bộ với hệ thống **Innova Automotive Detection** trên DGX, bảng nhãn được định nghĩa chuẩn tại [data_engine_bay.yaml](file:///e:/Research/GEN%20AI/AI%20Vision/Distillation/configs/data_engine_bay.yaml):

| Class ID | Mã linh kiện (`class_name`) | Tên tiếng Việt | Đặc điểm nhận diện & Prompt Text |
| :---: | :--- | :--- | :--- |
| **0** | `battery` | Bình ắc quy 12V | Khối chữ nhật đen, nhãn thông số CCA/Ah |
| **1** | `battery_terminal` | Cọc bình (+/-) | Đầu kẹp chì/đồng đỏ hoặc đen |
| **2** | `fuse_relay_box` | Hộp cầu chì & Rơ-le | Hộp nhựa đen có nắp sơ đồ mạch |
| **3** | `coolant_reservoir` | Bình nước làm mát phụ | Bình nhựa trắng mờ chứa dung dịch xanh/hồng |
| **4** | `radiator_cap` | Nắp két nước tản nhiệt | Nắp kim loại có van áp suất cảnh báo nhiệt |
| **5** | `brake_fluid_reservoir` | Bình dầu phanh (thắng) | Bình nhỏ gắn trên heo thắng tổng (master cylinder) |
| **6** | `washer_fluid_reservoir` | Bình nước rửa kính | Nắp nhựa màu xanh dương đặc trưng |
| **7** | `engine_cover` | Nắp che động cơ | Nắp nhựa trang trí dập logo hoặc nắp dàn cò |
| **8** | `oil_filler_cap` | Nắp châm nhớt máy | Nắp vặn có biểu tượng bình nhớt nhỏ giọt |
| **9** | `oil_dipstick` | Que thăm dầu động cơ | Tay nắm tròn màu vàng hoặc cam nổi bật |
| **10** | `air_filter_box` | Hộp lọc gió động cơ | Hộp nhựa lớn nối trực tiếp ống hút khí nạp |
| **11** | `air_intake_duct` | Đường ống dẫn khí nạp | Ống cao su đen có gân xếp đàn hồi |
| **12** | `maf_sensor` | Cảm biến lưu lượng gió (MAF) | Cụm cảm biến gắn giữa hộp lọc gió và họng ga |
| **13** | `throttle_body` | Cụm bướm ga | Khối nhôm sáng có đĩa xoay bướm ga |
| **14** | `alternator` | Máy phát điện (Dynamo) | Thân hợp kim có rãnh tản nhiệt và puly dây curoa |
| **15** | `ignition_coil` | Bô-bin đánh lửa | Cụm bô-bin cắm thẳng hàng trên nắp quy-lát |
| **16** | `radiator_hose_upper` | Ống nước két trên | Ống cao su đen uốn cong nối vào két tản nhiệt |
| **17** | `serpentine_belt` | Dây curoa tổng | Dây đai cao su có rãnh truyền động |
| **18** | `ecu_module` | Hộp điều khiển động cơ (ECU) | Hộp kim loại gắn giắc cắm nhiều chân (pin) |
| **19** | `multimeter_diagnostic_tool` | Đồng hồ đo VOM & que đo | Thiết bị đo chẩn đoán màu vàng/đỏ có màn hình LCD |

---

## 3. CHIẾN LƯỢC PHÂN CHIA TẬP DỮ LIỆU (GROUPED REQUEST SPLITTING)

> [!IMPORTANT]
> **Quy tắc chống rò rỉ dữ liệu (No Data Leakage)**:  
> Các ảnh trong cùng một `Request_ID` thuộc về cùng 1 xe tại 1 thời điểm. Nếu chia ngẫu nhiên từng ảnh (flat split), mô hình sẽ học vẹt góc nhìn và xe đó ở tập train rồi đem dự đoán ở tập val.  
> **Giải pháp**: Phân chia theo **Cụm Request (Group by Request_ID)** thông qua [split_dataset.py](file:///e:/Research/GEN%20AI/AI%20Vision/Distillation/scripts/data_pipeline/split_dataset.py).

- **Tập Train (75% Requests)**: 21 Request IDs (~980 ảnh) $\rightarrow$ Huấn luyện mô hình Teacher & Student.
- **Tập Validation (12.5% Requests)**: 3-4 Request IDs (~165 ảnh) $\rightarrow$ Đánh giá loss distillation và chọn checkpoint tốt nhất.
- **Tập Test (12.5% Requests)**: 3-4 Request IDs (~162 ảnh) $\rightarrow$ Benchmark độ chính xác (mAP@50, mAP@50-95 Mask) trên xe hoàn toàn mới.

---

## 4. QUY TRÌNH HUẤN LUYỆN KNOWLEDGE DISTILLATION (KD PIPELINE)

```
   [ 1,307 Ảnh Khoang Động Cơ ] 
                 │
                 ▼
   [ Pipeline Auto-Labeling ] ──► Grounding DINO + SAM 2 (Polygon Mask) ──► YOLO Segmentation Dataset
                 │
                 ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                      DGX GPU KNOWLEDGE DISTILLATION ENGINE                             │
 │                                                                                        │
 │      Teacher: D-FINE-L/X (Frozen)                                                      │
 │      ├── Multi-scale P3, P4, P5 Features (Encoder-Decoder)                             │
 │      └── Class Logits & FDR Bins                                                       │
 │               │                                                                        │
 │               ├──► [ Feature Distillation (FGD Loss) ] ◄── [ 1x1 Conv Adapters ]       │
 │               └──► [ Logit Distillation (KL Div Loss) ] ◄── [ Student Heads ]          │
 │                                                                        │               │
 │      Student: YOLO26s-seg (Trainable) ─────────────────────────────────┘               │
 │      └── Supervised Loss: TaskAligned BBox + BCE Cls + BCE/Dice Mask                   │
 └────────────────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
   [ Model Stripping & Export ] ──► Export ONNX Opset 18 ──► TensorRT INT8 (< 15ms trên Edge)
                 │
                 ▼
   [ Tích hợp DGX Qwen3-VL ] ──► OCR mã phụ tùng OEM & Hỗ trợ Chẩn đoán trực quan
```

---

## 5. KẾ HOẠCH THI CÔNG CHI TIẾT (6 TUẦN - 4 SPRINTS)

### Sprint 1: Tiền xử lý Dữ liệu & Auto-Labeling Pipeline (Tuần 1 - Tuần 2)
- [x] Khám phá & phân tích 1,307 ảnh và kết nối phần cứng DGX.
- [ ] **Data Preprocessing**: Viết pipeline resize thông minh từ $6000 \times 4000 \rightarrow 1280 \times 1280$ giữ nguyên tỷ lệ và chi tiết.
- [ ] **Auto-Labeling**: Chạy `auto_label_sam2_dino.py` kết hợp với prompt từ `data_engine_bay.yaml` để tạo tự động polygon segmentation mask cho 20 classes.
- [ ] **Group Split**: Phân chia 28 requests theo tỷ lệ 75 / 12.5 / 12.5 vào `data/car_parts_dataset/{images,labels}/{train,val,test}`.
- [ ] **QC/Review**: Kiểm tra nhanh 50 mẫu polygon masks bằng visualization script.

### Sprint 2: Huấn luyện Baseline Teacher D-FINE trên DGX (Tuần 3)
- [ ] Đồng bộ dataset lên DGX qua `remote_training_dgx.py --sync`.
- [ ] Huấn luyện mô hình Teacher D-FINE-L (hoặc D-FINE-X) trên tập train khoang máy.
- [ ] Đạt mAP@50 (Box) $\ge 80\%$ để làm chuẩn tri thức vững chắc cho việc chưng cất.

### Sprint 3: Huấn luyện Chưng cất Tri thức (Knowledge Distillation) (Tuần 4 - Tuần 5)
- [ ] Khởi tạo 1x1 Conv Adapters trên P3, P4, P5 giữa D-FINE và YOLO26s-seg.
- [ ] Kích hoạt `train_kd.py` trên GPU DGX với cấu hình:
  - $\alpha_{\text{feat}} = 1.0$ (FGD Feature Loss)
  - $\beta_{\text{cls}} = 0.5$ (KL Divergence Loss, $\tau = 3.0$)
  - Mixed Precision (AMP FP16) + AdamW optimizer.
- [ ] So sánh Student KD với Student Baseline thông thường để kiểm chứng độ tăng trưởng mAP ($+4-5\%$).

### Sprint 4: Xuất xưởng Edge, Tối ưu hóa TensorRT & Tích hợp VLM (Tuần 6)
- [ ] Tách bỏ Teacher và Adapter, xuất xưởng mô hình Student tinh gọn sang ONNX Opset 18.
- [ ] Build TensorRT Engine (FP16 / INT8 PTQ) đạt tốc độ suy luận $< 15$ ms/ảnh.
- [ ] Ghép nối với module VLM trên DGX (`qwen2_vl_ocr.py`) để đọc mã OEM phụ tùng tự động.

---

## 6. LỆNH THỰC THI NHANH (QUICK COMMANDS)

```powershell
# 1. Phân chia tập dữ liệu từ 28 Request IDs vào cấu trúc train/val/test:
python scripts/data_pipeline/split_dataset.py --source ./dataset --output ./data/car_parts_dataset --group-by-request

# 2. Sinh nhãn tự động với mô hình nền tảng (Grounding DINO + SAM 2):
python scripts/data_pipeline/auto_label_sam2_dino.py --image_dir ./dataset --output_dir ./data/car_parts_dataset --config configs/data_engine_bay.yaml

# 3. Đồng bộ dữ liệu sang máy chủ DGX (dgx-host):
python remote_training_dgx.py --sync

# 4. Kích hoạt huấn luyện Distillation trên GPU máy chủ DGX:
python remote_training_dgx.py --train --epochs 100 --batch-size 16 --device 0
```
