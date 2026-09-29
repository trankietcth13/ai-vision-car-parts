# KẾ HOẠCH TRIỂN KHAI CHI TIẾT (IMPLEMENTATION PLAN)
## HỆ THỐNG NHẬN DIỆN & PHÂN ĐOẠN LINH KIỆN XE HƠI (VEHICLE COMPONENT DETECTION & SEGMENTATION)
### ÁP DỤNG KNOWLEDGE DISTILLATION (D-FINE -> YOLO-SEG) & VISION-LANGUAGE MODELS (VLM)

---

## 1. TỔNG QUAN DỰ ÁN & MỤC TIÊU KỸ THUẬT

### 1.1. Bối cảnh & Mục tiêu
- **Mục tiêu cốt lõi**: Xây dựng hệ thống thị giác máy tính nhận diện và phân đoạn (instance segmentation) các chi tiết, linh kiện xe hơi với độ chính xác tương đương mô hình Foundation/DETR cỡ lớn, nhưng hoạt động với độ trễ siêu thấp (< 20ms/ảnh) trên thiết bị biên (Edge Hardware: Nvidia Jetson, Máy tính bảng gara, Camera chẩn đoán).
- **Chiến lược cốt lõi**:
  - **Knowledge Distillation (KD)**: Chuyển giao tri thức định vị chi tiết và biểu diễn đặc trưng ngữ cảnh từ mô hình **Teacher (D-FINE-L/X)** sang mô hình **Student (YOLO26s-seg / Ultralytics)**.
  - **Auto-Labeling Pipeline**: Tiết kiệm 90% chi phí gán nhãn thủ công bằng cách tự động hóa qua chuỗi **Grounding DINO + SAM 2** kết hợp quy trình thẩm định Human-in-the-loop (FiftyOne).
  - **Tích hợp VLM**: Mở rộng khả năng tương tác trực quan với **Florence-2** (hỏi đáp/tìm kiếm chi tiết theo ngôn ngữ tự nhiên) và **Qwen2-VL** (đọc mã phụ tùng / OCR số khung VIN bị trầy xước, mờ).

### 1.2. Chỉ số KPI & Tiêu chuẩn Nghiệm thu (Success Metrics)
| Tiêu chí | Mô hình Teacher (D-FINE-X) | Mô hình Student Baseline (YOLO26s-seg) | Mô hình Student KD (Mục tiêu) |
| :--- | :--- | :--- | :--- |
| **mAP@50 (Box)** | $\ge 82.0\%$ | $\approx 73.5\%$ | $\ge \mathbf{78.0\%}$ ($+4.5\%$) |
| **mAP@50-95 (Box)** | $\ge 64.0\%$ | $\approx 51.0\%$ | $\ge \mathbf{56.5\%}$ ($+5.5\%$) |
| **mAP@50-95 (Mask)** | N/A (Teacher box-only) | $\approx 46.0\%$ | $\ge \mathbf{50.5\%}$ ($+4.5\%$) |
| **Độ trễ suy luận (Jetson Orin FP16)** | $\approx 95$ ms | $\approx 12$ ms | $\le \mathbf{13}$ ms ($> 75$ FPS) |
| **Độ trễ suy luận (TensorRT INT8)** | N/A | $\approx 6$ ms | $\le \mathbf{7}$ ms ($> 140$ FPS) |
| **Kích thước mô hình (Weights)** | $\approx 245$ MB | $\approx 24$ MB | $\le \mathbf{24}$ MB |

---

## 2. KIẾN TRÚC HỆ THỐNG & CƠ CHẾ KNOWLEDGE DISTILLATION

```
+---------------------------------------------------------------------------------------------------+
|                                 AUTO-LABELING & DATA PREPARATION                                  |
|   Ảnh Xe Thô ---> Grounding DINO (Text Prompts) ---> SAM 2 (Polygon Mask) ---> FiftyOne (QA/QC)   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                             KNOWLEDGE DISTILLATION TRAINING PIPELINE                              |
|                                                                                                   |
|     +-----------------------------------------+                                                   |
|     |       Teacher: D-FINE-L/X (Frozen)      |                                                   |
|     |   (Backbone -> Hybrid Encoder -> Dec)   |                                                   |
|     +-----------------------------------------+                                                   |
|               | (P3, P4, P5 Features)                                                             |
|               v                                                                                   |
|        [ MSE / FGD Loss ] <==== [ 1x1 Conv Adapters ] <==== (P3, P4, P5 Features)                 |
|               ^                                                        |                          |
|               |                                                        v                          |
|     +-----------------------------------------------------------------------+                     |
|     |                      Student: YOLO26s-seg (Trainable)                 |                     |
|     |            Backbone & C3k2/C2f ---> Neck (PAN/FPN) ---> Seg Heads     |                     |
|     +-----------------------------------------------------------------------+                     |
|               |                                                                                   |
|               v                                                                                   |
|     [ Supervised Loss: TaskAligned BBox (CIoU + DFL) + Cls (BCE) + Mask (BCE + Dice) ]            |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                EDGE DEPLOYMENT & VLM INTEGRATION                                  |
|   Strip Teacher/Adapters ---> Export ONNX / TensorRT INT8 ---> Jetson Orin / Android Gara         |
|   Hỗ trợ Florence-2 (Interactive visual QA) + Qwen2-VL (Part Number / VIN OCR)                    |
+---------------------------------------------------------------------------------------------------+
```

### 2.1. Thiết kế Chi tiết Module Adapter
Do kích thước số kênh đặc trưng (channels) giữa D-FINE (Transformer-based) và YOLO26s-seg (CNN-based) có sự chênh lệch:
- **Tầng P3 (Stride 8)**: Chiếu $C_{\text{student}}^{P3} \rightarrow C_{\text{teacher}}^{P3}$
- **Tầng P4 (Stride 16)**: Chiếu $C_{\text{student}}^{P4} \rightarrow C_{\text{teacher}}^{P4}$
- **Tầng P5 (Stride 32)**: Chiếu $C_{\text{student}}^{P5} \rightarrow C_{\text{teacher}}^{P5}$

Mỗi Adapter bao gồm:
$$\text{Adapter}(X) = \text{GELU}(\text{BatchNorm2d}(\text{Conv2d}_{1\times 1}(X)))$$

### 2.2. Thiết kế Hàm Mất Mát (Loss Engine)
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{supervised}} + \alpha \cdot \mathcal{L}_{\text{feat\_kd}} + \beta \cdot \mathcal{L}_{\text{cls\_kd}}$$

1. **Supervised Loss (Student)**:
   $$\mathcal{L}_{\text{supervised}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{mask}} (\mathcal{L}_{\text{BCE\_mask}} + \mathcal{L}_{\text{Dice}})$$
2. **Feature Distillation Loss ($\mathcal{L}_{\text{feat\_kd}}$)**:
   Sử dụng **Focal and Global Distillation (FGD)** hoặc Masked Feature Loss để ưu tiên ép Student học sâu các vùng có đối tượng (Foreground attention mask $M$):
   $$\mathcal{L}_{\text{feat\_kd}} = \sum_{l \in \{P3, P4, P5\}} \frac{1}{H_l W_l} \sum_{h, w} M_{h,w} \cdot \left\| \phi_l(F_{\text{student}}^{(l)})_{h,w} - F_{\text{teacher}}^{(l)}{}_{h,w} \right\|_2^2$$
3. **Classification Logit Loss ($\mathcal{L}_{\text{cls\_kd}}$)**:
   Sử dụng Softmax với Temperature $\tau = 3.0$:
   $$\mathcal{L}_{\text{cls\_kd}} = \tau^2 \cdot D_{\text{KL}}\left(\sigma\left(\frac{Z_{\text{student}}}{\tau}\right) \parallel \sigma\left(\frac{Z_{\text{teacher}}}{\tau}\right)\right)$$

---

## 3. BẢNG PHÂN RÃ HẠNG MỤC CÔNG VIỆC (WBS) & TIẾN ĐỘ 7 TUẦN

```
Tuần 1: Thu thập dữ liệu & Setup Pipeline Auto-Labeling (SAM 2 + Grounding DINO)
Tuần 2: Hoàn thiện Dataset, Data Cleaning (FiftyOne) & Label Validation
Tuần 3: Huấn luyện & Đánh giá Baseline Teacher D-FINE-L/X
Tuần 4: Xây dựng Kiến trúc KD Pipeline, Adapter Layers & Hooks
Tuần 5: Huấn luyện Distillation Student YOLO26s-seg & Tuning Hyperparameters
Tuần 6: Benchmark, Model Stripping, Export TensorRT (FP16/INT8 PTQ)
Tuần 7: Tích hợp Edge VLM (Florence-2 / Qwen2-VL), Kiểm thử End-to-End & Bàn giao
```

### TUẦN 1: Chuẩn bị Dữ liệu & Xây dựng Auto-Labeling Pipeline
- **Mục tiêu**: Thiết lập hệ thống tự động gán nhãn cho 20,000+ ảnh linh kiện xe hơi.
- **Nhiệm vụ cụ thể**:
  1. Chuẩn hóa bộ nhãn ngoại thất & nội thất xe (24 classes):
     - `front_bumper`, `rear_bumper`, `hood`, `headlight_left`, `headlight_right`, `taillight_left`, `taillight_right`, `front_door_left`, `front_door_right`, `rear_door_left`, `rear_door_right`, `fender_front_left`, `fender_front_right`, `fender_rear_left`, `fender_rear_right`, `side_mirror_left`, `side_mirror_right`, `grille`, `windshield`, `rear_windshield`, `roof`, `wheel`, `quarter_panel`, `trunk_lid`.
  2. Viết module `auto_label_sam2_dino.py`:
     - Tích hợp Grounding DINO 1.5 với dynamic text prompts.
     - Nối output bounding box sang SAM 2 để sinh polygon segmentation masks chuẩn xác cao.
  3. Cài đặt pipeline phân tán/batching trên GPU để tăng tốc độ gán nhãn đạt > 5 ảnh/giây.

### TUẦN 2: Thẩm định Dữ liệu (Quality Control) & Dataset Partition
- **Mục tiêu**: Đảm bảo Ground Truth sạch, tỷ lệ gán nhãn sai/hallucination < 3%.
- **Nhiệm vụ cụ thể**:
  1. Triển khai script kiểm tra chất lượng nhãn bằng **FiftyOne**:
     - Lọc bỏ các polygon bị rách viền, diện tích quá nhỏ (< 100 pixels), hoặc box trùng nhau (IoU > 0.85).
     - Thiết lập giao diện Web UI FiftyOne cho đội ngũ QA review ngẫu nhiên 10% dataset.
  2. Chia tập dữ liệu chuẩn:
     - **Train**: 70% (~14,000 ảnh)
     - **Validation**: 15% (~3,000 ảnh)
     - **Test (Gara benchmark)**: 15% (~3,000 ảnh với các góc chụp thực tế, ánh sáng yếu, xe tai nạn biến dạng).
  3. Xuất dataset theo chuẩn **YOLO Segmentation Format** (`labels/*.txt` và `data_car_parts.yaml`).

### TUẦN 3: Huấn luyện & Tối ưu Teacher Model (D-FINE-L/X)
- **Mục tiêu**: Có mô hình Teacher chuyên gia đạt $mAP_{50-95} \ge 60\%$.
- **Nhiệm vụ cụ thể**:
  1. Thiết lập codebase D-FINE từ mã nguồn gốc, load trọng số pre-trained COCO.
  2. Tinh chỉnh (fine-tune) trên tập dữ liệu xe hơi 24 classes:
     - Kích thước ảnh: $640 \times 640$ (hoặc multi-scale $640 \sim 800$).
     - Optimizer: AdamW, LR ban đầu $1 \times 10^{-4}$, weight decay $1 \times 10^{-4}$.
     - Epochs: 60 - 80 epochs với Cosine Annealing scheduler.
  3. Trích xuất checkpoint tốt nhất (`dfine_teacher_best.pth`), kiểm thử độc lập trên test set.

### TUẦN 4: Thiết kế Framework Knowledge Distillation
- **Mục tiêu**: Hoàn thiện toàn bộ mã nguồn KD liên kết giữa D-FINE và YOLO26s-seg.
- **Nhiệm vụ cụ thể**:
  1. Xây dựng module trích xuất đặc trưng `teacher_wrapper.py`:
     - Đóng băng toàn bộ trọng số Teacher (`param.requires_grad = False`).
     - Đặt Forward Hooks tại 3 tầng output của FPN/Encoder tương ứng với Stride 8, 16, 32.
  2. Xây dựng module `adapters.py`:
     - Tạo các khối chiếu 1x1 Conv + BatchNorm + GELU để đồng bộ feature dimensions giữa Student và Teacher.
  3. Cài đặt `distillation_loss.py`:
     - Module tính toán Feature Loss (FGD / Masked L2) giữa Student Adapter outputs và Teacher features.
     - Module tính toán Soft-label Logit Loss qua KL-Divergence.

### TUẦN 5: Huấn luyện Student Model (Distillation Training)
- **Mục tiêu**: Huấn luyện YOLO26s-seg hấp thụ tri thức từ D-FINE, vượt trội baseline thông thường.
- **Nhiệm vụ cụ thể**:
  1. Huấn luyện 2 giai đoạn:
     - **Giai đoạn 1 (Warmup Feature Alignment - 15 epochs)**:
       Giữ trọng số task loss nhỏ, tập trung tối ưu $\mathcal{L}_{\text{feat\_kd}}$ để Adapter và Student Backbone đồng bộ biểu diễn không gian với Teacher.
     - **Giai đoạn 2 (Joint Full Distillation - 85 epochs)**:
       Mở toàn bộ hàm mất mát: Supervised Task Loss + Feature KD Loss ($\alpha = 0.5$) + Class Logit Loss ($\beta = 0.2$).
  2. Áp dụng Data Augmentation nâng cao:
     - Mosaic, Albumentations (ColorJitter mô phỏng ánh sáng gara, MotionBlur mô phỏng rung tay khi cầm tablet).
  3. Đánh giá song song: Huấn luyện đồng thời 1 model YOLO26s-seg thuần (không KD) trên cùng dữ liệu để so sánh đối chứng (Ablation Study).

### TUẦN 6: Stripping, Tối ưu hóa & Đóng gói Triển khai Edge
- **Mục tiêu**: Tối ưu hóa mô hình đạt độ trễ < 15ms trên Jetson Orin / < 25ms trên Android.
- **Nhiệm vụ cụ thể**:
  1. **Model Pruning & Stripping**:
     - Tách bỏ hoàn toàn Teacher model và các tầng Adapter.
     - Trích xuất checkpoint chỉ chứa cấu trúc YOLO26s-seg (`yolo26s_seg_distilled.pt`).
  2. **Xuất xưởng ONNX & TensorRT**:
     - Export sang ONNX với Dynamic batch size và Opset 18.
     - Chạy `onnxsim` để gấp các node hằng số và chuẩn hóa đồ thị tính toán.
     - Biên dịch sang TensorRT Engine (FP16).
     - Biên dịch sang TensorRT Engine (INT8) sử dụng **PTQ (Post-Training Quantization)** với tập calibration gồm 1,000 ảnh thực tế từ gara.
  3. **Đo đạc Benchmark**:
     - Kiểm tra tính tương thích, độ suy hao mAP sau lượng tử hóa (yêu cầu $\Delta mAP_{\text{INT8}} \le 0.8\%$).
     - Đo FPS, GPU Memory (VRAM), và nhiệt độ hoạt động liên tục trong 30 phút trên Jetson Orin Nano.

### TUẦN 7: Tích hợp VLM Đa Phương Thức & Kiểm thử Toàn diện
- **Mục tiêu**: Hoàn thiện tính năng thông minh bổ trợ (Florence-2 & Qwen2-VL) và đóng gói SDK.
- **Nhiệm vụ cụ thể**:
  1. **Tích hợp Florence-2 Microservice**:
     - Đóng gói Florence-2-base (~0.23B) chạy on-device.
     - Cho phép kỹ thuật viên gõ text: *"Tìm đèn pha trái bị nứt"* hoặc *"Kiểm tra khe hở nắp capo"* -> Florence-2 thực hiện Open-vocabulary Grounding.
  2. **Tích hợp Qwen2-VL OCR Agent**:
     - Sau khi YOLO26s-seg khoanh vùng các bộ phận có tem nhãn, crop bounding box độ phân giải cao gửi sang Qwen2-VL.
     - Đọc mã Part Number, số phụ tùng, số khung VIN chìm/bị mờ xước.
  3. **Kiểm thử chấp nhận người dùng (UAT)**:
     - Thử nghiệm thực tế tại gara: test trên 5 dòng xe khác nhau (Sedan, SUV, Bán tải, Xe điện, Xe tải nhẹ).
     - Hoàn thiện tài liệu API, hướng dẫn cài đặt, và bàn giao mã nguồn.

---

## 4. THIẾT KẾ CẤU TRÚC CODEBASE CHI TIẾT

Cấu trúc thư mục được thiết kế chuẩn module hóa cho môi trường Enterprise:

```text
Distillation/
├── configs/
│   ├── data_car_parts.yaml          # Định nghĩa 24 classes, đường dẫn train/val/test
│   ├── teacher_dfine_x.yaml         # Config kiến trúc D-FINE-X
│   ├── student_yolo26s.yaml         # Config kiến trúc YOLO26s-seg
│   └── kd_hyperparams.yaml          # Hệ số alpha, beta, temperature tau, epochs
├── scripts/data_pipeline/
│   ├── auto_label_sam2_dino.py      # Script gán nhãn tự động Grounding DINO + SAM 2
│   ├── visual_qa_qc.py              # Script FiftyOne kiểm tra và lọc nhãn bất thường
│   ├── split_dataset.py             # Chia tập dữ liệu Train/Val/Test
│   └── format_converter.py          # Chuyển đổi qua lại giữa COCO và YOLO Seg
├── models/
│   ├── adapters.py                  # Module Adapter 1x1 Conv + BN + GELU cho P3, P4, P5
│   ├── teacher_wrapper.py           # Wrapper bắt hooks P3/P4/P5 của D-FINE Teacher
│   └── student_yolo_seg.py          # Wrapper YOLO26s-seg trích xuất intermediate features
├── src/distillation/
│   ├── feature_loss.py              # FGD Loss, Masked MSE Loss cho feature map
│   ├── logit_loss.py                # Softmax KL-Divergence Loss
│   └── kd_engine.py                 # Custom Training Loop quản lý gradient & backprop
├── vlm_integration/
│   ├── florence2_agent.py           # Module tương tác ngôn ngữ tự nhiên tìm kiếm chi tiết
│   └── qwen2_vl_ocr.py              # Module đọc mã phụ tùng Part Number & số VIN
├── scripts/deployment/
│   ├── export_onnx.py               # Script xuất ONNX và tối ưu hóa graph
│   ├── build_tensorrt_engine.py     # Script build TensorRT FP16/INT8 Calibrator
│   └── edge_inference_demo.py       # Pipeline suy luận camera thời gian thực trên Edge
├── tests/
│   ├── test_adapters.py             # Unit test kiểm tra output shape của Adapter
│   └── test_distillation_loss.py    # Unit test kiểm tra tính toán loss
├── requirements.txt
└── README.md
```

---

## 5. QUẢN TRỊ RỦI RO & PHƯƠNG ÁN DỰ PHÒNG (RISK MITIGATION)

| Rủi ro Kỹ thuật | Mức độ | Hậu quả | Phương án Xử lý / Kế hoạch Dự phòng |
| :--- | :--- | :--- | :--- |
| **Bất đồng bộ Shape giữa DETR & YOLO** | Cao | Không tính được Feature Loss hoặc sập VRAM | Sử dụng Adaptive Average Pooling trong Adapter để chuẩn hóa spatial dimensions $(H, W)$ trước khi đưa vào hàm Loss. |
| **Grounding DINO bị Hallucination** | Trung bình | Nhãn sai nhiều chi tiết khó (cản trước vs lưới tản nhiệt) | Dùng Prompt Engineering cụ thể (kèm visual attributes) và đặt ngưỡng Box Confidence $\ge 0.45$. Thẩm định tự động qua FiftyOne. |
| **Suy hao độ chính xác khi Quantize INT8** | Trung bình | Giảm mAP trên các linh kiện nhỏ (đèn sương mù, gương) | Sử dụng **Quantization-Aware Training (QAT)** nếu PTQ làm giảm quá $1\%$ mAP, hoặc giữ FP16 cho các tầng Segmentation Head. |
| **Tràn VRAM khi train đồng thời Teacher & Student** | Cao | Không thể train batch size lớn trên GPU thương mại | Đóng băng hoàn toàn Teacher (`torch.no_grad()`), bật `torch.cuda.amp.autocast()` (Mixed Precision FP16), và áp dụng Gradient Accumulation. |

---

## 6. DANH MỤC TÀI NGUYÊN PHẦN CỨNG & MÔI TRƯỜNG KHUYẾN NGHỊ

### 6.1. Môi trường Huấn luyện (R&D Server / Cloud)
- **GPU**: $1 \times$ Nvidia RTX 4090 (24GB VRAM) hoặc $1 \times$ A100 (40GB/80GB).
- **CPU**: 16 Cores, RAM 64GB+.
- **Hệ điều hành**: Ubuntu 22.04 LTS hoặc Windows 11 với WSL2.
- **CUDA/cuDNN**: CUDA 12.2 / cuDNN 8.9.
- **Thư viện chính**: PyTorch 2.3+, Ultralytics, HuggingFace Transformers, Timm, FiftyOne, TensorRT 10.x.

### 6.2. Thiết bị Triển khai Thực tế (Edge Hardware)
- **Phương án tối ưu (Khuyến nghị)**: **Nvidia Jetson Orin Nano (8GB)** hoặc **Orin NX (16GB)**.
  - Vừa chạy YOLO26s-seg (TensorRT INT8) với tốc độ $> 100$ FPS.
  - Vừa có thể tải thêm Florence-2 (FP16) on-demand để phục vụ giao tiếp trực quan.
- **Phương án di động (Mobile Tablet Gara)**: Máy tính bảng hỗ trợ NPU/GPU Adreno hoặc máy tính bảng công nghiệp chạy Windows 11/Android (ONNX Runtime Execution Provider cho DirectML hoặc QNN).
