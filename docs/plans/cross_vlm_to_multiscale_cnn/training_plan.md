# Kế hoạch Huấn luyện Thực chiến: Cặp Cross-Architecture (Transformer-based VLM → CNN Multi-Scale)

**Ngày lập:** 2026-10-02  
**Mô hình chiến lược:** Transformer-based VLM (Qwen2.5-VL / Qwen3-VL / Grounding DINO EVA-02/ViT-H/Swin-L) $\longrightarrow$ CNN/Hybrid Multi-Scale (CSP-Darknet C3k2 trong YOLO11/YOLO26-seg hoặc HGNetv2 trong D-FINE/RT-DETR)  
**Trạng thái:** Sẵn sàng triển khai trên DGX Server A100/H100 & Edge TensorRT / ONNX  

---

## 1. Bối cảnh & Lý do Đây là Cặp Đôi "Thực Chiến Nhất" (The Most Pragmatic Battle-Tested Pair)

Trong bài toán thị giác máy tính công nghiệp và kiểm định ô tô (Car Parts & Engine-Bay Inspection), các kỹ sư thường đối mặt với một nghịch lý:
- **Mô hình VLM lớn (Qwen-VL, Grounding DINO ViT-H):** Có "bộ não" cực kỳ thông minh, hiểu sâu sắc ngữ cảnh không gian phức tạp và nhận diện được linh kiện lạ nhờ prompt văn bản (open-vocabulary), nhưng **tốc độ quá chậm** (300ms – 2s/ảnh), kích thước hàng GB, không thể gắn vào camera cầm tay hay chạy real-time ở trạm đăng kiểm.
- **Mô hình CNN đa tỉ lệ (YOLO11-seg, D-FINE HGNetv2):** Có "thân thủ" nhanh như chớp (< 2–4ms trên GPU/NPU), độ chính xác định vị pixel tuyệt vời nhờ **Inductive Bias** mạnh mẽ của tích chập (tính bất biến dịch chuyển, nhận diện vân bề mặt, góc cạnh cục bộ), nhưng **thiếu tầm nhìn toàn cảnh**, dễ nhầm lẫn các vật thể có bề ngoài tương tự nếu thiếu nhãn giám sát.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     TEACHER: VLM Transformer (Đầu óc Uyên bác)                    │
│   • Backbone: Qwen2.5-VL / Qwen3-VL / Grounding DINO (EVA-02 / ViT-H / Swin-L)   │
│   • Điểm mạnh: Hiểu ngữ nghĩa trừu tượng, liên kết Text-Image, zero-shot           │
│   • Thách thức: Quá nặng để chạy real-time (tốc độ ~FPS < 5, VRAM > 16GB)         │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                  Cross-Architecture Multi-Scale KD Pipeline
     ┌────────────────────────────────────┼────────────────────────────────────┐
     │                                    │                                    │
Multi-Scale Reconstruction          Inductive Bias                      Prompt-Guided
 FPN Adapter (1D ViT → 2D)         Affinity Transfer                 Decoupled Logit KD
     │                                    │                                    │
     └────────────────────────────────────┼────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────▼─────────────────────────────────────────┐
│                     STUDENT: Multi-Scale CNN (Thân thủ Tia chớp)                  │
│   • Backbone: C3k2-CSPDarknet (YOLO11/26-seg) hoặc HGNetv2 (D-FINE / RT-DETR)    │
│   • Điểm mạnh: Inductive Bias tích chập, định vị pixel sắc nét, tốc độ 2–4ms      │
│   • Triển khai: TensorRT / ONNX Runtime FP16/INT8 trên máy trạm xưởng & Mobile    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Sự kết hợp hoàn hảo:** Kỹ thuật chưng cất liên kiến trúc (Cross-Architecture KD) chắt lọc toàn bộ "trí tuệ toàn cảnh" của Transformer VLM đổ vào "bộ khung tích chập đa tỉ lệ" của Student CNN, giúp Student đạt độ chính xác tương đương mô hình khổng lồ trong khi vẫn giữ nguyên tốc độ siêu âm mili-giây.

---

## 2. Phân tích Sâu Kiến trúc Teacher & Student

### 2.1. Lựa chọn Teacher: Qwen-VL vs. Grounding DINO (EVA-02 / ViT-H / Swin-L)

| Tiêu chí kỹ thuật | Qwen2.5-VL / Qwen3-VL (Vision Tower) | Grounding DINO (EVA-02 / ViT-H / Swin-L) |
|---|---|---|
| **Cốt lõi Kiến trúc** | NaViT / ViT linh hoạt với 2D-RoPE (Rotary Position Embedding) động, xử lý mọi tỉ lệ khung hình | Hierarchical ViT (EVA-02/Swin-L) kết hợp Text-Vision Bi-directional Cross-Attention Decoder |
| **Đặc trưng Ngữ nghĩa** | Siêu mạnh về tương quan bối cảnh logic: liên hệ giữa mô tả lỗi chẩn đoán (`src/jev/`) và hình ảnh | Đạt đỉnh cao về Grounding: liên kết chính xác bounding box với từng cụm từ ("alternator behind serpentine belt") |
| **Sẵn sàng trong Hệ thống** | **Có sẵn trên DGX Server** (`vllm_model: "qwen3-vl-30b"` theo `configs/hardware_dgx.yaml`) | Sẵn sàng qua repository (`GroundingDINO` trong `requirements.txt`) |
| **Ứng dụng KD tốt nhất** | Trích xuất feature ngữ nghĩa tổng quát và sinh pseudo-labels tự động cho ảnh unlabeled | Trích xuất token affinity map và feature đa tỉ lệ có căn chỉnh prompt |

### 2.2. Lựa chọn Student: CSP-Darknet (C3k2) vs. HGNetv2 (D-FINE / RT-DETR)

| Tiêu chí kỹ thuật | CSP-Darknet với C3k2 (YOLO11 / YOLO26-seg) | HGNetv2 (D-FINE / RT-DETR) |
|---|---|---|
| **Cấu trúc khối cơ bản** | **C3k2 block**: Kết hợp CSP (Cross Stage Partial) với RepVGG 3x3 depthwise nhánh kép + SPPF | **HGNetv2 block**: Hierarchical Grained Conv với Learnable Affine Transform + Light-weight DW-Conv |
| **Cơ chế dự đoán** | Anchor-free decoupled head (Task-aligned Assigner) kết hợp Proto-head phân đoạn mask | Query-based Transformer Decoder (DETR style) với Fine-grained Box Boundary Refinement |
| **Độ trễ TensorRT (DGX)** | **2.5 – 3.8 ms** (cho bản small/medium) | **3.0 – 4.5 ms** (cho bản small/medium) |
| **Điểm mạnh cục bộ** | Mask segmentation cực kỳ ổn định, cộng đồng hỗ trợ lớn, deploy dễ dàng qua Ultralytics | Không cần NMS (Non-Maximum Suppression), độ chính xác biên bounding box cực bén nhờ D-FINE |

> **Khuyến nghị lựa chọn:**
> 1. **Dòng YOLO (YOLO11-seg C3k2):** Dành cho hệ thống cần cả Bounding Box + Pixel Mask phân đoạn chi tiết linh kiện.
> 2. **Dòng D-FINE (HGNetv2):** Dành cho hệ thống kiểm tra nhanh Bounding Box không cần NMS với độ trễ thấp nhất.

---

## 3. Ba Thách thức Kỹ thuật và Giải pháp Cốt lõi (Cross-Architecture Bridging)

### Thách thức 1: Token phẳng 1D (ViT) $\neq$ Feature Pyramid 2D đa tỉ lệ ($P_3, P_4, P_5$)
* **Bản chất:** ViT xuất ra một chuỗi token phẳng $[B, N_{patches}, D]$ (với $D = 1024$ hoặc $1280$), trong khi CNN phân cấp yêu cầu các feature map 2D ở 3 độ phân giải khác nhau:
  - $P_3$: $\frac{H}{8} \times \frac{W}{8} \times C_3$ (vật nhỏ: ốc, giắc cắm, kẹp ống).
  - $P_4$: $\frac{H}{16} \times \frac{W}{16} \times C_4$ (vật vừa: nắp dầu, bình dầu phanh, cảm biến).
  - $P_5$: $\frac{H}{32} \times \frac{W}{32} \times C_5$ (vật lớn: ắc quy, vỏ máy, bầu lọc gió).
* **Giải pháp: Multi-Scale Reconstruction FPN Adapter:**
  1. Unflatten chuỗi token ViT về không gian 2D gốc: $[B, \frac{H}{16}, \frac{W}{16}, D]$.
  2. Xây dựng một Adapter đa tầng gồm:
     - **Tầng $P_4$:** `Conv2d(1x1) -> LayerNorm -> GELU` chiếu thẳng kênh về $C_4$.
     - **Tầng $P_3$:** `Deconv2d(2x2, stride=2)` nội suy kích thước lên $\frac{H}{8} \times \frac{W}{8}$, sau đó qua `Conv2d(3x3)`.
     - **Tầng $P_5$:** `Conv2d(3x3, stride=2)` hạ độ phân giải xuống $\frac{H}{32} \times \frac{W}{32}$.

$$\begin{aligned}
\tilde{F}_T^{(4)} &= \text{Proj}_4(\text{Reshape}(Z_{ViT})) \\
\tilde{F}_T^{(3)} &= \text{Upsample}(\tilde{F}_T^{(4)}) + \text{Proj}_3(\text{Early\_ViT\_Tokens}) \\
\tilde{F}_T^{(5)} &= \text{Downsample}(\tilde{F}_T^{(4)})
\end{aligned}$$

---

### Thách thức 2: Dung hòa giữa Global Self-Attention và Inductive Bias của ConvNet
* **Bản chất:** Transformer chú ý tới toàn cảnh (global relations) nhưng yếu ở việc phân biệt ranh giới cục bộ. CNN có inductive bias cực tốt ở viền cạnh cục bộ nhưng dễ bị "mù" mối quan hệ không gian xa (ví dụ: không biết bình nước làm mát nối với két tản nhiệt ở vị trí nào).
* **Giải pháp: Cross-Attention to Convolutional Affinity Transfer (CAT):**
  - Trích xuất ma trận tương quan giữa các patch token của Teacher:
    $$A_T(i, j) = \frac{Q_T(i) K_T(j)^T}{\sqrt{d}}$$
  - Trích xuất ma trận tương quan vị trí của Student CNN sau tầng $1 \times 1$ conv:
    $$A_S(i, j) = \frac{F_S(i) \cdot F_S(j)^T}{\|F_S(i)\| \|F_S(j)\|}$$
  - Ép Student học theo ma trận tương quan $A_T$ thông qua L2 loss hoặc KL divergence:
    $$L_{affinity} = \frac{1}{N^2} \sum_{i,j} \|A_T(i, j) - A_S(i, j)\|^2$$
  - Nhờ đó, Student CNN **giữ nguyên khả năng bắt viền cực bén** của tích chập, nhưng **nạp thêm trực giác không gian toàn cảnh** từ Teacher ViT.

---

### Thách thức 3: Chi phí Tính toán Khổng lồ của Teacher VLM trong Quá trình Train
* **Bản chất:** Qwen3-VL 30B hoặc Grounding DINO EVA-02 tiêu tốn rất nhiều VRAM. Nếu chạy forward Teacher song song trong từng batch của Student thì GPU sẽ cạn kiệt bộ nhớ hoặc tốc độ train bị nghẽn (từ 120 FPS của YOLO giảm xuống còn 1.5 FPS).
* **Giải pháp: Chiến lược "Offline Feature & Label Caching" trên DGX:**
  1. Chạy Teacher VLM một lần duy nhất trên toàn bộ kho dữ liệu (bao gồm cả ảnh có nhãn và ảnh crawl chưa nhãn).
  2. Lưu (cache) sẵn ra ổ SSD NVMe:
     - Các feature map đa tỉ lệ $[\tilde{F}_T^{(3)}, \tilde{F}_T^{(4)}, \tilde{F}_T^{(5)}]$.
     - Ma trận Attention Affinity $A_T$.
     - Soft class logits và predicted boxes/masks có độ tin cậy cao.
  3. Khi huấn luyện Student: **GPU hoàn toàn giải phóng khỏi Teacher VLM**, nạp trực tiếp feature từ bộ nhớ cache NVMe. **Tốc độ train tăng 10 lần**, VRAM chỉ cần 6–8 GB!

---

## 4. Lộ trình Triển khai Huấn luyện 5 Giai đoạn

```
Tuần 1                 Tuần 2                 Tuần 3                 Tuần 4
[ Phase 0 & 1 ] ──────► [    Phase 2    ] ──────► [    Phase 3    ] ──────► [ Phase 4 & 5 ]
VLM Cache Pipeline     Student Baseline       Cross-Architecture KD  Export TRT & Edge
& FPN Adapters         (YOLO11 / D-FINE)      (Cached Feature + DKD) Field Validation
```

### Giai đoạn 0: Thiết kế Module Multi-Scale Adapter & Cache Extractor
- Xây dựng file script trích xuất: `scripts/training/extract_vlm_teacher_cache.py`.
- Thiết kế module `src/distillation/adapters/vlm_fpn_adapter.py`:
  - Khối chiếu token 1D sang $P_3, P_4, P_5$.
  - Module chuẩn hóa LayerNorm + Smooth L1 / Cosine Similarity loss.

### Giai đoạn 1: Chạy Offline Caching trên DGX Server
- Khởi chạy trích xuất đặc trưng bằng DGX GPU (A100/H100) tận dụng Qwen3-VL hoặc Grounding DINO EVA-02:
  ```bash
  python scripts/training/extract_vlm_teacher_cache.py \
      --vlm-endpoint http://172.16.110.221:8000/v1 \
      --data configs/data_engine_bay.yaml \
      --unlabeled data/unlabeled_exterior \
      --output-cache /data/cache/vlm_features_v1 \
      --imgsz 640
  ```
- Kết quả: Thư mục cache chứa nén `.h5` / memmap các tensor đặc trưng tương ứng với từng ảnh.

### Giai đoạn 2: Huấn luyện Baseline Student Độc lập (Không KD)
- Huấn luyện `yolo11m-seg` (hoặc `dfine_s`) 100 epoch trực tiếp trên dữ liệu để làm thước đo đối chứng tiêu chuẩn.
- Ghi nhận: mask mAP50-95, mAP50, Box mAP và Latency.

### Giai đoạn 3: Huấn luyện Cross-Architecture KD Đa Tỉ Lệ
- Khởi chạy script chưng cất:
  ```bash
  python scripts/training/train_cross_vlm_kd.py \
      --kd configs/kd_hyperparams_cross_vlm_cnn.yaml \
      --feature-cache /data/cache/vlm_features_v1 \
      --student yolo11m-seg.pt \
      --data configs/data_engine_bay.yaml \
      --epochs 120 --batch 16 --imgsz 640 \
      --name kd_cross_vlm_yolo11m
  ```
- Áp dụng kỹ thuật:
  - Multi-Scale FGD ($P_3, P_4, P_5$).
  - Decoupled Knowledge Distillation (TCKD + NCKD) trên class logits.
  - Trung bình 5 checkpoint tốt nhất (`average_checkpoints.py --top-k 5`).

### Giai đoạn 4: Tối ưu Hóa Biên & Export TensorRT / ONNX
- Export Student model sang ONNX và TensorRT engine trên DGX:
  ```bash
  python scripts/deployment/export_onnx.py --weights runs/train_kd/kd_cross_vlm_yolo11m/weights/avg5.pt --opset 18
  trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16
  ```
- Đo lường độ trễ chi tiết từng layer (kernel profiler).

### Giai đoạn 5: Thẩm định E2E & Thực chiến tại Trạm Kiểm Định
- Kiểm tra toàn bộ test suite trên 3 xe mới chưa từng thấy (`tests/test_benchmark_engine_bay.py`).
- Đánh giá khả năng nhận diện các trường hợp khó:
  - Dây điện ngoằn ngoèo bị che khuất (`wiring_harness`).
  - Cảm biến nhỏ nằm sâu dưới gầm động cơ (`o2_sensor`, `knock_sensor`).
  - Kiểm tra độ ổn định FPS trên video luồng RTSP camera xưởng.

---

## 5. Đặc tả Siêu tham số Huấn luyện (Hyperparameter Specifications)

File cấu hình được chuẩn hóa tại `configs/kd_hyperparams_cross_vlm_cnn.yaml`:

```yaml
distillation:
  # Cấu hình Teacher VLM
  teacher:
    name: "qwen3-vl-30b"                      # hoặc grounding_dino_eva02
    cache_path: "/data/cache/vlm_features_v1" # Sử dụng offline cache
    feature_levels: ["P3", "P4", "P5"]
    teacher_dim: 1024                         # Kích thước embedding của ViT

  # Cấu hình Student CNN Multi-scale
  student:
    architecture: "yolo11m-seg"               # hoặc dfine_hgnetv2_m
    backbone: "C3k2-CSPDarknet"
    channels:
      P3: 128
      P4: 256
      P5: 512

  # Trọng số Loss Chưng cất
  alpha_feature: 2.0                          # Trọng số chuyển giao đặc trưng đa tỉ lệ
  beta_cls: 1.0                              # Trọng số phân loại Decoupled KD
  gamma_loc: 1.0                             # Trọng số phân phối biên DFL
  lambda_affinity: 0.8                       # Trọng số học quan hệ không gian (CAT)

  # Cấu hình Feature KD Đa Tỉ Lệ (Multi-scale FGD)
  multi_scale_fgd:
    weights:
      P3: 1.2                                 # Tăng trọng số P3 để bắt vật nhỏ
      P4: 1.0
      P5: 0.8
    alpha_fg: 1.5
    alpha_bg: 0.2
    temperature: 0.5

  # Decoupled Class KD
  dkd:
    alpha_tckd: 1.0
    beta_nckd: 0.6
    temperature: 3.0

training:
  epochs: 120
  warmup_epochs: 5
  batch: 16
  imgsz: 640
  optimizer: "SGD"                            # SGD kết hợp Momentum rất ổn định cho CNN
  lr0: 0.01
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  cos_lr: true
  amp: true
  workers: 8
  seed: 42
  save_period: 5

  # Data Augmentation thực chiến
  mosaic: 1.0
  close_mosaic: 15
  copy_paste: 0.3
  mixup: 0.1
  degrees: 5.0
  scale: 0.5
```

---

## 6. Tiêu chuẩn Nghiệm thu & So sánh Hiệu năng

| Chỉ số | Baseline Độc lập (`yolo11m-seg`) | Mục tiêu sau khi Distill từ VLM | Mức Cải thiện |
|---|---|---|---|
| **Mask mAP50-95 (Xe mới test)** | 0.354 | **$\ge 0.385 - 0.395$** | **+3.1 đến +4.1 điểm** |
| **Mask mAP50** | 0.577 | **$\ge 0.620$** | **+4.3 điểm** |
| **Box mAP50-95** | 0.390 | **$\ge 0.430$** | **+4.0 điểm** |
| **Độ nhầm lẫn giữa các class tương tự** | 14.2% lỗi | **$\le 6.5\%$ lỗi** | **Giảm hơn 50% lỗi nhầm** |
| **Độ trễ TensorRT FP16 (DGX)** | 3.8 ms | **3.8 ms** (không đổi) | Tốc độ giữ nguyên 100% |
| **Độ trễ Edge CPU i7 (ONNX INT8)** | 22 ms | **22 ms** (không đổi) | Hoàn toàn đạt chuẩn real-time |

---

## 7. Quản trị Rủi ro & Phương án Dự phòng

1. **Rủi ro: Feature Cache quá lớn gây tràn ổ cứng SSD NVMe**  
   * *Giải pháp:* Nén tensor dạng FP16 hoặc BF16 thay vì FP32, chỉ lưu các feature map sau tầng FPN thay vì lưu toàn bộ hidden states của ViT.
2. **Rủi ro: Độ trễ I/O đọc ổ đĩa khi load cache làm nghẽn GPU**  
   * *Giải pháp:* Sử dụng `torch.multiprocessing` với `SharedMemory` hoặc tạo RAM disk (`/dev/shm`) trên DGX Server cho các batch đang huấn luyện.
3. **Rủi ro: Student CNN bị "nhiễu" do Teacher VLM dự đoán sai (Hallucination)**  
   * *Giải pháp:* Áp dụng cơ chế **Confidence Thresholding Filtering**: Chỉ chưng cất soft logits và features tại các vùng mà Teacher có độ tự tin $> 0.65$; các vùng còn lại chỉ dùng ground-truth supervised loss.
