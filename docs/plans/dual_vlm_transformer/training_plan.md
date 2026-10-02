# Kế hoạch Huấn luyện: Cặp Dual-VLM Transformer (DaViT / Swin-L) → Hybrid Edge ConvNet (MobileNetV4 / FastViT)

**Ngày lập:** 2026-10-02  
**Dự án:** Engine-Bay & Automotive Component Vision Distillation  
**Trạng thái:** Sẵn sàng triển khai trên DGX Server & Edge Deployment  

---

## 1. Bối cảnh & Mục tiêu Chiến lược

### 1.1. Hiện trạng & Giới hạn của Thế hệ v6 (YOLO11)
Trong giai đoạn trước, hệ thống sử dụng cặp ConvNet truyền thống:
- **Teacher:** `yolo11l-seg` (Teacher mAP50-95: 0.354, mAP50: 0.577).
- **Student:** `yolo11n-seg` (Student mAP50-95: 0.302, mAP50: 0.514, Latency: 4.0 ms).

Mặc dù giải pháp đạt hiệu năng thời gian thực tốt trên thiết bị biên, pipeline thuần ConvNet bộc lộ các rào cản kỹ thuật:
1. **Thiếu hiểu biết ngữ cảnh toàn cục (Long-range spatial context):** Khoang máy xe hơi là một hệ thống topology phức tạp (ví dụ: bình nước làm mát nối với ống tản nhiệt, cảm biến MAF gắn liền với bầu lọc gió, hộp cầu chì gần ắc quy). ConvNet bị giới hạn bởi receptive field cục bộ, dẫn đến nhầm lẫn các cụm giắc cắm hoặc nắp đậy tương đồng.
2. **Khả năng căn chỉnh Ngôn ngữ - Hình ảnh (Visual-Language Grounding) yếu:** ConvNet truyền thống không có không gian biểu diễn đa phương thức (multimodal latent space), khó mở rộng theo taxonomy phân tầng (taxonomy v2: 13 hệ thống, 36+ classes) hoặc nhận diện zero-shot / open-vocabulary khi gặp linh kiện của các dòng xe mới.

### 1.2. Bước nhảy vọt: Cặp Dual-VLM Transformer $\rightarrow$ Hybrid Edge ConvNet
Mô hình mới kết hợp sức mạnh tối đa của 2 thế giới:
- **Teacher (Foundation VLM Backbone):** DaViT (Dual Attention Vision Transformer) hoặc Swin-Transformer Large (như trong Florence-2, Grounding DINO).
- **Student (Edge Hybrid Backbone):** MobileNetV4-Hybrid hoặc FastViT (như trong Florence-2-nano, NanoVLM).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TEACHER (Server / DGX)                          │
│   DaViT / Swin-Transformer Large (Florence-2 / Grounding DINO)        │
│   • Spatial Window Attention + Channel Group Attention                 │
│   • Multi-scale Cross-modal Visual-Language Tokens                     │
│   • Pixel-level Context & Sub-component Topology                       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                     Knowledge Distillation (KD)
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   Feature KD                 Logit KD                   Mask / Loc KD
 (Cross-Attention          (Decoupled KD +             (DFL Distribution +
  FGD + Adapters)           Class Affinity)             Mask IoU / Dice)
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                         STUDENT (Edge / Mobile)                        │
│             FastViT / MobileNetV4-Hybrid (Universal UIB)               │
│   • Training: RepVGG / RepMixer multi-branch + Mobile Attention        │
│   • Deployment: Fused linear conv (0 cost) + Linearized Attention       │
│   • Zero runtime overhead, 100% ONNX Runtime / NPU / WebAssembly       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phân tích Kỹ thuật Cốt lõi của Kiến trúc

### 2.1. Teacher Backbone: DaViT vs. Swin-Transformer Large

| Tiêu chí | DaViT (Dual Attention ViT) | Swin-Transformer Large (Swin-L) |
|---|---|---|
| **Cơ chế Attention** | Xen kẽ **Spatial Window Attention** và **Channel Group Attention** trong từng block | **Shifted Window Attention** (W-MSA + SW-MSA) theo cấu trúc kim tự tháp 4 tầng |
| **Độ phân giải & Chi tiết** | Cực kỳ mạnh ở mức pixel, giữ nguyên độ phân giải cục bộ cao, bắt rất bén viền linh kiện | Bắt cấu trúc phân cấp (hierarchical) tốt từ vùng lớn đến chi tiết vừa |
| **Ứng dụng VLM thực tế** | **Florence-2** Image Encoder (Vision Foundation Model hàng đầu của Microsoft) | **Grounding DINO**, Co-DETR (tiêu chuẩn vàng cho Open-Vocabulary Detection) |
| **Ưu thế trên Khoang máy** | Phân biệt xuất sắc các chi tiết cực nhỏ: giắc cắm cảm biến, đầu cực ắc quy (`battery_terminal`), que thăm dầu (`oil_dipstick`), kẹp ống | Bắt quan hệ không gian toàn cục: toàn bộ khối động cơ, vị trí tương đối giữa vách ngăn động cơ (firewall) và két nước tản nhiệt |
| **VRAM & Training** | Tốn ~16 - 22 GB VRAM với input 640–800px; tối ưu tốt với PyTorch SDPA | Tốn ~20 - 24 GB VRAM với input 640–800px; yêu cầu gradient checkpointing nếu batch $\ge 8$ |

> **Khuyến nghị lựa chọn Teacher:**
> - **Lựa chọn chính:** **DaViT-Large** (chuẩn hóa theo Florence-2 vision tower) nhờ khả năng trích xuất đặc trưng hình ảnh cực mịn kết hợp tự nhiên với prompt văn bản của 36 class linh kiện.
> - **Lựa chọn so sánh / bổ trợ:** **Swin-L** (Grounding DINO backbone) cho bài toán bounding box và zero-shot grounding trên các xe hiếm.

### 2.2. Student Backbone: MobileNetV4-Hybrid vs. FastViT

| Tiêu chí | MobileNetV4-Hybrid (Universal UIB) | FastViT (Apple) |
|---|---|---|
| **Thành phần cấu tạo** | Universal Inverted Bottleneck (UIB) + RepVGG Conv + Mobile Multi-Head Attention ở P4/P5 | RepMixer (thay thế self-attention cục bộ) + Mobile Attention ở P4/P5 |
| **Cơ chế Reparameterization** | Nhánh 3x3 Conv + 1x1 Conv + Identity gập (fuse) toán học thành duy nhất 1 lớp Conv 3x3 khi export | RepMixer đa nhánh gập hoàn toàn thành Depthwise Conv chuẩn khi export |
| **Độ trễ & Throughput** | Thiết kế dựa trên Roofline Model tối ưu cho CPU, GPU và NPU di động (Qualcomm Snapdragon, Apple Silicon, MediaTek) | Thiết kế chuyên biệt cho latency cực thấp trên mobile/edge, loại bỏ overhead dịch chuyển bộ nhớ (reshape/transpose) |
| **Hỗ trợ ONNX / NPU** | 100% toán tử cơ bản (Conv2d, BatchNorm, SiLU, Linear Attention) tương thích hoàn hảo ONNX Runtime, CoreML, TFLite, WebGPU | 100% chuẩn hóa ONNX, không chứa toán tử lạ |
| **Thông số tham khảo** | MobileNetV4-Hybrid-M (~10M params, ~1.8 GFLOPs) hoặc Hybrid-L (~21M params, ~4.5 GFLOPs) | FastViT-S12 (~8.8M params, ~1.8 GFLOPs) hoặc FastViT-MA36 (~36M params) |

> **Khuyến nghị lựa chọn Student:**
> - **FastViT-S12 / FastViT-T8:** Dành cho Web browser (WASM/WebGPU) và ứng dụng di động cần phản hồi tức thì (< 3 ms).
> - **MobileNetV4-Hybrid-Medium:** Dành cho edge server / edge PC kiểm định tại xưởng dịch vụ với độ chính xác cao nhất trong nhóm < 10M params.

---

## 3. Khung Phương pháp Chưng cất Tri thức (Multi-Level Distillation Framework)

Do Teacher là mô hình thuần Transformer (attention-heavy, channel dimensions lớn: 768 / 1024 / 1536) trong khi Student là kiến trúc Hybrid (Conv-heavy ở tầng thấp, Mobile Attention ở tầng cao, channel dimensions nhỏ: 96 / 192 / 384 / 512), hệ thống áp dụng chiến lược chưng cất 4 tầng:

```
Loss Tổng = L_task + α * L_feat + β * L_cls + γ * L_loc + λ * L_mask
```

### 3.1. Tầng 1: Multi-Scale Feature Distillation (Cross-Attention FGD)
- **1x1 Projection Adaptor:** Đặt các module chiếu tuyến tính $P_k$ tại các stage P3, P4, P5:
  $$P_k: \mathbb{R}^{C_S^{(k)} \times H_k \times W_k} \longrightarrow \mathbb{R}^{C_T^{(k)} \times H_k \times W_k}$$
  Module gồm: `Conv2d(1x1) -> GroupNorm/LayerNorm -> GELU`.
- **Focal & Global Distillation (FGD):**
  - **Foreground Attention Transfer:** Tính ma trận năng lượng chú ý cục bộ từ Teacher $S_T = \frac{1}{C}\sum |F_T^c|$, ép Student tập trung vào ranh giới linh kiện thực (mask & box) thay vì nền kim loại trống.
  - **Background Distillation:** Học bối cảnh khoang máy và vị trí tương đối với trọng số giảm nhẹ ($\alpha_{bg} = 0.3$).
  - **Global Context Block (GcBlock):** Chưng cất quan hệ tương quan giữa các cặp pixel trên toàn ảnh để Student nắm được cấu trúc hình học của khoang máy.

### 3.2. Tầng 2: Visual-Language Token Alignment KD
- Teacher sở hữu không gian embedding liên kết giữa Text Prompt ("oil dipstick", "coolant reservoir", "fuse box") và Visual Tokens.
- **Affinity Distillation Loss:**
  - Tính ma trận đồng thuận ngữ nghĩa (Semantic Affinity Matrix) $M_T = Z_{text} \cdot (F_T)^T$.
  - Ép ma trận tương ứng của Student $M_S = Z_{text} \cdot (P(F_S))^T$ khớp với $M_T$ thông qua KL Divergence hoặc Smooth L1 Loss:
    $$L_{align} = \mathcal{D}_{KL}(\text{Softmax}(M_T / \tau) \parallel \text{Softmax}(M_S / \tau))$$
  - Giúp Student tự động phân biệt được các linh kiện có hình thái tương đồng nhưng thuộc hai hệ thống khác nhau (ví dụ: bình dầu phanh vs. bình nước làm mát phụ).

### 3.3. Tầng 3: Decoupled Knowledge Distillation (DKD) trên Class Logits
Thay vì dùng KL divergence thông thường (bị chi phối nặng bởi class nền chiếm đa số), áp dụng Decoupled KD:
- **TCKD (Target Class Knowledge Distillation):** Đo độ tin cậy của class mục tiêu.
- **NCKD (Non-target Class Knowledge Distillation):** Chuyển giao tri thức tương quan giữa các class phi mục tiêu (giúp Student biết rằng một giắc cắm nếu không phải MAF sensor thì nhiều khả năng là MAP sensor hơn là một cái bình nước).

### 3.4. Tầng 4: High-Precision Mask & DFL Localization KD
- **Distribution Focal Loss (DFL) Distillation:** Chuyển giao phân phối xác suất biên bounding box của Teacher sang Student với nhiệt độ $\tau_{loc} = 8.0$.
- **Mask Feature / Prototype KD:** Chuyển giao các prototype mask vector để Student học cách sinh mask sắc nét bao quanh các ống dẫn ngoằn ngoèo (`radiator_hose`, `wiring_harness`) và tấm chắn nhiệt (`exhaust_manifold_heat_shield`).

### 3.5. Tầng 5: Semi-Supervised Feature KD trên Ảnh Chưa Gán Nhãn (Unlabeled)
- Kho ảnh crawl hiện có hơn 2,000+ ảnh khoang máy chưa gắn nhãn chi tiết.
- Cho ảnh unlabeled đi qua Teacher để sinh soft features và pseudo-labels; tính $L_{feat}$ trên Student mà không cần ground-truth human label, tăng tính tổng quát hóa trên các dòng xe lạ.

---

## 4. Lộ trình Triển khai Huấn luyện 5 Giai đoạn (5-Phase Roadmap)

```
Tuần 1                 Tuần 2                 Tuần 3                 Tuần 4
[ Phase 0 & 1 ] ──────► [    Phase 2    ] ──────► [    Phase 3    ] ──────► [ Phase 4 & 5 ]
Môi trường, Base &     Domain Teacher         Distillation Training  Reparam, ONNX,
Model Registry         Fine-tuning + Avg5     (Teacher -> Student)   E2E Benchmarking
```

### Giai đoạn 0: Chuẩn bị Môi trường & Thiết kế Module Adaptor
- **Mục tiêu:** Tích hợp backbone DaViT, Swin-L, MobileNetV4-Hybrid và FastViT vào pipeline của repo.
- **Nội dung thực hiện:**
  1. Kiểm tra môi trường PyTorch 2.2+, `timm>=1.0.0`, `transformers>=4.42.0`, `accelerate`.
  2. Tạo module `src/distillation/backbones/`:
     - `davit_adapter.py`: Wrapper cho DaViT từ `timm` / Florence-2 vision tower, trích xuất feature maps tại 4 stages ($C_T = [96, 192, 384, 768]$ hoặc $[128, 256, 512, 1024]$).
     - `mobilenetv4_hybrid.py` & `fastvit_adapter.py`: Wrapper cho Student, trích xuất intermediate features.
     - `projection_heads.py`: Các module 1x1 projection adapter $P_k$ có thể học được.
  3. Viết unit test kiểm tra forward/backward pass và tính toán shape alignment.

### Giai đoạn 1: Huấn luyện & Hiệu chuẩn Teacher (DaViT-Base/Large hoặc Swin-L)
- **Mục tiêu:** Tạo Teacher model có độ chính xác vượt trội trên toàn bộ 20/36 class xe hơi.
- **Cấu hình & Dữ liệu:**
  - Tập dữ liệu: `configs/data_engine_bay.yaml` (full 28 xe hoặc tập gán nhãn hybrid v8).
  - Pretrained weights: DaViT pretrained trên ImageNet-22k / Florence-2 visual encoder; Swin-L pretrained trên COCO/Objects365.
  - Kích thước ảnh: $640 \times 640$ (hoặc $800 \times 800$ nếu VRAM cho phép).
  - Kỹ thuật nâng cao: Cosine LR, Weight Decay 0.001, Random Scale Jitter (0.5 - 1.5), Copy-Paste 0.3.
  - Checkpoint Averaging: Lưu checkpoint mỗi 5 epoch, chạy `average_checkpoints.py --top-k 5` để tạo `teacher_avg5.pt`.
- **Target KPI Teacher:** Mask mAP50-95 $\ge 0.380$ (tăng ít nhất +2.5% so với Teacher `yolo11l-seg` cũ là 0.354).

### Giai đoạn 2: Huấn luyện Student Baseline (Không Distill) làm Đối chứng
- **Mục tiêu:** Đo lường chính xác mức độ cải thiện (gain) do kiến trúc mới và do KD mang lại.
- **Nội dung thực hiện:**
  - Train FastViT-S12 và MobileNetV4-Hybrid-M trực tiếp trên cùng tập dữ liệu (100 epochs, cosine LR, không có Teacher).
  - Đánh giá trên tập test 3 xe độc lập chưa từng thấy:
    - Ghi nhận: Baseline mask mAP50-95, mAP50, class-wise recall.

### Giai đoạn 3: Huấn luyện Distillation Toàn diện (Teacher $\to$ Student)
- **Mục tiêu:** Truyền toàn bộ tri thức không gian và liên kết đa phương thức vào Student.
- **Kịch bản thực thi:**
  - Khởi chạy script:
    ```bash
    python scripts/training/train_vlm_kd.py \
        --kd configs/kd_hyperparams_dual_vlm.yaml \
        --teacher runs/segment/teacher_davit_large/weights/avg5.pt \
        --student mobilenetv4_hybrid_medium \
        --data configs/data_engine_bay.yaml \
        --unlabeled data/unlabeled_exterior \
        --epochs 120 --batch 16 --imgsz 640 \
        --device 0,1 --name kd_vlm_mobilenetv4_med
    ```
  - **Quy trình 2-phase warmup:**
    - Epoch 1 - 5: Đóng băng (freeze) backbone Student, chỉ train Projection Adapters và Task Heads để đồng bộ channel space với Teacher.
    - Epoch 6 - 120: Mở toàn bộ Student, áp dụng dynamic loss weights với Cosine Annealing.
  - Trung bình checkpoint: Chạy `average_checkpoints.py --run runs/train_kd/kd_vlm_mobilenetv4_med --top-k 5 --strip-kd`.

### Giai đoạn 4: Structural Reparameterization & Edge Optimization
- **Mục tiêu:** Chuyển đổi mô hình Student sang trạng thái suy luận tối ưu, xóa bỏ hoàn toàn chi phí tính toán đa nhánh.
- **Nội dung thực hiện:**
  1. **RepVGG / RepMixer Fuse:**
     - Gọi phương thức `switch_to_deploy()` để gập toàn bộ nhánh 1x1 conv, identity branch và batchnorm vào trọng số kernel $3 \times 3$ duy nhất thông qua phép biến đổi đại số tuyến tính:
       $$W_{fused} = W_{3\times3} \cdot \frac{\gamma_{3\times3}}{\sigma_{3\times3}} + \text{Pad}(W_{1\times1} \cdot \frac{\gamma_{1\times1}}{\sigma_{1\times1}}) + \text{Pad}(\frac{\gamma_{id}}{\sigma_{id}})$$
  2. **Export ONNX & Graph Optimization:**
     - Export sang ONNX Opset 18 với fixed dynamic batch.
     - Chạy `onnxsim` để loại bỏ các node thừa, constant folding.
  3. **Lượng tử hóa INT8 / FP16:**
     - Tạo bản FP16 cho GPU/Edge server.
     - Sử dụng ONNX Runtime Quantization Tools tạo bản INT8 PTQ (Post-Training Quantization) có calibration trên 500 ảnh khoang máy.

### Giai đoạn 5: Thẩm định E2E, Cross-Validation & Benchmark Tốc độ
- **Nội dung thực hiện:**
  - Chạy bộ test suite toàn diện: `scripts/evaluation/qa_test.py`, `tests/test_benchmark_engine_bay.py`.
  - Đo độ trễ pipeline đầy đủ trên các môi trường:
    - DGX GPU (TensorRT FP16).
    - Edge PC / Laptop Intel i7 CPU (ONNX Runtime CPU).
    - Web Browser (ONNX Runtime WebAssembly & WebGPU trên `apps/engine_bay_vercel/`).
    - Android Mobile NPU (TFLite / ONNX Runtime Mobile).

---

## 5. Đặc tả Siêu tham số Huấn luyện (Hyperparameter Specifications)

File cấu hình tiêu chuẩn được lưu tại `configs/kd_hyperparams_dual_vlm.yaml`:

```yaml
distillation:
  teacher:
    architecture: davit_large # hoặc swin_large_patch4_window7_224
    pretrained: true
    freeze: true
    feature_stages: [stage1, stage2, stage3, stage4] # P2, P3, P4, P5
    feature_channels: [128, 256, 512, 1024]

  student:
    architecture: mobilenetv4_hybrid_medium # hoặc fastvit_s12
    feature_channels: [64, 128, 256, 512]
    reparameterize_on_export: true

  # Cấu hình Loss Chưng cất
  alpha_feature: 1.5           # Trọng số Feature Loss
  beta_cls: 1.0               # Trọng số Decoupled Logit Loss
  gamma_loc: 1.0              # Trọng số Localization (DFL) Loss
  lambda_mask: 1.2            # Trọng số Prototype / Mask IoU Loss

  # Feature Distillation (Cross-Attention FGD)
  fgd:
    alpha_fg: 1.2             # Tập trung vào vùng linh kiện (foreground)
    alpha_bg: 0.3             # Giảm trọng số nền nhưng giữ bối cảnh
    lambda_attn: 0.6          # Chuyển giao Spatial Attention Map
    lambda_global: 0.4        # Chuyển giao Global Context tương quan xa
    temperature: 0.5

  # Logit Distillation (Decoupled KD)
  dkd:
    alpha_tckd: 1.0           # Target class KD
    beta_nckd: 0.5            # Non-target class correlation KD
    temperature: 2.5

  # Localization & Mask
  localization:
    temperature_ld: 8.0
    box_iou_weight: 0.5

  # Luyện bán giám sát với ảnh Unlabeled
  unlabeled:
    dir: "data/unlabeled_exterior"
    weight: 0.6
    batch: 8
    every: 1

training:
  epochs: 120
  warmup_epochs: 5            # Khởi động Adapter và Task Head
  batch: 16
  imgsz: 640
  optimizer: AdamW            # AdamW tối ưu tốt nhất cho hybrid transformer
  lr0: 0.001
  lrf: 0.01
  weight_decay: 0.01          # Tăng nhẹ weight decay để chống overfitting
  cos_lr: true
  amp: true                   # Mixed precision (bfloat16 trên DGX)
  workers: 8
  seed: 42
  save_period: 5

  # Data Augmentation chống quá khớp
  copy_paste: 0.35
  mixup: 0.15
  degrees: 7.0
  scale: 0.5
  close_mosaic: 15            # Tắt Mosaic 15 epoch cuối để hội tụ mask mịn
```

---

## 6. Tiêu chí Đánh giá & Chỉ số Nghiệm thu (KPIs & Acceptance Criteria)

| Chỉ số Đánh giá | Thế hệ cũ (YOLO11n-seg) | Mục tiêu Cặp Mới (MobileNetV4-H / FastViT) | Mức Cải thiện Kỳ vọng |
|---|---|---|---|
| **Mask mAP50-95 (3 xe mới test)** | 0.302 | **$\ge 0.335 - 0.345$** | **+3.3 đến +4.3 điểm** |
| **Mask mAP50** | 0.514 | **$\ge 0.560$** | **+4.6 điểm** |
| **Box mAP50-95** | 0.340 | **$\ge 0.380$** | **+4.0 điểm** |
| **Recall linh kiện nhỏ (< 32x32px)** | 0.390 (tăng 0.58 với tile) | **$\ge 0.480$ (không cần tile)** | **+9.0% recall trực tiếp** |
| **Độ trễ DGX (TensorRT FP16)** | 4.0 ms | **$\le 3.2$ ms** | Nhanh hơn 20% |
| **Độ trễ CPU Edge (ONNX INT8)** | 35.0 ms | **$\le 18.0 - 22.0$ ms** | Nhanh hơn gần 40% |
| **Kích thước Model triển khai** | ~6.2 MB | **$\le 8.5$ MB** (MobileNetV4-M) / **$\le 5.8$ MB** (FastViT-S12) | Siêu gọn cho mobile/web |

---

## 7. Quản trị Rủi ro & Giải pháp Dự phòng (Risk & Contingency Plan)

| Rủi ro Kỹ thuật | Xác suất | Mức ảnh hưởng | Giải pháp Dự phòng |
|---|---|---|---|
| **VRAM OOM khi forward cả Teacher DaViT và Student** | Trung bình | Cao | **Feature Caching:** Chạy Teacher một lần trên toàn bộ tập train và lưu (cache) các tensor feature map P3..P5 ra ổ cứng SSD/NVMe. Khi train Student, chỉ cần nạp feature từ cache, giảm 70% VRAM và tăng tốc độ train gấp 3 lần. |
| **Bất đồng Gradient giữa Transformer và ConvNet** | Trung bình | Trung bình | Sử dụng LayerNorm trong Projection Adapter; tách riêng learning rate cho Adapter (`lr = 2 * lr0`) và Student backbone (`lr = lr0`). |
| **Sai lệch số học (Numeric drift) sau khi Reparameterization** | Thấp | Cao | Viết script kiểm thử sai số `verify_reparam_equivalence.py` so sánh output tensor trước và sau khi gập nhánh với ngưỡng sai số tuyệt đối $L_\infty < 10^{-5}$. |
| **Overfitting trên tập xe ít mẫu** | Cao | Trung bình | Giữ nguyên chính sách k-fold theo xe; tăng cường độ Copy-Paste từ các ảnh đã review; tận dụng triệt để nguồn ảnh unlabeled qua semi-supervised feature KD. |

---

## 8. Danh mục Công việc Cần Thực hiện (Actionable Checklist)

- [ ] **Task 1:** Tạo file cấu hình `configs/kd_hyperparams_dual_vlm.yaml`.
- [ ] **Task 2:** Xây dựng module tích hợp Teacher DaViT/Swin-L và Student FastViT/MobileNetV4 trong `src/distillation/`.
- [ ] **Task 3:** Cài đặt 1x1 Projection Adaptor và cơ chế Reparameterization export.
- [ ] **Task 4:** Fine-tune Teacher DaViT trên domain xe hơi và thực hiện trung bình 5 checkpoint.
- [ ] **Task 5:** Huấn luyện baseline Student (không KD) để làm mốc đối chứng.
- [ ] **Task 6:** Chạy Distillation toàn diện (Feature + Decoupled Logit + DFL Loc + Unlabeled).
- [ ] **Task 7:** Thực hiện Reparameterization gập nhánh và export ONNX/TensorRT.
- [ ] **Task 8:** Chạy Benchmark đo đạc mAP và Latency, cập nhật vào Web UI demo (`apps/engine_bay_vercel/`).
