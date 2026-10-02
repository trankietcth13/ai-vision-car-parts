# Kế hoạch Huấn luyện: Cặp SigLIP / CLIP ViT → Tiny-CLIP / MobileCLIP (Multimodal On-Device Distillation)

**Ngày lập:** 2026-10-02  
**Mô hình chiến lược:** Foundation Multimodal VLM (SigLIP-SO400M / CLIP ViT-L/14@336px) $\longrightarrow$ Edge Multimodal (MobileCLIP-S0 / S2 hoặc Tiny-CLIP)  
**Mục tiêu cốt lõi:** Bảo toàn không gian nhúng đa phương thức (Multimodal Latent Space) phong phú, tối ưu hóa triệt để bằng hàm mất mát Sigmoid Loss, nén mô hình xuống kích thước vài chục MB để chạy Zero-shot Search & Diagnostics tức thì (< 3ms) trên điện thoại Android (`engine_bay_android`) và Web client.  

---

## 1. Bối cảnh & Mục tiêu Chiến lược

### 1.1. Tại sao Cần Mobile Multimodal Embedding trong Dự án Ô tô?
Trong hệ thống chẩn đoán và kiểm định khoang máy hiện nay:
1. **Tìm kiếm Linh kiện & Bộ phận Bằng Ngôn ngữ Tự nhiên (Natural Language Visual Search):**
   - Kỹ thuật viên tại xưởng có thể gõ hoặc nói: *"bình nước làm mát phụ"*, *"cảm biến lưu lượng khí nạp nứt giắc"*, *"nắp châm dầu màu vàng"* $\rightarrow$ Hệ thống lập tức khoanh vùng và định danh linh kiện trên ảnh chụp camera.
2. **Cầu nối Trực tiếp giữa Chẩn đoán Jev (`src/jev/`) và Hình ảnh:**
   - Jev tiếp nhận khiếu nại khách hàng ("tiếng rít khi nổ máy", "nổi đèn check engine P0101") $\rightarrow$ Tạo text query $\rightarrow$ MobileCLIP so khớp độ tương đồng ngữ nghĩa trực tiếp với hình ảnh thực tế từ camera để kiểm tra linh kiện nghi ngờ (`serpentine_belt`, `maf_sensor`).
3. **Phân loại Mở (Zero-Shot / Open-Vocabulary Classification):**
   - Khoang máy có hàng nghìn loại chi tiết nhỏ (nút cao su, kẹp ống, bu-lông gá, van một chiều...) mà các tập dữ liệu có nhãn bounding box không bao giờ bao phủ hết. Mô hình multimodal embedding cho phép nhận diện ngay các chi tiết lạ thông qua mô tả text mà không cần đào tạo lại mô hình phát hiện.

### 1.2. Thách thức Khi Đưa Foundation CLIP Lên Thiết bị Biên
- **SigLIP-SO400M** (~400M params, ~1.6 GB) và **CLIP ViT-L/14@336px** (~428M params, ~1.7 GB) có độ chính xác biểu diễn ngữ nghĩa vượt trội nhưng:
  - Quá nặng để nạp vào RAM điện thoại di động thông thường.
  - Thời gian tính toán embedding trên CPU điện thoại mất từ 300ms – 1.2s, gây giật lag trải nghiệm tương tác trực tiếp.
- **MobileCLIP-S0 / S2 (Apple 2024):**
  - Tối ưu hóa kiến trúc bằng **FastViT / MCi** kết hợp **Structural Reparameterization** (RepMixer) cho nhánh hình ảnh và transformer siêu gọn cho nhánh văn bản.
  - Kích thước tổng thể chỉ từ **35 MB – 70 MB**.
  - Tốc độ suy luận chỉ **1.8 – 3.5 ms** trên NPU di động (Qualcomm Snapdragon / Apple Neural Engine / Android NNAPI) và **12 ms** trên CPU điện thoại thông thường.

---

## 2. Phân tích Kỹ thuật Cốt lõi: Teacher vs. Student

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     TEACHER: Foundation Multimodal Encoder                        │
│   • Model: SigLIP-SO400M (Shape-Optimized 400M) hoặc OpenCLIP ViT-L/14@336px      │
│   • Đặc trưng: Pairwise Sigmoid Loss (BCE) không cần global batch normalization    │
│   • Không gian nhúng: D_T = 1152 (SigLIP) hoặc D_T = 768 (CLIP ViT-L)             │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
               Multimodal Cross-Encoder Knowledge Distillation
     ┌────────────────────────────────────┼────────────────────────────────────┐
     │                                    │                                    │
  Sigmoid Contrastive                Cross-Modal Feature                 Cross-Modal Relation
   Distillation (BCE)                Projection Alignment               Similarity Graph (CRD)
     │                                    │                                    │
     └────────────────────────────────────┼────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────▼─────────────────────────────────────────┐
│                     STUDENT: Edge Mobile Multimodal Model                         │
│   • Model: MobileCLIP-S0 / S2 (Apple) hoặc Tiny-CLIP                              │
│   • Image Tower: FastViT-MCi0 / MCi2 (RepMixer + Mobile Attention)                │
│   • Text Tower: 8-layer Transformer siêu nhẹ                                      │
│   • Dung lượng: ~38 MB (S0) đến ~65 MB (S2), tốc độ: 2.2 ms trên thiết bị biên    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1. So sánh Thông số Kỹ thuật Chi tiết

| Tiêu chí | SigLIP-SO400M (Teacher) | CLIP ViT-L/14@336px (Teacher) | MobileCLIP-S0 (Student) | MobileCLIP-S2 (Student) |
|---|---|---|---|---|
| **Backbone Hình ảnh** | ViT-SO400M (Shape-Optimized) | ViT-Large (Patch 14x14) | FastViT-MCi0 (Hybrid Conv-ViT) | FastViT-MCi2 (Hybrid Conv-ViT) |
| **Kích thước Input** | $378 \times 378$ (hoặc 224) | $336 \times 336$ | $256 \times 256$ | $256 \times 256$ |
| **Hàm mất mát gốc** | **Sigmoid Loss (Pairwise BCE)** | Softmax InfoNCE | Tương thích cả Sigmoid & InfoNCE | Tương thích cả Sigmoid & InfoNCE |
| **Số tham số (Params)** | 400 Triệu | 428 Triệu | **11.2 Triệu (Image) + 14 Triệu (Text)** | **24.5 Triệu (Image) + 16 Triệu (Text)** |
| **Dung lượng File** | ~1.6 GB | ~1.7 GB | **~38 MB (sau khi fuse RepVGG)** | **~65 MB (sau khi fuse RepVGG)** |
| **Độ trễ Android NPU** | Không thể chạy mượt | Không thể chạy mượt | **2.2 ms** | **4.1 ms** |
| **Zero-shot ImageNet top-1**| 83.2% | 76.2% | 67.8% (gốc) $\to$ **72.5% (sau KD)** | 74.4% (gốc) $\to$ **77.8% (sau KD)** |

> **Khuyến nghị lựa chọn:**
> - **Teacher:** **SigLIP-SO400M** là ưu tiên hàng đầu nhờ hàm mất mát Sigmoid loss tối ưu cho cặp ảnh-chữ cục bộ, không bị chi phối bởi độ lớn của batch size khi chưng cất trên DGX.
> - **Student:** **MobileCLIP-S0** cho ứng dụng di động Android (`apps/engine_bay_android/`) và **MobileCLIP-S2** cho máy trạm kiểm tra tại xưởng.

---

## 3. Khung Phương pháp Chưng cất Đa Phương thức (Multimodal Distillation Formulation)

Hệ thống thiết kế hàm mất mát tổng hợp gồm 4 thành phần chưng cất đồng thời:

$$\mathcal{L}_{total} = \mathcal{L}_{siglip\_kd} + \alpha \cdot \mathcal{L}_{feat\_img} + \beta \cdot \mathcal{L}_{feat\_txt} + \gamma \cdot \mathcal{L}_{relation}$$

### 3.1. Sigmoid Contrastive Distillation Loss ($\mathcal{L}_{siglip\_kd}$)
Khác với CLIP truyền thống sử dụng phân phối Softmax trên toàn bộ batch (đòi hỏi batch size phải từ 4096 đến 32768 mới ổn định), **SigLIP** coi mỗi cặp ảnh-chữ $(I_i, T_j)$ là một bài toán phân loại nhị phân độc lập:

$$z_{i,j}^{(T)} = \frac{1}{\tau_T} \left( \hat{e}_I^{(T)}(i) \cdot \hat{e}_T^{(T)}(j) \right) + b_T$$

Teacher sinh ra ma trận xác suất sigmoid: $P_{i,j}^{(T)} = \sigma(z_{i,j}^{(T)})$.  
Student tính toán ma trận logits tương ứng: $z_{i,j}^{(S)} = \frac{1}{\tau_S} \left( \hat{e}_I^{(S)}(i) \cdot \hat{e}_T^{(S)}(j) \right) + b_S$.  

Hàm mất mát chưng cất nhị phân Sigmoid KD:
$$\mathcal{L}_{siglip\_kd} = - \sum_{i=1}^B \sum_{j=1}^B \left[ P_{i,j}^{(T)} \log \sigma(z_{i,j}^{(S)}) + (1 - P_{i,j}^{(T)}) \log (1 - \sigma(z_{i,j}^{(S)})) \right]$$

*Lợi thế đột phá:* Giúp Student học chính xác phân phối tương quan mềm (soft similarity) giữa ảnh và văn bản mà không cần batch size khổng lồ trên GPU DGX.

---

### 3.2. Cross-Modal Feature Alignment ($\mathcal{L}_{feat\_img}, \mathcal{L}_{feat\_txt}$)
Do chiều vector embedding của Teacher ($D_T = 1152$) lớn hơn nhiều so với Student ($D_S = 512$):
- **Image Projection Head:** Đặt module chiếu tuyến tính $P_I: \mathbb{R}^{D_S} \to \mathbb{R}^{D_T}$ gồm `Linear -> LayerNorm -> GELU`.
- **Text Projection Head:** Đặt module chiếu tuyến tính $P_T: \mathbb{R}^{D_S} \to \mathbb{R}^{D_T}$.
- **Cosine Representation Distance:**
  $$\mathcal{L}_{feat\_img} = 1 - \frac{P_I(\hat{e}_I^{(S)}) \cdot \hat{e}_I^{(T)}}{\|P_I(\hat{e}_I^{(S)})\| \|\hat{e}_I^{(T)}\|}, \quad \mathcal{L}_{feat\_txt} = 1 - \frac{P_T(\hat{e}_T^{(S)}) \cdot \hat{e}_T^{(T)}}{\|P_T(\hat{e}_T^{(S)})\| \|\hat{e}_T^{(T)}\|}$$

---

### 3.3. Cross-Modal Relation Similarity Graph Distillation ($\mathcal{L}_{relation}$)
Bảo toàn đồ thị tương quan đa chiều trong không gian đặc trưng:
- **Image-to-Image Graph:** Ma trận mức độ giống nhau giữa các ảnh trong cùng một batch $G_{I,I}^{(T)} = Z_I^{(T)} (Z_I^{(T)})^T$.
- **Text-to-Text Graph:** Ma trận mức độ tương đồng giữa các mô tả bệnh / linh kiện $G_{T,T}^{(T)} = Z_T^{(T)} (Z_T^{(T)})^T$.
- Ép ma trận tương quan của Student khớp với Teacher bằng hàm Frobenius Norm:
  $$\mathcal{L}_{relation} = \frac{1}{B^2} \| G_{I,I}^{(T)} - G_{I,I}^{(S)} \|_F^2 + \frac{1}{B^2} \| G_{T,T}^{(T)} - G_{T,T}^{(S)} \|_F^2$$

---

## 4. Dữ liệu Đa Phương thức Chuyên ngành Ô tô (Automotive Multimodal Dataset)

Để mô hình nắm bắt trọn vẹn ngữ nghĩa xe hơi song ngữ Anh - Việt, hệ thống tổ chức pipeline sinh dữ liệu đa tầng:

```
[ Ảnh khoang máy / Ngoại thất / Hư hại ] 
                   │
                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│     Hệ thống Sinh Caption Tự động Theo Ngữ cảnh (Multi-layer Prompt)   │
│   • Tầng 1 (Định danh hệ thống): "Cooling system component"           │
│   • Tầng 2 (Linh kiện chi tiết): "Expansion coolant reservoir tank"    │
│   • Tầng 3 (Thuộc tính & Vị trí): "Translucent white tank with black   │
│     cap near the passenger side firewall"                             │
│   • Tầng 4 (Trạng thái kỹ thuật): "Fluid level visible between MIN/MAX"│
│   • Tầng 5 (Bilingual Translation): "Bình nước làm mát phụ màu trắng,  │
│     nắp đen gần vách ngăn khoang máy phía phụ"                         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                   [ Cặp Dữ liệu (Image, Captions) ]
```

- **Quy mô tập dữ liệu chưng cất:**
  1. **Tập chuyên ngành nội bộ (Domain-specific):** ~15,000 cặp ảnh - caption được sinh từ taxonomy v2 (36 class), dữ liệu kiểm tra xe thực tế và mô tả triệu chứng mã lỗi Jev (`src/jev/`).
  2. **Tập tổng quát (General Alignment):** ~100,000 cặp ảnh - văn bản kỹ thuật từ tập DataComp-1B đã được lọc theo danh mục "automotive, vehicle mechanics, engine components" để giữ nguyên năng lực zero-shot tổng quát.

---

## 5. Lộ trình Triển khai Huấn luyện 5 Giai đoạn

```
Tuần 1                 Tuần 2                 Tuần 3                 Tuần 4
[ Phase 0 & 1 ] ──────► [    Phase 2    ] ──────► [    Phase 3    ] ──────► [ Phase 4 & 5 ]
Data Prep & Offline    MobileCLIP Baseline    Sigmoid KD Training    Reparam, INT8 ONNX
Teacher Caching        (No KD Benchmark)      (Feature + Relation)   & Android App Deploy
```

### Giai đoạn 0: Chuẩn bị Dữ liệu & Môi trường Kỹ thuật
- Xây dựng script tạo bộ caption song ngữ tự động: `scripts/data_pipeline/build_multimodal_captions.py`.
- Tích hợp thư viện `open_clip` và bộ mã nguồn `mobileclip` từ Apple vào `src/distillation/multimodal/`.

### Giai đoạn 1: Trích xuất & Caching Vector Nhúng của Teacher trên DGX
- Chạy **SigLIP-SO400M** trên toàn bộ 115,000 ảnh để trích xuất trước:
  - Vector nhúng hình ảnh: $\hat{e}_I^{(T)} \in \mathbb{R}^{1152}$.
  - Vector nhúng văn bản: $\hat{e}_T^{(T)} \in \mathbb{R}^{1152}$.
- Lưu trữ vào thư mục cache `/data/cache/siglip_embeddings/` định dạng HDF5/Memmap.
- **Hiệu quả:** Quá trình huấn luyện Student MobileCLIP hoàn toàn không phải chạy Teacher nữa, giải phóng 100% tài nguyên GPU DGX cho Student.

### Giai đoạn 2: Huấn luyện Baseline MobileCLIP Độc lập
- Huấn luyện `mobileclip_s0` từ đầu (scratch) hoặc fine-tune trực tiếp từ weights pretrain gốc mà không có KD.
- Đo chỉ số: Zero-shot Top-1 Accuracy trên 36 class linh kiện khoang máy và Mean Reciprocal Rank (MRR) trong bài toán tìm kiếm ảnh từ text.

### Giai đoạn 3: Huấn luyện Chưng cất Đa phương thức Toàn diện
- Khởi chạy script chưng cất:
  ```bash
  python scripts/training/train_multimodal_kd.py \
      --kd configs/kd_hyperparams_siglip_mobileclip.yaml \
      --teacher-cache /data/cache/siglip_embeddings \
      --student mobileclip_s0 \
      --data data/multimodal_automotive_manifest.json \
      --epochs 80 --batch 64 --device 0,1 \
      --name kd_mobileclip_s0_siglip
  ```
- Áp dụng kỹ thuật:
  - 5 epoch đầu chỉ khởi động (warmup) Projection Head.
  - Sau đó huấn luyện toàn bộ Image + Text encoder với Cosine Annealing LR.

### Giai đoạn 4: Structural Reparameterization & Nén Lượng tử INT8
- Gập nhánh RepMixer thành tích chập chuẩn 3x3 duy nhất bằng phương thức `reparameterize_model()`.
- Export sang ONNX Opset 18:
  - `mobileclip_s0_image.onnx` (~38 MB).
  - `mobileclip_s0_text.onnx` (~25 MB).
- Lượng tử hóa Post-Training Quantization (PTQ) INT8 bằng ONNX Runtime / NNCF:
  - Bản INT8 Image Encoder: **~11 MB**.
  - Bản INT8 Text Encoder: **~8 MB**.

### Giai đoạn 5: Tích hợp vào Android App (`engine_bay_android`) & Web Demo
- Tích hợp 2 file ONNX INT8 vào thư mục assets của Android app: `apps/engine_bay_android/app/src/main/assets/models/`.
- Thực hiện kiểm thử tính năng:
  - Kỹ thuật viên gõ text tiếng Việt hoặc tiếng Anh $\rightarrow$ Android NPU tính embedding trong 2.2 ms $\rightarrow$ Tính cosine similarity với các linh kiện đang quét trên camera.
  - Hiển thị mô hình 3D tương ứng (như `air_filter_box.json`) khi độ tương đồng $> 0.75$.

---

## 6. Đặc tả Siêu tham số Huấn luyện (`kd_hyperparams_siglip_mobileclip.yaml`)

```yaml
distillation:
  teacher:
    name: "siglip_so400m_patch14_384"         # hoặc vit_large_patch14_336
    embedding_dim: 1152
    cache_path: "/data/cache/siglip_embeddings"
    temperature_init: 10.0
    bias_init: -10.0

  student:
    name: "mobileclip_s0"                     # hoặc mobileclip_s2
    image_backbone: "fastvit_mci0"
    embedding_dim: 512
    reparameterize_on_export: true

  # Trọng số Hàm Mất Mát Chưng Cất
  loss_weights:
    siglip_bce: 1.0                           # Sigmoid Binary Cross-Entropy KD
    feat_image: 1.5                           # Image embedding alignment (Cosine)
    feat_text: 1.0                            # Text embedding alignment
    relation_graph: 0.8                       # Similarity Graph (Frobenius)

  # Cấu hình Sigmoid KD
  siglip_kd:
    temperature: 1.5                          # Softening factor cho xác suất Teacher
    symmetric: true                           # Đối xứng cả Image-to-Text và Text-to-Image

training:
  epochs: 80
  batch: 64                                   # Batch size 64 lý tưởng trên DGX
  imgsz: 256                                  # Chuẩn kích thước của MobileCLIP
  optimizer: "AdamW"
  lr0: 0.0005
  lrf: 0.01
  weight_decay: 0.05                          # Regularization mạnh chống quá khớp
  warmup_epochs: 5
  cos_lr: true
  amp: true                                   # Bfloat16
  workers: 8
  seed: 42
  save_period: 5
```

---

## 7. Tiêu chí Đánh giá & Nghiệm thu Kỹ thuật (KPIs)

| Tiêu chí Đánh giá | Baseline Gốc (MobileCLIP-S0 chưa KD) | Mục tiêu Sau Distill từ SigLIP | Ý Nghĩa Thực Tế |
|---|---|---|---|
| **Zero-shot Top-1 Accuracy (36 Car Parts)** | 58.4% | **$\ge 73.5\% - 76.0\%$** | **Tăng +15.1% đến +17.6%** độ chính xác gọi tên linh kiện lạ |
| **Image-to-Text Retrieval Recall@1** | 42.1% | **$\ge 58.0\%$** | Tìm đúng linh kiện ngay từ mô tả đầu tiên |
| **Text-to-Image Retrieval Recall@5** | 68.3% | **$\ge 84.5\%$** | Kỹ thuật viên tìm kiếm linh kiện trong top 5 gợi ý |
| **Độ trễ suy luận Image (Android NPU)** | 2.2 ms | **2.2 ms** (không đổi) | Phản hồi tức thì trên camera 30 FPS |
| **Độ trễ suy luận Text (Android CPU)** | 1.8 ms | **1.8 ms** (không đổi) | Gõ phím tìm kiếm thời gian thực |
| **Dung lượng tổng thể (Bản INT8)** | ~19 MB | **~19 MB** | Dễ dàng nhúng thẳng vào APK mà không làm nặng app |

---

## 8. Quản trị Rủi ro & Phương án Dự phòng

1. **Rủi ro: Hiện tượng sụp đổ không gian nhúng (Representation Collapse)**  
   * *Nguyên nhân:* Nếu trọng số Feature Loss quá cao so với Contrastive Loss, Student có thể ép toàn bộ vector về một điểm hội tụ cục bộ.  
   * *Giải pháp:* Luôn duy trì trọng số Sigmoid BCE KD ($\ge 1.0$) và chuẩn hóa L2 độ dài vector ($L2\_norm = 1.0$) trước khi tính khoảng cách.
2. **Rủi ro: Sai lệch ngữ nghĩa tiếng Việt kỹ thuật (Vietnamese Automotive Terminology Gap)**  
   * *Nguyên nhân:* SigLIP ban đầu được huấn luyện chủ yếu trên tiếng Anh/đa ngôn ngữ chung, một số thuật ngữ chuyên ngành xe hơi (như *"nắp giàn cò"*, *"bầu trợ lực phanh"*) có thể bị dịch máy ngô nghê.  
   * *Giải pháp:* Sử dụng từ điển chuẩn hóa thuật ngữ từ `configs/engine_bay_class_map.yaml` và bộ tri thức `configs/diagnosis_knowledge_vi.yaml` để tạo từ vựng kỹ thuật chuẩn xác.
