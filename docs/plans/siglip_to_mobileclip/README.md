# Chưng Cất Nhúng Đa Phương Thức: SigLIP / CLIP ViT → Tiny-CLIP / MobileCLIP

Thư mục chứa toàn bộ tài liệu kế hoạch, thiết kế hàm mất mát Sigmoid KD, cấu hình siêu tham số và kịch bản triển khai mô hình multimodal siêu nhẹ trên thiết bị di động Android và Web:
**Teacher: SigLIP-SO400M (hoặc CLIP ViT-L/14@336px)**  
**$\longrightarrow$ Student: MobileCLIP-S0 / S2 (hoặc Tiny-CLIP)**

---

## Danh mục Tài liệu trong Thư mục

| File | Mô tả |
|---|---|
| [`training_plan.md`](./training_plan.md) | **Kế hoạch huấn luyện kỹ thuật chi tiết:** Cơ sở lý thuyết Sigmoid Loss KD, căn chỉnh đa phương thức (Dual-Encoder Transfer), pipeline sinh Caption song ngữ chuyên ngành ô tô, ứng dụng Zero-shot Search trên Android (`engine_bay_android`), lộ trình 5 giai đoạn, KPIs và ma trận rủi ro |
| [`kd_hyperparams_siglip_mobileclip.yaml`](./kd_hyperparams_siglip_mobileclip.yaml) | **Cấu hình siêu tham số:** Trọng số Sigmoid BCE loss, Cosine Image/Text Alignment, AdamW, Data Augmentation |

---

## Kiến trúc Tổng quan

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
│   • Dung lượng: ~38 MB (S0) đến ~65 MB (S2), bản INT8 chỉ ~19 MB                  │
│   • Độ trễ: 2.2 ms trên Android NPU / 12 ms trên Mobile CPU                        │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## Khởi động Nhanh

1. **Bước 1: Trích xuất Offline Embedding Cache của SigLIP trên DGX:**
   ```bash
   python scripts/training/extract_multimodal_embeddings.py \
       --model siglip_so400m_patch14_384 \
       --manifest data/multimodal_automotive_manifest.json \
       --output-cache /data/cache/siglip_embeddings
   ```

2. **Bước 2: Huấn luyện Chưng cất Đa Phương Thức Sigmoid KD:**
   ```bash
   python scripts/training/train_multimodal_kd.py \
       --kd configs/kd_hyperparams_siglip_mobileclip.yaml \
       --teacher-cache /data/cache/siglip_embeddings \
       --student mobileclip_s0 \
       --data data/multimodal_automotive_manifest.json \
       --epochs 80 --batch 64
   ```

3. **Bước 3: Export INT8 ONNX & Triển khai vào Android App:**
   ```bash
   python scripts/deployment/export_mobileclip_onnx.py --weights runs/train_kd/kd_mobileclip_s0/weights/best.pt --quantize int8
   # Sao chép vào thư mục assets của Android app:
   cp export/mobileclip_s0_int8.onnx apps/engine_bay_android/app/src/main/assets/models/
   ```
