# Dual-VLM Transformer to Hybrid Edge ConvNet Distillation

Thư mục chứa toàn bộ tài liệu kế hoạch, thiết kế kiến trúc, thông số siêu tham số và hướng dẫn triển khai cho mô hình huấn luyện chưng cất mới:
**Teacher (DaViT / Swin-Transformer Large) $\rightarrow$ Student (MobileNetV4-Hybrid / FastViT)**.

---

## Danh mục Tài liệu trong Thư mục

| File | Mô tả |
|---|---|
| [`training_plan.md`](./training_plan.md) | **Kế hoạch huấn luyện toàn diện** (Bối cảnh, Kiến trúc Teacher/Student, Khung phương pháp KD 5 tầng, Lộ trình 5 giai đoạn, Ma trận rủi ro & KPIs) |
| [`kd_hyperparams_dual_vlm.yaml`](./kd_hyperparams_dual_vlm.yaml) | **File cấu hình mẫu siêu tham số** (Loss weights, Feature FGD, DKD, Learning rate Cosine, Data Augmentation) |

---

## Kiến trúc Tổng quan

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TEACHER (Server / DGX)                          │
│   DaViT-Large / Swin-Transformer Large (Florence-2 / Grounding DINO)   │
│   • Spatial Window Attention + Channel Group Attention                 │
│   • Visual-Language Grounding đa tỷ lệ (Pixel-level Topology)          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                     Knowledge Distillation (KD)
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   Feature KD                 Logit KD                   Mask / Loc KD
 (Cross-Attention          (Decoupled KD +             (DFL Distribution +
  FGD + 1x1 Adapters)       Class Affinity)             Mask IoU / Dice)
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                         STUDENT (Edge / Mobile)                        │
│             MobileNetV4-Hybrid-Medium / FastViT-S12                    │
│   • Train: RepVGG / RepMixer đa nhánh + Mobile MHA                     │
│   • Deploy: Gập nhánh (Reparameterize) thành 1 Conv duy nhất (0 cost)  │
│   • 100% ONNX Runtime / NPU / WebAssembly / WebGPU                     │
└────────────────────────────────────────────────────────────────────────┘
```

## Khởi động Nhanh

File cấu hình chính thức đã được liên kết với `configs/kd_hyperparams_dual_vlm.yaml`:

```bash
# Huấn luyện chưng cất từ Teacher DaViT sang Student MobileNetV4-Hybrid trên DGX
python scripts/training/train_vlm_kd.py \
    --kd configs/kd_hyperparams_dual_vlm.yaml \
    --teacher weights/teacher_davit_large_avg5.pt \
    --student mobilenetv4_hybrid_medium \
    --data configs/data_engine_bay.yaml \
    --epochs 120 --batch 16 --imgsz 640
```
