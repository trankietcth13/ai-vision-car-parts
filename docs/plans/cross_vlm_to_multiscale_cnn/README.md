# Cặp Chưng Cất Thực Chiến Nhất: Transformer-based VLM → CNN Multi-Scale

Thư mục chứa toàn bộ tài liệu quy hoạch, phân tích kỹ thuật, kiến trúc FPN Adapter và bộ siêu tham số cho cặp mô hình chưng cất liên kiến trúc:
**Teacher: Qwen2.5-VL / Qwen3-VL / Grounding DINO (EVA-02 / ViT-H / Swin-L)**  
**$\longrightarrow$ Student: CSP-Darknet C3k2 (YOLO11/YOLO26-seg) hoặc HGNetv2 (D-FINE / RT-DETR)**

---

## Danh mục Tài liệu trong Thư mục

| File | Mô tả |
|---|---|
| [`training_plan.md`](./training_plan.md) | **Kế hoạch huấn luyện kỹ thuật chi tiết:** Phân tích lý do thực chiến, 3 bất đồng kiến trúc và giải pháp (FPN Resampling, CAT Affinity, Offline Caching), lộ trình 5 giai đoạn, KPIs và ma trận rủi ro |
| [`kd_hyperparams_cross_vlm_cnn.yaml`](./kd_hyperparams_cross_vlm_cnn.yaml) | **Cấu hình siêu tham số hoàn chỉnh:** Multi-scale FGD weights ($P_3, P_4, P_5$), Decoupled KD, SGD momentum, data augmentations |

---

## Kiến trúc Tổng quan

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     TEACHER: VLM Transformer (Đầu óc Uyên bác)                    │
│   • Backbone: Qwen2.5-VL / Qwen3-VL / Grounding DINO (EVA-02 / ViT-H / Swin-L)   │
│   • Điểm mạnh: Hiểu ngữ nghĩa trừu tượng, liên kết Text-Image, zero-shot           │
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
│   • Triển khai: TensorRT / ONNX Runtime FP16/INT8 trên trạm kiểm định & Mobile    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## Khởi động Nhanh

1. **Bước 1: Trích xuất Offline Feature Cache trên DGX (chạy 1 lần duy nhất):**
   ```bash
   python scripts/training/extract_vlm_teacher_cache.py \
       --vlm-endpoint http://172.16.110.221:8000/v1 \
       --data configs/data_engine_bay.yaml \
       --output-cache /data/cache/vlm_features_v1
   ```

2. **Bước 2: Huấn luyện Chưng cất Đa tỉ lệ (GPU giải phóng khỏi VLM):**
   ```bash
   python scripts/training/train_cross_vlm_kd.py \
       --kd configs/kd_hyperparams_cross_vlm_cnn.yaml \
       --feature-cache /data/cache/vlm_features_v1 \
       --student yolo11m-seg.pt \
       --data configs/data_engine_bay.yaml \
       --epochs 120 --batch 16 --imgsz 640
   ```
