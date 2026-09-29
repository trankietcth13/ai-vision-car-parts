# Engine-bay component segmentation · knowledge distillation

Nhận diện và phân đoạn 20 loại linh kiện khoang máy trong ảnh kiểm tra xe. Teacher **yolo11l-seg** học từ nhãn đã review;
student **yolo11n-seg** (6 MB) được distill từ teacher để chạy trên thiết bị biên.

Kết quả hiện tại (29/09/2026), mask mAP50-95 trên 3 xe chưa từng thấy:

| Model | mask mAP50-95 | mask mAP50 | Độ trễ (DGX, toàn pipeline) |
|---|---|---|---|
| Teacher tốt nhất (`p5_reg`, trung bình 5 checkpoint) | 0.354 | 0.577 | 9.2 ms (chỉ mạng) |
| Student tốt nhất (`kd_n_p5t` seed 0) | 0.302 | 0.514 | 4.0 ms |
| Model cuối trên cả 28 xe (`kd_n_full`) | đo bằng cross-validation | | 4.0 ms |

Chi tiết: `docs/reports/Engine_Bay_KD_Training_Report_{EN,VI}.pdf`, `docs/plans/model_versions_report.html`,
`docs/plans/distillation_explainer.html`.

## Cấu trúc

```text
apps/engine_bay_web/       Web UI + REST API triển khai model
src/distillation/          KD trainer và các loss FGD/KL/localization
scripts/training/          Train teacher/student và checkpoint averaging
scripts/evaluation/        QA, test cases, cross-validation và label-error mining
scripts/data_pipeline/     Thu thập, gán nhãn, review và dựng dataset
scripts/deployment/        Export ONNX, TensorRT và benchmark latency
scripts/operations/        Điều phối và theo dõi job DGX
tools/image_database_crawler/  Crawler ảnh độc lập
configs/                   Ontology, dataset manifests và KD hyperparameters
tests/                     Unit/integration tests
docs/plans/                Roadmap, nghiên cứu và handoff
docs/reports/              Script và báo cáo huấn luyện
```

Dữ liệu (`data/`, `dataset/`, `datasets/`), checkpoint, kết quả train/QA, model triển khai và secrets đều bị loại bởi
`.gitignore`. Sao model triển khai vào `apps/engine_bay_web/models/` ở máy chạy dịch vụ.

## Quy trình

1. Gán nhãn máy: `scripts/data_pipeline/qwen_grounding_annotator.py`, `hybrid_qwen_deepseek.py` → `refine_masks_sam2.py`.
2. Review: `build_review_packets.py` → skill `.claude/skills/engine-bay-label-review` → `apply_review.py`;
   kiểm chứng độc lập bằng các công cụ trong `scripts/evaluation/`.
3. Dựng dataset: `build_v8.py` (chia theo xe, đo trên xe mới) hoặc `build_full_dataset.py` (28 xe, k-fold theo ảnh).
4. Train trên DGX (một job mỗi lúc):

```bash
python scripts/operations/dgx_train.py push --dataset data/engine_bay_train_v8
# teacher: yolo11l-seg, 640 px, 100 epoch, save_period 5, weight_decay 0.001, scale 0.7, perspective 0.0005
python scripts/training/average_checkpoints.py --run runs/segment/<teacher> --top-k 5
python scripts/training/train_kd.py --kd configs/kd_hyperparams_p5.yaml --teacher runs/segment/<teacher>/weights/avg5.pt \
    --student yolo11n-seg.pt --data <data.yaml> --name <student>
python scripts/training/average_checkpoints.py --run runs/train_kd/<student> --top-k 5 --strip-kd
```

5. Đánh giá bằng `scripts/evaluation/qa_test.py`, `test_cases.py` và `cv_eval.py`.
6. Triển khai bằng `scripts/deployment/`; ứng dụng nằm trong `apps/engine_bay_web/`.

## Kiểm thử

```bash
python -m pytest tests -q
```

## Lưu ý

- Chỉ train trên DGX (`admin@dgx-host`), mỗi lúc một job vì bộ nhớ dùng chung với dịch vụ vLLM.
- `Note.md` chứa thông tin đăng nhập DGX: không đưa file này lên server hay kho mã dùng chung.
