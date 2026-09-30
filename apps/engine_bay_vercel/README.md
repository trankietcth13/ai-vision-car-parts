# Engine Bay Vision: demo trên Vercel

Trang tĩnh chạy model POC v1 (`kd_n_full`, yolo11n-seg) **ngay trong trình duyệt** bằng onnxruntime-web (WASM).
Không có server: ảnh không rời máy người dùng.

Vì sao không deploy app Python (`apps/engine_bay_web`) lên Vercel: PyTorch, Ultralytics và Gradio vượt giới hạn dung lượng
của một Vercel Function, và Gradio cần kết nối dài hạn. Student ở dạng ONNX chỉ nặng 11 MB và chạy tốt trên trình duyệt.

| File | Nguồn |
|---|---|
| `index.html`, `detector.js`, `vercel.json` | Trong git |
| `model/kd_n_full.onnx`, `config.json`, `examples/`, `vendor/ort/` | Sinh ra bằng `python scripts/deployment/prepare_vercel_demo.py` |

## Deploy

```bash
python scripts/deployment/prepare_vercel_demo.py     # export ONNX, ngưỡng từng lớp, ảnh mẫu, onnxruntime-web 1.30.0
cd apps/engine_bay_vercel
vercel deploy            # bản preview
vercel deploy --prod     # bản production
```

`vercel.json` bật COOP/COEP (cross-origin isolation) để WASM chạy đa luồng. Nhờ đó toàn bộ thư viện được self-host,
không phụ thuộc CDN.

## Đối chiếu với Ultralytics

Tiền xử lý giống app Python:
1. Thu nhỏ ảnh về cạnh dài 1600 px.
2. Letterbox 640 với resize `cv2.INTER_LINEAR` (dựng lại chính xác trong JS).

Hậu xử lý: NMS theo từng lớp với IoU 0,7, mask = hệ số × prototype, cắt theo box, ngưỡng logit 0.

Kiểm tra trên 6 ảnh mẫu bằng Chromium headless (ngưỡng 0,35):
- **72/72 box của Ultralytics khớp** (cùng lớp, IoU ≥ 0,9); trình duyệt ra dư 1 box có điểm sát ngưỡng.
- Điểm tin cậy lệch tối đa khoảng 0,13, do `.pt` dùng letterbox chữ nhật còn ONNX dùng khung vuông 640.
- Tốc độ khoảng 180 ms/ảnh với 4 luồng trên máy dev.

## Lưu ý

- Ai mở được trang cũng tải được file model (`model/kd_n_full.onnx`).
- Mô tả độ chính xác và giới hạn: [docs/releases/poc_v1.md](../../docs/releases/poc_v1.md).
