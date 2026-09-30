# Engine Bay Vision

Ứng dụng web và API nhận diện, phân đoạn 20 loại linh kiện khoang máy trong ảnh. Thư mục này độc lập: có sẵn model,
ảnh mẫu và cấu hình; chỉ cần Python hoặc Docker để chạy.

```text
apps/engine_bay_web/
├── app.py             web UI (Gradio) + REST API (FastAPI), chạy bằng uvicorn
├── models/            kd_n_full.pt (mặc định), teacher_full.pt
├── config/            class_thresholds/<model>.yaml (ngưỡng riêng từng lớp, hiệu chỉnh riêng cho từng model)
├── examples/          6 ảnh mẫu
├── requirements.txt
└── Dockerfile
```

## Model

Bản chốt hiện tại: **POC v1** (tag `poc-v1`). Độ chính xác chi tiết và các giới hạn xem [docs/releases/poc_v1.md](../../docs/releases/poc_v1.md).

| File | Loại | Kích thước | Tốc độ | Ghi chú |
|---|---|---|---|---|
| `kd_n_full.pt` | Student yolo11n-seg, distill từ teacher | 6 MB | khoảng 4 ms/ảnh trên GPU, 0,3–0,8 s trên CPU | Mặc định. Train trên cả 28 xe của dataset |
| `teacher_full.pt` | Teacher yolo11l-seg | 56 MB | khoảng 9 ms/ảnh trên GPU, 2–3 s trên CPU | Chính xác hơn, dùng khi server có GPU |

Độ chính xác đã đo (29/09/2026):
- Trên ảnh của các xe trong dataset (đã học): 1.017/1.081 ảnh đạt, recall 0,94, precision 0,75 ở ngưỡng 0,25.
- Trên xe mới chưa từng thấy (công thức tương tự, đo trên 3 xe giữ riêng): mask mAP50 khoảng 0,51, mAP50-95 khoảng 0,30.
  Với xe mới, nên dùng kết quả như gợi ý để kỹ thuật viên xác nhận.

## Chạy trực tiếp

```bash
# 1. PyTorch cho phần cứng của server (chọn một)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu      # CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128    # GPU NVIDIA
# 2. Thư viện còn lại
pip install -r requirements.txt
# 3. Chạy
python app.py                                   # http://127.0.0.1:7860, chỉ máy này truy cập
ENGINE_BAY_HOST=0.0.0.0 python app.py           # mở cho mạng nội bộ
```

Cấu hình bằng biến môi trường (hoặc cờ `--model`, `--host`, `--port`):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ENGINE_BAY_HOST` | `127.0.0.1` | Địa chỉ lắng nghe; `0.0.0.0` để mở ra mạng |
| `ENGINE_BAY_PORT` | `7860` | Cổng |
| `ENGINE_BAY_MODEL` | `kd_n_full.pt` | Model mặc định (tên file trong `models/` hoặc đường dẫn) |
| `ENGINE_BAY_DEVICE` | GPU `0` nếu có, không thì `cpu` | Thiết bị suy luận |
| `ENGINE_BAY_CONF` | `0.35` | Ngưỡng tin cậy mặc định |
| `ENGINE_BAY_MODELS_DIR` | `./models` | Thư mục chứa model |
| `ENGINE_BAY_THRESHOLDS` | `./config/class_thresholds.yaml` | File ngưỡng dùng chung, chỉ dùng khi không có `config/class_thresholds/<model>.yaml` |

## Chạy bằng Docker

```bash
docker build -t engine-bay-vision .
docker run -d --name engine-bay -p 7860:7860 --restart unless-stopped engine-bay-vision
curl http://localhost:7860/healthz
```

Image mặc định chạy CPU. Với server có GPU NVIDIA: đổi dòng cài PyTorch trong `Dockerfile` sang index `cu128`,
đặt `ENV ENGINE_BAY_DEVICE=0`, và chạy với `docker run --gpus all ...` (cần NVIDIA Container Toolkit).

## Giao diện web

Mở `http://<server>:7860/`, tải ảnh lên (hoặc chọn ảnh mẫu): kết quả hiện ngay. Cuộn để phóng to, kéo để di chuyển,
phím `O` hoặc nút **Ảnh gốc** để so sánh. Mục **Tùy chọn** đổi model, ngưỡng tin cậy, và bật ngưỡng riêng từng lớp khi có
file `config/class_thresholds.yaml`.

## REST API

`GET /healthz`

```json
{"status": "ok", "device": "cpu", "default_model": "kd_n_full", "models": ["kd_n_full", "teacher_full"], "per_class_thresholds": false}
```

`POST /api/detect` (multipart/form-data)

| Trường | Bắt buộc | Ý nghĩa |
|---|---|---|
| `file` | có | Ảnh JPG/PNG |
| `conf` | không | Ngưỡng tin cậy, mặc định `ENGINE_BAY_CONF` |
| `model` | không | `kd_n_full` hoặc `teacher_full` (tên trong `/healthz`, có hoặc không có `.pt`) |
| `per_class` | không | `true` (mặc định): dùng ngưỡng từng lớp nếu có file cấu hình |

```bash
curl -X POST -F "file=@examples/Request_ID_29_img_011.jpg" -F "conf=0.35" http://localhost:7860/api/detect
```

```json
{
  "model": "kd_n_full", "image_size": [1600, 1067], "inference_ms": 377.3, "count": 14,
  "detections": [
    {"class": "air_filter_box", "name_vi": "Hộp lọc gió", "confidence": 0.9256,
     "box_xyxy": [1147.4, 482.3, 1477.5, 687.3], "polygon": [[1150.0, 490.0], "..."]}
  ]
}
```

Toạ độ tính theo pixel của ảnh gửi lên. Lỗi: `400` nếu file không phải ảnh, `404` nếu không có model.
Tài liệu API tự sinh: `http://<server>:7860/docs`.

## Cập nhật model

Chép checkpoint mới (đã gỡ adapter KD, ví dụ `runs/train_kd/<run>/weights/avg5.pt` trong project train) vào `models/`
với tên mới, rồi đặt `ENGINE_BAY_MODEL=<tên>.pt` và khởi động lại. Khi có kết quả cross-validation, chép
`configs/class_thresholds.yaml` từ project train vào `config/`.

## Lưu ý vận hành

- Ứng dụng không có đăng nhập. Khi mở ra ngoài mạng nội bộ, đặt sau reverse proxy (nginx, Caddy) có HTTPS và xác thực.
- Mỗi lúc xử lý một yêu cầu (hàng đợi Gradio, một model trong bộ nhớ). Lần gọi đầu chậm hơn vì phải nạp model.
- Ảnh chỉ được xử lý trong bộ nhớ, không lưu xuống đĩa.
