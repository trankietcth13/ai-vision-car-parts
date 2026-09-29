# Vehicle Image Database Crawler

Project thu thập ảnh từ Wikimedia Commons, Openverse, iFixit và nguồn Cars & Bids đã
được cấp phép. Chỉ lưu ảnh được Jev hoặc model YOLO xác nhận là
`vehicle_component`, `engine_bay` hoặc `sensor`. Mỗi ảnh có ID toàn cục; metadata và tên
component được ghi vào Excel đặt ngay cạnh ảnh.

> **Cars & Bids bắt buộc có quyền sử dụng:** Terms of Use hiện cấm spidering/scraping
> không được phép và cấm trích xuất nội dung để train AI/ML nếu chưa được Cars & Bids
> cho phép rõ ràng. CLI vì vậy yêu cầu `--authorization-file` trước mọi crawl Cars & Bids.
> File này nên là bản lưu văn bản/email cho phép và chỉ đóng vai trò audit nội bộ.

## Output

```text
output/
└── 2014/
    └── Falcon/
        └── F7/
            └── 7.0L_V8/
                ├── 00000001.jpg
                ├── 00000002.jpg
                └── images.xlsx
```

`images.xlsx` có các cột `id`, `ymme_id`, `year`, `make`, `model`, `engine`,
`component_name`, `category`, `confidence`, URL nguồn, license, creator, attribution,
SHA-256 và thời gian thu thập.
Ví dụ `ymme_id`: `2014_Falcon_F7_7.0L_V8_00000001`.

## Cài đặt

```powershell
cd image_database_crawler
python -m pip install -e .
python -m playwright install chromium
```

## Crawl nguồn mở với model Jev/YOLO local

Mặc định project dùng model hiện có tại
`../runs/segment/engine_teacher_v7/weights/best.pt` làm adapter Jev/YOLO. Model detect/segment
sẽ chỉ chấp nhận ảnh khi tìm thấy ít nhất một component.

Wikimedia Commons yêu cầu User-Agent có email hoặc URL liên hệ thật:

```powershell
vehicle-image-crawler --source wikimedia `
  --contact "you@example.com" `
  --category "Automobiles with open hoods" `
  --limit 200 --output output --delay 1.5
```

Openverse chỉ dùng API chính thức và mặc định lọc `CC0`, `PDM`, `CC BY`, `CC BY-SA`:

```powershell
vehicle-image-crawler --source openverse `
  --query "car engine bay" --query "mass air flow sensor car" `
  --limit 100 --output output --delay 1.5
```

iFixit có ảnh sửa chữa cận cảnh nhưng giấy phép là CC BY-NC-SA 3.0. Chỉ bật cho nghiên
cứu phi thương mại:

```powershell
vehicle-image-crawler --source ifixit --allow-noncommercial `
  --query "car battery replacement" --query "car alternator replacement" `
  --limit 100 --output output --delay 1.5
```

Mọi candidate từ ba nguồn trên đều được tải vào RAM, chạy Jev/YOLO, và chỉ được ghi ra
đĩa khi classifier chấp nhận.

## Cars & Bids khi đã có văn bản cho phép

```powershell
vehicle-image-crawler --source carsandbids `
  --url-file examples/carsandbids_urls.txt `
  --authorization-file D:\permissions\carsandbids.txt `
  --output output --delay 1.5
```

Lần đầu Chromium sẽ mở có giao diện. Nếu Cars & Bids hiện Cloudflare, hoàn tất challenge
trong cửa sổ đó; cookie được giữ ở `.browser/carsandbids`. Chỉ dùng `--headless` sau khi
profile này truy cập được site.

Model Jev riêng ở định dạng Ultralytics:

```powershell
vehicle-image-crawler --source openverse --query "car engine bay" --model D:\models\jev.pt
```

## Chạy với Jev HTTP API

```powershell
$env:JEV_ENDPOINT = "http://localhost:8000/classify"
$env:JEV_API_KEY = "..."  # không cần nếu service nội bộ không yêu cầu
vehicle-image-crawler --source openverse --query "car engine bay" --classifier jev-api
```

Request gửi JSON `{ "image_base64": "..." }`. Response chuẩn:

```json
{
  "accepted": true,
  "category": "engine_bay",
  "component_name": "engine_cover, battery",
  "confidence": 0.93,
  "labels": ["engine_cover", "battery"]
}
```

## Kiểm thử

```powershell
python -m pytest
```

## Chính sách crawl

- Dùng API chính thức cho Wikimedia, Openverse và iFixit; Cars & Bids chỉ chạy khi có
  văn bản cho phép.
- License metadata được lọc và lưu cùng từng kết quả; vẫn phải mở `source_page_url` để
  audit lại provenance trước khi phát hành dataset hoặc model.
- Không có proxy rotation, CAPTCHA solver hay cơ chế vượt anti-bot.
- Giữ `--delay` ở mức hợp lý, bắt đầu với số listing nhỏ.
- URL nguồn và SHA-256 luôn được lưu để audit, deduplicate hoặc gỡ dữ liệu.
- Trước khi dùng ảnh cho huấn luyện/phân phối, cần kiểm tra Terms of Use và quyền đối với ảnh.
