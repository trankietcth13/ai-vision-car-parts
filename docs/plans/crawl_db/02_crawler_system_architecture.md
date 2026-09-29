# Kiến Trúc Kỹ Thuật Hệ Thống Thu Thập Dữ Liệu (Crawler Engine Architecture)

Tài liệu này mô tả chi tiết thiết kế phần mềm, cấu trúc dữ liệu, cơ chế chống chặn (anti-bot) và luồng xử lý bất đồng bộ cho hệ thống thu thập dữ liệu khoang động cơ.

---

## 1. Thiết Kế Module (Modular Architecture)

Hệ thống được tổ chức trong thư mục `scripts/data_pipeline/crawlers/` theo mô hình Decoupled Architecture (Tách rời khâu cào danh sách tin và khâu tải ảnh):

```
scripts/data_pipeline/crawlers/
├── core/
│   ├── base_crawler.py        # Lớp trừu tượng BaseCrawler (Session, Headers, Retry)
│   ├── browser_engine.py      # Bộ điều khiển Playwright Chrome Channel (cho các trang WAF)
│   ├── pre_filter.py          # Bộ lọc in-memory kiểm tra thumbnail bằng CLIP/YOLO-cls
│   ├── rate_limiter.py        # Token Bucket Rate Limiter & Jitter Delay
│   └── database.py            # SQLite ORM / Manager quản lý trạng thái xe và ảnh
│
├── spiders/
│   ├── bat_spider.py          # Bring a Trailer Crawler (S3 CDN parser)
│   ├── carsandbids_spider.py  # Cars & Bids Next.js state extractor
│   ├── beforward_spider.py    # Be Forward Japan bulk image collector
│   ├── carsome_spider.py      # Carsome 175-point inspection API client
│   └── bonbanh_spider.py      # Bonbanh & Oto.com.vn HTML scraper
│
├── storage/
│   ├── sqlite_schema.sql      # Schema cơ sở dữ liệu quản lý tiến độ
│   └── manifests/             # Các tệp JSON metadata theo phiên
│
└── run_crawler.py             # CLI Entrypoint điều phối toàn bộ tác vụ
```

---

## 2. Luồng Xử Lý 2 Giai Đoạn (Two-Phase Execution Flow)

Để tránh lãng phí băng thông và dung lượng đĩa, hệ thống thực thi theo 2 bước độc lập:

```
[GIAI ĐOẠN 1: HARVESTING (Thu Thập Metadata & Link Ảnh)]
   Sàn Xe ──▶ Parser ──▶ Trích xuất: Make, Model, Year, URL, Photo List
                         └──▶ Lưu vào SQLite: bảng `vehicles` và `raw_photos`
                                (Trạng thái: status = 'PENDING')

[GIAI ĐOẠN 2: PRE-FILTER IN-MEMORY & SELECTIVE DOWNLOADING]
   SQLite Queue ──▶ Tải Thumbnail vào RAM (io.BytesIO)
                 ──▶ Local Model (CLIP/YOLO-cls) dự đoán xác suất `engine_bay`
                 ──▶ Nếu >= 0.80: Tải ảnh Full-Res về đĩa `H:\AI_Datasets\raw_crawled\`
                 ──▶ Nếu < 0.80: BỎ QUA NGAY (Tiết kiệm >90% băng thông)
                 └──▶ Cập nhật SQLite: status = 'DOWNLOADED' hoặc 'SKIPPED'
```

---

## 3. Cấu Trúc Cơ Sở Dữ Liệu SQLite (`crawler_registry.db`)

Hệ thống sử dụng SQLite cục bộ để theo dõi tiến độ cào, đảm bảo **có thể dừng và tiếp tục (Resume/Pause)** bất kỳ lúc nào mà không bị tải trùng lặp.

```sql
-- Bảng quản lý từng xe độc lập
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,               -- 'bat', 'carsandbids', 'beforward', 'carsome', etc.
    source_vehicle_id TEXT NOT NULL,    -- ID xe trên sàn (vd: 'lot-123456')
    make TEXT,                          -- 'Toyota'
    model TEXT,                         -- 'Corolla'
    year INTEGER,                       -- 2021
    listing_url TEXT UNIQUE NOT NULL,   -- Link bài đăng
    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'HARVESTED'     -- 'HARVESTED', 'COMPLETED', 'FAILED'
);

-- Bảng quản lý danh sách ảnh
CREATE TABLE IF NOT EXISTS vehicle_photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER REFERENCES vehicles(id),
    original_url TEXT NOT NULL,
    photo_type_hint TEXT,               -- 'engine', 'exterior', 'interior', 'unknown'
    local_path TEXT,                    -- Đường dẫn file đã tải trên đĩa H:
    file_size_kb REAL,
    is_engine_bay_pred BOOLEAN,         -- Kết quả từ bộ lọc CLIP (NULL nếu chưa chạy)
    clip_confidence REAL,
    download_status TEXT DEFAULT 'PENDING', -- 'PENDING', 'DOWNLOADED', 'FAILED'
    downloaded_at TIMESTAMP,
    UNIQUE(vehicle_id, original_url)
);

CREATE INDEX idx_status ON vehicle_photos(download_status);
CREATE INDEX idx_engine ON vehicle_photos(is_engine_bay_pred);
```

---

## 4. Cơ Chế Xử Lý Anti-Bot & Phòng Ngừa Rate Limit

| Kỹ thuật | Cách thức triển khai | Mục đích |
| :--- | :--- | :--- |
| **Playwright Real Chrome Channel** | `playwright.chromium.launch(channel="chrome")` | Chạy trực tiếp trình duyệt Google Chrome cài đặt trên máy người dùng, vượt qua hầu hết các kiểm tra `navigator.webdriver` của Cloudflare Turnstile mà không cần giả lập. |
| **Token Bucket + Delay Jitter** | Độ trễ ngẫu nhiên: `random.uniform(0.8, 1.8)` giây giữa các lần tải ảnh; 2.5–4.5 giây giữa các trang tin. | Tránh kích hoạt ngưỡng phát hiện tần suất request bất thường của WAF. |
| **User-Agent Pool & Header Masquerading** | Xoay vòng danh sách 20 User-Agent trình duyệt hiện đại (Chrome 122+, Edge 122+ trên Windows/macOS), kèm theo `Sec-Ch-Ua`, `Accept-Language: vi,en-US;q=0.9`. | Giả lập lưu lượng truy cập của người dùng duyệt web thông thường. |
| **Exponential Backoff** | Khi gặp HTTP `429 (Too Many Requests)` hoặc `503`: tạm dừng $2^n \times 3$ giây (tối đa 3 lần thử lại). | Bảo vệ IP không bị đưa vào blacklist dài hạn. |

---

## 5. Tệp Cấu Hình Thực Thi (`configs/crawl_config.yaml`)

```yaml
storage:
  db_path: "data/crawlers/crawler_registry.db"
  download_dir: "H:/AI_Datasets/raw_crawled"
  max_storage_gb: 150

crawler_settings:
  concurrency_limit: 3
  timeout_seconds: 25
  max_retries: 3
  jitter_min: 0.8
  jitter_max: 1.8

spiders:
  bring_a_trailer:
    enabled: true
    target_count: 250
    category_filter: "engine"
  
  cars_and_bids:
    enabled: true
    target_count: 200
    category_filter: "engine"

  be_forward:
    enabled: true
    target_count: 400
    makes: ["toyota", "honda", "mazda", "hyundai", "kia"]
    photo_indices: [15, 16, 17, 18, 19, 20]

  carsome:
    enabled: true
    target_count: 200
    inspection_tab: "engine_compartment"
```
