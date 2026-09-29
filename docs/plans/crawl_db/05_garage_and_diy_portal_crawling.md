# Kế Hoạch Khai Thác Nguồn Ảnh Garage, Cổng Kỹ Thuật DIY & Xưởng Detailing (Garage & Technical Portals Crawling Plan)

Khác với các sàn bán xe (chỉ chụp tổng quan để bán), **các Garage sửa chữa, Cổng kỹ thuật DIY và Xưởng Detailing** sở hữu kho ảnh chụp cận cảnh, tháo lắp chi tiết từng bộ phận dưới nắp capo. Đây là **chìa khóa giải quyết các lớp linh kiện yếu nhất** trong lộ trình huấn luyện ([`../engine_bay_accuracy_roadmap_2026-09-28.md`](../engine_bay_accuracy_roadmap_2026-09-28.md#23-theo-từng-lớp-kd-v7-mask-ap50-trên-test)).

---

## 1. Mục Tiêu Trọng Tâm: Bổ Khuyết Các Lớp Linh Kiện Điểm Thấp

Trong đợt thử nghiệm v7, các lớp sau có điểm số rất thấp do thiếu mẫu và góc chụp bị che khuất:
- `coolant_reservoir` (AP 0.01)
- `radiator_hose` (AP 0.02)
- `ecu_module` (AP 0.20)
- `oil_dipstick` (AP 0.38) & `alternator` (AP 0.42)

**Ưu thế từ nguồn Garage & DIY**:
1. **CarCareKiosk**: Chụp chính xác vị trí que thăm dầu (`oil_dipstick`), bình nước phụ (`coolant_reservoir`), cọc bình (`battery_terminal`), hộp cầu chì (`fuse_relay_box`) cho từng dòng xe.
2. **Pelican Parts & FCP Euro**: Chụp chi tiết máy phát (`alternator`), mobin (`ignition coil`), cổ hút (`intake manifold`), ống nước làm mát trên/dưới (`radiator_hose_upper / lower`) trong quá trình đại tu.
3. **Xưởng Detailing (AMMO NYC, Hà Thành Garage, DPRO)**: Cung cấp cặp ảnh **Trước (bám bụi, dầu mỡ)** và **Sau (sạch bóng)** giúp mô hình không bị phụ thuộc vào màu sắc hay độ bóng của nhựa/kim loại.
4. **Diễn đàn OTO-HUI**: Ảnh thợ máy cầm đồng hồ đo (`multimeter_diagnostic_tool`), cờ lê tháo lắp hộp ECU và cảm biến MAF.

---

## 2. Kiến Trúc Khai Thác Chi Tiết Theo Nguồn

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. CARCAREKIOSK (Mục tiêu: 3.000 ảnh linh kiện có nhãn sẵn)           │
│ Cấu trúc: carcarekiosk.com/video/{Year}_{Make}_{Model}/{Component}     │
│ ➔ Bóc tách trực tiếp ảnh frame có mũi tên/khoanh tròn vị trí           │
├────────────────────────────────────────────────────────────────────────┤
│ 2. PELICAN PARTS & FCP EURO (Mục tiêu: 1.500 ảnh đại tu khoang máy)    │
│ Cấu trúc: pelicanparts.com/techarticles/{Make}_{Chassis}/{System}/...  │
│ ➔ Lấy ảnh top-down view tháo lắp máy phát, mobin, ống làm mát         │
├────────────────────────────────────────────────────────────────────────┤
│ 3. iFIXIT CAR GUIDES (Mục tiêu: 800 ảnh chuẩn Creative Commons)        │
│ API chính thức: https://www.ifixit.com/api/2.0/guides/                 │
│ ➔ Dữ liệu sạch, có sẵn tọa độ vùng khoanh cọc bình, bu-lông           │
├────────────────────────────────────────────────────────────────────────┤
│ 4. HÀ THÀNH GARAGE, DPRO & OTO-HUI (Mục tiêu: 1.200 ảnh xe tại VN)     │
│ Quét album dịch vụ 'Vệ sinh khoang máy' & box thảo luận kỹ thuật      │
│ ➔ Sát 100% với xe lưu hành tại Việt Nam (Vios, Accent, CX-5, Ranger)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Kỹ Thuật Crawl Chuyên Biệt

### 3.1. CarCareKiosk Component Harvester
Trang CarCareKiosk có cấu trúc URL hoàn toàn có thể dự đoán (predictable routing):

```python
# scripts/data_pipeline/crawlers/spiders/carcarekiosk_spider.py
import requests
from bs4 import BeautifulSoup

COMPONENTS_MAP = {
    "coolant_reservoir": "coolant/check_level",
    "oil_dipstick": "oil/check_oil_level",
    "battery": "battery/replace_battery",
    "fuse_relay_box": "fuse/engine",
    "brake_fluid_reservoir": "brake_fluid/check_fluid_level",
    "air_filter_box": "air_filter_engine/replace"
}

TARGET_VEHICLES = [
    ("Toyota", "Camry", "2018"),
    ("Toyota", "Corolla", "2019"),
    ("Honda", "Civic", "2020"),
    ("Honda", "CR-V", "2019"),
    ("Mazda", "CX-5", "2021"),
    ("Ford", "Ranger", "2020"),
    ("Hyundai", "Tucson", "2021")
]

def harvest_carcarekiosk(make, model, year, comp_key):
    path = COMPONENTS_MAP[comp_key]
    url = f"https://www.carcarekiosk.com/video/{year}_{make}_{model}/{path}"
    # Trích xuất các ảnh video frames chất lượng cao
    # Lưu metadata kèm nhãn class sơ bộ (comp_key)
```

### 3.2. iFixit API Bulk Ingestion
iFixit hỗ trợ REST API trả về định dạng JSON có sẵn tọa độ khoanh tròn linh kiện:

```python
import requests

def fetch_ifixit_car_guides():
    # Lấy danh sách guide thuộc category Car
    url = "https://www.ifixit.com/api/2.0/guides?filter=Car"
    resp = requests.get(url).json()
    for guide in resp:
        guide_id = guide['guideid']
        detail = requests.get(f"https://www.ifixit.com/api/2.0/guides/{guide_id}").json()
        for step in detail.get('steps', []):
            for media in step.get('media', {}).get('data', []):
                img_url = media.get('original')
                markers = media.get('markers', []) # Tọa độ khoanh linh kiện
                # Lưu ảnh và marker phục vụ train
```

---

## 4. Kế Hoạch Thực Hiện & Tích Hợp

| Tuần | Nhiệm vụ | Đầu ra |
| :---: | :--- | :--- |
| **Tuần 1** | Triển khai crawler cho **CarCareKiosk** và **iFixit API**. | 2.500 ảnh định danh rõ vị trí *coolant, dipstick, battery, fuse box*. |
| **Tuần 2** | Triển khai crawler cho **Pelican Parts** & **FCP Euro**. | 1.500 ảnh cận cảnh máy phát (`alternator`), mobin (`ignition_coil`), cổ hút. |
| **Tuần 3** | Cào dữ liệu từ **Hà Thành Garage**, **DPRO VN** và diễn đàn **OTO-HUI**. | 1.000 ảnh khoang máy xe Việt Nam (trước và sau dọn rửa). |
| **Tuần 4** | Chạy SAM 2 sinh mask đa giác tự động từ vị trí linh kiện $\rightarrow$ Nạp vào `engine_bay_train_v9`. | Tập train v9 có độ cân bằng hoàn hảo giữa 20 lớp linh kiện. |
