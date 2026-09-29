# Ma Trận Nguồn & Phân Bổ Mức Độ Ưu Tiên (Source Matrix & Priority Strategy)

Tài liệu xác định các nguồn dữ liệu mục tiêu, phân tầng ưu tiên dựa trên:
1. **Độ sạch và chất lượng ảnh khoang máy** (độ phân giải, độ sắc nét, góc chụp).
2. **Độ dễ khai thác kỹ thuật** (CDN mở, ít chặn bot, cấu trúc HTML/JSON ổn định).
3. **Mức độ tương thích với thị trường mục tiêu** (các dòng xe phổ biến tại Việt Nam và Châu Á).

---

## 1. Phân Tầng Ưu Tiên Triển Khai (Tiered Priority)

```
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1: TRIỂN KHAI NGAY (Dễ cào, chất lượng 2K-4K, CDN mở hoàn toàn)     │
│ ➔ Bring a Trailer, Cars & Bids, Be Forward Japan, SBT Japan, Wikimedia │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 1.5: GARAGE & DIY PORTALS (Đặc trị 20 linh kiện, fix AP thấp)     │
│ ➔ CarCareKiosk, Pelican Parts, iFixit API, FCP Euro, OTO-HUI, Hà Thành │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 2: TƯƠNG THÍCH ĐÔNG NAM Á & VN (Bắt buộc cho phân phối xe thực tế) │
│ ➔ Carsome (175-pt inspect), Carro, Bonbanh.com, Chợ Tốt Xe            │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 3: MÔI TRƯỜNG BỤI BẶM / THỰC TẾ (Chống Overfitting ánh sáng đẹp)  │
│ ➔ Copart, IAAI (Xe bảo hiểm / cứu hộ chụp ngoài trời)                 │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 4: MỞ RỘNG ĐẶC THÙ (Carvana 360°, Mobile.de EU)                   │
│ ➔ Khai thác chọn lọc frame xoay 360°                                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi Tiết Các Tầng Nguồn

### 🟢 Tier 1: Sàn Đấu Giá Chuyên Nghiệp & Xuất Khẩu JDM (Ưu tiên số 1)

| Nền tảng | Quốc gia | Target Quota | Kỹ thuật Crawl & CDN | Điểm mạnh cốt lõi |
| :--- | :--- | :---: | :--- | :--- |
| **Bring a Trailer (BaT)** | Mỹ / Toàn cầu | **1.200 ảnh** (250 xe) | CDN: AWS S3 / CloudFront.<br/>URL: Có mục `gallery` với filter tab `Engine`. Dễ parse HTML/JSON. | Ảnh siêu nét (2K–4K), chụp cực kỳ kỹ từng chi tiết bình dầu phanh, ắc quy, cổ hút, nhãn mác. |
| **Cars & Bids** | Bắc Mỹ | **800 ảnh** (200 xe) | CDN: Fastly.<br/>API: Trang web sử dụng Next.js, có state `__NEXT_DATA__` chứa sẵn danh sách ảnh phân loại mục `engine`. | Tập trung các dòng xe hiện đại từ 1990–nay (rất sát với xe đang lưu hành). |
| **Be Forward** | Nhật Bản | **1.500 ảnh** (400 xe) | CDN: Mở 100%, không rate-limit IP thông thường.<br/>Ảnh số 15–25 luôn là nắp capo mở. | Đầy đủ mọi dòng xe Toyota (Vios/Yaris, Corolla, Camry), Honda (Civic, CR-V), Mazda, Nissan. |
| **SBT Japan** | Nhật Bản | **600 ảnh** (150 xe) | CDN: Tải trực tiếp qua file ID xe.<br/>HTML tĩnh, dễ cào bằng `requests`/`aiohttp`. | Ảnh chụp kiểm định tại bãi cảng, ánh sáng ngoài trời chân thực. |
| **Wikimedia Commons** | Quốc tế | **500 ảnh** | MediaWiki API: `Category:Automobiles with open hoods`. Đã có script sẵn. | 100% giấy phép Creative Commons / Public Domain sạch, phân loại theo hãng xe. |

---

### 🟣 Tier 1.5: Garage, Cổng Kỹ Thuật DIY & Xưởng Detailing (Cứu Cánh Cho Linh Kiện Điểm Thấp)

*Mục đích: Không giống các sàn bán xe chỉ chụp nắp capo tổng quát, các nguồn Garage & DIY cung cấp ảnh macro/cận cảnh chụp từng linh kiện lúc tháo lắp, kiểm tra, sửa chữa. Đây là nguồn dữ liệu cứu cánh cho các lớp có mAP gần như bằng 0 trong bản v7 (`coolant_reservoir` 0.01, `radiator_hose` 0.02, `ecu_module` 0.20, `oil_dipstick` 0.38).*

| Nền tảng | Quốc gia | Target Quota | Kỹ thuật Crawl & CDN | Linh kiện mục tiêu & Điểm mạnh |
| :--- | :--- | :---: | :--- | :--- |
| **CarCareKiosk** | Mỹ / Toàn cầu | **3.000 ảnh** | URL theo pattern:<br/>`/video/{year}_{make}_{model}/{component}`.<br/>Frame video HD trên AWS CloudFront. | **Kho linh kiện số 1**: Chỉ đích danh vị trí `coolant_reservoir`, `oil_dipstick`, `battery`, `fuse_box`, `brake_fluid`. |
| **Pelican Parts** | Mỹ & Châu Âu | **1.500 ảnh** | HTML tĩnh cấu trúc đồng bộ:<br/>`/techarticles/{chassis}/{system}/...`. | Cận cảnh tháo lắp máy phát (`alternator`), mobin (`ignition_coil`), cổ hút, ống làm mát trên/dưới. |
| **iFixit Car Maintenance** | Toàn cầu | **800 ảnh** | REST API chính thức:<br/>`api.ifixit.com/2.0/guides?filter=Car`. | Giấy phép CC-BY. Ảnh có sẵn tọa độ khoanh tròn linh kiện (markers) cọc bình, bu-lông, ống nước. |
| **FCP Euro DIY** | Mỹ & Châu Âu | **800 ảnh** | Blog WordPress chuẩn, CDN mở, ảnh phân giải cao. | Chuyên sâu xe BMW, Mercedes, Volvo; ảnh khoang máy chụp dưới đèn xưởng cực chuẩn màu sắc. |
| **Diễn đàn OTO-HUI** | Việt Nam | **1.200 ảnh** | XenForo forum crawler: bóc tách ảnh đính kèm bài viết đại tu máy. | Chụp tại gara thực tế ở VN, thợ cầm đồng hồ đo multimeter, tháo rã hộp ECU, cảm biến MAF. |
| **Hà Thành Garage & DPRO VN** | Việt Nam | **800 ảnh** | Album dịch vụ vệ sinh khoang máy trên website/fanpage. | 100% dòng xe lưu hành tại VN (Vios, Accent, CX-5, SantaFe, Ranger) trước và sau khi dọn rửa. |
| **AMMO NYC & Chicago Auto Pros** | Mỹ | **600 ảnh** | Trích xuất video frame YouTube 4K & gallery ảnh bắn đá khô. | Khoang máy bám bẩn cực hạn (chuột cắn, rỉ sét, dầu loang) vs sạch bóng $\rightarrow$ chống overfitting độ bóng. |

---

### 🟡 Tier 2: Sàn Xe Đông Nam Á & Việt Nam (Ưu tiên số 2)

*Mục đích: Đảm bảo mô hình nhận diện chính xác các khoang động cơ của những dòng xe chiếm thị phần lớn nhất tại Việt Nam.*

| Nền tảng | Quốc gia | Target Quota | Kỹ thuật Crawl & CDN | Dòng xe mục tiêu |
| :--- | :--- | :---: | :--- | :--- |
| **Carsome** | Malaysia, Indo, Thái | **1.000 ảnh** (200 xe) | API: Báo cáo kiểm định 175 điểm trả về file JSON có mục `inspection_engine_bay`. | Vios, Yaris, City, Civic, Accent, Almera, Attrage. |
| **Carro** | Singapore, Malaysia | **500 ảnh** (100 xe) | Web CDN: Cloudflare.<br/>Trích xuất báo cáo kiểm tra 160 điểm. | Bán tải Hilux, Ranger, Triton, SUV CX-5, CR-V. |
| **Bonbanh.com** | Việt Nam | **800 ảnh** (200 xe) | HTML thuần, không bot protection.<br/>Ảnh JPG link trực tiếp. | Salon xe cũ VN: Vios, Accent, Morning, i10, Mazda 3, Fortuner. |
| **Chợ Tốt Xe** | Việt Nam | **600 ảnh** (150 xe) | Mobile API: `gateway.chotot.com/v1/public/ad-listing`. | Ảnh chụp thực tế bằng điện thoại của chủ xe, góc nghiêng lệch, thiếu sáng. |

---

### 🟠 Tier 3: Sàn Đấu Giá Xe Cứu Hộ / Bụi Bặm (Chống Overfitting)

*Mục đích: Khoang máy xe đời thực khi mang đi bảo dưỡng hoặc cứu hộ thường không bóng bẩy như xe phòng trưng bày; chúng có lớp bụi mỏng, dầu loang nhẹ hoặc dây điện câu thêm.*

| Nền tảng | Target Quota | Kỹ thuật & Đặc điểm |
| :--- | :---: | :--- |
| **Copart** | **800 ảnh** (200 xe) | Cào theo danh sách `Lot Number`. Ảnh chụp ngoài trời dưới nắng gắt hoặc bóng râm, phản ánh độ tương phản ánh sáng phức tạp. |
| **IAAI** | **500 ảnh** (120 xe) | Ảnh độ nét cao, zoom được chi tiết các bình chứa nước làm mát, két nước và hộp cầu chì khi có biến dạng nhẹ. |

---

### 🔵 Tier 4: Dữ Liệu 360 Độ & Thị Trường Châu Âu

| Nền tảng | Target Quota | Kỹ thuật & Đặc điểm |
| :--- | :---: | :--- |
| **Carvana** | **400 ảnh** (30 xe $\times$ 12 góc) | Trích xuất các frame xoay 360 độ của khoang máy. Cung cấp chuỗi ảnh liên tục xoay quanh nắp capo, giúp mô hình học tính bất biến theo góc nhìn (Viewpoint Invariance). |
| **Mobile.de** | **600 ảnh** (150 xe) | Chuyên các dòng xe Đức (BMW, Mercedes-Benz, Audi, Volkswagen, Porsche) với bố cục khoang máy bọc kín nắp chắn động cơ (engine cover). |

---

## 3. Ma Trận Phân Bổ Hãng Xe Mục Tiêu (Vehicle Distribution Target)

Để triệt tiêu khoảng cách mAP 0,47 giữa tập train và test, kế hoạch đặt hạn ngạch (quota) cân bằng giữa các phân khúc xe:

```
[TỔNG HẠN NGẠCH: ~8.000 ẢNH KHOANG MÁY ĐÃ LỌC SẠCH]

├── 1. Xe Phổ Thông Châu Á (40% - 3.200 ảnh)
│   ├── Toyota (Vios, Corolla, Camry, Hilux, Fortuner): 1.000 ảnh
│   ├── Honda (City, Civic, CR-V, Accord): 800 ảnh
│   ├── Hyundai & Kia (Accent, i10, Morning, Tucson, Santa Fe, Seltos): 800 ảnh
│   └── Mazda & Mitsubishi (Mazda 3, CX-5, Xpander, Triton): 600 ảnh
│
├── 2. Xe Bán Tải & SUV Mỹ (20% - 1.600 ảnh)
│   ├── Ford (Ranger, Everest, F-150, Explorer): 1.000 ảnh
│   └── Chevrolet & Jeep: 600 ảnh
│
├── 3. Xe Sang & Cao Cấp Châu Âu (25% - 2.000 ảnh)
│   ├── Mercedes-Benz (C-Class, E-Class, GLC): 700 ảnh
│   ├── BMW (3-Series, 5-Series, X3, X5): 700 ảnh
│   └── Audi, Porsche, Volkswagen: 600 ảnh
│
└── 4. Xe Cổ & Thể Thao Độc Đáo (15% - 1.200 ảnh)
    └── Các dòng xe có khoang máy lộ cơ khí, ống dẫn độ, turbocharger: 1.200 ảnh
```
