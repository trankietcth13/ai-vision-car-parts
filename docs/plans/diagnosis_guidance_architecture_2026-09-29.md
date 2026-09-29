# Kiến trúc cho chẩn đoán, hướng dẫn tháo lắp và phát hiện linh kiện (2026-09-29)

**Mục tiêu:** (1) chẩn đoán, (2) hướng dẫn tháo lắp, (3) phát hiện linh kiện trong khoang máy.
**Nền tảng:** model 20 class hiện có (student KD 3 ms, teacher trên server). Dữ liệu gồm 1.081 ảnh đã review, 28 xe, ảnh gốc 6000×4000.

## 1. Nguyên tắc thiết kế

1. **Quan hệ lấy từ tri thức sửa chữa, không học từ ảnh.**
   - Các câu hỏi như "bu-lông nào giữ MAF", "giắc nào thuộc cảm biến nào", "bước tháo theo thứ tự nào" đã có trong tài liệu sửa chữa theo từng dòng xe (YMME).
   - Phần thị giác chỉ cần **định vị** những thứ tài liệu nhắc tới.
   - Vì vậy không train GNN hay scene graph. Dữ liệu vài trăm ảnh không đủ, và quan hệ qua dây điện trong ảnh 2D thường bị che.
2. **Crop theo tác vụ, không theo giải phẫu.**
   - Đo trên nhãn hiện có (box cha nới 25%): cọc bình nằm trong ắc quy 86%. Nhưng nắp dầu trong nắp máy chỉ 21%, que thăm dầu 14%, mobin 2%, MAF trong ống gió 62%.
   - Nên crop quanh **linh kiện đang chẩn đoán hay tháo** (lấy từ mã lỗi DTC hoặc bước hướng dẫn). Chi tiết của nó (giắc, bu-lông, kẹp) nằm trong vùng đó theo đúng định nghĩa.
3. **Độ phân giải cao chỉ ở tầng crop.**
   - Model toàn cảnh giữ 640–1024. Chạy 1280 khi không train ở scale đó làm recall vật lớn giảm 0,50 → 0,34.
   - Crop lấy từ ảnh gốc 6000×4000. Tên ảnh `Request_ID_xx__img_nnn__*` ứng với `dataset/Request_ID_xx/img_nnn.jpg`.
4. **Một model cho tầng chi tiết.** Box và mask ra cùng lúc từ YOLO-seg; keypoint suy ra từ mask bằng hình học.
5. **Teacher và student cùng họ YOLO.** Không dùng D-FINE: nó chỉ phát hiện box, không có segmentation, và đầu box không hợp với YOLO26.

## 2. Kiến trúc

```
Mã lỗi DTC / bước hướng dẫn ──► [K] Tầng tri thức: DTC → linh kiện nghi ngờ → quy trình
        │                            (bước, chi tiết cần tháo, số bu-lông, giắc, lực siết, dụng cụ)
        ▼
Ảnh gốc 6000×4000
  │
  ├─► [L1] Model toàn cảnh (có sẵn, 640–1024): linh kiện chính + mask
  │        • edge: student n-seg 3 ms · server: teacher l-seg (+ SAM2 vẽ lại mask)
  │        • class nhỏ độc lập (que thăm dầu, nắp dầu, nắp két): tiled inference
  │
  ├─► [L2] Crop ROI theo tác vụ: box của linh kiện mục tiêu (từ [K]) + lề 25–50%,
  │        cắt ở độ phân giải gốc, resize về 640–1024
  │
  ├─► [L3] Model chi tiết trên crop (YOLO26-seg hoặc YOLO11-seg; thêm P2 nếu cần):
  │        electrical_connector · bolt_nut · hose_clamp · clip_push_pin (mở rộng dần)
  │
  ├─► [L4] Hình học + luật (không học):
  │        • gán chi tiết cho linh kiện: nằm trong mask/box đã nới
  │        • tâm bu-lông = trọng tâm mask; đầu ống/dây = điểm cuối skeleton
  │        • đối chiếu với [K]: "thấy 3/4 bu-lông, còn 1 bị che"
  │
  └─► [L5] Giao diện hướng dẫn: vẽ lên ảnh từng bước; kiểm tra bước bằng ảnh trước/sau (giai đoạn sau)
```

Ví dụ luồng chẩn đoán: DTC P0101 (lưu lượng MAF) → [K]: MAF sensor, giắc, kẹp ống, ống gió → [L1] tìm `maf_sensor` → [L2] crop → [L3] tìm giắc + 2 kẹp → [L5] "kiểm tra giắc (khoanh đỏ), kiểm tra ống gió sau MAF có hở không".

## 3. Khoảng trống lớn nhất: danh sách class chưa theo nhu cầu chẩn đoán

- Nhiều linh kiện hay gặp trong chẩn đoán **chưa có trong 20 class**: van xả EVAP (purge valve), van EGR, cảm biến MAP, cảm biến oxy (O2), cổ nhiệt / thermostat, kim phun và ống dầu, van PCV, cảm biến trục cam.
- Tập Phase 2 đã có ảnh cận cảnh EVAP, MAP, O2, giắc điện; trước đây DeepSeek gán nhầm chúng thành mobin hoặc ECU. Nên dùng lại những ảnh này khi mở rộng class.
- Nên chọn class theo **tần suất mã lỗi DTC** trên dữ liệu thật của Innova, không theo độ dễ nhận diện.

## 4. Lộ trình

| Giai đoạn | Nội dung | Nhãn mới | Điều kiện đi tiếp |
|---|---|---|---|
| **A. Củng cố L1** | E0 (đo `save_json`, theo cỡ vật), E1 (FGD theo kích thước), tiled inference cho class nhỏ, SAM2 vẽ lại mask trên server, luật ngữ cảnh + ngưỡng theo class | không | test mask mAP50-95 tăng; recall vật < 32 px tăng |
| **B. Tầng tri thức** | bảng DTC → linh kiện → quy trình cho khoảng 10 mã lỗi phổ biến, theo từng YMME có trong dữ liệu | không (dữ liệu văn bản) | ghép được với tên class của L1 |
| **C. Thử L2 + L3** | crop theo tác vụ từ ảnh gốc cho 3 linh kiện ưu tiên; quy định gán nhãn 2 class (`electrical_connector`, `bolt_nut`); khoảng 300 crop, SAM2 hỗ trợ và người review (VLM không tin được trên ảnh cận) | khoảng 300 crop | mask AP50 ≥ 0,5 trên crop của xe test |
| **D. Mở rộng** | thêm `hose_clamp`, `clip_push_pin`; thêm class chẩn đoán theo tần suất DTC; L4 dùng luật | theo kết quả C | đo end-to-end: tỉ lệ bước hướng dẫn vẽ đúng vị trí |
| **E. Sau cùng** | kiểm tra bước bằng ảnh trước/sau; keypoint model chỉ khi hình học không đủ | tuỳ | – |

## 5. Rủi ro

- **Lỗi cộng dồn.** L1 hiện chỉ khoảng 0,30 mask mAP50-95 trên xe mới. Nếu L1 bỏ sót linh kiện mục tiêu, cả chuỗi hỏng. Cần cho người dùng tự chọn hoặc chạm vào linh kiện để thay thế khi L1 không tìm thấy.
- **Công gán nhãn chi tiết nhỏ.** Mỗi crop có nhiều bu-lông và giắc. Chỉ gán trên crop của linh kiện mục tiêu, không gán toàn khoang máy.
- **Latency.** L1 + vài crop L3 + SAM2 khoảng vài trăm ms trên GPU server. Trên edge chỉ nên chạy L1 + L3 nano.
- **Giấy phép dữ liệu ngoài** (iFixit là NC) nếu dùng để train tầng chi tiết.
