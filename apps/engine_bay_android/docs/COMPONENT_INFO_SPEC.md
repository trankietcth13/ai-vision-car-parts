# Thông tin linh kiện: đặc tả `components.json`

Khi người dùng chạm vào một linh kiện, app hiển thị nội dung đọc từ `app/src/main/assets/components.json`.
File này được viết tay hoặc nhờ AI (Gemini) soạn, rồi được người có chuyên môn kiểm duyệt. Script
`prepare_android_app.py` chỉ tạo bản nháp ban đầu, và **không bao giờ ghi đè** file đã có.

Sau khi sửa file: chép đè vào `app/src/main/assets/components.json`, rồi build lại app. Không cần sửa code.

## Định dạng

```json
{
  "schema_version": 1,
  "language": "vi",
  "source": "Gemini draft, reviewed by <người duyệt>, <ngày>",
  "components": {
    "battery": {
      "name_vi": "Ắc quy",
      "name_en": "12 V battery",
      "draft": false,
      "summary": "1–2 câu: linh kiện này là gì.",
      "function": "Chức năng trong xe, 1–3 câu.",
      "location_hint": "Cách nhận ra và vị trí thường gặp trong khoang máy.",
      "inspection_checks": ["Việc kỹ thuật viên kiểm tra bằng mắt hoặc dụng cụ đơn giản", "..."],
      "common_symptoms": ["Triệu chứng khi linh kiện hỏng", "..."],
      "related_dtcs": [{"code": "P0562", "meaning": "Điện áp hệ thống thấp"}],
      "safety_notes": ["Cảnh báo an toàn khi thao tác", "..."]
    }
  }
}
```

- **Khoá** của mỗi linh kiện phải đúng tên class của model (bảng dưới). Khoá lạ sẽ bị bỏ qua; class thiếu sẽ hiện
  "Thông tin đang được biên soạn".
- Mọi trường đều tuỳ chọn. Trường để trống không được hiển thị.
- `draft: true` làm app hiện cảnh báo "Bản nháp – nội dung chưa được kiểm duyệt". Chỉ đặt `false` sau khi đã có người duyệt.
- `related_dtcs.code` có thể là một khoảng, ví dụ `"P0300/P0301-P0308"`.

## 21 class của model

| Khoá | Tên tiếng Việt trong app |
|---|---|
| `battery` | Ắc quy |
| `battery_terminal` | Cọc bình ắc quy |
| `fuse_relay_box` | Hộp cầu chì / rơ-le |
| `coolant_reservoir` | Bình nước làm mát |
| `radiator_cap` | Nắp két nước |
| `brake_fluid_reservoir` | Bình dầu phanh |
| `washer_fluid_reservoir` | Bình nước rửa kính |
| `engine_cover` | Nắp che động cơ |
| `oil_filler_cap` | Nắp châm dầu |
| `oil_dipstick` | Que thăm dầu |
| `air_filter_box` | Hộp lọc gió |
| `air_intake_duct` | Ống hút gió |
| `maf_sensor` | Cảm biến lưu lượng khí (MAF) |
| `throttle_body` | Cổ họng ga |
| `alternator` | Máy phát điện |
| `ignition_coil` | Bô-bin đánh lửa |
| `radiator_hose` | Ống két nước |
| `ecu_module` | Hộp ECU |
| `multimeter_diagnostic_tool` | Đồng hồ đo / thiết bị chẩn đoán |
| `intake_manifold` | Cổ hút |
| `oil_filter` | Lọc dầu |

## Yêu cầu nội dung

- Tiếng Việt, câu ngắn, viết cho kỹ thuật viên gara. Thuật ngữ dùng quen tay ở gara (bô-bin, cổ hút, két nước).
- Viết chung cho xe xăng phổ thông, không gắn với một hãng xe cụ thể. Mã DTC dùng mã OBD-II chung (P0xxx), không dùng mã riêng của hãng.
- Không đưa ra thông số cụ thể (mô-men siết, áp suất, điện áp chuẩn) nếu thông số đó thay đổi theo từng xe. Hãy ghi "theo tài liệu của hãng".
- `safety_notes` phải có cảnh báo khi linh kiện có nguy cơ thật. Ví dụ:
  - không mở nắp két nước khi máy còn nóng;
  - tháo cọc âm ắc quy trước;
  - dầu phanh ăn mòn sơn.
- Bản nháp lấy từ `configs/diagnosis_knowledge.yaml` đang có sẵn `name_en`, `summary` (tiếng Anh) và `related_dtcs` cho mỗi class, có thể dùng làm đầu vào.

## Prompt mẫu cho Gemini

> Bạn là kỹ thuật viên ô tô có kinh nghiệm. Hãy hoàn thiện file JSON sau theo đúng định dạng (giữ nguyên khoá và
> cấu trúc), viết bằng tiếng Việt cho kỹ thuật viên gara. Với mỗi linh kiện, điền `summary`, `function`,
> `location_hint`, 3–5 `inspection_checks`, 3–5 `common_symptoms`, `related_dtcs` (chỉ mã OBD-II chung, có thật,
> kèm ý nghĩa tiếng Việt) và `safety_notes`. Không bịa thông số kỹ thuật cụ thể; nếu không chắc, hãy bỏ trống.
> Giữ `"draft": true`. Chỉ trả về JSON hợp lệ.
>
> (dán nội dung `components.json` hiện tại)

Nên kiểm tra kết quả bằng `python -m json.tool components.json` trước khi chép vào app.
