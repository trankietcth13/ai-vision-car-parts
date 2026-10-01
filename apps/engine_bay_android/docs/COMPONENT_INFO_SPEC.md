# Thông tin linh kiện: đặc tả `components.json` (song ngữ Anh–Việt)

Khi người dùng chạm vào một linh kiện, app hiển thị nội dung đọc từ `app/src/main/assets/components.json`.
**Mọi đoạn văn bản phải có đủ tiếng Anh và tiếng Việt**, vì app chuyển được giữa hai ngôn ngữ. File được nhờ AI
(Gemini) soạn, rồi được người có chuyên môn kiểm duyệt. Script `prepare_android_app.py` chỉ tạo bản nháp, và không
ghi đè file đã có, trừ khi chạy với `--force-components`.

**Quy trình cập nhật:**
1. Chép đè file mới vào `app/src/main/assets/components.json`.
2. Chạy `./gradlew :app:testDebugUnitTest`. Test `BilingualDataTest` liệt kê mọi chỗ thiếu tiếng Anh hoặc tiếng Việt, và mọi class chưa có mục.
3. Build lại app. Không cần sửa code.

## Định dạng (schema 2)

Mỗi đoạn văn bản là một đối tượng `{"en": "...", "vi": "..."}`. Không dùng chuỗi đơn.

```json
{
  "schema_version": 2,
  "languages": ["en", "vi"],
  "source": "Gemini draft, reviewed by <người duyệt>, <ngày>",
  "components": {
    "battery": {
      "name": {"en": "Battery", "vi": "Ắc quy"},
      "draft": false,
      "summary": {"en": "1–2 sentences: what it is.", "vi": "1–2 câu: linh kiện này là gì."},
      "function": {"en": "...", "vi": "Chức năng trong xe, 1–3 câu."},
      "location_hint": {"en": "...", "vi": "Cách nhận ra, vị trí thường gặp trong khoang máy."},
      "inspection_checks": [{"en": "...", "vi": "Việc kiểm tra bằng mắt hoặc dụng cụ đơn giản"}],
      "common_symptoms": [{"en": "...", "vi": "Triệu chứng khi hỏng"}],
      "related_dtcs": [{"code": "P0562/P0563", "meaning": {"en": "System voltage low / high", "vi": "Điện áp hệ thống thấp / cao"}}],
      "safety_notes": [{"en": "...", "vi": "Cảnh báo an toàn khi thao tác"}]
    }
  }
}
```

- **Khoá** của mỗi linh kiện phải đúng tên class của model (bảng dưới). Cả 21 class đều phải có mục.
- Trường không có nội dung thì bỏ hẳn (hoặc để danh sách rỗng). Không được có trường chỉ điền một ngôn ngữ.
- `draft: true` làm app hiện cảnh báo bản nháp. Chỉ đặt `false` sau khi đã có người duyệt.
- `related_dtcs.code` là mã OBD-II chung; có thể ghi khoảng hoặc nhiều mã, ví dụ `"P0300/P0301-P0308"`. Mã không cần dịch.

## 21 class của model

| Khoá | English | Tiếng Việt |
|---|---|---|
| `battery` | Battery | Ắc quy |
| `battery_terminal` | Battery terminal | Cọc bình ắc quy |
| `fuse_relay_box` | Fuse / relay box | Hộp cầu chì / rơ-le |
| `coolant_reservoir` | Coolant reservoir | Bình nước làm mát |
| `radiator_cap` | Radiator cap | Nắp két nước |
| `brake_fluid_reservoir` | Brake fluid reservoir | Bình dầu phanh |
| `washer_fluid_reservoir` | Washer fluid reservoir | Bình nước rửa kính |
| `engine_cover` | Engine cover | Nắp che động cơ |
| `oil_filler_cap` | Oil filler cap | Nắp châm dầu |
| `oil_dipstick` | Oil dipstick | Que thăm dầu |
| `air_filter_box` | Air filter box | Hộp lọc gió |
| `air_intake_duct` | Air intake duct | Ống hút gió |
| `maf_sensor` | MAF sensor | Cảm biến lưu lượng khí (MAF) |
| `throttle_body` | Throttle body | Cổ họng ga |
| `alternator` | Alternator | Máy phát điện |
| `ignition_coil` | Ignition coil | Bô-bin đánh lửa |
| `radiator_hose` | Radiator hose | Ống két nước |
| `ecu_module` | ECU | Hộp ECU |
| `multimeter_diagnostic_tool` | Multimeter / scan tool | Đồng hồ đo / thiết bị chẩn đoán |
| `intake_manifold` | Intake manifold | Cổ hút |
| `oil_filter` | Oil filter | Lọc dầu |

Tên ngắn trên ảnh lấy từ `configs/diagnosis_knowledge_vi.yaml` (`display_names`). Trường `name` trong file này là tên hiển thị trong bảng thông tin, có thể đầy đủ hơn.

## Yêu cầu nội dung

- **Hai ngôn ngữ phải cùng nội dung**, không thêm bớt ý giữa bản Anh và bản Việt.
  - Tiếng Việt: câu ngắn, viết cho kỹ thuật viên gara, dùng thuật ngữ quen thuộc ở gara (bô-bin, cổ hút, két nước).
  - Tiếng Anh: thuật ngữ kỹ thuật chuẩn.
- Viết chung cho xe xăng phổ thông, không gắn với hãng xe cụ thể. Chỉ dùng mã OBD-II chung (P0xxx, U0xxx), không dùng mã riêng của hãng.
- Không đưa ra thông số cụ thể (mô-men siết, áp suất, điện áp chuẩn) nếu thông số đó thay đổi theo từng xe. Hãy ghi "theo tài liệu của hãng" / "per the manufacturer's service information".
- `safety_notes` phải có cảnh báo khi linh kiện có nguy cơ thật. Ví dụ:
  - không mở nắp két nước khi máy còn nóng;
  - tháo cọc âm ắc quy trước;
  - dầu phanh ăn mòn sơn.
- File hiện tại đã có `name`, `summary` và `related_dtcs` (Anh + Việt) cho từng class, lấy từ `configs/diagnosis_knowledge.yaml` và bản dịch trong `configs/diagnosis_knowledge_vi.yaml`. Có thể dùng làm đầu vào.

## Prompt mẫu cho Gemini

> You are an experienced automotive technician and a professional English–Vietnamese technical translator.
> Complete the JSON below. Keep every key and the structure exactly; every text is an object
> {"en": "...", "vi": "..."} and BOTH languages are required with the same meaning. For each component fill
> `summary`, `function`, `location_hint`, 3–5 `inspection_checks`, 3–5 `common_symptoms`, `related_dtcs`
> (only real, generic OBD-II codes) and `safety_notes`. Vietnamese is for garage technicians (use common workshop
> terms such as bô-bin, cổ hút, két nước). Do not invent vehicle-specific specifications; leave a field out if unsure.
> Keep `"draft": true`. Return valid JSON only.
>
> (dán nội dung `components.json` hiện tại)

Kiểm tra kết quả bằng `python -m json.tool components.json`, sau đó chạy `BilingualDataTest`.
