# Quy định gán nhãn: bình chứa, nắp và tấm chắn nhiệt (2026-09-29)

**Căn cứ:**
- Bảng thống kê bằng Jev trên 888 ghi chú của chuyên gia review (`docs/reports/reservoir_label_audit.md`).
- `component_catalog.md` của skill automotive-expert.

**Mục đích:** các class này đang bị gán nhãn theo nhiều kiểu khác nhau:
- Box coolant được chấp nhận: 63% bao cả thân bình, còn lại chỉ bao nắp hoặc một phần.
- Box washer: 77% chỉ bao nắp.
- 24% box coolant được chấp nhận chỉ dựa vào hình dạng.
- 56% box tấm chắn nhiệt do VLM đề xuất không phải linh kiện.

## 1. Quy tắc chung

1. **Mỗi nhãn cần ít nhất 2 dấu hiệu độc lập.** Ví dụ: ký hiệu hoặc chữ trên nắp, vạch MIN/MAX, màu dung dịch, vị trí so với mốc (xi-lanh phanh chính, vách máy, két nước), ống nối đi đâu.
2. **Chỉ dựa vào hình dạng thì không được chấp nhận ngay.** Trường hợp này đánh dấu "cần kiểm tra"; nếu không quyết được thì không gán tên cụ thể.
3. **Box bao toàn bộ phần nhìn thấy của thân bình, gồm cả nắp.** Chỉ bao riêng nắp khi thân bình thật sự bị che.
4. **Nắp nằm trên bình thuộc về bình**, không gán thành class nắp riêng. `radiator_cap` chỉ dùng cho nắp nằm trên cổ két nước.

## 2. Theo từng class

| class | Box bao | Dấu hiệu bắt buộc (ít nhất 2) | Dễ nhầm với / cách phân biệt |
|---|---|---|---|
| `coolant_reservoir` | cả thân bình nhìn thấy + nắp | nắp có ký hiệu hơi nóng / cảnh báo áp suất; MIN/MAX (hoặc LOW/FULL); dung dịch hồng/xanh/cam; ống nối về két nước hoặc cổ nước | **bình dầu phanh:** nhỏ, nằm **trên xi-lanh phanh chính ở vách máy**, nắp có chữ DOT. Đây là lỗi phổ biến nhất: 46% box coolant bị loại thực ra là bình dầu phanh. **Nước rửa kính:** nắp có ký hiệu kính chắn gió + tia nước, thường màu xanh |
| `brake_fluid_reservoir` | cả bình trên xi-lanh phanh chính | chữ DOT 3/4 hoặc ký hiệu phanh trên nắp; nằm trên xi-lanh phanh chính có 2 ống thép, ở vách máy phía người lái | bình dầu côn (xe số sàn): nhỏ hơn, nằm cạnh → `other_reservoir` (tier B) |
| `washer_fluid_reservoir` | **nắp + toàn bộ cổ đổ trong suốt nhìn thấy được** khi thân bình bị che (thường gặp: 11/11 mẫu soát lại); cả bình nếu thấy được | ký hiệu kính chắn gió + tia nước trên nắp; nắp thường màu xanh; ở góc trước / vè | bình nước làm mát (có MIN/MAX, nắp cảnh báo nóng) |
| `radiator_cap` | chỉ nắp, trên **cổ két nước** | nắp kim loại/nhựa có cảnh báo không mở khi nóng; nằm trên két nước ở đầu xe | nắp trên bình phụ thì gán nhãn cho bình, không gán nắp riêng; nắp dầu nằm trên nắp dàn cò |
| `power_steering_reservoir`, `clutch_reservoir` | cả bình | ký hiệu vô-lăng (PS); nằm cạnh bình dầu phanh (côn) | tier B → train dưới `other_reservoir` |
| `exhaust_manifold_heat_shield` | **chỉ tấm chắn kim loại bắt bu-lông trên cổ xả / turbo / đoạn ống xả đầu**, nhìn thấy từ trên xuống | kim loại mỏng sáng hoặc có gân; bu-lông bắt vào cổ xả; nằm ở mặt xả của nắp máy | **không** gán cho: tấm chắn dưới gầm (ảnh gầm xe bị loại), tấm cách nhiệt vách máy/capo, nắp máy bằng nhựa, nắp dàn cò, vỏ bảo vệ ống điều hòa |

## 3. Cách áp dụng

- **Skill review** (`engine-bay-label-review`) áp dụng bảng trên khi viết verdict: box sai quy định → `bad_geometry` kèm `fixed_box_norm`; box sai tên → `wrong_class`.
- **Prompt cho VLM** (`qwen_grounding_annotator` / `hybrid_qwen_deepseek`): thêm yêu cầu nêu dấu hiệu phân biệt cho 3 loại bình, đúng như cột "dễ nhầm với".
- **Soát lại dữ liệu cũ:** hàng đợi `docs/reports/reservoir_rereview_queue.jsonl` (75 box). Đợt 1 đã kiểm 34 box (24 coolant/brake + 10 washer lấy mẫu) trên pixel (`docs/reports/reservoir_rereview_round1.md`):
  - **7 box đổi tên (21%)**, 5 cần vẽ lại box, 6 có độ tin cậy thấp (ảnh mờ/tối, không đọc được nắp).
  - Mọi lỗi tên đều do đoán theo vị trí khi không đọc được nắp.
  - Cờ "chỉ bao nắp" của Jev báo thừa, vì ghi chú review mô tả mask chứ không mô tả box. Nên chỉ dùng cờ này để **xếp thứ tự** soát lại, không dùng để sửa nhãn.
  - Danh sách sửa cho lần build sau: `docs/reports/reservoir_corrections_round1.jsonl`.
- **Đợt tiếp theo:** soát toàn bộ nhãn coolant/brake đã chấp nhận mà người review chỉ dựa vào vị trí hoặc hình dạng (theo trường `evidence` của Jev). Tỉ lệ sai 21% cho thấy lỗi không chỉ nằm trong hàng đợi.
- **Đánh giá:** tập test chỉ có 3 instance coolant, nên AP gần 0 của class này trên test không có ý nghĩa thống kê. Hãy dùng AP theo class từ CV (Phase 3) để quyết định class này có thật sự yếu không.
