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

## 2a. Giới hạn box: ống nạp, cọc bình, ắc quy (D2, thêm 2026-09-30)

**Lý do:** 9–12% vật trên xe mới bị bỏ sót vì **box lệch** (IoU 0,1–0,5). Trong 40 lỗi loại này của teacher, 25 lỗi thuộc 3 class sau: ống nạp 10/47, cọc bình 9/53, ắc quy 6/37 (`docs/reports/miss_analysis_2026-09-30.md`).

| class | 1 instance = | Gồm | Không gồm |
|---|---|---|---|
| `air_intake_duct` | **một đoạn ống liên tục giữa hai linh kiện** (hộp lọc gió → bướm ga / turbo / thân MAF; ống hút gió trước hộp lọc là một instance riêng) | đoạn xếp nếp, đoạn cứng, đai kẹp ở hai đầu; nếu bị che một phần, các phần nhìn thấy vẫn là **một** instance (mask có thể rời) | thân cảm biến MAF, bầu cộng hưởng, hộp lọc gió, bướm ga, ống thông hơi nhỏ nối vào ống |
| `battery_terminal` | **một cọc** (+ hoặc −) | kẹp trên cọc + nắp chụp nhựa (đỏ/đen) + đoạn cáp ngắn ngay tại kẹp | đoạn cáp chạy xa, hộp cầu chì gắn trên cọc dương, cọc câu bình ở chỗ khác |
| `battery` | **thân vỏ ắc quy nhìn thấy được** (mặt trên + các mặt bên thấy được, cả vùng cọc) | vỏ bọc cách nhiệt ôm sát thân bình, khi thấy cọc hoặc tem ắc quy | giá kẹp, khay, nắp che rời không ôm thân bình; ắc quy bị che hoàn toàn thì không gán nhãn |

**Quy tắc chung:** box ôm sát phần nhìn thấy, không kéo dài qua vật khác. Hai ắc quy / hai cọc / hai đoạn ống thì là hai instance, không gộp một box.

## 2b. Linh kiện cao áp hybrid/EV (thêm 2026-09-30)

**Lý do:** trên ảnh Chevrolet Volt, cả 4 model hiện có đều gọi hộp biến tần cao áp là `engine_cover` hoặc `air_filter_box` (`docs/reports/model_comparison_2026-09-30.md`).

| linh kiện (taxonomy v2) | Box bao | Dấu hiệu | Train dưới |
|---|---|---|---|
| `hv_cable` | từng đoạn cáp/ống gen **màu cam** nhìn thấy được | màu cam, dày, nối biến tần / mô-tơ / máy nén điện | `hv_component` |
| `inverter_converter` | cả hộp | hộp nhôm lớn, cáp cam, ống nước làm mát, nhãn HIGH VOLTAGE | `hv_component` |
| `electric_ac_compressor` | cả máy nén | không có puly, có cáp cam | `hv_component` |
| `hv_service_plug` | chốt / tay nắm | màu cam | `hv_component` |
| `inverter_coolant_reservoir` | cả bình | bình nước làm mát thứ hai, nhỏ hơn, gần biến tần, xe có cáp cam | `other_reservoir` |

- **Mọi class cao áp mang cờ `safety: high_voltage`.** Teacher system ghi cảnh báo "chỉ nhận diện, không hướng dẫn tháo" vào `warnings`. Tầng hướng dẫn [K]/[L5] phải từ chối mọi bước tháo lắp chạm vào các linh kiện này.
- **Có cáp cam** → không gán bộ phận đó thành `engine_cover` / `air_filter_box` / `fuse_relay_box`.
- **Nguồn ảnh:** cần ảnh khoang máy xe hybrid/EV (Toyota THS, Honda i-MMD, Volt…). Hiện dataset gần như không có loại ảnh này.

## 3. Cách áp dụng

- **Skill review** (`engine-bay-label-review`) áp dụng bảng trên khi viết verdict: box sai quy định → `bad_geometry` kèm `fixed_box_norm`; box sai tên → `wrong_class`.
- **Prompt cho VLM** (`qwen_grounding_annotator` / `hybrid_qwen_deepseek`): thêm yêu cầu nêu dấu hiệu phân biệt cho 3 loại bình, đúng như cột "dễ nhầm với".
- **Soát lại dữ liệu cũ:** hàng đợi `docs/reports/reservoir_rereview_queue.jsonl` (75 box). Đợt 1 đã kiểm 34 box (24 coolant/brake + 10 washer lấy mẫu) trên pixel (`docs/reports/reservoir_rereview_round1.md`):
  - **7 box đổi tên (21%)**, 5 cần vẽ lại box, 6 có độ tin cậy thấp (ảnh mờ/tối, không đọc được nắp).
  - Mọi lỗi tên đều do đoán theo vị trí khi không đọc được nắp.
  - Cờ "chỉ bao nắp" của Jev báo thừa, vì ghi chú review mô tả mask chứ không mô tả box. Nên chỉ dùng cờ này để **xếp thứ tự** soát lại, không dùng để sửa nhãn.
  - Danh sách sửa cho lần build sau: `docs/reports/reservoir_corrections_round1.jsonl`.
- **Đợt 2** (100 box nhận dạng yếu, `docs/reports/reservoir_rereview_round2.md`):
  - 16/80 box coolant/PS sai loại. Trong đó 4 là **bình nước làm mát biến tần hybrid** và 3 là **chi tiết thân xe màu trắng** (tháp giảm xóc) bị tưởng là bình.
  - Bình dầu phanh: 19/20 đúng loại, nhưng **10/20 box cần vẽ lại**.
  - Tổng hai đợt: 134 box → 9 đổi tên, 14 bỏ, 31 vẽ lại box.
  - Bản sao verdict đã sửa nằm ở `data/<review>_rr/`, sinh ra bằng `apply_reservoir_corrections.py`.
- **Đợt tiếp theo:** soát toàn bộ nhãn coolant/brake đã chấp nhận mà người review chỉ dựa vào vị trí hoặc hình dạng (theo trường `evidence` của Jev). Tỉ lệ sai 21% cho thấy lỗi không chỉ nằm trong hàng đợi.
- **Đánh giá:** tập test chỉ có 3 instance coolant, nên AP gần 0 của class này trên test không có ý nghĩa thống kê. Hãy dùng AP theo class từ CV (Phase 3) để quyết định class này có thật sự yếu không.
