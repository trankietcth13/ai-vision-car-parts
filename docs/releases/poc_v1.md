# Engine Bay Vision: bản POC v1 (chốt ngày 30/09/2026)

Nhận diện và phân đoạn 20 loại linh kiện khoang máy trên ảnh chụp. Gói chạy được là [`apps/engine_bay_web`](../../apps/engine_bay_web/README.md)
(web + REST API, Docker). Git tag: `poc-v1`.

## Thành phần được chốt

| File trong `apps/engine_bay_web/models/` | Model | Nguồn (DGX) | md5 |
|---|---|---|---|
| `kd_n_full.pt` (mặc định) | Student yolo11n-seg, 2,8 triệu tham số, distill từ teacher | `runs/segment/runs/train_kd/kd_n_full/weights/avg5.pt` | `e005b93ecf6758a22d8da5ce3d8b032d` |
| `teacher_full.pt` | Teacher yolo11l-seg, 27,6 triệu tham số | `runs/segment/teacher_full/weights/avg5.pt` | `118f8098a9b73fec1585727a75e19ae9` |

- Cả hai model được train trên **toàn bộ 28 xe** của dataset hiện có (không giữ lại xe nào). Đây là bản dùng nhiều dữ liệu nhất.
- Ngưỡng tin cậy riêng cho từng class, **hiệu chỉnh riêng cho từng model**, nằm trong `config/class_thresholds/<model>.yaml`.
  Các ngưỡng này lấy từ dự đoán out-of-fold của cross-validation 3 fold (`cv_eval.py`), và được bật mặc định trong web và API.

## Độ chính xác

Đo bằng hai cách, ứng với hai tình huống sử dụng:

| Tình huống | Cách đo | Student `kd_n_full` | Teacher `teacher_full` |
|---|---|---|---|
| **Xe đã có trong dataset** (ảnh mới của cùng các xe) | CV 3 fold chia theo ảnh, mask mAP50-95 | 0,381 ± 0,010 | 0,418 ± 0,010 |
| | Cùng CV, với ngưỡng từng class: precision / recall / F1 | 0,715 / 0,612 / 0,66 | 0,74 / 0,674 / 0,705 |
| **Xe mới chưa từng thấy** | Cùng công thức train, nhưng giữ riêng 3 xe để test (125 ảnh), mask mAP50-95 | 0,27–0,30 (2 seed) | 0,354 |
| | mask mAP50 | 0,47–0,51 | 0,58 |

Cách đọc các con số:
- Trên **xe mới**, kết quả chỉ nên dùng như **gợi ý để kỹ thuật viên xác nhận**, chưa đủ tin cậy để tự động kết luận.
- Tập test xe mới chỉ gồm 3 xe, nên sai số lớn: khoảng tin cậy 95% theo ảnh là khoảng **±0,05** mAP.
- Một phần "lỗi" trên tập test là do **nhãn test sai**. Khi soát lại hình học, 18/26 ca box lệch hoá ra là lỗi nhãn. Sửa 18 ảnh đó làm F1 của teacher tăng từ 0,612 lên 0,647 mà không đổi model. Vì vậy độ chính xác thật cao hơn một chút so với bảng trên.
- Tốc độ:
  - Student: khoảng 3 ms/ảnh trên GPU, 0,35–0,6 s/ảnh trên CPU (đo trên máy dev).
  - Teacher: khoảng 9 ms/ảnh trên GPU, 1,8–2,7 s/ảnh trên CPU.

## Giới hạn đã biết

- **Class yếu** (mask AP50 trong CV của student dưới 0,5): máy phát điện (0,35), ống két nước (0,35), hộp ECU (0,37),
  bình nước làm mát (0,47). Trên xe mới, bình nước làm mát gần như không nhận được (mAP50-95 khoảng 0,05): model hay nhầm nó với bình dầu phanh, và bản thân nhãn cũng thường nhầm hai loại này.
- Chỉ hỗ trợ **khoang máy**, 20 class. Linh kiện ngoài danh sách (lọc dầu, turbo, lốc điều hoà…) sẽ không được nhận ra, hoặc bị gán nhầm sang class gần nhất.
- Dataset chỉ có 28 xe. Với dòng xe có bố cục khoang máy khác xa các xe này, độ chính xác sẽ thấp hơn bảng trên.

## Các thử nghiệm đã chạy nhưng không đưa vào bản này

| Thử nghiệm | Kết quả trên xe mới (mask mAP50-95) | Lý do không dùng |
|---|---|---|
| v9: thêm dữ liệu Phase 3 | Teacher 0,355 (p5: 0,354). KD 0,277 (p5: 0,286) | Không cải thiện |
| E1: KD theo kế hoạch cải tiến | 0,281 / 0,280 (p5: 0,302 / 0,270) | Không cải thiện, dù chênh lệch giữa 2 seed nhỏ hơn |
| Taxonomy v2 (27 class, có nhóm `other_*`) | Teacher 0,335 trên bộ nhãn khác | Chưa so công bằng được; phép so sánh trên 117 ảnh và 20 class chung đang chạy |
| v10: nhãn đã sửa (D2 + một phần bình chứa) | Đang train (30/09 đêm) | Nếu cải thiện rõ, sẽ train lại bản "full" trên nhãn đã sửa để làm **POC v1.1** |

## Chạy và thay model

- Cách chạy (Python hoặc Docker) và mô tả API: xem [`apps/engine_bay_web/README.md`](../../apps/engine_bay_web/README.md).
- Để thay model mà không sửa code:
  1. Chép file `.pt` mới vào `models/`.
  2. Chép file ngưỡng tương ứng vào `config/class_thresholds/<tên model>.yaml`.
  3. Chọn model mặc định bằng biến `ENGINE_BAY_MODEL`.
