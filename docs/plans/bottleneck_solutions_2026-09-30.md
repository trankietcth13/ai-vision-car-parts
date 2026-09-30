# Giải pháp cho các nút thắt còn lại (2026-09-30)

**Đầu vào:**
- Phân tích bỏ sót trên 125 ảnh test (`docs/reports/miss_analysis_2026-09-30.md`).
- Bản tóm tắt gửi ra ngoài (`docs/reports/bottleneck_brief_2026-09-30.md`).
- Ý kiến tham khảo: DeepSeek v4-pro (có suy luận), Codex (đọc repo ở chế độ chỉ đọc), Jev (chấm điểm từng phương án). Câu trả lời gốc nằm ở `artifacts/consult/`.

## 1. Chẩn đoán: teacher bỏ sót gì

| Kiểu | Teacher | Student | Ý nghĩa |
|---|---|---|---|
| Phát hiện đúng | 58% | 48% | |
| Có thấy nhưng điểm < 0,25 | 15% | 18% | model thấy vật nhưng không tự tin (khác vật với độ bất định thật trên xe lạ) |
| Box lệch (IoU 0,1–0,5) | 9% | 12% | chủ yếu ống nạp, cọc bình, ắc quy: vật dài hoặc khó xác định giới hạn; nhãn hình học cũng không nhất quán (25% box bình chứa phải vẽ lại) |
| Nhầm class | 4% | 3% | hộp cầu chì → hộp lọc gió (7/11) |
| Hoàn toàn không thấy | 14% | 19% | mobin, ống nạp, bướm ga, ECU, MAF; xe Request_ID_29 tệ nhất (19%) |

## 2. Ba bên xếp hạng thế nào

| Phương án | Claude | DeepSeek | Codex | Jev (mức tăng / chi phí) |
|---|---|---|---|---|
| #7 Thêm xe mới + active learning | 1 | 1 | 1 | 2,78 / 1,9 (cao nhất) |
| #5 Self-training trên ảnh chưa nhãn | 2 | 3 | 2 | 2,62 / 1,95 |
| #3 Soát hình học nhãn (ống, cọc, ắc quy) | 3 | 6 | 3 | 1,53 / 1,09 |
| #9 Train 960 / tile (vật nhỏ) | 4 | 4 | 5 | 1,55 / 1,42 |
| #8 Teacher backbone foundation model | 5 | 2 | 4 | 2,04 / 1,97 |
| #2 TTA / ensemble teacher | 6 | 5 (thí nghiệm đầu tiên) | 6 | 1,96 / 1,39 |
| #6 Copy-paste chéo xe | 7 | 7 | 7 | 1,40 / 1,62 |
| #4 Hộp cầu chì ↔ hộp lọc gió | 8 | 9 | 8 | 1,52 / 0,95 |
| #1 Ngưỡng theo class | 9 | 8 | 9 | 1,59 / 0,46 (rẻ nhất) |

**Đồng thuận:** đa dạng xe là nút thắt số 1. Cả bốn bên đều xếp #7 đứng đầu và #5 trong top 3.

**Bất đồng và cách tôi quyết:**
- **#1 ngưỡng theo class:** Jev xếp cao vì rẻ. Codex chỉ ra ngưỡng chỉ đổi điểm vận hành, **không cải thiện AP**. Thêm nữa, CV hiện tại chia theo ảnh, nên ngưỡng tính ra là cho xe đã biết. → Chỉ dùng khi deploy, và phải tính trên fold **chia theo xe**.
- **#3 soát hình học:** DeepSeek xếp thấp. Nhưng số đo ủng hộ Codex: 9–12% lỗi box lệch, 25% box bình chứa phải vẽ lại, và mAP50-95 phạt nặng box lệch ở ngưỡng IoU cao. Quy trình soát bằng subagent cũng đã có sẵn. → Giữ ở hạng 3.
- **#8 foundation model:** DeepSeek xếp 2, Codex cho là còn phỏng đoán. Probe DINOv2 (kNN trên crop chỉ đạt 0,60) là bằng chứng yếu, không đủ để kết luận. → Để sau khi đã có thêm dữ liệu, và so trên cùng dữ liệu.
- **#2 TTA:** DeepSeek muốn chạy đầu tiên, Codex cho là chỉ chữa triệu chứng. Số đo khi chia ô ảnh (recall +3,9, precision −8) cho thấy lợi ích có hạn. → Chỉ dùng như một phần của #5, để tạo pseudo-label.

**Góp ý về phương pháp (Codex và DeepSeek), tôi chấp nhận:**
1. **Tập test quá nhỏ và dồn cụm:** 3 xe, riêng 192/443 vật thuộc 1 xe. Chênh lệch ±1–2 điểm mAP không có sức thuyết phục. → Báo cáo **trung bình theo xe (vehicle-macro)** kèm khoảng tin cậy bootstrap, và thêm **CV nhóm theo xe** cho mục tiêu xe mới.
2. **Pseudo-label phải có gate precision:** duyệt 200 box, cần ≥ 95% đúng trước khi đưa vào train. Riêng AUROC 0,84 thì chưa đủ.
3. **Mọi so sánh giữ cố định:** cùng dữ liệu, khởi tạo, augmentation, cách lấy trung bình checkpoint, tối thiểu 2 seed.

## 3. Kế hoạch

| Bước | Việc | Gate | Chi phí |
|---|---|---|---|
| **M1 (đo lường, làm trước)** | `vehicle_split`/`build_full_dataset --group-by-vehicle`: CV 4 fold theo xe (khoảng 7 xe/fold); báo cáo vehicle-macro + bootstrap CI; ngưỡng deploy tính trên fold theo xe | — | 4 teacher × ~75 phút (chạy sau hàng đợi hiện tại) |
| **E1** (đang chờ DGX) | FGD theo kích thước vật | trung bình 2 seed ≥ 0,296 và recall vật < 32 px +≥ 10 điểm, recall vật ≥ 96 px không giảm quá 1 điểm | 2 × 50 phút |
| **D2** | Soát hình học nhãn + quy định giới hạn cho ống nạp, cọc bình, ắc quy (cùng quy trình như bình chứa) | tỉ lệ box phải vẽ lại; sau build: lỗi box lệch giảm | subagent, không cần GPU |
| **D1** | **Thu thập xe mới** (mục tiêu 60–100 xe) theo `docs/plans/huong_dan_chup_xe_moi.md`, gán nhãn trước bằng teacher (`teacher_prelabel.py`), mô hình tin cậy xếp thứ tự duyệt, chuyên gia duyệt | vehicle-macro mAP tăng theo số xe | cần bạn cung cấp ảnh |
| **D3** | Thí điểm self-training: ≥ 10 xe mới **không nhãn**, pseudo-label bằng teacher (+ tile/TTA, + SAM2), lọc theo độ tin cậy, **duyệt 200 box ≥ 95%** | 2 seed, vehicle-macro mAP +≥ 0,02, tốt hơn trên ≥ 2/3 xe test | nguồn ảnh không nhãn: crawler + nguồn ngoài |
| **E3** | Fine-tune trên tile 800 px (nếu E1 chưa đủ) | mAP +≥ 0,015 và recall vật < 32 px +≥ 10 | 2 lần chạy |
| **Sau cùng** | Teacher foundation model (#8), sau khi đã có D1/D3 | so trên cùng dữ liệu | cao |

**Không ưu tiên:** copy-paste chéo xe (#6; ngoại cảnh không thật), riêng cặp hộp cầu chì ↔ hộp lọc gió (#4; ảnh hưởng hẹp, sẽ được D1/D2 xử lý gián tiếp).
