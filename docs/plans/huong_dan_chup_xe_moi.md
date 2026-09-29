# Hướng dẫn chụp ảnh khoang máy cho xe mới

Tài liệu dành cho kỹ thuật viên thu thập ảnh. Mục tiêu là có thêm **40–60 xe mới**, giúp model nhận diện chính xác linh kiện trên xe chưa từng thấy.

## Nguyên tắc quan trọng nhất

- **Nhiều xe khác nhau quan trọng hơn nhiều ảnh của một xe.** Một xe mới có giá trị hơn 50 ảnh thêm của xe đã chụp.
- Mỗi xe chụp khoảng **30 ảnh** là đủ.
- Mỗi xe là **một Request_ID riêng**, một thư mục riêng. Không trộn ảnh của hai xe vào cùng một thư mục.
- Chỉ chụp **khoang máy và linh kiện trong khoang máy**. Không chụp ngoại thất, nội thất, biển số hay giấy tờ xe.

## Danh sách ảnh cho mỗi xe (khoảng 30 ảnh)

| Nhóm | Số ảnh | Cách chụp |
|---|---|---|
| Toàn cảnh từ trên xuống | 4–5 | Mở nắp capo hết cỡ, đứng trước đầu xe, chụp thấy trọn khoang máy |
| Toàn cảnh chéo trái | 3–4 | Đứng ở góc trước bên trái, chụp chéo vào khoang máy |
| Toàn cảnh chéo phải | 3–4 | Đứng ở góc trước bên phải, chụp chéo vào khoang máy |
| Cận cảnh linh kiện | 15–18 | Mỗi linh kiện trong danh sách dưới đây 1–2 ảnh, linh kiện chiếm khoảng 1/3 khung hình |
| Điều kiện ánh sáng khác | 3–4 | Chụp lại vài góc toàn cảnh khi có bóng râm, ánh nắng gắt, hoặc bật đèn flash |

## Linh kiện cần chụp cận

Chụp những linh kiện nhìn thấy được trên xe. Không cần tháo nắp che để chụp.

- Ắc quy và cọc bình (cả cọc dương và cọc âm)
- Hộp cầu chì, rơ-le
- Bình nước làm mát và nắp két nước
- Bình dầu phanh
- Bình nước rửa kính
- Nắp che động cơ, cổ hút
- Nắp châm dầu, que thăm dầu
- Hộp lọc gió, ống hút gió, cảm biến lưu lượng khí (MAF), cổ họng ga
- Máy phát điện, bô-bin đánh lửa
- Ống két nước
- Hộp ECU (nếu nhìn thấy)

Nếu dùng đồng hồ đo hoặc thiết bị chẩn đoán Innova trong lúc kiểm tra, hãy chụp thêm 1–2 ảnh có thiết bị trong khoang máy.

## Yêu cầu kỹ thuật

- Ảnh rõ nét, không rung. Chụp lại nếu ảnh bị mờ.
- Không để ngón tay, dây đeo hay vật khác che ống kính.
- Giữ nguyên độ phân giải gốc của máy ảnh, không nén hay cắt ảnh.
- Hạn chế để người đứng chắn trong khung hình.

## Đa dạng xe

Khi chọn xe để chụp, ưu tiên những xe khác với dữ liệu hiện có:

- Nhiều hãng khác nhau, như Toyota, Honda, Hyundai, Kia, Mazda, Ford, Mitsubishi, VinFast, Mercedes, BMW.
- Nhiều đời xe, cả xe cũ và xe mới.
- Nhiều loại động cơ: xăng, dầu, turbo, hybrid.
- Cả khoang máy sạch và khoang máy bẩn.

## Cách gửi dữ liệu

1. Mỗi xe một thư mục, đặt tên theo Request_ID nội bộ, ví dụ `Request_ID_61`.
2. Ghi chú kèm theo mỗi xe: hãng, dòng xe, năm sản xuất nếu biết.
3. Chép thư mục vào `dataset/` hoặc gửi cho nhóm dữ liệu.

Khi có đủ xe mới, nhóm dữ liệu sẽ giữ riêng ít nhất 6 xe làm tập kiểm tra và 6 xe làm tập kiểm định. Những xe này không bao giờ được dùng để train, nhờ vậy số đo độ chính xác trên xe mới là đáng tin cậy.
