# Rà soát mở rộng sáng chế sang toàn xe

## Phạm vi bản cập nhật

Bản vehicle-wide thay bản engine bay làm bản chính cho hướng mở rộng. Bản engine bay trước được giữ để đối chiếu và làm ví dụ thực hiện. Tài liệu mới giảm bảng class, công thức, hyperparameter và JSON chi tiết; giữ cơ chế xử lý và các ví dụ đại diện cần để hỗ trợ claims.

Phạm vi gồm ngoại thất, nội thất/khoang hàng, bánh/lốp/phanh nhìn thấy được, gầm/chassis, khoang động cơ hoặc hệ truyền động và điện/năng lượng. Các profile bao gồm ICE, hybrid, EV, xe con, xe thương mại, bus và xe máy. “Toàn xe” là phạm vi kiến trúc và khả năng biểu diễn; không có nghĩa một ảnh chứng minh mọi bộ phận hay chức năng ẩn đã được kiểm tra.

## Các bổ sung quan trọng

1. Region routing: xác định vùng ảnh và chọn mô hình phù hợp.
2. Phân biệt hợp nhất trong một ảnh với liên kết cùng linh kiện qua nhiều góc nhìn. Không dùng IoU của hai ảnh khác viewpoint như thể cùng hệ tọa độ.
3. Coverage: đã quan sát, quan sát một phần, chưa kiểm tra, không áp dụng; chỉ thông báo hoàn tất khi đạt coverage policy.
4. Expected-component reasoning theo cấu hình xe; EV không bị báo thiếu động cơ đốt trong. Generic chưa rõ danh tính được giữ như bằng chứng chưa giải quyết.
5. Targeted recapture: yêu cầu thêm đúng vùng/góc nhìn khi thiếu bằng chứng hoặc ảnh bị glare/blur.
6. Condition findings gắn với component hoặc surface, giữ mức nghi ngờ; không suy ra lỗi cơ khí ẩn từ hình dáng.
7. DTC/OBD, lịch sử và sensor là các nguồn bằng chứng riêng; mâu thuẫn được giữ để review.
8. Theo dõi thay đổi qua lần kiểm tra, ghi nhận thay linh kiện và provenance.
9. Curation và synthetic training mở theo region, vẫn giữ real-review hysteresis và geometry/uniqueness gates.
10. Tám figure vector tự vẽ, không ảnh thật, không ROC giả; có sơ đồ vùng xe, pipeline, hierarchy/graph, multi-view, coverage/recapture, feedback, condition/history, synthetic engine.

## Claims và mức độ chi tiết

30 claims là bộ đề xuất để rà soát, không phải số lượng tối ưu cho mọi cơ quan nộp. Claim 1 chuyển trọng tâm sang multi-scale evidence + region hierarchy + observation coverage; không bắt buộc engine bay, fixed 2x2, model vendor, class count, DTC, synthetic training hoặc 3D. Claims phụ tạo các vị trí thu hẹp theo từng cơ chế. Claim 23 là system, 24 là medium; Claim 25 là curation method với 28/29 là system/medium.

Mô tả high level vẫn phải giải thích cách cơ chế hoạt động và có ví dụ đại diện cho phạm vi. WIPO ISPE 5.45-5.58 yêu cầu claims được mô tả hỗ trợ và có thể thực hiện; không chỉ liệt kê mọi bộ phận xe rồi kết luận có bảo hộ toàn xe.

Nguồn: https://www.wipo.int/en/web/pct-system/texts/ispe/5_45_58

## Còn cần xác minh trước khi nộp

- Inventor xác nhận các embodiment vehicle-wide mới thuộc ý tưởng đã hình thành: multi-view association, region coverage, recapture, condition linkage, longitudinal comparison và profile các loại xe. Đây là nội dung mở rộng đề xuất, chưa có thử nghiệm tương ứng trong repo.
- Nếu đã nộp đơn: đối chiếu disclosure ban đầu và ngày ưu tiên; nội dung mới không mặc nhiên hưởng ngày ưu tiên cũ hoặc được thêm như amendment.
- Tra cứu patent/non-patent literature về vehicle inspection, damage assessment, multi-view, coverage-driven capture, condition history, curation và synthetic data. Bản này chưa kết luận tính mới/trình độ sáng tạo.
- Rà claim breadth/support/unity/eligibility theo nước nộp. Curation độc lập có thể cần phân nhóm đơn tùy đối chứng và kết luận thẩm định.
- Kết quả repo chỉ là engine bay: không chuyển AUROC/recall đó thành kết quả toàn xe. Không thêm số liệu ngoài bằng chứng đã có.
- Chưa bao phủ đo chức năng ẩn chỉ bằng ảnh: battery health, electrical isolation, internal brake wear và failure diagnosis cần bằng chứng hoặc đo riêng.

## Mức ưu tiên tra cứu đối chứng

| Nhóm | Cần so sánh cụ thể |
| --- | --- |
| Vehicle inspection/damage assessment | Taxonomy vùng xe + visual finding + coverage/uncertainty |
| Multi-view recognition | Semantic identity theo side/region + evidence association, tránh merge sai |
| Guided capture | Coverage deficit -> requested viewpoint -> updated evidence |
| Longitudinal inspection | Corresponding part/surface + capture differences + replacement provenance |
| Label review/active learning | Verdict reference margin + class/localization/uniqueness gates + real support hysteresis |
| Synthetic data | Region-linked asset labels + rare-class sampling + exclusion from real-review promotion |
