# Đánh giá distillation và hướng tăng accuracy — 27/09/2026

Kết luận: pipeline đã có các thành phần KD hợp lý, nhưng chưa chứng minh được cải thiện box mAP so với student không KD. Ưu tiên sửa tính đúng đắn của resume, chuẩn hóa benchmark, rồi ablation loss. Không thay đổi training code hoặc chạy lại training trong lần đánh giá này.

## Bằng chứng thực nghiệm hiện có

Nguồn: `runs/**/results.csv`, `args.yaml`, metadata trong `weights/best.pt`. Các số dưới đây là validation đã lưu, không phải phép đánh giá test mới.

| Exterior parts, 21 lớp | Epoch có box mAP cao nhất | Box mAP50 | Box mAP50–95 |
|---|---:|---:|---:|
| YOLOv8n-seg thường, exterior_parts_auto | 40 | 87.578% | 65.104% |
| YOLOv8n-seg KD, kd_yolov8n_seg | 47 | 87.680% | 64.750% |
| YOLO11m-seg teacher | 39 | 90.545% | 69.680% |

KD thấp hơn baseline 0.354 điểm phần trăm ở box mAP50–95; cao hơn 0.102 điểm ở mAP50 tại các epoch trên. Không đủ bằng chứng thống kê để quy chênh lệch cho KD: mới một seed, KD có resume, workers khác nhau, warmup_bias_lr khác, và một epoch adapter warm-up làm lịch cập nhật student khác baseline.

Đối với file **best.pt thực sự được lưu**, metadata cho thấy:

| Checkpoint | Box mAP50–95 | Mask mAP50–95 |
|---|---:|---:|
| Baseline | 65.104% | 59.493% |
| KD | 64.648% | 58.860% |
| Teacher | 69.670% | 63.851% |

Stock segmentation fitness đang kết hợp box và mask mAP; do đó best.pt không nhất thiết là epoch có box mAP cao nhất. Nếu mục tiêu chính là detection, cần lưu riêng best_box.pt theo box mAP50–95.

Pipeline đang chạy là YOLO11m-seg → YOLOv8n-seg, không phải sơ đồ D-FINE → YOLO26 trong phần mở đầu README. `train_kd.py` cố định task segment; chưa phải trainer KD cho model detect thuần.

## Phát hiện về implementation

### 1. Resume không khôi phục adapter của model đang train — ưu tiên cao

`src/distillation/ultralytics_kd.py:431`: get_model() gọi model.load(weights) trước attach_kd(). Khi load, model chưa có kd_adapters/kd_fgd; loader của Ultralytics chỉ nạp các key giao nhau. Sau đó attach_kd() tạo lại các module ngẫu nhiên.

Đã tái hiện trên CPU qua chính get_model(): adapter checkpoint được gán hằng 0.123, adapter sau load có mean -0.0049275; tensor không bằng nhau. Ultralytics có khôi phục EMA riêng, nhưng điều đó không khôi phục adapter trong model đang nhận gradient. Optimizer state được resume nên có thể ghép momentum cũ với adapter mới. Run KD hiện có args.resume trỏ tới last.pt; chưa xác định mức đóng góp của lỗi này vào mAP đã ghi.

Sửa đề xuất: dựng các module KD đúng cấu hình trước khi nạp đầy đủ state_dict; kiểm tra key/shape; lưu KD config và nhận dạng teacher cùng checkpoint. Tách checkpoint dùng resume khỏi bản export đã strip. Test so sánh adapter, GcBlock, BN buffers và optimizer state trước/sau resume.

### 2. Warm-up mở lại cả tham số phải đóng băng

`src/distillation/ultralytics_kd.py:356`: callback đặt requires_grad=True cho mọi tham số ngoài KD sau warm-up, bỏ qua freeze policy của trainer, gồm DFL projection. Đã tái hiện DFL requires_grad từ False thành True.

Không khẳng định DFL đã bị học sai trong run cũ: checkpoint KD được kiểm tra vẫn có projection đúng [0,1,...,15]. Sửa callback để lưu và khôi phục trạng thái requires_grad ban đầu; nếu muốn student thực sự đứng yên khi warm-up thì phải quản lý cả BatchNorm running statistics, vì chỉ tắt gradient không đóng băng các buffer đó.

### 3. Kiểm tra tương thích teacher/student chưa đủ

Code chỉ kiểm tra số lớp và tổng số anchor trước logit KD. Exterior parts và engine_bay_train_v3 đều có 21 lớp nhưng ý nghĩa/index hoàn toàn khác nhau. Cần fail fast khi names/class order, task, stride từng level, grid hoặc thứ tự anchor không tương thích; số lớp và số anchor bằng nhau không đủ.

Đây là rủi ro khi đổi dataset, không phải bằng chứng run exterior hiện tại dùng nhầm teacher. Default teacher trong kd_hyperparams.yaml là exterior; không dùng nguyên default này cho khoang máy.

### 4. KD chưa phân biệt teacher đáng tin và teacher sai

`ultralytics_kd.py:281–301`: classification dùng mask trong GT box và background weight cố định 0.05; localization dùng các positive anchor do student assigner chọn, không truyền confidence/IoU weight mặc dù loss đã hỗ trợ weights.

Đề xuất thử: trọng số mềm kết hợp teacher score cho GT class và IoU teacher box với assigned GT; giảm/tắt localization KD tại điểm teacher kém hơn student (tính gate bằng tensor detach), giữ task loss theo GT. Thử positive và vùng lân cận được chọn có kiểm soát, không mở rộng sang mọi background anchor. Đây là biến thể đề xuất cần ablation, không phải cải tiến đã được chứng minh trên dữ liệu này.

### 5. FGD hiện tại là biến thể giản lược, có thể bất lợi với vật nhỏ

`feature_loss.py:170,278`: foreground là hợp các box nhị phân rồi chia theo tổng số pixel foreground. Box lớn đóng góp nhiều pixel hơn; không cân bằng theo từng instance. Việc lấy tâm cell cũng có thể bỏ lọt box rất nhỏ ở level thô.

FGD gốc dùng foreground weight theo nghịch đảo diện tích box và chuẩn hóa background theo từng ảnh. Nguồn: [implementation của tác giả](https://github.com/yzd-v/FGD/blob/master/mmdet/src/distillation/losses/fgd.py), [paper CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html).

Thử cân bằng instance/area, kiểm tra độ phủ vật nhỏ ở P3; ablation trọng số từng level. Không chép nguyên hệ số từ paper vì loss reduction hiện tại khác. CWD là ứng viên thay thế feature loss nếu FGD vẫn không giúp: chuẩn hóa phân phối không gian theo từng channel. Nguồn: [CWD, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Shu_Channel-Wise_Knowledge_Distillation_for_Dense_Prediction_ICCV_2021_paper.pdf).

### 6. Trọng số KD cố định và thiếu phép đo ảnh hưởng gradient

Config hiện tại alpha=beta=gamma=1, T_cls=2, T_LD=10. Ở epoch 47: kd_feat=1.56761, kd_cls=0.17988, kd_loc=0.34148, trong khi task box/seg/cls/dfl tương ứng 0.77887/1.16360/0.69120/1.02676. Feature KD là thành phần lớn, nhưng độ lớn loss không chứng minh gradient chi phối.

Đề xuất log gradient norm và cosine giữa task/KD ở một số batch; thử ramp-up KD 5–10 epoch và giảm KD ở cuối lịch. Không đổi cả ba trọng số cùng lúc. Log hiện tại xác nhận optimizer=auto chọn AdamW lr=0.0004 và bỏ qua lr0=0.01; muốn sweep learning rate phải chỉ định optimizer rõ ràng.

## Dữ liệu khoang máy là vấn đề riêng

Không so trực tiếp các mAP sau để suy ra tiến bộ do kiến trúc, vì khác dataset/version/resolution:

- baseline_engine_bay_clean: detect YOLO11m, 36 lớp, 640px; best box mAP50–95 7.466%.
- engine_teacher_v1: segment, train_v2, 1024px; best 33.067%, cuối 23.032%.
- engine_teacher_v2: segment, train_v3, 1024px; best 34.811% tại epoch 47, cuối 28.698% tại epoch 77. Train loss tiếp tục giảm trong khi validation xấu đi: dấu hiệu phù hợp với overfitting, cần đối chiếu chất lượng nhãn và domain shift.
- engine_teacher_v4: local CSV chỉ có 1 epoch tại thời điểm audit; chưa đánh giá được convergence. Chưa xác minh trạng thái tiến trình hoặc kết quả trên DGX.

Thống kê trực tiếp label files:

| Dataset | Train | Val | Test | Label rỗng trong train |
|---|---:|---:|---:|---:|
| Exterior | 748 | 124 | 126 | 0 |
| Engine train_v3 | 1043 | 80 | 125 | 356 |
| Engine train_v4 | 1033 | 80 | 125 | 343 |

Engine v3: oil_filter chỉ 4 instance train, 1 val; ecu_module có 43 train nhưng không có instance val. Engine v4 tăng oil_filter lên 9 train, val vẫn chỉ 1. Nhãn rỗng có thể là negative hợp lệ hoặc ảnh chưa được gán hết nhãn; chưa kiểm tra ảnh thủ công nên không kết luận tất cả đều sai.

Ưu tiên audit các ảnh rỗng, sửa missing labels/class confusion; bổ sung ảnh thật cho lớp hiếm và tập validation đại diện. Chỉ oversample sau khi kiểm tra nhãn. Giữ các góc chụp cùng xe/Request_ID, ảnh trùng và biến thể augmentation trong cùng split; audit hash/near-duplicate trước khi cố định benchmark. Lần kiểm tra này chưa chạy kiểm tra hash hoặc near-duplicate.

## Kế hoạch thí nghiệm đề xuất

Giai đoạn 1: sửa resume/freeze, pin môi trường hiện có (torch 2.11.0+cu128, ultralytics 8.4.75), lưu manifest/hash dataset, teacher checkpoint và cấu hình từng run. Xác minh teacher/student cùng ontology.

Giai đoạn 2: trên exterior hiện tại, cùng initialization/dataset/augmentation/optimizer/epochs/batch/imgsz; chạy liên tục để tránh nhiễu resume. Dùng seed 0 cho sàng lọc, sau đó xác nhận ứng viên tốt với seed 0/1/2:

| Run | Thay đổi so với baseline |
|---|---|
| A | Student không KD |
| B | KD hiện tại sau sửa correctness |
| C | Chỉ localization KD, có quality weighting |
| D | Chỉ feature KD, cân bằng diện tích instance |
| E | Chỉ binary class KD, có confidence weighting |
| F | Ghép các thành phần có lợi + lịch ramp-up/decay |

Sàng lọc hệ số quanh alpha_feature 0.1/0.25/0.5/1.0; gamma_loc 0.25/0.5/1.0; beta_cls 0.1/0.25/0.5/1.0 theo từng thành phần. T_cls 1/2/4 và T_LD 2/5/10 chỉ thử sau khi chọn loss có ích. Đây là search space đề xuất, không phải cấu hình đã tối ưu.

Ưu tiên localization có chọn lọc vì box mAP50–95 chưa tăng và teacher còn khoảng cách khoảng 4.9 điểm. LD hỗ trợ truyền phân phối localization, nhưng paper cũng nhấn mạnh lựa chọn vùng distillation có giá trị: [Localization Distillation, CVPR 2022](https://arxiv.org/abs/2102.12252). Không suy ra mức tăng trên repo từ mức tăng COCO trong paper.

Giai đoạn 3: nếu lỗi tập trung ở vật nhỏ, thử 640→832→1024 và cân nhắc P2 head sau khi có AP theo kích thước. Đo latency/VRAM ở phần cứng triển khai. Nếu student vẫn thiếu capacity, thử student cỡ s; nếu chỉ cần box, benchmark detector thuần và bổ sung DetectionTrainer KD tương ứng. Các thay đổi kiến trúc/resolution phải báo riêng với lợi ích của KD.

Chọn bằng box mAP50–95, AP75, AP theo lớp/kích thước và recall; báo mean±std trên các seed. Chỉ dùng validation để chọn hyperparameter, đánh giá test khóa một lần khi chốt. Không hứa mức tăng accuracy trước khi có kết quả.

## Kiểm chứng đã chạy

- `python -m pytest tests/test_ultralytics_kd.py tests/test_distillation_loss.py -q`: **13 passed**, 25.81s.
- CPU reproduction: get_model không giữ adapter checkpoint; callback warm-up mở requires_grad của DFL.
- Đọc metadata ba best.pt, đối chiếu CSV/args và đếm nhãn các dataset.
- Không train lại, không chạy inference test mới, không thay đổi checkpoint/config/source training. Unit tests hiện có xác nhận loss/gradient/strip hoạt động, nhưng chưa bao phủ lỗi resume/freeze nêu trên.
