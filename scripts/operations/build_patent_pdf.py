import os
import subprocess
import base64

patents_dir = r"e:\Research\GEN AI\AI Vision\Distillation\docs\patents"
figures_dir = os.path.join(patents_dir, "figures")
html_path = os.path.join(patents_dir, "patent_specification.html")
pdf_path = os.path.join(patents_dir, "patent_specification_engine_bay_inspection.pdf")

# Convert images to base64 data URIs so the HTML is 100% self-contained
def get_base64_img(filename):
    path = os.path.join(figures_dir, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return ""

img1 = get_base64_img("fig1_system_architecture.png")
img2 = get_base64_img("fig2_taxonomy_hysteresis.png")
img3 = get_base64_img("fig3_multiscale_inference.png")
img4 = get_base64_img("fig4_confidence_calibration_roc.png")
img5 = get_base64_img("fig5_functional_decomposition.png")
img6 = get_base64_img("fig6_domain_randomization.png")

html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Bản Mô Tả Đơn Đăng Ký Sáng Chế</title>
    <style>
        @page {{
            size: A4;
            margin: 20mm 20mm 20mm 25mm;
            @bottom-right {{
                content: counter(page);
                font-family: 'Times New Roman', Times, serif;
                font-size: 10pt;
            }}
        }}
        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 11pt;
            line-height: 1.4;
            color: #111;
            text-align: justify;
            margin: 0;
            padding: 0;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #111;
            padding-bottom: 12px;
            margin-bottom: 24px;
        }}
        .country-title {{
            font-size: 12pt;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .sub-header {{
            font-size: 10.5pt;
            font-style: italic;
            margin-top: 4px;
            color: #333;
        }}
        .doc-title {{
            font-size: 15pt;
            font-weight: bold;
            text-transform: uppercase;
            margin-top: 16px;
            line-height: 1.3;
        }}
        h2 {{
            font-size: 12pt;
            font-weight: bold;
            text-transform: uppercase;
            border-bottom: 1px solid #444;
            padding-bottom: 4px;
            margin-top: 24px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 11pt;
            font-weight: bold;
            margin-top: 14px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        p {{
            margin-top: 0;
            margin-bottom: 8px;
            text-indent: 20px;
        }}
        .no-indent {{
            text-indent: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0 16px 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }}
        table, th, td {{
            border: 1px solid #333;
        }}
        th {{
            background-color: #f2f2f2;
            padding: 6px 8px;
            font-weight: bold;
            text-align: center;
        }}
        td {{
            padding: 6px 8px;
            vertical-align: top;
        }}
        .fig-box {{
            text-align: center;
            margin: 18px 0;
            page-break-inside: avoid;
        }}
        .fig-box img {{
            max-width: 95%;
            height: auto;
            border: 1px solid #cbd5e1;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .fig-caption {{
            font-size: 10pt;
            font-weight: bold;
            margin-top: 6px;
            text-align: center;
        }}
        .claim-item {{
            margin-bottom: 12px;
            padding-left: 24px;
            text-indent: -24px;
        }}
        .formula {{
            text-align: center;
            font-style: italic;
            font-family: 'Cambria Math', 'Times New Roman', serif;
            margin: 10px 0;
            background: #fafafa;
            padding: 6px;
            border: 1px dashed #ccc;
        }}
        .page-break {{
            page-break-before: always;
        }}
    </style>
</head>
<body>

    <div class="header">
        <div class="country-title">CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM</div>
        <div class="country-title">CỤC SỞ HỮU TRÍ TUỆ</div>
        <div class="sub-header">Mẫu văn bản theo chuẩn Đơn đăng ký Sáng chế / Giải pháp hữu ích & Hiệp ước PCT</div>
        <div class="doc-title">BẢN MÔ TẢ SÁNG CHẾ</div>
    </div>

    <h2>1. TÊN SÁNG CHẾ</h2>
    <p class="no-indent"><strong>Tiếng Việt:</strong> HỆ THỐNG VÀ PHƯƠNG PHÁP NHẬN DIỆN, PHÂN RÃ CHỨC NĂNG PHÂN CẤP VÀ HIỆU CHUẨN ĐỘ TIN CẬY CÁC LINH KIỆN KHOANG ĐỘNG CƠ PHƯƠNG TIỆN SỬ DỤNG THỊ GIÁC MÁY TÍNH ĐA TỶ LỆ</p>
    <p class="no-indent"><strong>Tiếng Anh:</strong> METHOD AND SYSTEM FOR HIERARCHICAL FUNCTIONAL DECOMPOSITION AND MULTI-SCALE VISUAL INSPECTION WITH CONFIDENCE CALIBRATION FOR VEHICLE ENGINE BAY COMPONENTS</p>

    <h2>2. LĨNH VỰC KỸ THUẬT ĐƯỢC ĐỀ CẬP</h2>
    <p>Sáng chế đề cập đến lĩnh vực thị giác máy tính (computer vision), học máy (machine learning) và hệ thống chẩn đoán kỹ thuật ô tô tự động. Cụ thể hơn, sáng chế đề cập đến một phương pháp và hệ thống tích hợp cho phép nhận diện, phân đoạn (instance segmentation) mọi linh kiện cơ khí, điện tử nhìn thấy được trong khoang động cơ phương tiện giao thông theo các hệ thống chức năng kỹ thuật ô tô, kết hợp cơ chế suy luận đa tỷ lệ để tối ưu hóa độ thu hồi (recall) cho linh kiện kích thước nhỏ, và phương pháp hiệu chuẩn độ tin cậy nhãn phục vụ quy trình gán nhãn tự động hóa cao, cùng quy trình sinh dữ liệu huấn luyện tổng hợp bằng ngẫu nhiên hóa miền (domain randomization) trong môi trường ba chiều (3D).</p>

    <h2>3. TÌNH TRẠNG KỸ THUẬT CỦA SÁNG CHẾ (PRIOR ART)</h2>
    <p>Trong công tác kiểm định, bảo dưỡng và chẩn đoán lỗi phương tiện giao thông (sau bán hàng - Aftermarket), việc nhận diện chính xác các bộ phận trong khoang động cơ đóng vai trò tiên quyết. Tuy nhiên, các giải pháp kỹ thuật hiện nay tồn tại những nhược điểm lớn sau:</p>
    <p><strong>Thứ nhất,</strong> các hệ thống thị giác máy tính công nghiệp tại dây chuyền lắp ráp (OEM Assembly Vision) như của Bosch hay BMW chỉ hoạt động trong môi trường được kiểm soát tuyệt đối (buồng chiếu sáng đồng nhất, camera cố định một góc) và chỉ áp dụng cho một model động cơ duy nhất. Chúng hoàn toàn mất tác dụng đối với ảnh chụp tự nhiên bằng thiết bị di động trong điều kiện ánh sáng môi trường phức tạp và trên các dòng xe thương mại đa dạng ngoài thị trường.</p>
    <p><strong>Thứ hai,</strong> các giải pháp quét kiểm định xe thương mại tự động hiện nay (như UVeye, Ravin AI) chỉ tập trung vào ngoại thất, sơn và gầm xe. Khoang máy bị bỏ qua do có mức độ che khuất phức tạp và kích thước linh kiện chênh lệch quá lớn từ vài milimet (giắc cắm điện, van cảm biến) đến hàng chục centimet (nắp che động cơ, bình ắc quy, bầu lọc gió).</p>
    <p><strong>Thứ ba,</strong> các mô hình nhận diện đối tượng tiêu chuẩn (như YOLO, Faster R-CNN) khi áp dụng vào ảnh toàn cảnh khoang máy ở kích thước tiêu chuẩn (ví dụ 640x640 px) bị hiện tượng nén suy giảm độ phân giải, dẫn đến tỷ lệ bỏ sót linh kiện nhỏ (recall) thường dưới 40%. Đồng thời, điểm tin cậy (confidence score) nguyên bản của mô hình thường bị sai lệch (miscalibrated), không thể tự động hóa việc gán nhãn dữ liệu mà không gây nhiễu tập huấn luyện.</p>
    <p><strong>Thứ tư,</strong> việc huấn luyện mạng nơ-ron sâu nhận diện hàng chục loại linh kiện chi tiết trên hàng nghìn biến thể xe (Năm - Hãng - Dòng xe - Động cơ, YMME) đòi hỏi tập dữ liệu gán nhãn ở mức điểm ảnh rất lớn; việc gán nhãn thủ công tốn kém, dễ sai sót, và các linh kiện hiếm hoặc kích thước nhỏ vẫn thiếu mẫu. Ảnh tổng hợp kết xuất từ mô hình CAD tĩnh không chuyển giao tốt sang ảnh thật do khoảng cách mô phỏng - thực tế (reality gap: khác biệt về ánh sáng, bóng đổ, kết cấu bề mặt, bụi bẩn và méo ống kính). Kỹ thuật ngẫu nhiên hóa miền đã được đề xuất cho bài toán định vị vật thể của robot trong cảnh đơn giản (Tobin và cộng sự, IROS 2017), nhưng chưa được áp dụng cho bài toán phân đoạn thực thể đa lớp, chi tiết cao đối với linh kiện khoang động cơ bị che khuất nhiều, kết hợp với dữ liệu thật đã được chuyên gia thẩm định.</p>

    <h2>4. BẢN CHẤT KỸ THUẬT CỦA SÁNG CHẾ</h2>
    <p>Sáng chế giải quyết triệt để các nhược điểm trên thông qua 5 giải pháp kỹ thuật cốt lõi:</p>
    <p><strong>(1) Cơ chế phân tầng Taxonomy 3 tầng kết hợp Lớp dự phòng và Thăng hạng trễ:</strong> Khoang máy được phân chia thành 14 hệ thống kỹ thuật chức năng (Làm mát, Bôi trơn, Đánh lửa, Nạp khí, Nhiên liệu & EVAP, Điện & Khởi động, Khí xả, Phanh, v.v.), nhóm thành các cấu trúc hình học và 63 linh kiện chi tiết. Hệ thống bổ sung 6 lớp bao đóng hình dạng chung (Tier C - Generic Fallbacks: <em>other_reservoir, other_sensor_actuator, other_hose_line, other_cap_plug, other_module_box, other_pulley_device</em>) để bắt mọi vật thể lạ nhìn thấy được. Cơ chế thăng hạng có độ trễ đảm bảo một linh kiện chỉ được nâng lên thành lớp huấn luyện riêng khi đạt tối thiểu &ge; 60 mẫu từ &ge; 8 xe khác nhau, và chỉ hạ cấp khi mẫu giảm dưới 40.</p>
    <p><strong>(2) Pipeline suy luận đa tỷ lệ kết hợp hậu xử lý ràng buộc không gian:</strong> Kết hợp luồng xử lý song song giữa ảnh toàn cảnh và lưới 4 ô con (2x2 Tiling) có độ chồng lấn 10% - 20%. Tọa độ phát hiện được chiếu về hệ quy chiếu gốc và áp dụng thuật toán gộp không phân biệt lớp (Class-Agnostic NMS) với ngưỡng IoU &ge; 0.85 kết hợp giới hạn số lượng tối đa (Count Cap) và phân vị không gian [p10, p90] của từng loại linh kiện.</p>
    <p><strong>(3) Mô hình hiệu chuẩn độ tin cậy nhãn phục vụ Active Learning:</strong> Huấn luyện mô hình Gradient Boosting trên phán quyết đúng/sai từ chuyên gia ô tô kết hợp véc-tơ đặc trưng gồm điểm VLM, biên độ tương đồng embedding DINOv2 của crop, mật độ vị trí không gian và tỷ lệ hình học. Mô hình đạt AUROC 0.844 (so với 0.617 của VLM thông thường), cho phép tự động duyệt 21.8% số lượng mẫu với độ chính xác đạt 94.1%.</p>
    <p><strong>(4) Phân rã theo cụm chức năng và suy luận linh kiện bị che khuất:</strong> Hệ thống đối chiếu dữ liệu nhận diện với cấu trúc động cơ (theo YMME) để trả về cây danh mục 14 hệ thống chức năng kèm danh sách linh kiện bị che khuất (<em>expected_not_visible</em>) hỗ trợ quy trình sửa chữa theo mã lỗi DTC, mà không cần kết xuất 3D hay xử lý đám mây điểm theo thời gian thực trên thiết bị người dùng.</p>
    <p><strong>(5) Bộ sinh dữ liệu tổng hợp bằng ngẫu nhiên hóa miền 3D (giai đoạn huấn luyện):</strong> Kết xuất mô hình CAD/lưới 3D của linh kiện khoang máy với tư thế camera, ánh sáng, vật liệu bề mặt, bụi bẩn và vật che khuất được ngẫu nhiên hóa; tự động sinh mặt nạ thực thể chính xác đến từng điểm ảnh, khung bao và nhãn lớp; kết hợp ảnh tổng hợp với ảnh thật đã được chuyên gia thẩm định (huấn luyện sơ bộ trên ảnh tổng hợp, sau đó tinh chỉnh trên ảnh thật) để bổ sung mẫu cho các lớp linh kiện hiếm và nhỏ. Toàn bộ quá trình kết xuất 3D chỉ diễn ra ở giai đoạn huấn luyện ngoại tuyến.</p>

    <div class="page-break"></div>

    <h2>5. MÔ TẢ VẮN TẮT CÁC HÌNH VẼ KỸ THUẬT</h2>
    <p class="no-indent">Kèm theo bản mô tả này là các hình vẽ kỹ thuật thể hiện các khía cạnh ưu tiên của sáng chế:</p>
    
    <div class="fig-box">
        <img src="{img1}" alt="Hình 1">
        <div class="fig-caption">HÌNH 1: Sơ đồ khối tổng thể hệ thống nhận diện và chẩn đoán khoang động cơ (100)</div>
    </div>
    <p>Hình 1 thể hiện kiến trúc hệ thống gồm Thiết bị thu nhận hình ảnh (102), Bộ tiền xử lý đa tỷ lệ (104), Mạng nơ-ron phân đoạn (106), Cơ sở dữ liệu ràng buộc không gian (108), Bộ xử lý hậu kiểm gộp Class-Agnostic NMS (110), Bộ suy luận chẩn đoán (112), và Giao diện trực quan hóa (114).</p>

    <div class="fig-box">
        <img src="{img2}" alt="Hình 2">
        <div class="fig-caption">HÌNH 2: Cơ chế phân tầng Taxonomy và thăng hạng có độ trễ (200)</div>
    </div>
    <p>Hình 2 mô tả cây dữ liệu 3 tầng: 14 Hệ thống kỹ thuật chức năng (202), Nhóm hình thái hình học và 6 lớp bao đóng chung Tier C (204), các lớp huấn luyện Tier A (206), lớp chờ Tier B (208) cùng vòng lặp thăng hạng trễ (ngưỡng 60 instance/8 xe so với sàn 40 instance).</p>

    <div class="page-break"></div>

    <div class="fig-box">
        <img src="{img3}" alt="Hình 3">
        <div class="fig-caption">HÌNH 3: Quy trình suy luận đa tỷ lệ và hậu xử lý ràng buộc không gian (300)</div>
    </div>
    <p>Hình 3 minh họa luồng xử lý: Nhánh toàn cảnh (302) bắt vật thể lớn, Nhánh lưới chia ô 2x2 có độ chồng lấn (304) bắt chi tiết nhỏ, kết quả gộp Class-Agnostic NMS và Count Cap (306) cùng thuật toán hậu xử lý kỹ thuật (308) giúp triệt tiêu dương tính giả và tăng recall.</p>

    <div class="fig-box">
        <img src="{img4}" alt="Hình 4">
        <div class="fig-caption">HÌNH 4: Mô hình hiệu chuẩn độ tin cậy nhãn tự động W1e (400)</div>
    </div>
    <p>Hình 4 thể hiện quy trình trích xuất 5 đặc trưng kết hợp mô hình GBDT (402, 404) và đồ thị so sánh đường cong ROC thực nghiệm (406) chứng minh AUROC tăng vượt bậc từ 0.617 lên 0.844, đạt độ chuẩn xác 94.1% tại vùng điểm cao.</p>

    <div class="page-break"></div>

    <div class="fig-box">
        <img src="{img5}" alt="Hình 5">
        <div class="fig-caption">HÌNH 5: Sơ đồ đầu ra phân rã theo 14 hệ thống và thành phần bị che khuất (500)</div>
    </div>
    <p>Hình 5 mô tả cấu trúc dữ liệu đầu ra: Cây linh kiện nhìn thấy phân nhóm theo hệ thống chức năng (502), danh sách linh kiện bị che khuất theo động cơ (504), và cơ chế tích hợp chỉ dẫn chẩn đoán mã lỗi DTC (506).</p>

    <div class="page-break"></div>

    <div class="fig-box">
        <img src="{img6}" alt="Hình 6">
        <div class="fig-caption">HÌNH 6: Bộ sinh dữ liệu tổng hợp bằng ngẫu nhiên hóa miền 3D (600)</div>
    </div>
    <p>Hình 6 mô tả bộ sinh dữ liệu ngoại tuyến gồm: Thư viện mô hình 3D (602), Bộ ngẫu nhiên hóa miền (604) cho camera, ánh sáng, vật liệu và vật che khuất, Bộ kết xuất (606), Bộ sinh nhãn tự động (608), Bộ trộn tập huấn luyện (610) và khối Huấn luyện hai giai đoạn (612); đầu ra là Mạng nơ-ron phân đoạn (106) được triển khai để suy luận chỉ trên ảnh 2D.</p>

    <h2>6. MÔ TẢ CHI TIẾT SÁNG CHẾ VÀ KẾT QUẢ THỰC NGHIỆM</h2>
    <p>Cơ chế thăng hạng có độ trễ được vận hành theo công thức toán học:</p>
    <div class="formula">
        T(c_i) = Tier A &nbsp; khi &nbsp; N(c_i) &ge; 60 &nbsp; và &nbsp; V(c_i) &ge; 8;<br>
        T(c_i) = Tier C &nbsp; khi &nbsp; T_cũ(c_i) = Tier A &nbsp; và &nbsp; N(c_i) &lt; 40
    </div>

    <p>Bảng 1 dưới đây tổng hợp kết quả thực nghiệm kiểm chứng quy trình suy luận đa tỷ lệ của sáng chế trên 125 ảnh kiểm định khoang máy độc lập (mô hình nền tảng YOLO11l-seg tại kích thước 640 px):</p>
    
    <table>
        <thead>
            <tr>
                <th>Chế độ suy luận</th>
                <th>Recall</th>
                <th>Precision</th>
                <th>Số dự đoán</th>
                <th>Hiệu quả kỹ thuật</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Ảnh toàn cảnh đơn lẻ (Full image)</td>
                <td>0.582</td>
                <td>0.632</td>
                <td>408</td>
                <td>Baseline ban đầu</td>
            </tr>
            <tr>
                <td>Toàn cảnh + Lưới chia ô 2x2</td>
                <td>0.621</td>
                <td>0.552</td>
                <td>498</td>
                <td>Tăng mạnh độ bắt vật nhỏ</td>
            </tr>
            <tr>
                <td>Lưới 2x2 + Gộp Class-Agnostic NMS</td>
                <td>0.616</td>
                <td>0.575</td>
                <td>475</td>
                <td>Khử trùng lặp đa tỷ lệ</td>
            </tr>
            <tr>
                <td><strong>Phương pháp Sáng chế (Tiles + NMS + Count Cap)</strong></td>
                <td><strong>0.614</strong></td>
                <td><strong>0.581</strong></td>
                <td>468</td>
                <td><strong>Recall tăng +3.2%, khử false positive</strong></td>
            </tr>
        </tbody>
    </table>

    <p>Đặc biệt, độ thu hồi (Recall) theo từng hệ thống chức năng ghi nhận sự tăng trưởng vượt bậc: Hệ Bôi trơn tăng từ 0.76 lên 0.83 (+7%), Hệ Làm mát tăng từ 0.56 lên 0.62 (+6%), Hệ Nạp khí tăng từ 0.55 lên 0.59 (+4%), Hệ Điện & Khởi động tăng từ 0.55 lên 0.58 (+3%).</p>

    <h3>6.1 Bộ sinh dữ liệu tổng hợp bằng ngẫu nhiên hóa miền 3D (600)</h3>
    <p>Trong một phương án thực hiện tùy chọn ở giai đoạn huấn luyện (Hình 6), dữ liệu huấn luyện của mạng nơ-ron phân đoạn (106) được bổ sung bởi bộ sinh dữ liệu tổng hợp (600). Thư viện mô hình 3D (602) lưu trữ mô hình CAD hoặc lưới 3D của các linh kiện khoang máy, mỗi mô hình gắn với một lớp linh kiện trong cây phân tầng của Hình 2. Trong một phương án, các mô hình được lắp ghép thành cảnh khoang máy hoàn chỉnh theo cấu hình YMME; trong phương án khác, từng mô hình linh kiện riêng lẻ (ví dụ ắc quy, bình nước làm mát phụ, bô-bin đánh lửa, hộp ECU, hộp lọc gió) được kết xuất như các thực thể độc lập.</p>
    <p>Với mỗi khung hình tổng hợp, bộ ngẫu nhiên hóa miền (604) lấy mẫu tham số từ phân phối đều hoặc phân phối Gauss: (i) vị trí camera (x, y, z), góc quay (roll, pitch, yaw) và trường nhìn (FOV) trong phạm vi vật lý hợp lý phía trên khoang máy đang mở; (ii) từ một đến N nguồn sáng với vị trí, cường độ, nhiệt độ màu và độ mềm bóng đổ ngẫu nhiên, mô phỏng nắng ngoài trời, đèn huỳnh quang xưởng và đèn pin cầm tay; (iii) tham số vật liệu bề mặt (độ phản xạ, tính kim loại, độ nhám) kèm lớp phủ thủ tục mô phỏng gỉ sét, vết dầu, bụi và cặn bẩn; và (iv) các vật che khuất không phải mục tiêu như dây điện, ống dẫn, dụng cụ và bàn tay kỹ thuật viên.</p>
    <p>Bộ kết xuất (606) xuất ra cho mỗi khung hình một ảnh màu 2D cùng ảnh định danh thực thể (instance-ID) và tùy chọn ảnh độ sâu. Trong phương án ghép ảnh, các thực thể linh kiện đã kết xuất được ghép theo kênh alpha lên ảnh chụp khoang máy thật để nền ảnh giữ được đặc điểm thực tế. Bộ sinh nhãn tự động (608) suy ra từ ảnh định danh thực thể một mặt nạ thực thể chính xác đến từng điểm ảnh, một khung bao sát và một nhãn lớp cho mỗi thực thể nhìn thấy được, đồng thời loại bỏ các thực thể có tỷ lệ phần nhìn thấy (diện tích mặt nạ nhìn thấy chia cho diện tích mặt nạ khi không bị che) thấp hơn ngưỡng xác định trước, ví dụ 10%.</p>
    <p>Bộ trộn tập huấn luyện (610) kết hợp ảnh tổng hợp với ảnh thật mang nhãn đã được chuyên gia thẩm định. Số lượng thực thể tổng hợp sinh ra cho mỗi lớp giảm dần theo số mẫu thật đã thẩm định của lớp đó, nhờ vậy việc sinh dữ liệu tập trung vào các linh kiện Tier B, linh kiện hiếm và linh kiện nhỏ. Thực thể tổng hợp không bao giờ được tính vào ngưỡng số mẫu thẩm định của quy tắc thăng hạng có độ trễ. Khối huấn luyện hai giai đoạn (612) huấn luyện sơ bộ mạng trên dữ liệu tổng hợp hoặc dữ liệu trộn, sau đó tinh chỉnh trên ảnh thật đã thẩm định. Đóng góp của dữ liệu tổng hợp được đánh giá bằng cách so sánh, trên cùng một tập xe kiểm thử thật không xuất hiện trong huấn luyện, mạng đã tinh chỉnh với mạng chỉ huấn luyện trên ảnh thật.</p>
    <p>Do toàn bộ quá trình kết xuất 3D chỉ diễn ra ở giai đoạn huấn luyện ngoại tuyến, quy trình kiểm tra trực tuyến ở Hình 1 và Hình 3 chỉ xử lý ảnh 2D và không phát sinh chi phí kết xuất hay xử lý đám mây điểm trên thiết bị người dùng.</p>

    <div class="page-break"></div>

    <h2>7. YÊU CẦU BẢO HỘ (PATENT CLAIMS)</h2>
    
    <div class="claim-item">
        <strong>1.</strong> Một phương pháp xử lý dữ liệu và nhận diện thị giác máy tính đối với khoang động cơ phương tiện giao thông, phương pháp bao gồm các bước:<br>
        (a) Tiếp nhận một ảnh chụp hai chiều (2D) của khoang động cơ từ thiết bị thu nhận hình ảnh;<br>
        (b) Thực hiện suy luận đa tỷ lệ song song trên ảnh chụp thông qua: một nhánh toàn cảnh xử lý toàn bộ ảnh để trích xuất các đề xuất vùng và mặt nạ phân đoạn cho các linh kiện có kích thước lớn; và một nhánh phân mảnh tự động chia ảnh chụp thành lưới gồm nhiều ô con có độ chồng lấn xác định trước và phóng đại từng ô con để trích xuất các đề xuất vùng cho các linh kiện có kích thước nhỏ;<br>
        (c) Chiếu tọa độ của các đề xuất vùng từ các ô con về hệ quy chiếu không gian chung của ảnh toàn cảnh;<br>
        (d) Gộp các đề xuất vùng thông qua thuật toán triệt tiêu không cực đại phi phân loại (Class-Agnostic Non-Maximum Suppression) dựa trên ngưỡng diện tích giao nhau trên diện tích hợp (IoU) xác định trước để triệt tiêu hiện tượng phát hiện trùng lặp của cùng một vật thể qua các tỷ lệ khác nhau;<br>
        (e) Áp dụng bộ lọc ràng buộc vị trí không gian xác suất và ngưỡng giới hạn số lượng tối đa trên mỗi phương tiện cho từng loại linh kiện để loại bỏ các dương tính giả;<br>
        (f) Phân nhóm các linh kiện đã được nhận diện thành một cây phân cấp bao gồm danh mục 14 hệ thống kỹ thuật chức năng của phương tiện; và<br>
        (g) Xuất cấu trúc dữ liệu biểu diễn các linh kiện nhìn thấy kèm theo danh sách các linh kiện bị che khuất dựa trên việc đối chiếu loại động cơ của phương tiện với cơ sở tri thức kỹ thuật.
    </div>

    <div class="claim-item">
        <strong>2.</strong> Phương pháp theo Điểm 1, trong đó tại Bước (f), hệ thống phân loại linh kiện tuân theo một cơ chế phân tầng dữ liệu 3 cấp gồm: Tầng hệ thống chức năng, Tầng nhóm hình thái hình học, và Tầng linh kiện chi tiết; trong đó các linh kiện chi tiết chưa đủ số lượng mẫu được định danh theo các lớp bao đóng hình dạng dự phòng (Generic Fallback Classes) bao gồm: nhóm bình chứa chung, nhóm van và cảm biến chung, nhóm đường ống chung, nhóm nắp đậy chung, và nhóm hộp điều khiển chung.
    </div>

    <div class="claim-item">
        <strong>3.</strong> Phương pháp theo Điểm 2, trong đó các linh kiện chi tiết được quản lý bằng cơ chế thăng hạng có độ trễ (Promotion Hysteresis), theo đó một linh kiện chi tiết chỉ được nâng cấp thành một lớp phân loại độc lập khi đạt số lượng mẫu thẩm định tối thiểu từ một số lượng phương tiện phân biệt xác định trước; và chỉ bị giáng cấp về lớp bao đóng hình dạng dự phòng khi số lượng mẫu giảm xuống dưới một ngưỡng sàn thấp hơn ngưỡng nâng cấp.
    </div>

    <div class="claim-item">
        <strong>4.</strong> Phương pháp theo Điểm 1, trong đó ngưỡng diện tích giao nhau trên diện tích hợp (IoU) của thuật toán triệt tiêu không cực đại phi phân loại tại Bước (d) được thiết lập bằng hoặc lớn hơn 0.85.
    </div>

    <div class="claim-item">
        <strong>5.</strong> Phương pháp theo Điểm 1, trong đó bộ lọc ràng buộc vị trí không gian tại Bước (e) sử dụng phân bố xác suất tọa độ từ phân vị thứ 10 (p10) đến phân vị thứ 90 (p90) được xây dựng từ tập dữ liệu tham chiếu của từng loại linh kiện cụ thể.
    </div>

    <div class="claim-item">
        <strong>6.</strong> Phương pháp theo Điểm 1, trong đó phương pháp bao gồm thêm quy trình hiệu chuẩn độ tin cậy nhãn tự động phục vụ học chủ động (Active Learning), bao gồm:<br>
        • Trích xuất véc-tơ đặc trưng đa chiều từ mỗi đề xuất vùng, bao gồm điểm tin cậy thị giác ban đầu, biên độ tương đồng véc-tơ đặc trưng hình ảnh trích xuất từ mô hình thị giác nền tảng (crop margin similarity), mật độ phân bố vị trí không gian, và tỷ lệ hình học;<br>
        • Ước tính xác suất chính xác của đề xuất vùng thông qua một mô hình học máy cây quyết định tăng cường độ dốc (Gradient Boosting Decision Tree) được huấn luyện trên nhãn phán quyết đúng/sai từ chuyên gia; và<br>
        • Tự động phê duyệt các đề xuất vùng có xác suất chính xác vượt ngưỡng xác định trước mà không cần sự can thiệp thủ công của con người.
    </div>

    <div class="claim-item">
        <strong>7.</strong> Một phương pháp sinh dữ liệu huấn luyện cho mạng nơ-ron phân đoạn linh kiện khoang động cơ phương tiện giao thông, phương pháp bao gồm các bước:<br>
        (a) Cung cấp một thư viện mô hình 3D chứa các mô hình số ba chiều của linh kiện khoang động cơ, mỗi mô hình được gắn với một lớp linh kiện trong cây phân tầng linh kiện;<br>
        (b) Với mỗi khung hình trong nhiều khung hình tổng hợp, áp dụng ngẫu nhiên hóa miền bằng cách lấy mẫu ngẫu nhiên: (i) vị trí, hướng và trường nhìn của camera; (ii) số lượng, vị trí, cường độ, nhiệt độ màu và độ mềm bóng đổ của các nguồn sáng; (iii) thuộc tính vật liệu bề mặt gồm độ phản xạ và độ nhám cùng lớp phủ thủ tục mô phỏng gỉ sét, dầu, bụi hoặc cặn bẩn; và (iv) vị trí đặt các vật che khuất không phải mục tiêu gồm dây điện, ống dẫn, dụng cụ hoặc bàn tay người;<br>
        (c) Kết xuất mỗi khung hình tổng hợp thành một ảnh màu hai chiều (2D) và một ảnh định danh thực thể;<br>
        (d) Tự động suy ra từ ảnh định danh thực thể một mặt nạ phân đoạn, một khung bao và một nhãn lớp cho mỗi thực thể linh kiện nhìn thấy được; và<br>
        (e) Huấn luyện mạng nơ-ron phân đoạn bằng các ảnh màu đã kết xuất và nhãn đã suy ra, kết hợp với ảnh khoang động cơ thật mang nhãn đã được chuyên gia thẩm định.
    </div>

    <div class="claim-item">
        <strong>8.</strong> Phương pháp theo Điểm 7, trong đó các mô hình số ba chiều được xây dựng từ dữ liệu thiết kế có hỗ trợ máy tính (CAD) tương ứng với các cấu hình phương tiện cụ thể theo Năm - Hãng - Dòng xe - Động cơ (YMME) và được lắp ghép thành cảnh khoang động cơ hoàn chỉnh.
    </div>

    <div class="claim-item">
        <strong>9.</strong> Phương pháp theo Điểm 7, trong đó bước kết xuất bao gồm kết xuất từng mô hình linh kiện riêng lẻ và ghép các thực thể linh kiện đã kết xuất theo kênh alpha lên ảnh chụp khoang động cơ thật, mặt nạ phân đoạn được suy ra từ kênh alpha đã kết xuất.
    </div>

    <div class="claim-item">
        <strong>10.</strong> Phương pháp theo Điểm 7, trong đó tại bước (d) loại bỏ các thực thể linh kiện có tỷ lệ phần nhìn thấy, được tính bằng tỷ số giữa diện tích mặt nạ nhìn thấy và diện tích mặt nạ khi không bị che, thấp hơn một ngưỡng xác định trước.
    </div>

    <div class="claim-item">
        <strong>11.</strong> Phương pháp theo Điểm 7, trong đó bước (e) bao gồm huấn luyện sơ bộ mạng nơ-ron phân đoạn trên các ảnh màu đã kết xuất rồi tinh chỉnh mạng trên các ảnh khoang động cơ thật; và trong đó số lượng thực thể tổng hợp được sinh ra cho mỗi lớp linh kiện giảm dần theo số lượng mẫu thật đã được chuyên gia thẩm định của lớp đó, các thực thể tổng hợp không được tính vào số mẫu thẩm định dùng để thăng hạng lớp linh kiện thành lớp huấn luyện độc lập.
    </div>

    <div class="claim-item">
        <strong>12.</strong> Phương pháp theo Điểm 1, trong đó mạng nơ-ron phân đoạn thực hiện Bước (b) được huấn luyện bằng phương pháp theo Điểm 7, toàn bộ quá trình kết xuất ba chiều được thực hiện trước khi tiếp nhận ảnh chụp tại Bước (a).
    </div>

    <div class="claim-item">
        <strong>13.</strong> Một hệ thống nhận diện và chẩn đoán linh kiện khoang động cơ phương tiện giao thông, hệ thống bao gồm:<br>
        • Một bộ thu nhận hình ảnh được cấu hình để chụp ảnh khoang động cơ phương tiện;<br>
        • Một bộ nhớ lưu trữ các chỉ thị chương trình máy tính và cơ sở dữ liệu ràng buộc không gian linh kiện;<br>
        • Một hoặc nhiều bộ xử lý được liên kết với bộ nhớ, được cấu hình để thực thi các chỉ thị chương trình nhằm triển khai phương pháp được xác định trong bất kỳ Điểm nào từ Điểm 1 đến Điểm 12; và<br>
        • Một giao diện hiển thị đồ họa trực quan hóa cây thành phần linh kiện theo các hệ thống chức năng kỹ thuật và hiển thị cảnh báo vị trí các linh kiện bị che khuất.
    </div>

    <div class="claim-item">
        <strong>14.</strong> Một phương tiện lưu trữ phi tạm thời có thể đọc được bằng máy tính, chứa các lệnh có thể thực thi bởi một bộ xử lý để khiến bộ xử lý thực hiện các bước của phương pháp được xác định trong bất kỳ Điểm nào từ Điểm 1 đến Điểm 12.
    </div>

    <h2>8. TÓM TẮT SÁNG CHẾ</h2>
    <p>Sáng chế đề cập đến một hệ thống và phương pháp tự động nhận diện, phân đoạn và phân rã chức năng các linh kiện khoang động cơ phương tiện giao thông sử dụng thị giác máy tính đa tỷ lệ. Hệ thống tiếp nhận ảnh 2D khoang máy, thực hiện suy luận song song qua nhánh toàn cảnh và nhánh chia ô 2x2 có độ chồng lấn, sau đó gộp các phát hiện bằng thuật toán Class-Agnostic NMS (IoU &ge; 0.85) kết hợp bộ lọc ràng buộc vị trí không gian và giới hạn số lượng để tối ưu hóa độ thu hồi (recall) cho linh kiện kích thước nhỏ mà không gây trùng lặp. Dữ liệu được tổ chức theo cây Taxonomy 3 tầng kết hợp các lớp bao đóng hình dạng chung và quy tắc thăng hạng trễ nhằm giải quyết bài toán linh kiện hiếm mẫu. Hệ thống tích hợp mô hình Gradient Boosting hiệu chuẩn độ tin cậy nhãn từ phán quyết chuyên gia (AUROC 0.844) phục vụ gán nhãn tự động, đồng thời trích xuất danh sách linh kiện bị che khuất (<em>expected_not_visible</em>) dựa trên loại động cơ để hỗ trợ chẩn đoán và bảo dưỡng kỹ thuật. Trong một phương án tùy chọn ở giai đoạn huấn luyện, bộ sinh dữ liệu tổng hợp kết xuất mô hình 3D của linh kiện khoang máy với tư thế camera, ánh sáng, vật liệu bề mặt và vật che khuất được ngẫu nhiên hóa, tự động sinh mặt nạ và nhãn chính xác đến từng điểm ảnh, rồi kết hợp với ảnh thật đã thẩm định để huấn luyện hai giai đoạn, nhờ đó mọi xử lý 3D chỉ diễn ra ngoại tuyến.</p>

</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print("HTML generated at:", html_path)

# Call Microsoft Edge in headless mode to render perfect PDF
msedge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if os.path.exists(msedge_exe):
    cmd = [
        msedge_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    print("Running Edge headless print-to-pdf...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_path):
        print("SUCCESS: Patent PDF generated at:", pdf_path)
        print("Size:", os.path.getsize(pdf_path), "bytes")
    else:
        print("Error printing PDF:", res.stderr)
else:
    print("Edge executable not found at:", msedge_exe)
