# BẢN MÔ TẢ SÁNG CHẾ - DỰ THẢO KIỂM TRA TOÀN XE

Ngày cập nhật: 30 tháng 9 năm 2026. Khoang động cơ là một phương án thực hiện đại diện trong kiến trúc kiểm tra toàn xe. Các phương án mở rộng dưới đây là nội dung mô tả đề xuất; số liệu thực nghiệm hiện có chỉ liên quan đến khoang động cơ. Bản tiếng Việt tương ứng với bản tiếng Anh cùng phiên bản.

## 1. TÊN SÁNG CHẾ

PHƯƠNG PHÁP VÀ HỆ THỐNG KIỂM TRA TRỰC QUAN ĐA TỶ LỆ THEO CẤU TRÚC PHÂN CẤP VÀ ĐÁNH GIÁ DỰA TRÊN BẰNG CHỨNG ĐỐI VỚI LINH KIỆN VÀ CÁC VÙNG CỦA PHƯƠNG TIỆN

## 2. LĨNH VỰC KỸ THUẬT

Sáng chế liên quan đến kiểm tra phương tiện bằng máy tính sử dụng hình ảnh, biểu diễn phân cấp các vùng và linh kiện, phân tích thị giác đa tỷ lệ, mức độ bao phủ quan sát, thẩm định dựa trên độ tin cậy và thông tin chẩn đoán tùy chọn. Phạm vi kiểm tra có thể bao gồm bề mặt ngoại thất, khoang hành khách và khoang hàng, khoang động cơ hoặc hệ truyền động, bánh xe và lốp, kết cấu gầm, hệ thống điện và các bộ phận bảo dưỡng có thể tiếp cận. Các phương án thực hiện áp dụng cho xe động cơ đốt trong, xe lai và xe điện; xe con, xe thương mại, xe buýt và xe máy, với cấu trúc vùng được điều chỉnh theo cấu hình phương tiện.

## 3. TÌNH TRẠNG KỸ THUẬT

Việc kiểm tra phương tiện thường tách thành các quy trình về ngoại hình, trang bị nội thất, lốp, bộ phận cơ khí và thông tin chẩn đoán. Các ảnh thu nhận khác nhau về tỷ lệ, góc nhìn, khả năng quan sát, thiết bị và cấu hình phương tiện. Một chi tiết có thể được phát hiện trong ảnh cận cảnh nhưng chưa xác định rõ quan hệ với vùng trên xe. Ngược lại, ảnh toàn cảnh có thể xác định vùng nhưng làm mất chi tiết linh kiện nhỏ hoặc tình trạng bề mặt. Kết hợp các ảnh này có thể tạo quan sát trùng hoặc hợp nhất sai các linh kiện khác nhau.

Đánh giá toàn xe còn đòi hỏi phân biệt vùng chưa kiểm tra với vùng đã kiểm tra nhưng không phát hiện một linh kiện. Linh kiện có thể không thuộc cấu hình xe, nằm ngoài góc chụp, bị che khuất hoặc chưa được phân tích thị giác xác định. Mã chẩn đoán hoặc hồ sơ bảo dưỡng cung cấp ngữ cảnh bổ sung nhưng tự nó không xác lập một hư hỏng đã được xác nhận bằng hình ảnh.

Phát hiện đa tỷ lệ, phân đoạn, liên kết ảnh, phân loại phân cấp, ước lượng độ tin cậy và ngẫu nhiên hóa miền cho dữ liệu tổng hợp là các nhóm kỹ thuật đã biết. Kiến trúc được mô tả phối hợp các thao tác này thông qua bản ghi bằng chứng gắn với vùng xe, định danh linh kiện và mức độ bao phủ quan sát. Không khẳng định rằng tất cả hệ thống hiện có đều thiếu một thao tác riêng lẻ nào. Các yêu cầu bảo hộ được đề xuất để đối chiếu với tình trạng kỹ thuật và rà soát theo cơ quan nộp đơn.

## 4. BẢN CHẤT KỸ THUẬT CỦA SÁNG CHẾ

Bộ xử lý kiểm tra tiếp nhận một hoặc nhiều ảnh gắn với phương tiện và xác định vùng được thể hiện trong từng ảnh. Bộ xử lý phân tích ảnh ở tỷ lệ ngữ cảnh và một hoặc nhiều tỷ lệ cục bộ, xử lý các quan sát lặp trong cùng ảnh bằng hệ tọa độ ảnh chung, rồi gắn bằng chứng thu được với cấu trúc phân cấp của phương tiện. Bằng chứng từ các ảnh riêng biệt sau đó có thể được liên kết với cùng một linh kiện dựa trên vùng, đặc điểm hình ảnh, cấu hình và ngữ cảnh hình học.

Biểu diễn kiểm tra thu được phân biệt các phát hiện về linh kiện hoặc bề mặt đã quan sát với mức độ bao phủ bằng chứng của các vùng áp dụng. Một vùng có thể đã quan sát, quan sát một phần hoặc chưa kiểm tra; một linh kiện được kỳ vọng có thể đã quan sát, chưa xác định hoặc được kỳ vọng nhưng chưa quan sát thấy. Các trạng thái này quyết định việc báo cáo phát hiện, yêu cầu chụp bổ sung có mục tiêu hoặc giữ trạng thái bất định. Biểu diễn phân cấp có thể dùng cây cho quan hệ thuộc về và đồ thị cho chức năng dùng chung hoặc các quan hệ khác.

Các phương án tùy chọn bổ sung mô hình theo vùng, đánh giá tình trạng, liên kết nhiều góc nhìn, theo dõi thay đổi, thông tin chẩn đoán, chấm điểm độ tin cậy được huấn luyện từ kết quả thẩm định, cơ chế thăng hạng lớp có ngưỡng trễ và huấn luyện bằng dữ liệu tổng hợp. Các thao tác này sử dụng cùng bản ghi bằng chứng về vùng và linh kiện. Phương pháp không cần tiếp nhận ảnh của mọi vùng để kiểm tra một vùng trên xe, và không đòi hỏi kích thước lưới, nhà cung cấp mô hình, số hệ thống trên xe, ngưỡng điểm hoặc việc tái dựng ba chiều cụ thể.

## 5. MÔ TẢ VẮN TẮT CÁC HÌNH VẼ

HÌNH 1 minh họa phạm vi vùng kiểm tra toàn xe: ngoại thất và thân xe (120), khoang hành khách và khoang hàng (130), bánh xe và lốp (140), gầm và khung xe (150), khoang hệ truyền động (160), và các bộ phận điện/năng lượng (170).

HÌNH 2 minh họa thu nhận ảnh (202), định tuyến theo vùng (204), phân tích ngữ cảnh/cục bộ (206), xử lý bằng chứng (208), kho phân cấp và mức độ bao phủ (210), đánh giá (212), và đầu ra hoặc chụp lại có mục tiêu (214).

HÌNH 3 minh họa cấu trúc phân cấp phương tiện-vùng-hệ thống-linh kiện (300) và đồ thị quan hệ riêng biệt (320).

HÌNH 4 minh họa phân tích ngữ cảnh (402), phân tích vùng cục bộ (404), xử lý trong hệ tọa độ chung (406), và tùy chọn liên kết bằng chứng giữa các ảnh riêng biệt (408).

HÌNH 5 minh họa mức độ bao phủ quan sát (502), tính áp dụng (504), trạng thái bằng chứng (506) và chụp bổ sung có mục tiêu (508).

HÌNH 6 minh họa thẩm định và phản hồi huấn luyện (600), bao gồm thẩm định riêng về lớp và hình học, điều kiện chấp nhận, và trạng thái lớp dựa trên mẫu thực đã thẩm định.

HÌNH 7 minh họa bằng chứng tình trạng gắn với linh kiện (700), ngữ cảnh chẩn đoán tùy chọn (720) và so sánh qua thời gian có bảo toàn bằng chứng (740).

HÌNH 8 minh họa bộ sinh dữ liệu tổng hợp tùy chọn (800), sử dụng mô hình đối tượng thuộc nhiều vùng xe và huấn luyện ngoại tuyến cho hệ thống kiểm tra được triển khai.

Các hình vẽ là sơ đồ vector. Kích thước, vị trí linh kiện và số góc nhìn minh họa không phải phép đo vật lý hoặc giới hạn của yêu cầu bảo hộ.

## 6. MÔ TẢ CÁC PHƯƠNG ÁN THỰC HIỆN ĐẠI DIỆN

### 6.1 Phạm vi kiểm tra và hồ sơ vùng

Hồ sơ phương tiện xác định các vùng, hệ thống, loại linh kiện áp dụng và tùy chọn biến thể trang bị. Siêu dữ liệu có thể gồm cấu hình suy ra từ VIN, năm sản xuất/hãng/mẫu xe, loại hệ truyền động, bố trí trục, phiên bản trang bị hoặc mã tài sản trong đội xe. Hồ sơ chưa xác định được giữ ở trạng thái chưa biết hoặc có điều kiện thay vì thay bằng cấu hình giả định.

Bảng 1 nêu các vùng kiểm tra đại diện. Vùng là ngữ cảnh ngữ nghĩa về thu nhận và quan hệ thuộc về, không nhất thiết là một hình chữ nhật cố định. Xe có thể có nhiều khoang hệ truyền động hoặc khoang hàng; các tên vùng khác nhau có thể ánh xạ vào cùng nút phân cấp. Hồ sơ xe máy có thể bỏ khoang hành khách nhưng giữ hệ truyền động lộ ra, bánh xe, khung và cơ cấu điều khiển. Hồ sơ xe buýt hoặc xe tải có thể gồm nhiều trục và các khoang hàng hoặc thiết bị riêng.

| Nhóm vùng | Đối tượng đại diện | Giới hạn theo góc nhìn |
| --- | --- | --- |
| Ngoại thất/thân xe | Tấm thân vỏ, cản, cửa, kính, đèn, gương, chi tiết trang trí, tình trạng bề mặt nhìn thấy | Phản xạ, ánh sáng và góc nhìn ảnh hưởng tình trạng bề mặt biểu kiến |
| Nội thất/khoang hàng | Ghế, hệ thống giữ người, bảng táp-lô, điều khiển, màn hình, trang trí, chi tiết khoang hàng tiếp cận được | Hình thức bên ngoài không xác lập trạng thái cơ cấu ẩn hoặc điện tử bên trong |
| Bánh/lốp/phanh | Lốp, vành, vùng van, chi tiết siết bánh, bộ phận phanh nhìn thấy | Mòn bên trong và kích thước định lượng cần góc nhìn hoặc phép đo phù hợp |
| Gầm/khung xe | Khung, treo, liên kết lái, khí xả, tấm chắn, đường ống tiếp cận được | Khả năng tiếp cận, chất bẩn và che khuất giới hạn mức độ quan sát |
| Khoang hệ truyền động | Động cơ, nạp khí, làm mát, bôi trơn, phụ kiện, điểm bảo dưỡng | Bộ phận bị che hoặc bên trong có thể vẫn được kỳ vọng nhưng chưa quan sát thấy |
| Điện/năng lượng | Dây dẫn nhìn thấy, đầu nối, cổng sạc, vỏ bộ pin, vỏ hệ thống năng lượng | Ngoại hình không xác lập sức khỏe bên trong bộ pin hoặc khả năng cách điện |

Phạm vi toàn xe nghĩa là kiến trúc có thể biểu diễn các vùng áp dụng nêu trên. Chỉ thị hoàn tất kiểm tra chỉ được phát khi đáp ứng các tiêu chí bao phủ đã cấu hình cho phương tiện và nhiệm vụ kiểm tra. Không suy ra hoàn tất từ một ảnh hoặc chỉ từ việc có cấu trúc phân cấp toàn xe.

### 6.2 Thu nhận ảnh và định tuyến theo vùng

Đầu vào gồm ảnh tĩnh, khung hình video được chọn hoặc ảnh từ camera cầm tay, cố định, trạm xe chạy qua, camera gầm, robot hoặc camera nội soi tại vị trí tiếp cận được. Bản ghi ảnh lưu định danh xe/phiên, định danh góc nhìn, thời điểm thu nhận, hướng ảnh, nguồn và vùng được gán. Vùng có thể do người vận hành cung cấp, suy ra từ mốc thị giác hoặc xác định bằng trạm chụp đã biết. Gán vùng chưa rõ được giữ như phương án ứng viên cho đến khi thẩm định.

Chất lượng và mức độ bao phủ ảnh được kiểm tra riêng với suy luận linh kiện. Nhòe, chói, khoảng cách quá xa, cắt ảnh hoặc khoang chưa mở có thể giảm trạng thái quan sát của vùng. Định tuyến vùng chọn mô hình phát hiện, phân đoạn, đánh giá tình trạng phù hợp hoặc mô hình dùng chung có ngữ cảnh vùng. Không bắt buộc tái dựng các ảnh thành mô hình phương tiện ba chiều.

### 6.3 Phân tích thị giác theo ngữ cảnh và cục bộ

Nhánh ngữ cảnh đánh giá toàn ảnh hoặc ảnh tổng quan vùng. Các nhánh cục bộ đánh giá vùng con chồng lấn, vùng quan tâm được chọn hoặc tháp ảnh có mức chi tiết cao hơn. Các nhánh có thể chạy tuần tự hoặc đồng thời nhằm giữ cả ngữ cảnh vùng và chi tiết hình ảnh; chia ô hai-hai là một phương án thực hiện.

Mỗi ứng viên mang định danh lớp hoặc nhóm chung, điểm, khung bao hoặc mặt nạ, nguồn nhánh/góc nhìn và ngữ cảnh vùng. Kết quả cục bộ được chuyển sang hệ tọa độ ảnh gốc tương ứng bằng độ lệch vùng cắt và phép biến đổi ngược của việc đổi kích thước. Mặt nạ, nếu dùng, nhận cùng phép biến đổi với khung bao. Ứng viên bị cắt tại biên nội bộ của vùng cắt có thể được đánh dấu hoặc loại để ưu tiên quan sát đầy đủ hơn.

Xử lý trong một ảnh dùng mức chồng lấn và bằng chứng cá thể tương thích để loại ứng viên lặp hoặc giữ khung bao/mặt nạ có bằng chứng tốt nhất. Triệt tiêu không cực đại (NMS) theo lớp hoặc không phân biệt lớp là các ví dụ; hợp nhất mặt nạ tương thích là lựa chọn khác. Các đối tượng khác nhau lồng nhau, như bánh xe và bộ phận phanh, được giữ bằng quy tắc tương thích xét phân cấp, thay vì tự động gộp do khung bao chồng lấn. Giới hạn số lượng theo vùng và bằng chứng vị trí có thể đánh dấu ứng viên khó hợp lý; giới hạn thiếu cơ sở được vô hiệu hóa. Tiên nghiệm không gian phụ thuộc góc nhìn được dùng như bằng chứng mềm, trừ khi việc đăng ký ảnh đủ cơ sở cho ràng buộc mạnh hơn.

### 6.4 Bằng chứng phân cấp và liên kết giữa các ảnh

Kho bằng chứng gắn phát hiện với cấu trúc như phương tiện -> vùng -> hệ thống chức năng -> nhóm linh kiện -> cá thể linh kiện. Quan hệ thuộc về có thể dùng cây; các cạnh connected_to, part_of, serves_system, adjacent_to và possible_same_instance biểu diễn quan hệ bổ sung. Một đầu nối điện có thể thuộc bó dây vật lý nhưng phục vụ hệ truyền động hoặc chiếu sáng. Phát hiện tình trạng được gắn với linh kiện hoặc vùng bề mặt tương ứng, không coi là nhãn cấp xe không có liên hệ.

Giữa các ảnh riêng biệt, chồng lấn điểm ảnh đơn thuần không đủ để đối sánh định danh. Khối liên kết tùy chọn so sánh định danh xe/phiên, vùng và phía xe, lớp linh kiện, biểu diễn nhúng hình ảnh, mốc nhìn thấy và tương ứng hình học có sẵn. Không gộp bánh trước trái với bánh trước phải chỉ vì chúng giống nhau. Nhãn trạm/góc nhìn đã biết, đối sánh mốc hoặc vị trí do người vận hành xác nhận giúp phân biệt. Đối sánh chưa rõ được giữ như giả thuyết có nguồn gốc thay vì định danh vật lý đã xác nhận.

Thao tác tọa độ chung ở Mục 6.3 áp dụng trong từng ảnh gốc. Liên kết giữa ảnh nối bằng chứng cục bộ của từng ảnh với cùng cá thể theo ngữ nghĩa, không giả định mọi ảnh chung hệ tọa độ điểm ảnh. Quan sát độc lập và mâu thuẫn được giữ để thẩm định. Sự tách biệt này cho phép camera khác góc nhìn đóng góp bằng chứng mà không bắt buộc đăng ký ba chiều.

### 6.5 Mức độ bao phủ, tính áp dụng và linh kiện kỳ vọng

Với từng vùng áp dụng, bộ xử lý lưu mức độ bao phủ quan sát từ chất lượng ảnh được chấp nhận, mức chồng lấn ảnh/vùng và khả năng nhìn thấy mốc hoặc bề mặt mục tiêu liên quan nhiệm vụ. Chính sách có thể yêu cầu nhiều góc nhìn, mặt nạ bề mặt đủ đầy hoặc khả năng tiếp cận do người vận hành xác nhận. Chính sách phân biệt đã quan sát, quan sát một phần, chưa kiểm tra và không áp dụng. Các trạng thái phản ánh bao phủ bằng chứng, không phải tình trạng vật lý linh kiện.

Bản ghi linh kiện kỳ vọng được chọn từ hồ sơ xe đã xác định hoặc mẫu kiểm tra rồi đối chiếu với bằng chứng được giữ. Linh kiện có thể đã quan sát, unresolved_generic_match, expected_not_observed hoặc applicability_unknown. Có thể giữ tên expected_not_visible để tương thích, nhưng tên này nghĩa là được kỳ vọng và chưa quan sát thấy trong các ảnh có sẵn. Không xác nhận che khuất, thiếu linh kiện hoặc hỏng. Phát hiện nhóm chung có thể tương ứng với linh kiện được liên kết như bằng chứng chưa giải quyết.

Nếu mục tiêu chưa đạt mức bao phủ chấp nhận được, bộ xử lý tạo chỉ dẫn chụp lại có mục tiêu. Chỉ dẫn xác định vùng hoặc mốc, góc nhìn hoặc chi tiết cần có và lý do: chói, thiếu phía xe, linh kiện chưa rõ hoặc khả năng che khuất. Lớp phủ vị trí của mục tiêu chưa quan sát cần mốc có cơ sở, sơ đồ cấu hình phù hợp hoặc mô hình tọa độ trạm đã xác lập; nếu không, chỉ dẫn ở dạng chữ. Sau chụp lại, hệ thống cập nhật mức bao phủ và đối chiếu bằng chứng, vẫn giữ nguồn ảnh trước đó.

### 6.6 Tình trạng linh kiện và ngữ cảnh chẩn đoán

Khối đánh giá tình trạng tùy chọn phân loại dấu hiệu nhìn thấy như móp tấm thân vỏ, nứt bề mặt, vết chất lỏng, đầu nối tách rời, ăn mòn, thiếu chi tiết trang trí nhìn thấy hoặc bất thường bề mặt lốp. Có thể dùng phân đoạn, mô hình ảnh đã huấn luyện, quy tắc cấu hình hoặc bằng chứng thị giác đã thẩm định. Bản ghi lưu loại phát hiện, cá thể/vùng mục tiêu, ảnh hỗ trợ, vị trí, độ tin cậy và trạng thái đánh giá. Phát hiện chưa chắc chắn giữ trạng thái nghi ngờ hoặc review_required. Không tự động chuyển bất thường nhìn thấy thành hư hỏng cơ khí bên trong.

Trường mức độ nghiêm trọng tùy chọn là định tính, trừ khi phép đo có cơ sở hoặc mô hình đã hiệu chuẩn xác lập thang định lượng. Độ sâu gai lốp, kích thước biến dạng và phép đo tương tự cần chuẩn tỷ lệ hoặc bố trí đo đã kiểm chứng. Không khẳng định dung lượng bên trong bộ pin, cách điện hoặc độ dày phanh bị che chỉ từ hình thức RGB.

Đầu vào OBD/DTC tùy chọn, lịch sử bảo dưỡng, số km hoặc quan sát cảm biến có cơ sở được giữ như loại bằng chứng riêng với thời điểm và nguồn gốc. Quan hệ cấu hình có thể nối bằng chứng này với linh kiện và đề xuất bước kiểm tra. Mâu thuẫn giữa thị giác và chẩn đoán được lưu rõ ràng. Không khẳng định DTC chỉ từ không phát hiện, và không suy ra chỉ dẫn tháo bộ phận cao áp từ nhận diện vỏ bao.

### 6.7 Thẩm định, bất định và phản hồi huấn luyện

Mô hình huấn luyện từ thẩm định có thể chấm điểm đúng lớp bằng độ tin cậy của bộ phát hiện hoặc mô hình thị giác-ngôn ngữ, tương đồng hình ảnh vùng cắt tham chiếu, hình học, ngữ cảnh không gian, chồng lấn và đặc trưng theo vùng. Tương đồng có thể dùng hiệu giữa mức phù hợp với mẫu đúng lớp đã thẩm định và mẫu bị bác bỏ. Đúng lớp, chấp nhận định vị, đúng tình trạng và trạng thái trùng là các mục tiêu thẩm định riêng. Điểm lớp cao không xác nhận tất cả mục tiêu.

Nhãn ứng viên được chấp nhận, thẩm định hoặc loại theo ngưỡng điểm có cơ sở và điều kiện hình học/không trùng. Hiệu chuẩn xác suất tùy chọn được khớp cho mục tiêu xác định bằng xe kiểm định riêng. Kho tham chiếu và tiên nghiệm học được trong từng lượt đánh giá chỉ xây từ xe huấn luyện; đánh giá ngưỡng cuối dùng xe chưa tham gia các bước trước. Hệ thống lưu quyết định người thẩm định và có thể thu hồi nhãn đã chấp nhận.

Định danh chi tiết vẫn có trong bản ghi chú giải dù nhãn huấn luyện dùng lớp dự phòng chung như vỏ bao khác, đường dẫn khác, chi tiết siết khác hoặc bất thường bề mặt khác. Chỉ thăng hạng lớp độc lập khi số mẫu thực đủ điều kiện đã được chuyên gia thẩm định và độ đa dạng xe đạt ngưỡng. Hạ hạng dùng ngưỡng duy trì thấp hơn, giữ định danh và thay ánh xạ huấn luyện. Thay đổi đi vào danh mục huấn luyện có phiên bản và có hiệu lực trong mô hình cập nhật tương ứng. Mẫu tổng hợp không tăng số mẫu thực đã thẩm định. Ngưỡng và danh mục vùng có thể cấu hình, không cố định theo một số lượng toàn xe.

### 6.8 Bằng chứng qua thời gian và tạo báo cáo

Phương án theo dõi qua thời gian tùy chọn liên kết các phiên với cùng xe và đối sánh định danh vùng/linh kiện tương ứng. Hệ thống so sánh trạng thái phát hiện đã chuẩn hóa hoặc bề mặt tương ứng đã đăng ký, xét khác biệt góc nhìn, ánh sáng, thay thế và chất lượng chụp. Dấu vết mới được báo như thay đổi ứng viên cho đến khi bằng chứng hỗ trợ liên kết và so sánh tình trạng. Linh kiện thay thế tạo sự kiện định danh/phiên bản thay vì giả định cá thể vẫn liên tục tồn tại.

Đầu ra gồm cây hoặc đồ thị vùng/hệ thống, mức bao phủ quan sát, phát hiện có định vị, bất định, ngữ cảnh chẩn đoán liên kết và yêu cầu góc nhìn bổ sung. Báo cáo giữ phiên bản mô hình/danh mục phân loại, tham chiếu ảnh nguồn, trạng thái hồ sơ xe và lịch sử thẩm định. Giao diện đội xe hoặc bảo dưỡng có thể nhận bản ghi có cấu trúc nhưng không phải thành phần bắt buộc của phương pháp. Xử lý có thể tại thiết bị, máy chủ hoặc chia giữa thiết bị thu nhận và xử lý.

### 6.9 Huấn luyện tổng hợp toàn xe tùy chọn

Thư viện mô hình đối tượng ngoại tuyến gắn tấm ngoại thất, trang bị nội thất, bánh xe, bộ phận khung gầm, hệ truyền động và vỏ điện với cùng định danh vùng/linh kiện dùng khi kiểm tra. Cảnh theo vùng, linh kiện riêng và tình trạng thị giác đề xuất có thể được kết xuất với tư thế camera, ánh sáng, vật liệu, chất bẩn và che khuất ngẫu nhiên. Bộ kết xuất tạo ảnh màu và định danh cá thể/bề mặt để suy ra khung bao, mặt nạ, nhãn vùng và nhãn tình trạng đã cấu hình.

Nhãn tuân theo khả năng nhìn thấy và thao tác ghép ảnh cuối. Cá thể không quan sát được không nhận mặt nạ cá thể nhìn thấy. Bản ghi tổng hợp giữ nguồn mô hình, vùng, cấu hình và lần kết xuất. Lấy mẫu có thể tập trung linh kiện hoặc tình trạng hiếm, trong khi chỉ mẫu thực đã thẩm định điều khiển thăng hạng lớp. Tiền huấn luyện hoặc huấn luyện hỗn hợp được tiếp nối bằng thích nghi với ảnh thực đã thẩm định. Xe thực giữ riêng đánh giá chuyển miền; dự thảo không cung cấp số liệu toàn xe hoặc chuyển miền tổng hợp được tạo giả. Kết xuất ngoại tuyến không đòi hỏi tái dựng ba chiều khi vận hành.

### 6.10 Tình huống kiểm tra đại diện

Tình huống A - khoang hệ truyền động: ảnh tổng quan xác định các vỏ bao lớn, trong khi vùng cắt cục bộ nhận diện nắp bảo dưỡng nhỏ. Tọa độ chung của ảnh gốc xử lý trùng. Bugi được kỳ vọng nhưng bị che vẫn chưa quan sát với bất định; nắp có định danh hợp lý nhưng hình học khung bao chưa rõ vẫn cần thẩm định. Đây là phương án khoang động cơ ban đầu.

Tình huống B - ngoại thất và bánh xe: ảnh tổng quan bên trái và ảnh chi tiết cung cấp bằng chứng bề mặt cửa trước và bánh trước trái. Tình trạng cửa gắn với vùng bề mặt; bánh được liên kết bằng vị trí trước trái và bằng chứng góc nhìn. Thiếu ảnh bên phải giữ phía này chưa kiểm tra và tạo yêu cầu chụp bên phải. Hệ thống không báo phía chưa kiểm tra là không có hư hỏng.

Tình huống C - gầm xe: ảnh gầm được chấp nhận xác định liên kết hệ treo tiếp cận được và dấu vết giống chất lỏng. Dấu vết là tình trạng nghi ngờ có định vị gắn với các linh kiện gần đó. Bề mặt phanh bị che vẫn chỉ quan sát một phần; cần ảnh gần hơn. Riêng dấu vết không xác lập nguồn rò rỉ hoặc mòn phanh bên trong.

Tình huống D - xe điện và khoang hành khách: hồ sơ xác định động cơ đốt trong không áp dụng và xác định vùng cổng sạc/vỏ bộ pin. Dấu hiệu trên vỏ nhìn thấy và mã chẩn đoán được cung cấp vẫn là bằng chứng riêng. Ảnh khoang hành khách xác định ghế và điều khiển; điểm gắn hệ thống giữ người bị che chưa được quan sát. Không suy ra sức khỏe bộ pin từ hình thức vỏ. Đây là các phương án minh họa đề xuất, không phải thử nghiệm đã báo cáo.

### 6.11 Bằng chứng hiện có và giới hạn áp dụng

Báo cáo mô hình giáo viên khoang động cơ trong kho mã ghi 125 ảnh kiểm tra, kích thước đầu vào mạng 640 và IoU khung bao >=0.5 với lớp trùng khớp. Độ thu hồi (recall)/độ chính xác dự đoán (precision) của toàn ảnh là 0.582/0.632; suy luận chia ô cùng xử lý trùng và giới hạn số lượng đạt 0.614/0.581. Độ thu hồi tăng 3.2 điểm phần trăm, độ chính xác giảm 5.1 điểm phần trăm. Không xác lập hiệu năng ngoại thất, khoang hành khách, bánh xe, gầm hoặc xe điện.

Báo cáo thẩm định trước đây ghi 4627 khung bao đã thẩm định từ 177 xe và AUROC 0.844 so với 0.617 của độ tin cậy VLM thô. Với điểm >=0.9, tỷ lệ mẫu được chọn là 21.8% và độ chính xác lớp là 0.941. Mục tiêu coi bad_geometry là đúng lớp; mẫu cắt tham chiếu được tính trước bước chia lượt mô hình theo xe. Vì vậy đây là quan sát thăm dò về chấm điểm lớp, không phải kiểm chứng độc lập đầy đủ về hiệu chuẩn, chấp nhận định vị hoặc toàn xe. Không tái dựng đường ROC thực nghiệm trong hình vẽ.

Kiểm chứng toàn xe cần đo riêng sai số bao phủ vùng, hợp nhất sai giữa ảnh, sai lớp/tình trạng, ích lợi của chụp lại, bất định và chuyển giao giữa hồ sơ xe. Phương pháp có thể kiểm tra các vùng áp dụng từ tập ảnh được chấp nhận; chức năng nội bộ ẩn vẫn ngoài bằng chứng thị giác nếu không có phép đo riêng được xác định rõ hỗ trợ.

## 7. YÊU CẦU BẢO HỘ ĐỀ XUẤT

1. Phương pháp kiểm tra trực quan phương tiện được thực hiện bằng máy tính, bao gồm: (a) tiếp nhận một hoặc nhiều ảnh gắn với phương tiện; (b) gán từng ảnh cho một hoặc nhiều vùng của phương tiện được biểu diễn trong cấu trúc phân cấp phương tiện; (c) phân tích ít nhất một ảnh ở tỷ lệ ngữ cảnh và một hoặc nhiều tỷ lệ cục bộ để tạo các quan sát ứng viên về linh kiện hoặc bề mặt; (d) chuyển vị trí quan sát tỷ lệ cục bộ sang tọa độ ảnh gốc tương ứng và xử lý quan sát lặp trong ảnh đó; (e) gắn quan sát được giữ với định danh vùng và linh kiện hoặc bề mặt trong cấu trúc phân cấp; (f) xác định trạng thái bao phủ quan sát cho vùng được biểu diễn từ bằng chứng ảnh được chấp nhận; và (g) tạo đầu ra kiểm tra có cấu trúc phân biệt phát hiện đã quan sát với vùng chưa xác định hoặc chưa kiểm tra theo trạng thái bao phủ quan sát.

2. Phương pháp theo yêu cầu bảo hộ 1, trong đó các vùng được biểu diễn gồm ngoại thất thân xe, nội thất hoặc khoang hàng, bánh xe hoặc lốp, gầm hoặc khung xe, khoang hệ truyền động và vùng hệ thống điện hoặc năng lượng; phương pháp áp dụng cho một tập con hoặc tất cả vùng áp dụng đối với cấu hình phương tiện.

3. Phương pháp theo yêu cầu bảo hộ 1, trong đó hồ sơ phương tiện xác định tính áp dụng của vùng và linh kiện cho xe động cơ đốt trong, xe lai hoặc xe điện; linh kiện không áp dụng được phân biệt với linh kiện áp dụng nhưng chưa quan sát thấy trong ảnh có sẵn.

4. Phương pháp theo yêu cầu bảo hộ 1, trong đó thu nhận gồm chụp cầm tay, trạm cố định, khung hình video được chọn, chụp gầm, chụp bằng robot hoặc camera nội soi tại vị trí tiếp cận được, với định danh góc nhìn/phiên và gán vùng được lưu giữ.

5. Phương pháp theo yêu cầu bảo hộ 1, trong đó khối định tuyến vùng chọn mô hình thị giác theo vùng hoặc cung cấp ngữ cảnh vùng cho mô hình thị giác dùng chung.

6. Phương pháp theo yêu cầu bảo hộ 1, trong đó phân tích tỷ lệ cục bộ gồm ô chồng lấn, vùng quan tâm được chọn hoặc biểu diễn ảnh đa độ phân giải; các nhánh ngữ cảnh và cục bộ chạy tuần tự hoặc đồng thời.

7. Phương pháp theo yêu cầu bảo hộ 1, trong đó quan sát gồm mặt nạ phân đoạn; mặt nạ và khung bao tương ứng được chuyển bằng cùng độ lệch vùng cắt và phép biến đổi ngược của việc đổi kích thước trước khi xử lý trùng.

8. Phương pháp theo yêu cầu bảo hộ 1, trong đó xử lý quan sát lặp gồm NMS theo lớp hoặc không phân biệt lớp, hoặc hợp nhất mặt nạ tương thích; quan hệ linh kiện ngăn hợp nhất các mục tiêu khác nhau lồng nhau chỉ vì vị trí chồng lấn.

9. Phương pháp theo yêu cầu bảo hộ 1, trong đó bằng chứng vị trí hoặc ràng buộc số lượng linh kiện phụ thuộc vùng hoặc hồ sơ xe; ràng buộc thiếu cơ sở được vô hiệu hóa hoặc dùng như bằng chứng mềm.

10. Phương pháp theo yêu cầu bảo hộ 1, còn bao gồm liên kết quan sát từ các ảnh riêng với cùng định danh linh kiện bằng định danh xe/phiên, vùng hoặc phía xe, đặc điểm hình ảnh, mốc hoặc tương ứng hình học, đồng thời giữ liên kết chưa rõ như giả thuyết.

11. Phương pháp theo yêu cầu bảo hộ 10, trong đó xử lý tọa độ trong một ảnh tách biệt với liên kết cá thể theo ngữ nghĩa giữa ảnh; liên kết không đòi hỏi cùng hệ tọa độ điểm ảnh hoặc tái dựng phương tiện ba chiều.

12. Phương pháp theo yêu cầu bảo hộ 1, trong đó biểu diễn phân cấp gồm nút phương tiện, vùng, hệ thống, nhóm linh kiện và cá thể, với cạnh quan hệ xác định chức năng dùng chung, kết nối hoặc quan hệ thuộc về.

13. Phương pháp theo yêu cầu bảo hộ 1, trong đó trạng thái bao phủ quan sát phân biệt đã quan sát, quan sát một phần, chưa kiểm tra và không áp dụng theo chính sách kiểm tra và chất lượng ảnh được chấp nhận.

14. Phương pháp theo yêu cầu bảo hộ 1, còn bao gồm đối chiếu quan sát được giữ với kỳ vọng từ hồ sơ xe và biểu diễn linh kiện kỳ vọng nhưng chưa quan sát với bất định, phân biệt chưa quan sát với thiếu linh kiện vật lý hoặc hỏng đã xác nhận.

15. Phương pháp theo yêu cầu bảo hộ 14, trong đó quan sát nhóm chung liên kết với linh kiện chi tiết kỳ vọng chưa xác định; lớp phủ vị trí cho linh kiện chưa quan sát chỉ được tạo khi có mốc, sơ đồ cấu hình hoặc mô hình tọa độ trạm hỗ trợ.

16. Phương pháp theo yêu cầu bảo hộ 1, còn bao gồm tạo chỉ dẫn chụp bổ sung có mục tiêu xác định vùng, góc nhìn hoặc chi tiết cần có và bằng chứng còn thiếu, rồi cập nhật trạng thái bao phủ sau tiếp nhận ảnh bổ sung.

17. Phương pháp theo yêu cầu bảo hộ 1, còn bao gồm gắn phát hiện tình trạng nhìn thấy có định vị với định danh linh kiện hoặc bề mặt được giữ, đồng thời lưu nguồn ảnh hỗ trợ và trạng thái đánh giá tình trạng.

18. Phương pháp theo yêu cầu bảo hộ 17, trong đó mã chẩn đoán được cung cấp, hồ sơ bảo dưỡng hoặc quan sát cảm biến được giữ như loại bằng chứng riêng gắn với linh kiện hoặc vùng; mâu thuẫn với phát hiện thị giác được bảo toàn.

19. Phương pháp theo yêu cầu bảo hộ 1, còn bao gồm so sánh bằng chứng từ các phiên khác nhau của cùng phương tiện bằng định danh vùng hoặc linh kiện tương ứng, và ghi nhận thay đổi tình trạng hoặc thay linh kiện với nguồn ảnh.

20. Phương pháp theo yêu cầu bảo hộ 1, còn bao gồm chấm điểm nhãn huấn luyện đề xuất bằng bằng chứng kết quả thẩm định và chỉ chấp nhận nhãn khi điều kiện điểm lớp, hình học và không trùng đáp ứng chính sách cấu hình.

21. Phương pháp theo yêu cầu bảo hộ 20, trong đó linh kiện chi tiết giữ định danh chú giải trong khi nhãn huấn luyện ánh xạ vào lớp dự phòng chung; thăng hạng thành lớp huấn luyện độc lập đòi hỏi số mẫu thực đã thẩm định và độ đa dạng xe, còn hạ hạng dùng ngưỡng duy trì thấp hơn.

22. Phương pháp theo yêu cầu bảo hộ 1, trong đó mô hình thị giác triển khai được huấn luyện bằng ảnh tổng hợp ngoại tuyến từ mô hình đối tượng gắn với danh mục phân loại của nhiều vùng xe, được kết xuất với góc nhìn, ánh sáng, vật liệu và che khuất ngẫu nhiên, rồi thích nghi với ảnh thực đã chuyên gia thẩm định; mẫu tổng hợp không được tính vào số mẫu thực hỗ trợ thăng hạng lớp.

23. Hệ thống kiểm tra trực quan phương tiện bao gồm giao diện thu nhận, bộ nhớ lưu biểu diễn phân cấp phương tiện và bản ghi mức bao phủ vùng, và một hoặc nhiều bộ xử lý được cấu hình để: gắn ảnh xe tiếp nhận với các vùng; tạo quan sát linh kiện hoặc bề mặt bằng phân tích ngữ cảnh và tỷ lệ cục bộ; chuyển và xử lý quan sát cục bộ trong tọa độ ảnh gốc; gắn quan sát được giữ với định danh vùng và linh kiện hoặc bề mặt; xác định mức bao phủ vùng từ bằng chứng ảnh được chấp nhận; và xuất phát hiện đã quan sát riêng với vùng chưa xác định hoặc chưa kiểm tra theo mức bao phủ; cùng màn hình hoặc giao diện đầu ra cho biểu diễn kiểm tra có cấu trúc thu được.

24. Vật mang lưu trữ đọc được bằng máy tính không mang tính tạm thời, lưu các lệnh khi được một hoặc nhiều bộ xử lý thực thi sẽ khiến các bộ xử lý thực hiện phương pháp theo yêu cầu bảo hộ 1.

25. Phương pháp được thực hiện bằng máy tính để tuyển chọn nhãn huấn luyện kiểm tra trực quan trên các vùng phương tiện, bao gồm: tiếp nhận bản ghi ứng viên linh kiện hoặc bề mặt đã thẩm định với kết quả về lớp và định vị được xác định riêng; tính đặc trưng ứng viên theo ngữ cảnh vùng, gồm độ tin cậy thị giác và hiệu giữa độ tương đồng biểu diễn nhúng với mẫu đúng lớp đã thẩm định và với mẫu bị bác bỏ; huấn luyện mô hình chấm điểm kết quả thẩm định bằng các đặc trưng đó; chọn ngưỡng định tuyến bằng dữ liệu kiểm định tách theo xe, với kho tham chiếu chỉ gồm xe huấn luyện; và định tuyến ứng viên mới vào chấp nhận, thẩm định hoặc loại theo điểm và điều kiện riêng về định vị và không trùng, đồng thời giữ nguồn gốc thẩm định.

26. Phương pháp theo yêu cầu bảo hộ 25, còn bao gồm khớp ánh xạ hiệu chuẩn xác suất cho mục tiêu thẩm định xác định bằng xe kiểm định riêng và đánh giá ánh xạ bằng dữ liệu xe được giữ riêng.

27. Phương pháp theo yêu cầu bảo hộ 25, còn bao gồm duy trì định danh chi tiết với nhãn huấn luyện dự phòng chung và thay trạng thái lớp độc lập theo số cá thể thực đã thẩm định và số xe, dùng ngưỡng thăng hạng và duy trì khác nhau, không tính mẫu tổng hợp.

28. Hệ thống tuyển chọn nhãn huấn luyện bao gồm bộ nhớ lưu bản ghi ứng viên đã thẩm định gắn với vùng và các biểu diễn nhúng tham chiếu, cùng một hoặc nhiều bộ xử lý được cấu hình thực hiện phương pháp theo yêu cầu bảo hộ 25.

29. Vật mang lưu trữ đọc được bằng máy tính không mang tính tạm thời, lưu các lệnh khi được một hoặc nhiều bộ xử lý thực thi sẽ khiến các bộ xử lý thực hiện phương pháp theo yêu cầu bảo hộ 25.

30. Phương pháp theo yêu cầu bảo hộ 1, trong đó chỉ thị hoàn tất kiểm tra chỉ được phát khi đáp ứng các tiêu chí bao phủ quan sát đã cấu hình cho vùng áp dụng của hồ sơ phương tiện và nhiệm vụ kiểm tra.

## 8. TÓM TẮT

Sáng chế đề xuất hệ thống kiểm tra trực quan các vùng phương tiện bằng phân tích ảnh theo ngữ cảnh và tỷ lệ cục bộ. Quan sát cục bộ được chuyển về tọa độ ảnh gốc và xử lý trùng, rồi gắn với biểu diễn phân cấp các vùng ngoại thất, nội thất, bánh xe, gầm, hệ truyền động và điện áp dụng cho xe. Mức bao phủ phân biệt vùng đã quan sát, quan sát một phần và chưa kiểm tra. Hồ sơ xe xác định linh kiện áp dụng nhưng chưa quan sát với bất định; thiếu bằng chứng có thể tạo yêu cầu chụp bổ sung. Các phương án tùy chọn liên kết nhiều góc nhìn, tình trạng và chẩn đoán, so sánh các phiên, tuyển chọn nhãn từ thẩm định với điều kiện định vị và không trùng. Lớp dự phòng dùng mẫu thực đã thẩm định và ngưỡng thăng hạng trễ. Huấn luyện tổng hợp ngoại tuyến không làm tăng số mẫu thực dùng thăng hạng. Đầu ra bảo toàn nguồn bằng chứng và phân biệt quan sát với đánh giá chưa xác định.
