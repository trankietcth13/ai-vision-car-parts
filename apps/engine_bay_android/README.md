# Engine Bay Vision cho Android

App Android nhận diện linh kiện khoang máy, chạy **hoàn toàn trên điện thoại**. App dùng model POC v1 (`kd_n_full`,
yolo11n-seg) với ONNX Runtime và không cần mạng.

Cách dùng:
1. Chụp ảnh khoang máy bằng camera của máy, hoặc chọn ảnh từ thư viện hay ảnh mẫu.
2. App hiện mask, box và tên linh kiện bằng tiếng Việt.
3. Chạm vào một linh kiện, trên ảnh hoặc trong danh sách, để xem thông tin của linh kiện đó.

## Cấu trúc

```text
app/src/main/java/com/enginebay/vision/
├── core/Preprocess.kt    letterbox giống Ultralytics (resize cv2.INTER_LINEAR, làm tròn half-even)
├── core/Postprocess.kt   giải mã YOLO11-seg, NMS theo lớp, mask (crop ở độ phân giải prototype rồi upsample)
├── Detector.kt           ONNX Runtime (CPU, 4 luồng), ngưỡng riêng từng lớp, overlay mask
├── PhotoLoader.kt        giải mã ảnh: xoay theo EXIF, thu về cạnh dài 1600 px như app Python
├── ResultView.kt         vẽ kết quả; chạm để chọn linh kiện
├── ComponentInfo.kt      đọc assets/components.json (song ngữ)
├── AppState.kt           model, ảnh, kết quả giữ nguyên khi đổi ngôn ngữ
├── core/ModelBuilder.kt  dựng mô hình 3D ngay trên máy từ bản mô tả JSON (cùng thuật toán với Python)
├── core/Glb.kt           đọc mô hình 3D dạng GLB (khi có file mô hình thật: CAD, quét 3D)
├── ModelView.kt          xem 3D bằng OpenGL ES 2.0: xoay, phóng to, chạm chọn chi tiết, tách rời
├── ModelPanel.kt         khối 3D trong bảng thông tin: mô hình, danh sách chi tiết, nội dung cần kiểm tra
├── ModelStore.kt         dựng (JSON) hoặc đọc (GLB) mô hình từ assets/models, lưu đệm
└── MainActivity.kt       chụp (app camera hệ thống), thư viện, ảnh mẫu, danh sách, bảng thông tin
app/src/main/assets/
├── components.json       thông tin linh kiện (trong git; đặc tả: docs/COMPONENT_INFO_SPEC.md)
├── models/<linh_kiện>.json  34 bản mô tả mô hình 3D, tổng ~230 KB (trong git; sinh bằng scripts/deployment/build_component_models.py)
├── model/kd_n_full.onnx, config.json, examples/   sinh ra bằng script, không nằm trong git
```

## Build

Cần JDK 17 trở lên (đã dùng JDK 21) và Android SDK 36. Phiên bản: AGP 9.1.1, Gradle 9.5, ONNX Runtime Android 1.30.0.

```bash
python scripts/deployment/prepare_android_app.py   # (từ thư mục gốc dự án) model, config, ảnh mẫu, fixture test
cd apps/engine_bay_android
./gradlew :app:testDebugUnitTest                   # so khớp với Ultralytics
./gradlew :app:assembleDebug                       # app/build/outputs/apk/debug/app-debug.apk
./gradlew :app:installDebug                        # cài lên thiết bị đang cắm (đã bật USB debugging)
```

Nếu không có Android Studio, cần tạo `local.properties` với dòng `sdk.dir=C\:/Users/<user>/AppData/Local/Android/Sdk`.

## Độ chính xác so với Python

Unit test `UltralyticsParityTest` dùng ảnh thật và kết quả do chính Ultralytics xuất ra:
- **Letterbox:** lệch tối đa 1/255 so với `cv2`, do `cv2` dùng trọng số dấu phẩy tĩnh.
- **Hậu xử lý:** cả 11/11 detection trùng lớp, điểm (sai số < 1e-5) và box (sai số < 1e-3). Mask IoU bằng 1,0000.

## Song ngữ Anh–Việt

- **Đổi ngôn ngữ:** nút **EN / VI** trên thanh tiêu đề. App mặc định tiếng Việt và nhớ lựa chọn. Đổi ngôn ngữ không làm mất ảnh, kết quả hay mã lỗi đang xem.
- **Chữ trên giao diện:** tiếng Anh trong `res/values/strings.xml`, tiếng Việt trong `res/values-vi/strings.xml`. Code Kotlin không chứa chuỗi hiển thị viết cứng.
- **Dữ liệu:** mọi đoạn văn bản là cặp `{"en", "vi"}`. Cụ thể:
  - tên linh kiện: `config.json` có `names_en` / `names_vi`;
  - thông tin linh kiện: `components.json`;
  - mã lỗi và lưu ý an toàn: `diagnosis.json`.
- **Nguồn bản dịch:** `configs/diagnosis_knowledge_vi.yaml`, đi kèm `configs/diagnosis_knowledge.yaml`.
- **Kiểm tra:** `BilingualDataTest` báo lỗi nếu có đoạn văn bản nào thiếu một ngôn ngữ.

## Mô hình 3D linh kiện

Mỗi linh kiện trong cơ sở tri thức OBD2 (`configs/diagnosis_knowledge.yaml`) có một mô hình 3D. Có tất cả 34 mô hình: 33 linh kiện của bảng tri thức và nắp che động cơ.

**Cách dựng mô hình (ngay trên máy):** script `scripts/deployment/build_component_models.py` mô tả mỗi linh kiện bằng các khối hình học cơ bản và ghi ra `assets/models/<linh_kiện>.json`. Script dùng `tools/model3d/` lấy từ skill `pro-product-video`. Mỗi mô hình gồm:
- các chi tiết có tên;
- vật liệu PBR;
- hướng tách rời của từng chi tiết (thứ tự tháo);
- tên chi tiết và nội dung cần kiểm tra trên chi tiết đó, song ngữ `{"en", "vi"}`.

App **sinh lưới tam giác ngay trên máy** bằng `ModelBuilder.kt`, khi người dùng mở mô hình. `ModelBuilder.kt` chép lại đúng từng thuật toán Python: lấy mẫu, thứ tự tam giác, hướng mặt, pháp tuyến góc 35°.

`ComponentModelsTest` so từng chi tiết của cả 34 mô hình với hình học do Python sinh (`fixtures/model_stats.json`): số tam giác phải khớp, diện tích, thể tích và khung bao lệch không quá 1e-4. Nếu có file `<linh_kiện>.glb` (mô hình CAD hoặc quét 3D thật), app đọc file đó thay vì tự dựng.

Mô hình là hình minh hoạ chung, không phải đúng chi tiết của một xe cụ thể.

```bash
python scripts/deployment/build_component_models.py                       # bản mô tả -> assets/models + fixture so khớp
python scripts/deployment/build_component_models.py --only spark_plug --glb output/models3d/glb --preview output/models3d
```

**Trong app:**
- Bảng thông tin của linh kiện có khối 3D. Trên mô hình: kéo để xoay, chụm hai ngón để phóng to, chạm đúp để đặt lại. Chạm vào một chi tiết (trên mô hình hoặc trong danh sách) thì chi tiết đó nhấp nháy và app hiện nội dung cần kiểm tra. Nút **Tách rời** tách các chi tiết theo thứ tự tháo.
- Trong bảng chẩn đoán theo mã lỗi, mỗi linh kiện cần kiểm tra có nút **3D**. Nút này hữu ích nhất cho các linh kiện mà model chưa nhận diện được trên ảnh (bugi, kim phun, van EGR...).
- Nút **Linh kiện 3D** trên thanh tiêu đề mở thư viện các mô hình. Thư viện dùng được cả khi chưa chụp ảnh.

**Kiểm thử:**
- `adb shell am start -n com.enginebay.vision/.MainActivity --es model spark_plug` mở thẳng một mô hình.
- `ComponentModelsTest` kiểm tra ba điều: mọi linh kiện của bảng tri thức đều có mô hình, mô hình đọc được, và mọi nhãn cùng nội dung kiểm tra đều đủ hai ngôn ngữ.

## Kiểm tra offline: ảnh, checklist và lịch sử

Trên màn hình chính, nhóm thao tác kiểm tra gồm:

1. **Thông tin xe**: nhập biển số/tên xe/mã công việc và ghi chú.
2. **Tìm linh kiện**: chọn tên để làm nổi bật trên ảnh. Có thể bỏ chọn để trở lại các linh kiện liên quan mã lỗi. “Chưa nhận diện” không phải “bị thiếu/bị hỏng”.
3. **Checklist kiểm tra**: chọn linh kiện, hoặc mở từ bảng thông tin linh kiện. Checklist có bước xác nhận danh tính, ghi nhận quan sát, và hướng dẫn từ `components.json` hoặc các chi tiết của mô hình 3D khi có. Trạng thái: chưa kiểm tra / đã kiểm tra / có vấn đề / bỏ qua, kèm ghi chú riêng. Các hướng dẫn theo linh kiện vẫn là **bản nháp**, không phải quy trình sửa chữa đã được chuyên gia phê duyệt.
4. **Lưu kiểm tra**: lưu bản chụp bất biến gồm xe, thời gian, ảnh, mã lỗi, tên/điểm/box nhận diện, thông tin model, trạng thái checklist và ghi chú. Bản nháp tự lưu và được khôi phục khi mở lại app; đổi EN/VI giữ nguyên dữ liệu.
5. **Lịch sử**: lọc theo mã xe/công việc, xem lại và **Xuất báo cáo HTML** qua trình chọn nơi lưu của Android. File chứa ảnh JPEG nhúng, kết quả AI và quan sát kỹ thuật viên; không cần tài nguyên mạng. Mở bằng trình duyệt để xem/in. App không tự gửi báo cáo.

Mỗi bản ghi lưu **một ảnh khoang máy**. Thay ảnh đặt lại checklist của bản nháp; hãy lưu trước khi chụp góc khác và dùng cùng mã xe cho các bản ghi liên quan. Lần kiểm tra mới xóa bản nháp sau bước xác nhận trong app, giữ nguyên lịch sử. Các ảnh và JSON nằm trong bộ nhớ riêng `files/inspections/`; gỡ app/xóa dữ liệu sẽ mất các bản lưu chưa xuất.

Mã lỗi nhập tay qua nút DTC (chưa có quét ảnh mã lỗi bằng OCR). Manifest loại bỏ quyền INTERNET và ACCESS_NETWORK_STATE nếu dependency thêm vào. Chức năng 5–8 trong đề xuất ban đầu (hỏi đáp LLM, giọng nói, hướng dẫn chụp, model phát hiện bất thường) chưa thuộc đợt triển khai này.

Kiểm thử:

```bash
./gradlew :app:testDebugUnitTest :app:assembleDebug :app:lintDebug
./gradlew :app:assembleDebugAndroidTest
# Trên thiết bị QA riêng: Gradle connectedDebugAndroidTest có thể gỡ app sau khi chạy.
# Nếu cần giữ dữ liệu trên thiết bị đang dùng, cài cả hai APK bằng adb install -r rồi chạy:
adb shell am instrument -w com.enginebay.vision.test/androidx.test.runner.AndroidJUnitRunner
```

`InspectionTest` kiểm tra lọc mã, lưu/khôi phục, lịch sử bất biến, xử lý bản ghi lỗi và escape nội dung báo cáo. `OfflineInspectionTest` nhận diện ảnh mẫu, kiểm tra app không có quyền mạng, và lưu/đọc dữ liệu trên hệ thống file Android; dùng thư mục cache thử nghiệm riêng.

## Ghi chú vận hành

- **Quyền:** app không xin quyền CAMERA, vì ảnh được chụp bằng app camera của hệ thống. App cũng không xin quyền INTERNET.
- **Kích thước:** APK debug khoảng 115 MB (thư viện native cho `arm64-v8a`, `armeabi-v7a` cho máy 32-bit như Innova Spark, và `x86_64`; cộng model 11,6 MB). Bỏ bớt kiến trúc không dùng trong `abiFilters` để APK nhỏ hơn.
- **Kiểm thử không cần camera:** lệnh `adb shell am start -n com.enginebay.vision/.MainActivity --es sample Request_ID_23_img_004.jpg` mở thẳng một ảnh mẫu. Logcat với tag `EngineBay` ghi số linh kiện và thời gian xử lý.
- Kết quả là gợi ý để kỹ thuật viên xác nhận. Độ chính xác và giới hạn của model: `docs/releases/poc_v1.md`.
