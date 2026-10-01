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
└── MainActivity.kt       chụp (app camera hệ thống), thư viện, ảnh mẫu, danh sách, bảng thông tin
app/src/main/assets/
├── components.json       thông tin linh kiện (trong git; đặc tả: docs/COMPONENT_INFO_SPEC.md)
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

## Ghi chú

- **Quyền:** app không xin quyền CAMERA, vì ảnh được chụp bằng app camera của hệ thống. App cũng không xin quyền INTERNET.
- **Kích thước:** APK debug khoảng 115 MB (thư viện native cho `arm64-v8a`, `armeabi-v7a` cho máy 32-bit như Innova Spark, và `x86_64`; cộng model 11,6 MB). Bỏ bớt kiến trúc không dùng trong `abiFilters` để APK nhỏ hơn.
- **Kiểm thử không cần camera:** lệnh `adb shell am start -n com.enginebay.vision/.MainActivity --es sample Request_ID_23_img_004.jpg` mở thẳng một ảnh mẫu. Logcat với tag `EngineBay` ghi số linh kiện và thời gian xử lý.
- Kết quả là gợi ý để kỹ thuật viên xác nhận. Độ chính xác và giới hạn của model: `docs/releases/poc_v1.md`.
