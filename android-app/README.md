# MASt3R Scanner for Android 16

Open this directory as an Android Studio project. It requires JDK 17 and Android
SDK 36. The checked-in wrapper provides Gradle 8.13 (AGP 8.11.1):

```bash
./gradlew :app:assembleDebug
```

The Protobuf plugin reads the canonical schema directly from `../protocol`; do
not copy or edit a second Android-specific schema. Protocol v2 requires this
APK to be rebuilt and installed together with the updated Linux server.

Start the Linux TUI, choose **New**, then pair from the app's built-in QR
scanner. The manual connection screen accepts the complete
`mast3r-slam://pair?...` URI as a fallback. Camera permission is requested only
when opening the QR scanner or starting/resuming capture.
