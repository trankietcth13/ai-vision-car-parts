plugins {
    alias(libs.plugins.android.application)  // AGP 9 compiles Kotlin itself (built-in Kotlin)
}

android {
    namespace = "com.enginebay.vision"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.enginebay.vision"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0-poc-v1"
        // ONNX Runtime ships native code for 4 ABIs: arm64 phones, 32-bit ARM devices (Innova Spark tablet), x86_64 emulator
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64") }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources {
        noCompress += "onnx"  // memory-map friendly, and the model is already dense
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.onnxruntime.android)
    testImplementation(libs.junit)
    testImplementation(libs.json)
}
