package org.mast3r.slam.camera

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import org.mast3r.slam.net.CameraMetadata
import org.mast3r.slam.net.ScannerSession
import java.util.concurrent.Executors

@androidx.annotation.OptIn(markerClass = [ExperimentalCamera2Interop::class])
class CameraPipeline(
    private val context: Context,
    private val owner: LifecycleOwner,
    private val previewView: PreviewView,
    private val session: ScannerSession,
) {
    private val executor = Executors.newSingleThreadExecutor()
    private var provider: ProcessCameraProvider? = null
    @Volatile private var stopped = false

    fun start() {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (stopped) return@addListener
            val provider = future.get().also { this.provider = it }
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetResolution(android.util.Size(1280, 720))
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(executor) { image ->
                try { session.offerFrame(image.toBitmap(), image.imageInfo.rotationDegrees) }
                finally { image.close() }
            }
            provider.unbindAll()
            val camera = provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            val cameraId = Camera2CameraInfo.from(camera.cameraInfo).cameraId
            val manager = context.getSystemService(CameraManager::class.java)
            val logical = manager.getCameraCharacteristics(cameraId)
            val physical = logical.physicalCameraIds.firstOrNull() ?: cameraId
            val characteristics = manager.getCameraCharacteristics(physical)
            val resolution = analysis.resolutionInfo?.resolution ?: android.util.Size(1280, 720)
            val active = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                ?: android.graphics.Rect(0, 0, resolution.width, resolution.height)
            val outputRatio = resolution.width.toFloat() / resolution.height
            val sensorRatio = active.width().toFloat() / active.height()
            val cropWidth: Int; val cropHeight: Int
            if (sensorRatio > outputRatio) {
                cropHeight = active.height(); cropWidth = (cropHeight * outputRatio).toInt()
            } else {
                cropWidth = active.width(); cropHeight = (cropWidth / outputRatio).toInt()
            }
            val cropLeft = active.left + (active.width() - cropWidth) / 2
            val cropTop = active.top + (active.height() - cropHeight) / 2
            val rawIntrinsic = characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
            val intrinsic = rawIntrinsic?.let {
                val sx = resolution.width.toFloat() / cropWidth
                val sy = resolution.height.toFloat() / cropHeight
                listOf(it[0] * sx, it[1] * sy, (it[2] - cropLeft) * sx, (it[3] - cropTop) * sy)
            } ?: emptyList()
            val distortion = characteristics.get(CameraCharacteristics.LENS_DISTORTION)?.toList() ?: emptyList()
            session.updateCamera(
                CameraMetadata(
                    Build.MODEL, physical, resolution.width, resolution.height,
                    cropLeft, cropTop, cropWidth, cropHeight, intrinsic.take(4), distortion,
                )
            )
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        stopped = true
        provider?.unbindAll()
        executor.shutdown()
    }
}
