package org.mast3r.slam.net

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import okhttp3.*
import okhttp3.HttpUrl.Companion.toHttpUrl
import okio.ByteString
import org.mast3r.slam.protocol.*
import java.io.ByteArrayOutputStream
import java.security.MessageDigest
import java.security.SecureRandom
import java.security.cert.X509Certificate
import java.util.UUID
import java.util.concurrent.TimeUnit
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager

data class ScanUiState(
    val connected: Boolean = false,
    val connection: String = "DISCONNECTED",
    val phase: String = "PAIRING",
    val tracking: String = "INITIALIZING",
    val latencyMs: Int = 0,
    val liveFrame: Long = 0,
    val processedFrame: Long = 0,
    val guidance: String = "",
    val preview: Bitmap? = null,
    val cloudReady: Boolean = false,
    val cameraMismatch: Boolean = false,
    val cameraActive: Boolean = false,
    val saveProgress: Float = 0f,
    val operation: String = "",
    val alignmentStatus: String = "SEARCHING",
    val alignmentGuidance: String = "",
    val exportError: String = "",
)

data class CameraMetadata(
    val model: String,
    val cameraId: String,
    val width: Int,
    val height: Int,
    val cropLeft: Int,
    val cropTop: Int,
    val cropWidth: Int,
    val cropHeight: Int,
    val intrinsics: List<Float>,
    val distortion: List<Float>,
)

private data class ArFramePayload(
    val jpeg: ByteArray,
    val timestampNs: Long,
    val intrinsics: FloatArray,
    val arWorldFromCamera: FloatArray,
    val width: Int,
    val height: Int,
)

class ScannerSession(private val context: Context) : WebSocketListener() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val mutableState = MutableStateFlow(ScanUiState())
    val state: StateFlow<ScanUiState> = mutableState
    private var socket: WebSocket? = null
    private var endpoint: HttpUrl? = null
    private var token = ""
    private var fingerprint = ""
    private var client: OkHttpClient? = null
    private var camera: CameraMetadata? = null
    private var pendingCameraCommand: CommandType? = null
    private var frameId = 0L
    @Volatile private var quality = 80
    @Volatile private var targetFps = 10
    private var lastSentNs = 0L
    private val alignmentFrames = Channel<ArFramePayload>(Channel.CONFLATED)

    init {
        ArAlignmentBridge.sendFrame = ::offerArAlignmentFrame
        ArAlignmentBridge.leaveAr = { command(CommandType.LEAVE_AR) }
        ArAlignmentBridge.stateChanged = { status, detail ->
            mutableState.update { it.copy(alignmentStatus = status, alignmentGuidance = detail) }
        }
        scope.launch {
            for (frame in alignmentFrames) sendArAlignmentFrame(frame)
        }
    }

    @Synchronized
    fun connect(pairing: String): Boolean {
        val uri = Uri.parse(pairing.trim())
        val validScheme = uri.scheme == "mast3r-slam" && uri.host == "pair"
        val host = uri.getQueryParameter("host")
        val port = uri.getQueryParameter("port")?.toIntOrNull() ?: 8765
        val nextToken = uri.getQueryParameter("token")
        val nextFingerprint = uri.getQueryParameter("fingerprint")?.lowercase()
        if (
            !validScheme || host.isNullOrBlank() || port !in 1..65535 || nextToken.isNullOrBlank() ||
            nextFingerprint == null || !nextFingerprint.matches(Regex("[0-9a-f]{64}"))
        ) {
            mutableState.update {
                it.copy(guidance = "Enter or scan a valid MASt3R pairing URI.", connection = "DISCONNECTED")
            }
            return false
        }
        val nextEndpoint = "https://$host:$port/".toHttpUrl()
        if (
            socket != null && state.value.connected && endpoint == nextEndpoint &&
            token == nextToken && fingerprint == nextFingerprint
        ) {
            mutableState.update { it.copy(guidance = "") }
            return true
        }
        val previousSocket = socket
        socket = null
        previousSocket?.cancel()
        client?.dispatcher?.cancelAll()
        token = nextToken
        fingerprint = nextFingerprint
        endpoint = nextEndpoint
        client = pinnedClient(fingerprint)
        val request = Request.Builder().url(endpoint!!.newBuilder().addPathSegment("ws").build()).build()
        mutableState.update { it.copy(connected = false, connection = "CONNECTING", guidance = "") }
        socket = client!!.newWebSocket(request, this)
        return true
    }

    fun cancelConnection() {
        val current = socket
        socket = null
        current?.cancel()
        client?.dispatcher?.cancelAll()
        mutableState.update {
            it.copy(connected = false, connection = "DISCONNECTED", guidance = "", cameraActive = false)
        }
    }

    override fun onOpen(webSocket: WebSocket, response: Response) {
        if (socket !== webSocket) {
            webSocket.cancel()
            return
        }
        send(
            ClientMessage.newBuilder().setHello(
                Hello.newBuilder().setProtocolVersion(2).setToken(token)
                    .setClientName("Android").setDeviceId(android.os.Build.FINGERPRINT)
                    .setAppVersion("0.1.0")
            ).build()
        )
        camera?.let(::sendCameraConfig)
        mutableState.update { it.copy(connected = false, connection = "AUTHENTICATING") }
        scope.launch {
            while (isActive && socket === webSocket) {
                delay(750)
                send(ClientMessage.newBuilder().setHeartbeat(Heartbeat.newBuilder().setMonotonicNs(System.nanoTime())).build())
            }
        }
    }

    override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
        if (socket !== webSocket) return
        val message = try { ServerMessage.parseFrom(bytes.toByteArray()) } catch (_: Exception) { return }
        when (message.payloadCase) {
            ServerMessage.PayloadCase.SESSION_STATE -> {
                val phase = message.sessionState.phase.name
                mutableState.update {
                    it.copy(
                        connected = true,
                        phase = phase,
                        connection = "CONNECTED",
                        cameraActive = when (phase) {
                            "SCANNING" -> true
                            "FINALIZING", "SAVED", "ERROR", "DISCONNECTED" -> false
                            else -> it.cameraActive
                        },
                    )
                }
            }
            ServerMessage.PayloadCase.PREVIEW -> {
                val item = message.preview
                val bitmap = if (item.jpeg.isEmpty) null else BitmapFactory.decodeByteArray(item.jpeg.toByteArray(), 0, item.jpeg.size())
                adapt(item.latencyMs, item.latencyMs >= 900)
                mutableState.update {
                    it.copy(
                        processedFrame = item.processedFrameId,
                        tracking = item.tracking.name,
                        latencyMs = item.latencyMs,
                        guidance = item.guidance,
                        preview = bitmap ?: it.preview,
                    )
                }
            }
            ServerMessage.PayloadCase.FRAME_ACK -> adapt(mutableState.value.latencyMs, !message.frameAck.accepted)
            ServerMessage.PayloadCase.CLOUD_MANIFEST -> scope.launch { downloadCloud(message.cloudManifest) }
            ServerMessage.PayloadCase.PROGRESS -> mutableState.update {
                it.copy(
                    guidance = message.progress.detail,
                    operation = message.progress.operation,
                    saveProgress = if (message.progress.operation == "save") message.progress.fraction else it.saveProgress,
                )
            }
            ServerMessage.PayloadCase.AR_ALIGNMENT_RESULT -> {
                val result = message.arAlignmentResult
                ArAlignmentBridge.update(
                    result.status.name,
                    result.arWorldFromSlamMapList.toFloatArray(),
                    result.guidance,
                )
                mutableState.update {
                    it.copy(alignmentStatus = result.status.name, alignmentGuidance = result.guidance)
                }
            }
            ServerMessage.PayloadCase.ERROR -> {
                val error = message.error
                Log.e(TAG, "Server error ${error.code}: ${error.message}")
                mutableState.update {
                    it.copy(guidance = error.message, cameraMismatch = error.code == "CAMERA_MISMATCH")
                }
            }
            else -> Unit
        }
    }

    override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
        if (socket !== webSocket) return
        socket = null
        val detail = "WebSocket closed ($code)${reason.takeIf(String::isNotBlank)?.let { ": $it" } ?: ""}"
        Log.w(TAG, detail)
        disconnected(detail)
    }

    override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
        if (socket !== webSocket) return
        socket = null
        val detail = buildString {
            append(t::class.java.simpleName)
            t.message?.takeIf(String::isNotBlank)?.let { append(": ").append(it) }
            response?.let { append(" (HTTP ").append(it.code).append(')') }
        }
        Log.e(TAG, "WebSocket failure: $detail", t)
        disconnected(detail)
    }

    fun updateCamera(metadata: CameraMetadata) {
        camera = metadata
        if (state.value.connected) {
            sendCameraConfig(metadata)
            pendingCameraCommand?.let { type ->
                pendingCameraCommand = null
                command(type)
            }
        }
    }

    fun offerFrame(bitmap: Bitmap, rotationDegrees: Int) {
        if (state.value.phase != "SCANNING") return
        val now = System.nanoTime()
        if (now - lastSentNs < 1_000_000_000L / targetFps) return
        lastSentNs = now
        val metadata = camera ?: return
        val scaled = if (bitmap.width > 1280 || bitmap.height > 720) {
            val factor = minOf(1280f / bitmap.width, 720f / bitmap.height)
            Bitmap.createScaledBitmap(bitmap, (bitmap.width * factor).toInt(), (bitmap.height * factor).toInt(), true)
        } else bitmap
        val output = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, quality, output)
        val id = ++frameId
        val packet = FramePacket.newBuilder()
            .setFrameId(id).setTimestampNs(now).setOrientationDegrees(rotationDegrees)
            .setPhysicalCameraId(metadata.cameraId).setWidth(scaled.width).setHeight(scaled.height)
            .setCropLeft(metadata.cropLeft).setCropTop(metadata.cropTop)
            .setCropWidth(metadata.cropWidth).setCropHeight(metadata.cropHeight)
            .addAllIntrinsics(metadata.intrinsics).addAllDistortion(metadata.distortion)
            .setJpeg(com.google.protobuf.ByteString.copyFrom(output.toByteArray())).build()
        send(ClientMessage.newBuilder().setFrame(packet).build())
        mutableState.update { it.copy(liveFrame = id) }
    }

    fun prepareStart() = prepareCameraCommand(CommandType.START)
    fun pause() = command(CommandType.PAUSE)
    fun prepareResume() = prepareCameraCommand(CommandType.RESUME)
    fun confirmResume() = command(CommandType.RESUME, cameraChangeConfirmed = true)
    fun enterAr() {
        ArAlignmentBridge.reset()
        command(CommandType.ENTER_AR)
    }
    fun stop() = command(CommandType.STOP)
    fun pauseForBackground() { if (state.value.phase == "SCANNING") pause() }

    private fun offerArAlignmentFrame(
        jpeg: ByteArray,
        timestampNs: Long,
        intrinsics: FloatArray,
        arWorldFromCamera: FloatArray,
        width: Int,
        height: Int,
    ) {
        alignmentFrames.trySend(
            ArFramePayload(jpeg, timestampNs, intrinsics, arWorldFromCamera, width, height)
        )
    }

    private fun sendArAlignmentFrame(frame: ArFramePayload) {
        val id = ++frameId
        val packet = ArAlignmentFrame.newBuilder()
            .setFrameId(id).setTimestampNs(frame.timestampNs)
            .setJpeg(com.google.protobuf.ByteString.copyFrom(frame.jpeg))
            .addAllIntrinsics(frame.intrinsics.asList())
            .addAllArWorldFromCamera(frame.arWorldFromCamera.asList())
            .setWidth(frame.width).setHeight(frame.height).build()
        send(ClientMessage.newBuilder().setArAlignmentFrame(packet).build())
    }

    private fun prepareCameraCommand(type: CommandType) {
        pendingCameraCommand = type
        camera = null
        mutableState.update { it.copy(cameraActive = true, guidance = "") }
    }

    private fun command(type: CommandType, cameraChangeConfirmed: Boolean = false) {
        send(
            ClientMessage.newBuilder().setCommand(
                Command.newBuilder().setCommandId(UUID.randomUUID().toString()).setType(type)
                    .setCameraChangeConfirmed(cameraChangeConfirmed)
            ).build()
        )
        if (cameraChangeConfirmed) mutableState.update { it.copy(cameraMismatch = false) }
    }

    private fun sendCameraConfig(item: CameraMetadata) {
        send(
            ClientMessage.newBuilder().setCameraConfig(
                CameraConfig.newBuilder().setDeviceModel(item.model).setPhysicalCameraId(item.cameraId)
                    .setWidth(item.width).setHeight(item.height)
                    .setCropLeft(item.cropLeft).setCropTop(item.cropTop)
                    .setCropWidth(item.cropWidth).setCropHeight(item.cropHeight)
                    .addAllIntrinsics(item.intrinsics).addAllDistortion(item.distortion)
            ).build()
        )
    }

    private fun send(message: ClientMessage) { socket?.send(ByteString.of(*message.toByteArray())) }

    private fun adapt(latency: Int, dropped: Boolean) {
        if (dropped || latency >= 900) {
            if (targetFps > 3) targetFps-- else quality = (quality - 5).coerceAtLeast(50)
        } else if (latency in 1..350) {
            if (quality < 90) quality = (quality + 2).coerceAtMost(90) else targetFps = (targetFps + 1).coerceAtMost(15)
        }
    }

    private suspend fun downloadCloud(manifest: CloudManifest) {
        val http = client ?: return
        val base = endpoint ?: return
        try {
            val chunks = manifest.chunksList.map { item ->
                var verified: ByteArray? = null
                repeat(3) {
                    if (verified == null) {
                        val request = Request.Builder().url(base.resolve(item.url)!!)
                            .header("Authorization", "Bearer $token").header("Range", "bytes=0-").build()
                        val candidate = http.newCall(request).execute().use { response ->
                            if (!response.isSuccessful) error("chunk ${item.index} download failed")
                            response.body!!.bytes()
                        }
                        val digest = MessageDigest.getInstance("SHA-256").digest(candidate).joinToString("") { "%02x".format(it) }
                        if (digest == item.sha256) verified = candidate
                    }
                }
                requireNotNull(verified) { "chunk ${item.index} checksum failed after retry" }
            }
            SnapshotCache.install(manifest, chunks)
            mutableState.update { it.copy(cloudReady = true) }
        } catch (error: Exception) {
            mutableState.update { it.copy(guidance = error.message ?: "Snapshot download failed") }
        }
    }

    private fun disconnected(detail: String) {
        pendingCameraCommand = null
        mutableState.update {
            it.copy(
                connected = false,
                connection = "DISCONNECTED",
                phase = "DISCONNECTED",
                guidance = detail,
                cameraActive = false,
            )
        }
    }

    private fun pinnedClient(expectedHex: String): OkHttpClient {
        val trust = object : X509TrustManager {
            override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
            override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) = Unit
            override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) {
                require(chain.isNotEmpty()) { "server did not provide a certificate" }
                val actual = MessageDigest.getInstance("SHA-256").digest(chain[0].encoded).joinToString("") { "%02x".format(it) }
                require(actual.equals(expectedHex, ignoreCase = true)) { "TLS certificate fingerprint mismatch" }
            }
        }
        val ssl = SSLContext.getInstance("TLS").apply { init(null, arrayOf<TrustManager>(trust), SecureRandom()) }
        return OkHttpClient.Builder().sslSocketFactory(ssl.socketFactory, trust).hostnameVerifier { _, _ -> true }
            .pingInterval(30, TimeUnit.SECONDS).build()
    }

    private companion object {
        const val TAG = "Mast3rScannerSession"
    }
}

object SnapshotCache {
    @Volatile var manifest: CloudManifest? = null
        private set
    @Volatile var compressedChunks: List<ByteArray> = emptyList()
        private set
    fun install(value: CloudManifest, chunks: List<ByteArray>) { manifest = value; compressedChunks = chunks }
    fun clear() { manifest = null; compressedChunks = emptyList() }
}

object ArAlignmentBridge {
    @Volatile var status: String = "SEARCHING"
        private set
    @Volatile var modelMatrix: FloatArray? = null
        private set
    @Volatile var guidance: String = "Наведите камеру на уже отсканированную область"
        private set
    @Volatile var sendFrame: ((ByteArray, Long, FloatArray, FloatArray, Int, Int) -> Unit)? = null
    @Volatile var leaveAr: (() -> Unit)? = null
    @Volatile var stateChanged: ((String, String) -> Unit)? = null

    @Synchronized fun reset() {
        status = "SEARCHING"
        modelMatrix = null
        guidance = "Наведите камеру на уже отсканированную область"
        stateChanged?.invoke(status, guidance)
    }

    @Synchronized fun update(next: String, matrix: FloatArray, detail: String) {
        if (status == "ALIGNED" || status == "ALIGNMENT_LOST") return
        status = next
        modelMatrix = matrix.takeIf { next == "ALIGNED" && it.size == 16 }?.copyOf()
        guidance = detail
        stateChanged?.invoke(status, guidance)
    }

    @Synchronized fun trackingLost() {
        if (status == "ALIGNMENT_LOST") return
        status = "ALIGNMENT_LOST"
        modelMatrix = null
        guidance = "ARCore потерял позицию. Верните камеру к отсканированной области"
        stateChanged?.invoke(status, guidance)
    }

    @Synchronized fun trackingRestored() {
        if (status == "ALIGNMENT_LOST") reset()
    }
}
