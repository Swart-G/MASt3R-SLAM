package org.mast3r.slam.ar

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.opengl.GLES30
import android.opengl.GLES11Ext
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.util.Log
import com.github.luben.zstd.Zstd
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import org.mast3r.slam.net.ArAlignmentBridge
import org.mast3r.slam.net.SnapshotCache
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sqrt

class MeshRenderer(
    private val session: Session,
    private val cache: SnapshotCache,
    private val displayRotation: () -> Int,
) : GLSurfaceView.Renderer {
    private var program = 0
    private var backgroundProgram = 0
    private var cameraTexture = 0
    private var meshBuffer: FloatBuffer? = null
    private var vertexCount = 0
    private var width = 1
    private var height = 1
    private var lastAlignmentFrameNs = 0L

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0.04f, 0.05f, 0.06f, 1f)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glDisable(GLES30.GL_CULL_FACE)
        program = createProgram(VERTEX, FRAGMENT)
        backgroundProgram = createProgram(BACKGROUND_VERTEX, BACKGROUND_FRAGMENT)
        val textures = IntArray(1)
        GLES30.glGenTextures(1, textures, 0); cameraTexture = textures[0]
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        session.setCameraTextureName(cameraTexture)
        meshBuffer = decodeMesh().also {
            vertexCount = it.remaining() / 6
            Log.i(TAG, "Loaded translucent mesh with $vertexCount vertices")
        }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        this.width = width; this.height = height
        GLES30.glViewport(0, 0, width, height)
        session.setDisplayGeometry(displayRotation(), width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        session.setDisplayGeometry(displayRotation(), width, height)
        val frame = try { session.update() } catch (_: Exception) { return }
        drawCamera(frame)
        GLES30.glClear(GLES30.GL_DEPTH_BUFFER_BIT)
        if (frame.camera.trackingState != TrackingState.TRACKING) {
            ArAlignmentBridge.trackingLost()
            return
        }
        ArAlignmentBridge.trackingRestored()
        if (ArAlignmentBridge.status == "SEARCHING") offerAlignmentFrame(frame)
        val model = ArAlignmentBridge.modelMatrix?.copyOf() ?: return
        val projection = FloatArray(16); frame.camera.getProjectionMatrix(projection, 0, 0.1f, 100f)
        val view = FloatArray(16); frame.camera.getViewMatrix(view, 0)
        val pv = FloatArray(16); val mvp = FloatArray(16)
        Matrix.multiplyMM(pv, 0, projection, 0, view, 0)
        Matrix.multiplyMM(mvp, 0, pv, 0, model, 0)
        val buffer = meshBuffer ?: return
        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(program, "uMvp"), 1, false, mvp, 0)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(program, "uModel"), 1, false, model, 0)
        val position = GLES30.glGetAttribLocation(program, "aPosition")
        val normal = GLES30.glGetAttribLocation(program, "aNormal")
        buffer.position(0); GLES30.glEnableVertexAttribArray(position)
        GLES30.glVertexAttribPointer(position, 3, GLES30.GL_FLOAT, false, 24, buffer)
        buffer.position(3); GLES30.glEnableVertexAttribArray(normal)
        GLES30.glVertexAttribPointer(normal, 3, GLES30.GL_FLOAT, false, 24, buffer)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, vertexCount)
    }

    private fun decodeMesh(): FloatBuffer {
        val chunks = cache.compressedChunks
        val manifest = cache.manifest
        require(manifest != null && manifest.encoding.startsWith("m3mesh-v1")) { "AR snapshot is not a mesh" }
        var total = 0
        val decoded = chunks.map { compressed ->
            val size = Zstd.decompressedSize(compressed).toInt()
            val raw = Zstd.decompress(compressed, size)
            val input = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(5); input.get(magic)
            require(String(magic, Charsets.US_ASCII) == "M3MS1")
            val triangleCount = input.int
            val count = triangleCount * 3
            total += count
            val low = FloatArray(3) { input.float }; val high = FloatArray(3) { input.float }
            val xyz = Array(count) { FloatArray(3) }
            repeat(count) { i -> repeat(3) { axis ->
                val q = input.short.toInt() and 0xffff
                xyz[i][axis] = low[axis] + q / 65535f * (high[axis] - low[axis])
            } }
            val values = FloatArray(count * 6)
            repeat(triangleCount) { triangle ->
                val base = triangle * 3
                val p0 = xyz[base]
                val p1 = xyz[base + 1]
                val p2 = xyz[base + 2]
                val ux = p1[0] - p0[0]; val uy = p1[1] - p0[1]; val uz = p1[2] - p0[2]
                val vx = p2[0] - p0[0]; val vy = p2[1] - p0[1]; val vz = p2[2] - p0[2]
                var nx = uy * vz - uz * vy
                var ny = uz * vx - ux * vz
                var nz = ux * vy - uy * vx
                val length = sqrt(nx * nx + ny * ny + nz * nz).coerceAtLeast(1e-8f)
                nx /= length; ny /= length; nz /= length
                arrayOf(p0, p1, p2).forEachIndexed { vertex, point ->
                    val offset = (base + vertex) * 6
                    values[offset] = point[0]; values[offset + 1] = point[1]; values[offset + 2] = point[2]
                    values[offset + 3] = nx; values[offset + 4] = ny; values[offset + 5] = nz
                }
            }
            values
        }
        val output = ByteBuffer.allocateDirect(total * 6 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        decoded.forEach { output.put(it) }; output.flip(); return output
    }

    private fun offerAlignmentFrame(frame: com.google.ar.core.Frame) {
        val now = System.nanoTime()
        if (now - lastAlignmentFrameNs < 333_000_000L) return
        val sender = ArAlignmentBridge.sendFrame ?: return
        val image = try { frame.acquireCameraImage() } catch (_: Exception) { return }
        try {
            lastAlignmentFrameNs = now
            val intrinsics = frame.camera.imageIntrinsics
            val focal = intrinsics.focalLength
            val principal = intrinsics.principalPoint
            val dimensions = intrinsics.imageDimensions
            val pose = FloatArray(16).also { frame.camera.pose.toMatrix(it, 0) }
            sender(
                image.toJpeg(75),
                image.timestamp,
                floatArrayOf(focal[0], focal[1], principal[0], principal[1]),
                pose,
                dimensions[0],
                dimensions[1],
            )
        } finally {
            image.close()
        }
    }

    private fun drawCamera(frame: com.google.ar.core.Frame) {
        val ndc = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        val uv = FloatArray(8)
        frame.transformCoordinates2d(
            Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
            FloatBuffer.wrap(ndc),
            Coordinates2d.TEXTURE_NORMALIZED,
            FloatBuffer.wrap(uv),
        )
        val vertices = ByteBuffer.allocateDirect(16 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        repeat(4) { index ->
            vertices.put(ndc[index * 2]); vertices.put(ndc[index * 2 + 1])
            vertices.put(uv[index * 2]); vertices.put(uv[index * 2 + 1])
        }
        vertices.flip()
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glUseProgram(backgroundProgram)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture)
        val position = GLES30.glGetAttribLocation(backgroundProgram, "aPosition")
        val texture = GLES30.glGetAttribLocation(backgroundProgram, "aTexCoord")
        vertices.position(0); GLES30.glEnableVertexAttribArray(position)
        GLES30.glVertexAttribPointer(position, 2, GLES30.GL_FLOAT, false, 16, vertices)
        vertices.position(2); GLES30.glEnableVertexAttribArray(texture)
        GLES30.glVertexAttribPointer(texture, 2, GLES30.GL_FLOAT, false, 16, vertices)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
    }

    private fun createProgram(vertex: String, fragment: String): Int {
        fun shader(type: Int, source: String): Int = GLES30.glCreateShader(type).also {
            GLES30.glShaderSource(it, source); GLES30.glCompileShader(it)
        }
        return GLES30.glCreateProgram().also {
            GLES30.glAttachShader(it, shader(GLES30.GL_VERTEX_SHADER, vertex))
            GLES30.glAttachShader(it, shader(GLES30.GL_FRAGMENT_SHADER, fragment))
            GLES30.glLinkProgram(it)
        }
    }

    companion object {
        private const val TAG = "Mast3rMeshRenderer"
        private const val VERTEX = """#version 300 es
            uniform mat4 uMvp; uniform mat4 uModel; in vec3 aPosition; in vec3 aNormal; out float vLight;
            void main(){
                gl_Position=uMvp*vec4(aPosition,1.0);
                vec3 normal=normalize(mat3(uModel)*aNormal);
                vLight=0.38+0.62*abs(dot(normal,normalize(vec3(0.35,0.8,0.45))));
            }
        """
        private const val FRAGMENT = """#version 300 es
            precision mediump float; in float vLight; out vec4 outColor;
            void main(){ outColor=vec4(vec3(0.72)*vLight,0.42); }
        """
        private const val BACKGROUND_VERTEX = """#version 300 es
            in vec2 aPosition; in vec2 aTexCoord; out vec2 vTexCoord;
            void main(){ gl_Position=vec4(aPosition,0.9999,1.0); vTexCoord=aTexCoord; }
        """
        private const val BACKGROUND_FRAGMENT = """#version 300 es
            #extension GL_OES_EGL_image_external_essl3 : require
            precision mediump float; uniform samplerExternalOES cameraTexture;
            in vec2 vTexCoord; out vec4 outColor;
            void main(){ outColor=texture(cameraTexture,vTexCoord); }
        """
    }
}

private fun Image.toJpeg(quality: Int): ByteArray {
    require(format == ImageFormat.YUV_420_888)
    val output = ByteArray(width * height * 3 / 2)
    val yPlane = planes[0]
    val yStart = yPlane.buffer.position()
    var offset = 0
    repeat(height) { row ->
        repeat(width) { column ->
            output[offset++] = yPlane.buffer.get(yStart + row * yPlane.rowStride + column * yPlane.pixelStride)
        }
    }
    val uPlane = planes[1]
    val vPlane = planes[2]
    val uStart = uPlane.buffer.position()
    val vStart = vPlane.buffer.position()
    repeat(height / 2) { row ->
        repeat(width / 2) { column ->
            output[offset++] = vPlane.buffer.get(vStart + row * vPlane.rowStride + column * vPlane.pixelStride)
            output[offset++] = uPlane.buffer.get(uStart + row * uPlane.rowStride + column * uPlane.pixelStride)
        }
    }
    return ByteArrayOutputStream().use { stream ->
        YuvImage(output, ImageFormat.NV21, width, height, null)
            .compressToJpeg(Rect(0, 0, width, height), quality, stream)
        stream.toByteArray()
    }
}

object MeshCoordinates {
    fun opencvToAr(point: FloatArray, centerX: Float, floorY: Float, centerZ: Float): FloatArray =
        floatArrayOf(point[0] - centerX, floorY - point[1], centerZ - point[2])
}
