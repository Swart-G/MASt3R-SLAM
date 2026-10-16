package org.mast3r.slam.ar

import android.app.Activity
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.widget.Toast
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import org.mast3r.slam.net.SnapshotCache
import org.mast3r.slam.net.ArAlignmentBridge

class ArActivity : Activity() {
    private var session: Session? = null
    private lateinit var surface: GLSurfaceView
    private lateinit var renderer: MeshRenderer
    private var installRequested = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (ArCoreApk.getInstance().checkAvailability(this).isUnsupported) {
            Toast.makeText(this, "ARCore is not supported on this device", Toast.LENGTH_LONG).show()
            finish(); return
        }
    }

    private fun createSessionAndView(): Boolean {
        if (session != null) return true
        try {
            when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> { installRequested = true; return false }
                ArCoreApk.InstallStatus.INSTALLED -> Unit
            }
            session = Session(this).also { arSession ->
                val config = Config(arSession)
                if (arSession.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) config.depthMode = Config.DepthMode.AUTOMATIC
                config.planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
                config.instantPlacementMode = Config.InstantPlacementMode.LOCAL_Y_UP
                arSession.configure(config)
            }
        } catch (error: Exception) {
            Toast.makeText(this, "Install or update Google Play Services for AR", Toast.LENGTH_LONG).show()
            finish(); return false
        }
        surface = GLSurfaceView(this).apply {
            setEGLContextClientVersion(3)
            preserveEGLContextOnPause = true
        }
        renderer = MeshRenderer(session!!, SnapshotCache) {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.rotation
        }
        surface.setRenderer(renderer)
        surface.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        setContentView(surface)
        Toast.makeText(this, ArAlignmentBridge.guidance, Toast.LENGTH_LONG).show()
        return true
    }

    override fun onResume() {
        super.onResume()
        if (!createSessionAndView()) return
        session?.resume(); surface.onResume()
    }
    override fun onPause() {
        if (::surface.isInitialized) surface.onPause()
        session?.pause(); super.onPause()
    }
    override fun onDestroy() {
        ArAlignmentBridge.leaveAr?.invoke()
        session?.close()
        super.onDestroy()
    }
}
