package org.mast3r.slam

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import org.mast3r.slam.ar.ArActivity
import org.mast3r.slam.camera.CameraPipeline
import org.mast3r.slam.net.ScanUiState
import org.mast3r.slam.net.ScannerSession

private val AppBackground = Color(0xFF080D12)
private val Panel = Color(0xFF0D171D)
private val PanelRaised = Color(0xFF14242C)
private val Teal = Color(0xFF72E0CE)
private val TealDark = Color(0xFF174F4A)
private val Muted = Color(0xFF9CB7B3)
private val Warning = Color(0xFFFFC857)
private val Danger = Color(0xFFFF6B6B)

private val ScannerColors = darkColorScheme(
    primary = Teal,
    onPrimary = Color(0xFF06110F),
    secondary = Color(0xFF8DBCB5),
    background = AppBackground,
    surface = Panel,
    surfaceVariant = PanelRaised,
    onSurface = Color(0xFFE7F4F3),
    onSurfaceVariant = Muted,
    error = Danger,
)

class MainActivity : ComponentActivity() {
    private val session by lazy { ScannerSession(applicationContext) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        intent?.data?.toString()?.let(session::connect)
        lifecycle.addObserver(object : DefaultLifecycleObserver {
            override fun onStop(owner: LifecycleOwner) { session.pauseForBackground() }
        })
        setContent {
            MaterialTheme(colorScheme = ScannerColors) {
                Surface(Modifier.fillMaxSize(), color = AppBackground) { ScanApp(session) }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        intent.data?.toString()?.let(session::connect)
    }
}

private enum class ConnectPage { HOME, MANUAL, QR }
private enum class CameraRequest { QR, START, RESUME }

@Composable
internal fun ScanApp(session: ScannerSession) {
    val state by session.state.collectAsState()
    val context = LocalContext.current
    var connectPage by remember { mutableStateOf(ConnectPage.HOME) }
    var pairing by remember { mutableStateOf("") }
    var requestedCamera by remember { mutableStateOf<CameraRequest?>(null) }
    var permissionDenied by remember { mutableStateOf(false) }
    var stopConfirmation by remember { mutableStateOf(false) }
    var launchedAr by remember { mutableStateOf(false) }

    fun runCameraAction(action: CameraRequest) {
        when (action) {
            CameraRequest.QR -> connectPage = ConnectPage.QR
            CameraRequest.START -> session.prepareStart()
            CameraRequest.RESUME -> session.prepareResume()
        }
    }

    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        requestedCamera?.let { action ->
            if (granted) runCameraAction(action)
            else permissionDenied = true
        }
        requestedCamera = null
    }
    fun withCamera(action: CameraRequest) {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            runCameraAction(action)
        } else {
            requestedCamera = action
            permission.launch(Manifest.permission.CAMERA)
        }
    }

    val destination = when {
        !state.connected && state.connection in setOf("CONNECTING", "AUTHENTICATING") -> "connecting"
        !state.connected -> "connect-${connectPage.name}"
        state.phase == "READY" && !state.cameraActive -> "ready"
        state.phase in setOf("READY", "SCANNING", "PAUSED", "AR_VIEW") -> "workspace"
        state.phase == "FINALIZING" || state.phase == "AR_PREPARING" -> "working"
        state.phase == "SAVED" -> "saved"
        state.phase == "ERROR" -> "error"
        else -> "ready"
    }

    AnimatedContent(
        targetState = destination,
        transitionSpec = { fadeIn(tween(180)) togetherWith fadeOut(tween(140)) },
        label = "scanner navigation",
    ) { screen ->
        when {
            screen == "connecting" -> CenterMessage(
                icon = { CircularProgressIndicator(Modifier.size(34.dp), strokeWidth = 3.dp) },
                title = if (state.connection == "AUTHENTICATING") "Securing connection" else "Connecting",
                message = "Verifying the Linux scanner and pairing credentials…",
                action = {
                    TextButton(onClick = { session.cancelConnection(); connectPage = ConnectPage.HOME }) {
                        Text("Cancel")
                    }
                },
            )
            screen == "connect-QR" -> QrScannerScreen(
                onBack = { connectPage = ConnectPage.HOME },
                onDetected = { value ->
                    session.connect(value).also { valid -> if (valid) connectPage = ConnectPage.HOME }
                },
            )
            screen == "connect-MANUAL" -> ManualConnection(
                pairing = pairing,
                guidance = state.guidance,
                onPairingChange = { pairing = it },
                onBack = { connectPage = ConnectPage.HOME },
                onConnect = { session.connect(pairing) },
            )
            screen.startsWith("connect-") -> ConnectionHome(
                guidance = state.guidance,
                onQr = { withCamera(CameraRequest.QR) },
                onManual = { connectPage = ConnectPage.MANUAL },
            )
            screen == "ready" -> ReadyScreen(
                state,
                primaryLabel = "Start scanning",
                onStart = { withCamera(CameraRequest.START) },
            )
            screen == "workspace" -> ScanWorkspace(
                state = state,
                session = session,
                onPause = session::pause,
                onResume = { withCamera(CameraRequest.RESUME) },
                onAr = session::enterAr,
                onStop = { stopConfirmation = true },
            )
            screen == "working" -> CenterMessage(
                icon = { CircularProgressIndicator(Modifier.size(34.dp), strokeWidth = 3.dp) },
                title = if (state.phase == "FINALIZING") "Saving project" else "Preparing AR view",
                message = "This can take a moment. Keep the app open.",
            )
            screen == "saved" -> SavedScreen(onAr = session::enterAr)
            else -> CenterMessage(
                icon = { Icon(Icons.Rounded.ErrorOutline, null, tint = Danger, modifier = Modifier.size(40.dp)) },
                title = "Scanner error",
                message = state.guidance.ifBlank { "The scanning session could not continue." },
            )
        }
    }

    if (stopConfirmation) {
        AlertDialog(
            onDismissRequest = { stopConfirmation = false },
            icon = { Icon(Icons.Rounded.StopCircle, null) },
            title = { Text("Finish this scan?") },
            text = { Text("The current reconstruction will be finalized and saved on Linux.") },
            confirmButton = {
                Button(
                    onClick = { stopConfirmation = false; session.stop() },
                    colors = ButtonDefaults.buttonColors(containerColor = Danger),
                ) { Text("Finish and save") }
            },
            dismissButton = { TextButton(onClick = { stopConfirmation = false }) { Text("Cancel") } },
        )
    }
    if (permissionDenied) {
        AlertDialog(
            onDismissRequest = { permissionDenied = false },
            icon = { Icon(Icons.Rounded.NoPhotography, null) },
            title = { Text("Camera access required") },
            text = { Text("Allow camera access to scan a QR code or capture a reconstruction.") },
            confirmButton = { Button(onClick = { permissionDenied = false }) { Text("Got it") } },
        )
    }

    LaunchedEffect(state.phase, state.cloudReady) {
        if (state.phase != "AR_VIEW") launchedAr = false
        if (state.phase == "AR_VIEW" && state.cloudReady && !launchedAr) {
            launchedAr = true
            context.startActivity(Intent(context, ArActivity::class.java))
        }
    }
}

@Composable
private fun Brand() {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Surface(shape = RoundedCornerShape(12.dp), color = TealDark) {
            Icon(Icons.Rounded.ViewInAr, null, Modifier.padding(9.dp), tint = Teal)
        }
        Spacer(Modifier.width(12.dp))
        Column {
            Text("MASt3R", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleLarge)
            Text("SLAM SCANNER", color = Muted, style = MaterialTheme.typography.labelMedium)
        }
    }
}

@Composable
private fun ConnectionHome(guidance: String, onQr: () -> Unit, onManual: () -> Unit) {
    Row(Modifier.fillMaxSize().padding(36.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f).padding(horizontal = 24.dp)) {
            Brand()
            Spacer(Modifier.height(28.dp))
            Text("Connect to Linux", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Text(
                "Start a New scan in the Linux app, then pair this phone using its secure QR code.",
                color = Muted,
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 10.dp).widthIn(max = 480.dp),
            )
            if (guidance.isNotBlank()) StatusNotice(guidance, isWarning = true, Modifier.padding(top = 18.dp))
        }
        Surface(
            Modifier.weight(.72f).fillMaxHeight().padding(16.dp),
            shape = RoundedCornerShape(28.dp),
            color = Panel,
            tonalElevation = 8.dp,
        ) {
            Column(Modifier.padding(28.dp), verticalArrangement = Arrangement.Center) {
                Button(onClick = onQr, Modifier.fillMaxWidth().height(58.dp)) {
                    Icon(Icons.Rounded.QrCodeScanner, null)
                    Spacer(Modifier.width(10.dp))
                    Text("Scan QR code")
                }
                Spacer(Modifier.height(14.dp))
                OutlinedButton(onClick = onManual, Modifier.fillMaxWidth().height(54.dp)) {
                    Icon(Icons.Rounded.Link, null)
                    Spacer(Modifier.width(10.dp))
                    Text("Enter pairing URI")
                }
            }
        }
    }
}

@Composable
private fun ManualConnection(
    pairing: String,
    guidance: String,
    onPairingChange: (String) -> Unit,
    onBack: () -> Unit,
    onConnect: () -> Unit,
) {
    Column(Modifier.fillMaxSize().padding(30.dp)) {
        IconButton(onClick = onBack) { Icon(Icons.Rounded.ArrowBack, "Back") }
        Column(Modifier.fillMaxSize().padding(horizontal = 80.dp), verticalArrangement = Arrangement.Center) {
            Text("Enter pairing URI", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Text("Paste the complete mast3r-slam:// link shown on Linux.", color = Muted, modifier = Modifier.padding(top = 8.dp))
            OutlinedTextField(
                value = pairing,
                onValueChange = onPairingChange,
                modifier = Modifier.fillMaxWidth().padding(top = 22.dp),
                singleLine = true,
                label = { Text("Pairing URI") },
                leadingIcon = { Icon(Icons.Rounded.Link, null) },
            )
            if (guidance.isNotBlank()) Text(guidance, color = Warning, modifier = Modifier.padding(top = 10.dp))
            Button(onClick = onConnect, enabled = pairing.isNotBlank(), modifier = Modifier.align(Alignment.End).padding(top = 18.dp)) {
                Text("Connect")
                Spacer(Modifier.width(8.dp))
                Icon(Icons.Rounded.ArrowForward, null)
            }
        }
    }
}

@Composable
private fun ReadyScreen(state: ScanUiState, primaryLabel: String, onStart: () -> Unit) {
    Row(Modifier.fillMaxSize().padding(36.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f).padding(horizontal = 28.dp)) {
            Brand()
            Spacer(Modifier.height(28.dp))
            StatusChip("SECURELY CONNECTED", Color(0xFF64D98B))
            Text("Ready to capture", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 16.dp))
            Text("Move slowly and keep textured surfaces in view for a stable reconstruction.", color = Muted, modifier = Modifier.padding(top = 8.dp))
        }
        Surface(Modifier.weight(.72f).padding(20.dp), shape = RoundedCornerShape(28.dp), color = Panel) {
            Column(Modifier.padding(28.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(Icons.Rounded.PhotoCamera, null, tint = Teal, modifier = Modifier.size(52.dp))
                Text("Camera starts only when you are ready", color = Muted, modifier = Modifier.padding(vertical = 16.dp))
                Button(onClick = onStart, Modifier.fillMaxWidth().height(58.dp)) {
                    Icon(Icons.Rounded.PlayArrow, null)
                    Spacer(Modifier.width(8.dp))
                    Text(primaryLabel)
                }
            }
        }
    }
}

@Composable
private fun ScanWorkspace(
    state: ScanUiState,
    session: ScannerSession,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onAr: () -> Unit,
    onStop: () -> Unit,
) {
    ScannerScaffold(state, session) {
        when (state.phase) {
        "SCANNING" -> {
            FilledTonalButton(onClick = onPause) {
                Icon(Icons.Rounded.Pause, "Pause")
                Spacer(Modifier.width(8.dp))
                Text("Pause")
            }
            FilledIconButton(onClick = onStop, colors = IconButtonDefaults.filledIconButtonColors(containerColor = Danger)) {
                Icon(Icons.Rounded.Stop, "Finish scan")
            }
        }
        "PAUSED", "AR_VIEW" -> {
            Button(onClick = onResume) {
                Icon(Icons.Rounded.PlayArrow, null)
                Spacer(Modifier.width(8.dp))
                Text("Resume")
            }
            FilledTonalButton(onClick = onAr, enabled = state.phase == "PAUSED") {
                Icon(Icons.Rounded.ViewInAr, null)
                Spacer(Modifier.width(8.dp))
                Text("View in AR")
            }
            FilledIconButton(onClick = onStop, colors = IconButtonDefaults.filledIconButtonColors(containerColor = Danger)) {
                Icon(Icons.Rounded.Stop, "Finish scan")
            }
        }
        else -> {
            CircularProgressIndicator(Modifier.size(28.dp), strokeWidth = 3.dp)
            Text("Preparing camera…", color = Muted)
        }
        }
    }
}

@Composable
private fun ScannerScaffold(
    state: ScanUiState,
    session: ScannerSession,
    controls: @Composable RowScope.() -> Unit,
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    Box(Modifier.fillMaxSize().background(AppBackground)) {
        Column(Modifier.fillMaxSize().padding(16.dp)) {
            ScanTopBar(state)
            Row(Modifier.weight(1f).fillMaxWidth().padding(top = 12.dp)) {
                Surface(
                    Modifier.weight(1.7f).fillMaxHeight(),
                    shape = RoundedCornerShape(22.dp),
                    color = Color.Black,
                ) {
                    Box {
                        state.preview?.let {
                            Image(it.asImageBitmap(), "Processed 3D scene", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
                        } ?: EmptyPreview("Waiting for the first 3D preview")
                        Surface(
                            Modifier.align(Alignment.TopStart).padding(12.dp),
                            shape = RoundedCornerShape(10.dp),
                            color = Color(0xB30D171D),
                        ) { Text("3D MAP", Modifier.padding(10.dp, 5.dp), color = Teal, style = MaterialTheme.typography.labelMedium) }
                    }
                }
                Surface(
                    Modifier.weight(.8f).fillMaxHeight().padding(start = 12.dp),
                    shape = RoundedCornerShape(22.dp),
                    color = Color.Black,
                ) {
                    if (state.cameraActive) AndroidCamera(session, lifecycleOwner, Modifier.fillMaxSize())
                    else EmptyPreview("Camera paused")
                }
            }
            Row(
                Modifier.fillMaxWidth().height(66.dp).padding(top = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.End),
                verticalAlignment = Alignment.CenterVertically,
                content = controls,
            )
        }
        if (state.tracking in setOf("LOST", "RELOCALIZING") || state.cameraMismatch) {
            TrackingOverlay(state, session, Modifier.align(Alignment.Center))
        }
    }
}

@Composable
private fun ScanTopBar(state: ScanUiState) {
    Surface(shape = RoundedCornerShape(18.dp), color = Panel, modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(16.dp, 10.dp), verticalAlignment = Alignment.CenterVertically) {
            StatusChip(state.phase.replace('_', ' '), if (state.phase == "PAUSED") Warning else Teal)
            Spacer(Modifier.width(18.dp))
            Metric("Tracking", state.tracking)
            Metric("Latency", "${state.latencyMs} ms")
            Metric("Frames", "${state.processedFrame} / ${state.liveFrame}")
            Spacer(Modifier.weight(1f))
            Icon(Icons.Rounded.Wifi, null, tint = Color(0xFF64D98B))
            Text(" Connected", color = Muted, style = MaterialTheme.typography.labelLarge)
        }
    }
}

@Composable
private fun Metric(label: String, value: String) {
    Column(Modifier.padding(end = 22.dp)) {
        Text(label.uppercase(), color = Muted, style = MaterialTheme.typography.labelSmall)
        Text(value, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.labelLarge)
    }
}

@Composable
private fun TrackingOverlay(state: ScanUiState, session: ScannerSession, modifier: Modifier = Modifier) {
    Surface(
        modifier.widthIn(max = 460.dp).padding(20.dp),
        shape = RoundedCornerShape(24.dp),
        color = Color(0xF20D171D),
        shadowElevation = 16.dp,
        border = androidx.compose.foundation.BorderStroke(1.dp, Warning),
    ) {
        Column(Modifier.padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(Icons.Rounded.CenterFocusStrong, null, tint = Warning, modifier = Modifier.size(42.dp))
            Text(
                if (state.cameraMismatch) "Camera configuration changed" else "Tracking lost",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(top = 10.dp),
            )
            Text(
                if (state.cameraMismatch) "Confirm the camera change to continue this reconstruction."
                else "Return the camera to an area you scanned recently, then move slowly.",
                color = Muted,
                modifier = Modifier.padding(top = 8.dp),
            )
            if (state.cameraMismatch) Button(onClick = session::confirmResume, Modifier.padding(top = 16.dp)) {
                Text("Confirm and resume")
            }
        }
    }
}

@Composable
private fun SavedScreen(onAr: () -> Unit) {
    CenterMessage(
        icon = { Icon(Icons.Rounded.CheckCircle, null, tint = Color(0xFF64D98B), modifier = Modifier.size(48.dp)) },
        title = "Project saved",
        message = "The reconstruction is available from Projects in the Linux app.",
        action = {
            FilledTonalButton(onClick = onAr) {
                Icon(Icons.Rounded.ViewInAr, null)
                Spacer(Modifier.width(8.dp))
                Text("View result in AR")
            }
        },
    )
}

@Composable
private fun CenterMessage(
    icon: @Composable () -> Unit,
    title: String,
    message: String,
    action: (@Composable () -> Unit)? = null,
) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Surface(shape = RoundedCornerShape(28.dp), color = Panel, modifier = Modifier.widthIn(max = 520.dp)) {
            Column(Modifier.padding(34.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                icon()
                Text(title, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 16.dp))
                Text(message, color = Muted, modifier = Modifier.padding(top = 8.dp))
                action?.let { Box(Modifier.padding(top = 20.dp)) { it() } }
            }
        }
    }
}

@Composable
private fun StatusChip(text: String, color: Color) {
    Surface(shape = CircleShape, color = color.copy(alpha = .16f), border = androidx.compose.foundation.BorderStroke(1.dp, color.copy(alpha = .55f))) {
        Text(text, color = color, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(12.dp, 6.dp))
    }
}

@Composable
private fun StatusNotice(text: String, isWarning: Boolean, modifier: Modifier = Modifier) {
    Surface(modifier, shape = RoundedCornerShape(14.dp), color = (if (isWarning) Warning else Teal).copy(alpha = .12f)) {
        Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Rounded.Info, null, tint = if (isWarning) Warning else Teal)
            Text(text, Modifier.padding(start = 10.dp), color = if (isWarning) Warning else Teal)
        }
    }
}

@Composable
private fun EmptyPreview(text: String) {
    Box(Modifier.fillMaxSize().background(Color.Black), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(Icons.Rounded.ViewInAr, null, tint = Muted, modifier = Modifier.size(34.dp))
            Text(text, color = Muted, modifier = Modifier.padding(top = 8.dp))
        }
    }
}

@Composable
private fun AndroidCamera(session: ScannerSession, owner: LifecycleOwner, modifier: Modifier) {
    var pipeline by remember { mutableStateOf<CameraPipeline?>(null) }
    androidx.compose.ui.viewinterop.AndroidView(
        factory = { context ->
            androidx.camera.view.PreviewView(context).also { view ->
                pipeline = CameraPipeline(context, owner, view, session).also { it.start() }
            }
        },
        modifier = modifier
            .background(Color.Black)
            .semantics { contentDescription = "Live camera preview" },
        onRelease = { pipeline?.stop() },
    )
}
