# Mobile scanner implementation

## Data and control flow

Android renders its CameraX preview locally and sends at most the current
analysis frame. The server's `NetworkFrameSource` has exactly one waiting slot;
a newer frame atomically replaces an older waiting frame. `SlamEngine` owns the
lifecycle and emits transport-neutral events. `Mast3rProcessor` is the only new
component that imports the MASt3R tracker, retrieval database, factor graph and
CUDA backend.

The WebSocket uses one binary `ClientMessage` or `ServerMessage` per frame. Its
schema is `protocol/mast3r_slam.proto`; the checked-in Python binding and the
Android Gradle-generated lite binding both originate there. Large mobile cloud
chunks use authenticated HTTPS with Range support, retry and SHA-256 checking.

Tracking loss and failed continuation relocalization do not append a keyframe.
The engine sends the last good processed preview and asks the operator to return
to a scanned region. A resumed revision stays in relocalization until a frame
matches the restored retrieval database and factor graph.

## Project layout

Projects default to `projects/<project-id>/`. `manifest.json` points at immutable
revision directories. Each revision contains a ZIP/NPY CPU checkpoint whose
arrays are written and loaded one at a time, metadata, RGB keyframes and
checksums. A revision is written into a sibling temporary directory, fsynced,
renamed, then published by an atomic manifest replacement. Stop does not build a
PLY; Export copies an older PLY or streams one from the checkpoint.

The mobile snapshot is independent of revisions and contains no more than
750,000 mesh vertices. It is generated one keyframe at a time; every chunk
stores local 16-bit quantized triangle vertices before zstd compression.

## Android build

Open `android-app/` in Android Studio with JDK 17, Android SDK 36 and Gradle 8.13.
The app uses an AR-optional manifest so ordinary scan/preview remains usable on
unsupported devices; entering the AR screen performs an ARCore availability
check and enables Depth automatically where available.

Protocol v2 sends throttled ARCore camera images and exact poses to the server.
The server performs retrieval plus pose-only MASt3R matching without modifying
the graph, then returns a Sim3 model matrix with the OpenCV/OpenGL axis change.
The mesh remains hidden until alignment and is hidden/relocalized after ARCore
tracking loss; manual anchoring and artificial centering are not used.

## Verification

Run the GPU-independent suite with:

```bash
python -m unittest discover -s tests -v
```

It covers protocol round trips, state transitions/idempotency, latest-frame
backpressure, bitrate policy, camera mismatch, atomic project recovery,
checkpoint checksums, streaming PLY export, mesh chunks, Sim3 axis math and a
TLS WebSocket fake-phone flow. Android instrumentation runs through
`connectedDebugAndroidTest`; CUDA/TUM equivalence and Depth occlusion remain
hardware acceptance tests.

## Operator interface

The Linux TUI has separate main-menu, live-dashboard and project-browser
screens. During capture the dashboard reports session/tracking state, frame
latency, processing time, queue drops and keyframes alongside an ASCII version
of the processed map preview. Project actions are available from the row popup
or the C/O/E/D shortcuts.

Android does not activate the camera on launch. Pairing, ready, scanning,
paused/AR and saved states are separate views, so only valid actions are shown.
The AR action first appears after pausing. Tracking-loss guidance is presented
as a centered high-contrast overlay over the split map/live-camera workspace.
