from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import pathlib
import secrets
import socket
import ssl
import subprocess
import tempfile
import time
from typing import Any, Callable

from protocol import PROTOCOL_VERSION
from protocol import mast3r_slam_pb2 as wire

from .camera import CameraIdentity
from .engine import ArAlignmentFrame, EngineEvent, SlamEngine
from .frame_source import NetworkFrame
from .state_machine import InvalidTransition, SessionPhase


PHASES = {name: value for name, value in wire.SessionPhase.items()}
TRACKING = {name: value for name, value in wire.TrackingState.items()}
AR_ALIGNMENT = {
    "SEARCHING": wire.SEARCHING,
    "ALIGNED": wire.ALIGNED,
    "LOST": wire.ALIGNMENT_LOST,
}
COMMANDS = {
    wire.START: "start",
    wire.PAUSE: "pause",
    wire.RESUME: "resume",
    wire.ENTER_AR: "enter_ar",
    wire.LEAVE_AR: "leave_ar",
    wire.STOP: "stop",
}


class MobileServer:
    """Single-phone TLS WebSocket/HTTPS transport for a `SlamEngine`."""

    def __init__(
        self,
        engine: SlamEngine,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        certificate_dir: str | pathlib.Path | None = None,
        token: str | None = None,
        project_name: str = "Mobile scan",
    ) -> None:
        self.engine = engine
        self.project_name = project_name
        self.host = host
        self.port = port
        self.token = token or secrets.token_hex(16)
        self._certificate_dir = pathlib.Path(certificate_dir) if certificate_dir else None
        self._app: Any = None
        self._runner: Any = None
        self._socket: Any = None
        self._socket_authenticated = False
        self._device_id: str | None = None
        self._token_valid = True
        self._camera: CameraIdentity | None = None
        self._last_heartbeat = 0.0
        self._command_in_progress = 0
        self._control_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._disconnect_generation = 0
        self._connection_listeners: list[Callable[[bool, str | None], None]] = []
        self._alignment_queue: asyncio.Queue[ArAlignmentFrame] = asyncio.Queue(maxsize=1)
        self._alignment_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._disconnect_tasks: set[asyncio.Task] = set()
        self._stopping = False
        self.fingerprint = ""
        self.engine.on("*", self._on_engine_event)

    @property
    def client_connected(self) -> bool:
        return self._socket_authenticated

    @property
    def device_id(self) -> str | None:
        return self._device_id

    def on_connection_changed(
        self, callback: Callable[[bool, str | None], None]
    ) -> None:
        self._connection_listeners.append(callback)

    @property
    def pairing_uri(self) -> str:
        address = self._advertised_address()
        return (
            f"mast3r-slam://pair?host={address}&port={self.port}"
            f"&token={self.token}&fingerprint={self.fingerprint}&v={PROTOCOL_VERSION}"
        )

    async def start(self) -> None:
        try:
            from aiohttp import web
        except ImportError as error:
            raise RuntimeError("install the 'mobile' dependencies to run the server") from error
        cert, key = self._create_certificate()
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        der = ssl.PEM_cert_to_DER_cert(pathlib.Path(cert).read_text())
        self.fingerprint = hashlib.sha256(der).hexdigest()
        self._loop = asyncio.get_running_loop()
        self._app = web.Application(client_max_size=16 * 1024 * 1024)
        self._app.router.add_get("/ws", self._websocket)
        self._app.router.add_get("/cloud/{snapshot}/{name}", self._cloud_chunk)
        self._app.router.add_get("/health", self._health)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port, ssl_context=context).start()
        self._watchdog_task = asyncio.create_task(self._heartbeat_watchdog())
        self._alignment_task = asyncio.create_task(self._alignment_worker())
        await asyncio.to_thread(self._publish_mdns)

    async def stop(self) -> None:
        self._stopping = True
        background = list(self._disconnect_tasks)
        if self._watchdog_task:
            background.append(self._watchdog_task)
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)
        self._watchdog_task = None
        if self._socket is not None:
            await self._socket.close()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if hasattr(self, "_zeroconf"):
            await asyncio.to_thread(self._zeroconf.close)
        if self._alignment_task is not None:
            self._alignment_task.cancel()
            await asyncio.gather(self._alignment_task, return_exceptions=True)
            self._alignment_task = None
        await asyncio.to_thread(self.engine.close)

    async def _websocket(self, request):
        from aiohttp import WSMsgType, web

        if self._socket is not None and not self._socket.closed:
            raise web.HTTPConflict(text="a phone is already connected")
        socket_ = web.WebSocketResponse(heartbeat=30.0, max_msg_size=16 * 1024 * 1024)
        await socket_.prepare(request)
        authenticated = False
        self._socket = socket_
        self._socket_authenticated = False
        self._disconnect_generation += 1
        generation = self._disconnect_generation
        try:
            async for message in socket_:
                if message.type != WSMsgType.BINARY:
                    await self._send_error("BINARY_REQUIRED", "only binary Protobuf messages are accepted")
                    continue
                incoming = wire.ClientMessage()
                try:
                    incoming.ParseFromString(message.data)
                except Exception:
                    await self._send_error("INVALID_PROTOBUF", "message cannot be decoded")
                    continue
                kind = incoming.WhichOneof("payload")
                if not authenticated:
                    if kind != "hello" or not self._authenticate(incoming.hello):
                        await self._send_error("PAIRING_FAILED", "invalid token, protocol or device")
                        await socket_.close()
                        break
                    authenticated = True
                    self._socket_authenticated = True
                    self._last_heartbeat = time.monotonic()
                    self.engine.begin_frame_stream()
                    if self.engine.session.phase == SessionPhase.DISCONNECTED:
                        self.engine.session.paired()
                    self._notify_connection(True)
                    await self._send_state("connected")
                    continue
                self._last_heartbeat = time.monotonic()
                await self._dispatch(incoming, kind)
        finally:
            if self._socket is socket_:
                self._socket = None
                self._socket_authenticated = False
                self._notify_connection(False)
            if not self._stopping:
                task = asyncio.create_task(self._delayed_disconnect(generation))
                self._disconnect_tasks.add(task)
                task.add_done_callback(self._disconnect_tasks.discard)
        return socket_

    def _notify_connection(self, connected: bool) -> None:
        for callback in tuple(self._connection_listeners):
            callback(connected, self._device_id)

    def _authenticate(self, hello) -> bool:
        if (
            not self._token_valid
            or hello.protocol_version != PROTOCOL_VERSION
            or not secrets.compare_digest(hello.token, self.token)
        ):
            return False
        if self._device_id is not None and hello.device_id != self._device_id:
            return False
        self._device_id = hello.device_id
        return True

    async def _dispatch(self, incoming, kind: str | None) -> None:
        if kind == "heartbeat":
            return
        if kind == "camera_config":
            item = incoming.camera_config
            self._camera = CameraIdentity(
                item.device_model,
                item.physical_camera_id,
                item.width,
                item.height,
                (item.crop_left, item.crop_top, item.crop_width, item.crop_height),
                tuple(item.intrinsics[:4]),
                tuple(item.distortion),
            )
            return
        if kind == "frame":
            item = incoming.frame
            result = self.engine.submit(
                NetworkFrame(
                    item.frame_id,
                    item.timestamp_ns,
                    item.jpeg,
                    item.orientation_degrees,
                    item.physical_camera_id,
                    item.width,
                    item.height,
                    (item.crop_left, item.crop_top, item.crop_width, item.crop_height),
                    tuple(item.intrinsics),
                    tuple(item.distortion),
                )
            )
            await self._send(
                wire.ServerMessage(
                    frame_ack=wire.FrameAck(
                        frame_id=item.frame_id,
                        accepted=result.accepted,
                        dropped_frames=result.dropped_total,
                    )
                )
            )
            return
        if kind == "ar_alignment_frame":
            item = incoming.ar_alignment_frame
            frame = ArAlignmentFrame(
                item.frame_id,
                item.timestamp_ns,
                item.jpeg,
                tuple(item.intrinsics),
                tuple(item.ar_world_from_camera),
                item.width,
                item.height,
            )
            if self._alignment_queue.full():
                try:
                    self._alignment_queue.get_nowait()
                    self._alignment_queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            self._alignment_queue.put_nowait(frame)
            return
        if kind != "command":
            await self._send_error("UNEXPECTED_MESSAGE", "unsupported message")
            return
        command = incoming.command
        name = COMMANDS.get(command.type)
        if not name:
            await self._send_error("UNKNOWN_COMMAND", "unknown command", command.command_id)
            return
        try:
            async with self._control_lock:
                if name == "start":
                    if self._camera is None:
                        raise ValueError("CameraConfig is required before Start")
                    self.engine.start(command.command_id, self.project_name, self._camera)
                elif name == "pause":
                    await self._run_blocking_command(self.engine.pause, command.command_id)
                elif name == "resume":
                    if self._camera is None:
                        raise ValueError("CameraConfig is required before Resume")
                    self.engine.resume(
                        command.command_id,
                        self._camera,
                        camera_change_confirmed=command.camera_change_confirmed,
                    )
                elif name == "enter_ar":
                    await self._run_blocking_command(self.engine.enter_ar, command.command_id)
                elif name == "leave_ar":
                    self.engine.leave_ar(command.command_id)
                elif name == "stop":
                    await self._run_blocking_command(self.engine.stop, command.command_id)
        except (InvalidTransition, ValueError, RuntimeError) as error:
            code = "CAMERA_MISMATCH" if "camera configuration changed" in str(error) else "COMMAND_REJECTED"
            await self._send_error(code, str(error), command.command_id)

    async def _run_blocking_command(self, function, *args):
        self._command_in_progress += 1
        try:
            return await asyncio.to_thread(function, *args)
        finally:
            self._last_heartbeat = time.monotonic()
            self._command_in_progress -= 1

    async def execute_command(self, name: str, command_id: str) -> None:
        """Run a panel command using the same heartbeat protection as Android."""
        async with self._control_lock:
            if name == "resume":
                if self._camera is None or not self.client_connected:
                    raise ValueError("Подключите телефон перед продолжением")
                await self._run_blocking_command(self.engine.resume, command_id, self._camera)
            elif name in {"pause", "stop"}:
                await self._run_blocking_command(getattr(self.engine, name), command_id)
            else:
                raise ValueError("unknown panel command")

    async def _send_state(self, detail: str = "", command_id: str = "") -> None:
        await self._send(
            wire.ServerMessage(
                session_state=wire.SessionState(
                    phase=PHASES[self.engine.session.phase.value],
                    project_id=self.engine.project_id or "",
                    revision=self.engine.revision,
                    detail=detail,
                    command_id=command_id,
                )
            )
        )

    async def _send_error(self, code: str, message: str, command_id: str = "") -> None:
        await self._send(
            wire.ServerMessage(error=wire.Error(code=code, message=message, recoverable=True, command_id=command_id))
        )

    async def _send(self, message) -> None:
        socket_ = self._socket
        if socket_ is None or socket_.closed:
            return
        try:
            await socket_.send_bytes(message.SerializeToString())
        except ConnectionError:
            # The peer may close after the state check and before the write.
            # WebSocket cleanup owns the actual disconnect transition.
            return

    def _on_engine_event(self, event: EngineEvent) -> None:
        if not self._loop or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._send_engine_event(event), self._loop)

    async def _send_engine_event(self, event: EngineEvent) -> None:
        data = event.data
        if event.name == "state_changed":
            await self._send_state(command_id=data.get("command_id", ""))
        elif event.name == "save_completed":
            await self._send_state("saved revision")
        elif event.name == "frame_processed":
            await self._send(
                wire.ServerMessage(
                    preview=wire.ProcessedPreview(
                        input_frame_id=data["input_frame_id"],
                        processed_frame_id=data["processed_frame_id"],
                        tracking=TRACKING[data["tracking"]],
                        latency_ms=data["latency_ms"],
                        jpeg=data.get("preview_jpeg") or b"",
                        guidance=data.get("guidance", ""),
                    )
                )
            )
        elif event.name in {"tracking_lost", "camera_mismatch", "capacity_warning", "progress"}:
            await self._send(
                wire.ServerMessage(
                    progress=wire.Progress(
                        operation=data.get("operation", event.name),
                        fraction=data.get("fraction", 0),
                        detail=data.get("detail", str(data)),
                    )
                )
            )
        elif event.name == "error":
            await self._send_error(data["code"], data["message"])
        elif event.name == "cloud_ready":
            manifest = data["manifest"]
            chunks = [
                wire.CloudChunk(
                    index=item.index,
                    compressed_size=item.compressed_size,
                    uncompressed_size=item.uncompressed_size,
                    sha256=item.sha256,
                    url=f"/cloud/{manifest.snapshot_id}/{item.filename}?token={self.token}",
                )
                for item in manifest.chunks
            ]
            await self._send(
                wire.ServerMessage(
                    cloud_manifest=wire.CloudManifest(
                        snapshot_id=manifest.snapshot_id,
                        point_count=manifest.point_count,
                        bounds_min=manifest.bounds_min,
                        bounds_max=manifest.bounds_max,
                        chunks=chunks,
                        encoding=manifest.encoding,
                    )
                )
            )
        elif event.name == "finalized":
            await self._send_state("saved")
            self._token_valid = False

    async def _delayed_disconnect(self, generation: int) -> None:
        await asyncio.sleep(2)
        if generation == self._disconnect_generation and self._socket is None:
            await asyncio.to_thread(self.engine.disconnect)

    async def _heartbeat_watchdog(self) -> None:
        while self._runner is not None:
            await asyncio.sleep(0.5)
            if (
                self._socket is not None
                and self._socket_authenticated
                and not self._command_in_progress
                and time.monotonic() - self._last_heartbeat > 2
            ):
                await self._socket.close(code=1001, message=b"heartbeat timeout")

    async def _alignment_worker(self) -> None:
        while True:
            frame = await self._alignment_queue.get()
            try:
                result = await asyncio.to_thread(self.engine.align_ar, frame)
                await self._send(
                    wire.ServerMessage(
                        ar_alignment_result=wire.ArAlignmentResult(
                            status=AR_ALIGNMENT[result.status.value],
                            quality=result.quality,
                            ar_world_from_slam_map=result.ar_world_from_slam_map,
                            guidance=result.guidance,
                            frame_id=result.frame_id,
                        )
                    )
                )
            except (RuntimeError, ValueError) as error:
                await self._send_error("AR_ALIGNMENT_FAILED", str(error))
            finally:
                self._alignment_queue.task_done()

    async def _cloud_chunk(self, request):
        from aiohttp import web

        supplied = request.query.get("token", "")
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            supplied = auth[7:]
        if not secrets.compare_digest(supplied, self.token) or not self.engine.project_id:
            raise web.HTTPUnauthorized()
        name = pathlib.Path(request.match_info["name"]).name
        snapshot = pathlib.Path(request.match_info["snapshot"]).name
        path = self.engine.projects.project_path(self.engine.project_id) / "mobile_snapshots" / snapshot / name
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)  # aiohttp provides Range and conditional requests.

    async def _health(self, request):
        from aiohttp import web

        return web.json_response(
            {
                "protocol": PROTOCOL_VERSION,
                "phase": self.engine.session.phase.value,
                "error": self.engine.last_error,
            }
        )

    def _create_certificate(self) -> tuple[str, str]:
        directory = self._certificate_dir or pathlib.Path(tempfile.mkdtemp(prefix="mast3r-slam-tls-"))
        directory.mkdir(parents=True, exist_ok=True)
        cert, key = directory / "certificate.pem", directory / "private-key.pem"
        if not cert.exists() or not key.exists():
            subprocess.run(
                [
                    "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(key), "-out", str(cert), "-days", "1",
                    "-subj", "/CN=mast3r-slam.local",
                    "-addext", "subjectAltName=DNS:mast3r-slam.local,IP:127.0.0.1",
                ],
                check=True,
                capture_output=True,
            )
        return str(cert), str(key)

    def _publish_mdns(self) -> None:
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError:
            return
        address = self._advertised_address()
        info = ServiceInfo(
            "_mast3r-slam._tcp.local.",
            "MASt3R SLAM._mast3r-slam._tcp.local.",
            addresses=[ipaddress.ip_address(address).packed],
            port=self.port,
            properties={"version": str(PROTOCOL_VERSION), "tls": "1"},
            server="mast3r-slam.local.",
        )
        self._zeroconf = Zeroconf()
        self._zeroconf.register_service(info, allow_name_change=True)

    @staticmethod
    def _advertised_address() -> str:
        # Asking the kernel for a route to an arbitrary public address can select
        # a policy-routed VPN/TUN interface (for example Mihomo's 198.18.0.1).
        # The normal default route is a better representation of the LAN address
        # that a phone can actually reach.
        try:
            result = subprocess.run(
                ["ip", "-json", "route", "show", "default"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            routes = json.loads(result.stdout)
            routes.sort(key=lambda item: int(item.get("metric", 0)))
            for route in routes:
                address = route.get("prefsrc")
                if address and not ipaddress.ip_address(address).is_loopback:
                    return address
        except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
            pass

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            try:
                sock.connect(("192.0.2.1", 80))
                return sock.getsockname()[0]
            except OSError:
                return "127.0.0.1"
