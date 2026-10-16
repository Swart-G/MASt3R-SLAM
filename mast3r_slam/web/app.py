from __future__ import annotations

import asyncio
import gc
import hashlib
import io
import logging
import pathlib
import re
import secrets
import tempfile
import uuid
from collections import deque

from aiohttp import web

from mast3r_slam.mobile.projects import ProjectError, ProjectStore

STATIC = pathlib.Path(__file__).parent / "static"
PANEL_KEY = web.AppKey("panel", object)


def validate_project_id(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{8}-[0-9]{6}-[0-9a-f]{8}", value
    ):
        raise ProjectError("Некорректный идентификатор проекта")
    return value


class Panel:
    def __init__(
        self, store, *, config="config/base.yaml", device="cuda:0", phone_port=8765
    ):
        self.store = store
        self.config = config
        self.device = device
        self.phone_port = phone_port
        self.engine = self.server = None
        self.task = None
        self.busy = False
        self.phase = "IDLE"
        self.error = ""
        self.preview = None
        self.stats = {}
        self.project_name = None
        self._project_name_id = None
        self.events = deque(maxlen=30)
        self.lock = asyncio.Lock()
        self.csrf = secrets.token_urlsafe(32)

    def event(self, event):
        # Engine callbacks run on worker threads; mutate UI state on the HTTP loop.
        self.loop.call_soon_threadsafe(self.receive, event.name, dict(event.data))

    def receive(self, name, data):
        if name == "frame_processed":
            self.preview = data.pop("preview_jpeg", None) or self.preview
            self.stats.update(data)
        elif name == "map_updated":
            self.stats.update(data)
        elif name == "error":
            self.error = data.get("message", "Ошибка сканера")
        if name != "frame_processed":
            self.events.append(
                {
                    "name": name,
                    "detail": str(
                        data.get("detail", data.get("message", data.get("phase", name)))
                    ),
                }
            )

    async def open(self, project_id=None, *, name=None):
        self.phase, self.error = "LOADING", ""
        self.project_name = name
        self.preview, self.stats = None, {}
        self.events.clear()
        self.loop = asyncio.get_running_loop()
        try:
            from mast3r_slam.mobile.engine import SlamEngine
            from mast3r_slam.mobile.mast3r_processor import Mast3rProcessor
            from mast3r_slam.mobile.server import MobileServer

            def build():
                config = pathlib.Path(self.config)
                return SlamEngine(
                    Mast3rProcessor(str(config), self.device),
                    self.store,
                    model_id="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                    config_fingerprint=hashlib.sha256(config.read_bytes()).hexdigest(),
                )

            self.engine = await asyncio.to_thread(build)
            self.engine.on("*", self.event)
            if project_id:
                manifest = await asyncio.to_thread(self.store.read_manifest, project_id)
                self.project_name = manifest.get("name") or project_id
                self._project_name_id = project_id
                await asyncio.to_thread(self.engine.load_project, project_id)
            self.server = MobileServer(
                self.engine, port=self.phone_port, project_name=name or "Mobile scan"
            )
            await self.server.start()
            self.phase = "READY"
        except Exception as error:
            logging.getLogger(__name__).exception("Could not open scanner session")
            self.error, self.phase = str(error) or type(error).__name__, "ERROR"
            await self.close()
            self.phase = "ERROR"

    async def close(self):
        if self.server:
            await self.server.stop()
        elif self.engine:
            await asyncio.to_thread(self.engine.close)
        self.server = self.engine = None
        self.project_name = self._project_name_id = None
        self.phase = "IDLE"
        # Server/engine callbacks form cycles. Release the previous model before
        # loading another project into the same GPU.
        await asyncio.to_thread(gc.collect)

    def status(self):
        engine, server = self.engine, self.server
        project_id = engine.project_id if engine else None
        if project_id and project_id != self._project_name_id:
            try:
                self.project_name = (
                    self.store.read_manifest(project_id).get("name") or project_id
                )
            except ProjectError:
                self.project_name = project_id
            self._project_name_id = project_id
        return {
            "phase": engine.session.phase.value
            if engine and self.phase != "LOADING"
            else self.phase,
            "connected": bool(server and server.client_connected),
            "busy": self.busy,
            "pairing_uri": server.pairing_uri
            if server and server.fingerprint
            else None,
            "project_id": project_id,
            "project_name": self.project_name,
            "revision": engine.revision if engine else 0,
            "error": self.error,
            "preview": bool(self.preview),
            "stats": self.stats,
            "events": list(self.events),
            "csrf": self.csrf,
        }


def create_app(store: ProjectStore, **options):
    panel = Panel(store, **options)

    @web.middleware
    async def errors(request, handler):
        if request.method not in {"GET", "HEAD"}:
            if not secrets.compare_digest(
                request.headers.get("X-Panel-Token", ""), panel.csrf
            ):
                return web.json_response(
                    {"error": "Обновите страницу панели и повторите действие"},
                    status=403,
                )
        try:
            response = await handler(request)
        except (ProjectError, ValueError, RuntimeError, OSError) as error:
            return web.json_response({"error": str(error)}, status=409)
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    app = web.Application(middlewares=[errors], client_max_size=1024 * 1024)
    app[PANEL_KEY] = panel

    async def index(request):
        return web.FileResponse(STATIC / "index.html")

    async def health(request):
        return web.json_response({"status": "ok"})

    async def status(request):
        return web.json_response(panel.status())

    async def projects(request):
        items = await asyncio.to_thread(store.list_projects)
        return web.json_response(items)

    async def session(request):
        async with panel.lock:
            if panel.busy or panel.phase == "LOADING":
                raise web.HTTPConflict(text="Сканер занят")
            if panel.engine and panel.engine.session.phase.value not in {
                "READY",
                "SAVED",
                "ERROR",
            }:
                raise RuntimeError("Сначала завершите и сохраните текущее сканирование")
            data = await request.json()
            if not isinstance(data, dict):
                raise ValueError("Ожидается JSON-объект")
            project_id = data.get("project_id")
            if project_id is not None:
                validate_project_id(project_id)
            if project_id:
                store.read_manifest(project_id)
            name = None
            if not project_id:
                name = data.get("name")
                if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
                    raise ValueError("Введите название проекта: от 1 до 120 символов")
                name = name.strip()
            await panel.close()
            panel.phase = "LOADING"
            panel.project_name = name
            panel.task = asyncio.create_task(panel.open(project_id, name=name))
        return web.json_response({"phase": "LOADING"}, status=202)

    async def command(request):
        async with panel.lock:
            if panel.busy or panel.phase == "LOADING" or not panel.server:
                raise RuntimeError("Сканер ещё не готов")
            name = request.match_info["command"]
            if name not in {"pause", "resume", "stop"}:
                raise web.HTTPNotFound()
            panel.busy = True
            try:
                await panel.server.execute_command(name, str(uuid.uuid4()))
            finally:
                panel.busy = False
        return web.json_response(panel.status())

    async def qr(request):
        if not panel.server or not panel.server.fingerprint:
            raise web.HTTPNotFound()
        import qrcode

        output = io.BytesIO()
        qrcode.make(panel.server.pairing_uri).save(output, format="PNG")
        return web.Response(body=output.getvalue(), content_type="image/png")

    async def preview(request):
        if not panel.preview:
            raise web.HTTPNotFound()
        return web.Response(body=panel.preview, content_type="image/jpeg")

    async def export(request):
        project_id = validate_project_id(request.match_info["project_id"])
        # Per-request temporary directories avoid collisions between downloads.
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "reconstruction.ply"
            await asyncio.to_thread(store.export_ply, project_id, target)
            response = web.StreamResponse(
                headers={
                    "Content-Type": "application/octet-stream",
                    "Cache-Control": "no-store",
                    "Content-Disposition": 'attachment; filename="reconstruction.ply"',
                }
            )
            await response.prepare(request)
            with target.open("rb") as stream:
                while block := await asyncio.to_thread(stream.read, 1024 * 1024):
                    await response.write(block)
            await response.write_eof()
            return response

    async def delete(request):
        async with panel.lock:
            project_id = validate_project_id(request.match_info["project_id"])
            if panel.phase == "LOADING" or (
                panel.engine and panel.engine.project_id == project_id
            ):
                raise RuntimeError(
                    "Проект открыт в сканере. Откройте новый сеанс перед удалением"
                )
            await asyncio.to_thread(store.delete_project, project_id)
        return web.json_response({"deleted": True})

    async def cleanup(app):
        if panel.task and not panel.task.done():
            await panel.task
        await panel.close()

    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/projects", projects)
    app.router.add_post("/api/session", session)
    app.router.add_post("/api/commands/{command}", command)
    app.router.add_get("/api/pairing.png", qr)
    app.router.add_get("/api/preview.jpg", preview)
    app.router.add_get("/api/projects/{project_id}/export", export)
    app.router.add_delete("/api/projects/{project_id}", delete)
    app.router.add_static("/static/", STATIC)
    app.on_cleanup.append(cleanup)
    return app


def run_panel(store, *, host="127.0.0.1", port=8080, **options):
    web.run_app(create_app(store, **options), host=host, port=port)
    return 0
