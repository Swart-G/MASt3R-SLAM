from __future__ import annotations

import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from mast3r_slam.cli import _parser, main
from mast3r_slam.mobile.camera import CameraIdentity
from mast3r_slam.mobile.engine import SlamEngine
from mast3r_slam.mobile.projects import Checkpoint, ProjectStore
from mast3r_slam.mobile.server import MobileServer
from mast3r_slam.web.app import PANEL_KEY, create_app
from protocol import mast3r_slam_pb2 as wire

CAMERA = CameraIdentity(
    "phone", "0", 1280, 720, (0, 0, 1280, 720), (900, 900, 640, 360), ()
)


class PanelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = ProjectStore(self.directory.name)
        self.app = create_app(self.store)
        self.panel = self.app[PANEL_KEY]
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        response = await self.client.get("/api/status")
        self.headers = {"X-Panel-Token": (await response.json())["csrf"]}

    async def asyncTearDown(self):
        await self.client.close()
        self.directory.cleanup()

    def project(self):
        project = self.store.create(
            "<script>alert(1)</script>", CAMERA, model_id="m", config_fingerprint="c"
        )
        self.store.save_revision(
            project,
            Checkpoint({"x": np.arange(1)}, {"keyframe_count": 1}),
            state="saved",
            ply=b"ply\nformat ascii 1.0\nend_header\n",
        )
        return project

    async def test_static_health_and_empty_projects_without_gpu(self):
        for path in [
            "/",
            "/health",
            "/static/panel.js",
            "/static/panel.css",
            "/static/vendor/bootstrap.min.css",
        ]:
            response = await self.client.get(path)
            self.assertEqual(response.status, 200, path)
        self.assertEqual(await (await self.client.get("/api/projects")).json(), [])
        self.assertEqual(
            (await (await self.client.get("/api/status")).json())["phase"], "IDLE"
        )
        self.assertEqual((await self.client.get("/api/pairing.png")).status, 404)
        self.assertEqual((await self.client.get("/api/preview.jpg")).status, 404)

    async def test_status_reports_current_project_name_and_clears_it_on_close(self):
        project_id = self.project()
        engine = SlamEngine(None, self.store, model_id="m", config_fingerprint="c")
        self.panel.engine = engine
        engine.project_id = project_id
        response = await self.client.get("/api/status")
        status = await response.json()
        self.assertEqual(status["project_id"], project_id)
        self.assertEqual(status["project_name"], "<script>alert(1)</script>")
        await self.panel.close()
        status = await (await self.client.get("/api/status")).json()
        self.assertIsNone(status["project_id"])
        self.assertIsNone(status["project_name"])

    async def test_new_session_trims_and_exposes_its_name(self):
        with patch.object(self.panel, "open", new=AsyncMock()) as open_session:
            response = await self.client.post(
                "/api/session", json={"name": "  Кухня  "}, headers=self.headers
            )
            self.assertEqual(response.status, 202)
            await self.panel.task
            open_session.assert_awaited_once_with(None, name="Кухня")
            self.assertEqual(self.panel.status()["project_name"], "Кухня")

    async def test_new_session_requires_a_valid_name(self):
        for payload in [
            {},
            {"name": ""},
            {"name": "   "},
            {"name": 42},
            {"name": "x" * 121},
        ]:
            response = await self.client.post(
                "/api/session", json=payload, headers=self.headers
            )
            self.assertEqual(response.status, 409)
        self.assertIsNone(self.panel.task)

    async def test_mutations_require_page_token(self):
        project = self.project()
        response = await self.client.delete("/api/projects/" + project)
        self.assertEqual(response.status, 403)
        self.assertTrue(self.store.project_path(project).exists())
        response = await self.client.post("/api/session", json={})
        self.assertEqual(response.status, 403)

    async def test_project_export_and_delete(self):
        project = self.project()
        listed = await (await self.client.get("/api/projects")).json()
        self.assertEqual(listed[0]["project_id"], project)
        response = await self.client.get("/api/projects/" + project + "/export")
        self.assertEqual(response.status, 200)
        self.assertTrue((await response.read()).startswith(b"ply\n"))
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        response = await self.client.delete(
            "/api/projects/" + project, headers=self.headers
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(self.store.list_projects(), [])

    async def test_active_project_cannot_be_deleted_or_replaced(self):
        project = self.project()
        engine = SlamEngine(None, self.store, model_id="m", config_fingerprint="c")
        engine.start("start", "active", CAMERA)
        # Keep our saved project marked as opened by the active scanner.
        engine.project_id = project
        self.panel.engine = engine
        response = await self.client.delete(
            "/api/projects/" + project, headers=self.headers
        )
        self.assertEqual(response.status, 409)
        response = await self.client.post(
            "/api/session", json={"name": "Кухня"}, headers=self.headers
        )
        self.assertEqual(response.status, 409)
        engine.project_id = None

    async def test_session_loading_is_async_and_rejects_duplicate_start(self):
        gate = asyncio.Event()

        async def slow_open(project_id, *, name=None):
            await gate.wait()

        with patch.object(self.panel, "open", side_effect=slow_open):
            response = await self.client.post(
                "/api/session", json={"name": "Кухня"}, headers=self.headers
            )
            self.assertEqual(response.status, 202)
            self.assertEqual(
                (await (await self.client.get("/api/status")).json())["phase"],
                "LOADING",
            )
            response = await self.client.post(
                "/api/session", json={"name": "Кухня"}, headers=self.headers
            )
            self.assertEqual(response.status, 409)
            gate.set()
            await self.panel.task

    async def test_failed_session_can_be_retried(self):
        with patch(
            "mast3r_slam.mobile.mast3r_processor.Mast3rProcessor",
            side_effect=RuntimeError("CUDA unavailable"),
        ):
            response = await self.client.post(
                "/api/session", json={"name": "Кухня"}, headers=self.headers
            )
            self.assertEqual(response.status, 202)
            await self.panel.task
        status = self.panel.status()
        self.assertEqual(status["phase"], "ERROR")
        self.assertIn("CUDA unavailable", status["error"])
        self.assertIsNone(self.panel.engine)

    async def test_commands_and_preview(self):
        server = AsyncMock()
        server.engine = None
        server.client_connected = False
        server.fingerprint = ""
        self.panel.server = server
        response = await self.client.post(
            "/api/commands/pause", json={"name": "Кухня"}, headers=self.headers
        )
        self.assertEqual(response.status, 200)
        server.execute_command.assert_awaited_once()
        self.assertEqual(server.execute_command.await_args.args[0], "pause")
        self.panel.server = None
        self.panel.preview = b"jpeg bytes"
        response = await self.client.get("/api/preview.jpg")
        self.assertEqual(await response.read(), b"jpeg bytes")

    async def test_malformed_session_and_path_are_rejected(self):
        for value in [[], {"project_id": ".."}, {"project_id": 42}]:
            response = await self.client.post(
                "/api/session", json=value, headers=self.headers
            )
            self.assertEqual(response.status, 409)
        self.assertIsNone(self.panel.task)

    async def test_slow_command_keeps_status_responsive(self):
        started, release = asyncio.Event(), asyncio.Event()
        server = AsyncMock()
        server.client_connected, server.fingerprint = False, ""

        async def slow_command(*args):
            started.set()
            await release.wait()

        server.execute_command.side_effect = slow_command
        self.panel.server = server
        task = asyncio.create_task(
            self.client.post(
                "/api/commands/stop", json={"name": "Кухня"}, headers=self.headers
            )
        )
        await asyncio.wait_for(started.wait(), 1)
        response = await asyncio.wait_for(self.client.get("/api/status"), 1)
        self.assertTrue((await response.json())["busy"])
        release.set()
        self.assertEqual((await task).status, 200)
        self.panel.server = None

    async def test_invalid_project_does_not_start_a_session(self):
        response = await self.client.post(
            "/api/session", json={"project_id": "missing"}, headers=self.headers
        )
        self.assertEqual(response.status, 409)
        self.assertIsNone(self.panel.task)


class PanelCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_android_start_saves_the_name_chosen_in_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            engine = SlamEngine(None, store, model_id="m", config_fingerprint="c")
            server = MobileServer(engine, project_name="Кухня")
            server._camera = CAMERA
            try:
                message = wire.ClientMessage(
                    command=wire.Command(command_id="named-start", type=wire.START)
                )
                await server._dispatch(message, "command")
                self.assertEqual(
                    store.read_manifest(engine.project_id)["name"], "Кухня"
                )
            finally:
                engine.project_id = None
                await server.stop()

    async def test_resume_requires_connected_phone(self):
        server = MobileServer(Mock())
        with self.assertRaises(ValueError):
            await server.execute_command("resume", "resume")

    async def test_shutdown_cancels_delayed_disconnect(self):
        engine = Mock()
        server = MobileServer(engine)
        task = asyncio.create_task(server._delayed_disconnect(0))
        server._disconnect_tasks.add(task)
        await server.stop()
        self.assertTrue(task.cancelled())
        engine.disconnect.assert_not_called()
        engine.close.assert_called_once()

    def test_legacy_forwards_options_after_separator(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("mast3r_slam.cli.subprocess.call", return_value=0) as call:
                self.assertEqual(
                    main(
                        [
                            "--projects-dir",
                            directory,
                            "legacy",
                            "--",
                            "--dataset",
                            "clip.mp4",
                        ]
                    ),
                    0,
                )
                self.assertEqual(call.call_args.args[0][-2:], ["--dataset", "clip.mp4"])

    def test_explicit_launch_modes(self):
        self.assertEqual(_parser().parse_args(["tui"]).command, "tui")
        args = _parser().parse_args(["web", "--port", "8090"])
        self.assertEqual(args.command, "web")
        self.assertEqual(args.port, 8090)
        self.assertEqual(args.host, "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
