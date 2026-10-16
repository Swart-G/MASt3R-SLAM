from __future__ import annotations

import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from mast3r_slam.mobile.camera import CameraIdentity
from mast3r_slam.mobile.projects import Checkpoint, ProjectStore
from mast3r_slam.tui import MainMenuScreen, ProjectActions, ProjectsScreen, ScanDashboard, ScannerApp


CAMERA = CameraIdentity("phone", "0", 1280, 720, (0, 0, 1280, 720), (900, 900, 640, 360), ())


class TuiNavigationTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_menu_and_project_action_dialog(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = store.create("Kitchen", CAMERA, model_id="m", config_fingerprint="c")
            store.save_revision(project, Checkpoint({"x": np.arange(1)}, {"keyframe_count": 1}), state="paused")
            app = ScannerApp(store)
            async with app.run_test(size=(120, 40)) as pilot:
                self.assertIsInstance(app.screen, MainMenuScreen)
                await pilot.click("#projects")
                await pilot.pause()
                self.assertIsInstance(app.screen, ProjectsScreen)
                self.assertIn(project, app.screen.items)
                await pilot.press("enter")
                await pilot.pause()
                self.assertIsInstance(app.screen, ProjectActions)
                await pilot.press("escape")
                await pilot.pause()
                self.assertIsInstance(app.screen, ProjectsScreen)

    async def test_pairing_qr_uses_the_full_dashboard(self):
        with tempfile.TemporaryDirectory() as directory:
            app = ScannerApp(ProjectStore(directory))
            async with app.run_test(size=(120, 44)) as pilot:
                screen = ScanDashboard(app.store)
                with patch.object(ScanDashboard, "_start_server", new=AsyncMock()):
                    app.push_screen(screen)
                    await pilot.pause()
                uri = (
                    "mast3r-slam://pair?host=192.168.1.10&port=8765&token="
                    + "a" * 32
                    + "&fingerprint="
                    + "b" * 64
                    + "&v=1"
                )
                qr = screen._qr_text(uri)
                screen.query_one("#pairing").update(qr)
                screen.phase = "READY"
                screen._render_state()
                await pilot.pause()
                pair = screen.query_one("#pair-panel")
                self.assertFalse(screen.query_one("#preview-panel").display)
                self.assertFalse(screen.query_one("#scan-actions").display)
                self.assertGreaterEqual(pair.content_region.height - 2, len(qr.splitlines()))

    async def test_unmounted_dashboard_ignores_connection_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            screen = ScanDashboard(ProjectStore(directory))
            screen.connected = True
            screen._connection_changed(False, "phone")
            self.assertTrue(screen.connected)

    async def test_dashboard_ignores_disconnect_after_children_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            app = ScannerApp(ProjectStore(directory))
            async with app.run_test(size=(120, 44)) as pilot:
                screen = ScanDashboard(app.store)
                with patch.object(ScanDashboard, "_start_server", new=AsyncMock()):
                    app.push_screen(screen)
                    await pilot.pause()
                await screen.query_children("*").remove()
                self.assertTrue(screen.is_attached)
                screen._connection_changed(False, "phone")

    async def test_unmount_stops_the_mobile_server(self):
        with tempfile.TemporaryDirectory() as directory:
            app = ScannerApp(ProjectStore(directory))
            async with app.run_test(size=(120, 44)) as pilot:
                screen = ScanDashboard(app.store)
                with patch.object(ScanDashboard, "_start_server", new=AsyncMock()):
                    app.push_screen(screen)
                    await pilot.pause()
                server = AsyncMock()
                screen.server = server
                await app.pop_screen()
                server.stop.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
