from __future__ import annotations

import asyncio
import hashlib
import io
import pathlib
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Header, Label, Static


ACTIVE_PHASES = {
    "SCANNING",
    "PAUSED",
    "AR_PREPARING",
    "AR_VIEW",
    "FINALIZING",
    "DISCONNECTED",
}


class ConfirmDialog(ModalScreen[bool]):
    def __init__(self, title: str, message: str, *, destructive: bool = False) -> None:
        super().__init__()
        self.dialog_title = title
        self.dialog_message = message
        self.destructive = destructive

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(classes="dialog"):
                yield Label(self.dialog_title, classes="dialog-title")
                yield Static(self.dialog_message, classes="dialog-copy")
                with Horizontal(classes="dialog-actions"):
                    yield Button("Cancel", id="cancel")
                    yield Button(
                        "Delete" if self.destructive else "Continue",
                        id="confirm",
                        variant="error" if self.destructive else "primary",
                    )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class MainMenuScreen(Screen):
    BINDINGS = [Binding("escape", "quit", "Exit", show=False)]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Center():
            with Vertical(id="menu-card"):
                yield Label("MASt3R", id="brand")
                yield Label("SLAM MOBILE SCANNER", id="tagline")
                yield Static("Reconstruct spaces from your phone", id="menu-copy")
                yield Button("New", id="new", variant="success", classes="menu-button")
                yield Button("Projects", id="projects", classes="menu-button")
                yield Button("Exit", id="exit", classes="menu-button")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#new", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new":
            self.app.push_screen(ScanDashboard(self.app.store))
        elif event.button.id == "projects":
            self.app.push_screen(ProjectsScreen(self.app.store))
        elif event.button.id == "exit":
            self.app.exit()

    def action_quit(self) -> None:
        self.app.exit()


class ProjectActions(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss_dialog", "Back", show=False)]

    def __init__(self, item: dict[str, Any]) -> None:
        super().__init__()
        self.item = item

    def compose(self) -> ComposeResult:
        has_revision = int(self.item["revision"]) > 0
        has_ply = has_revision and self.item.get("state") == "saved"
        with Center():
            with Vertical(classes="dialog project-dialog"):
                yield Label(self.item["name"], classes="dialog-title")
                yield Static(self.item["project_id"], classes="project-id")
                yield Button("[C]  Continue", id="continue", disabled=not has_revision)
                yield Button("[O]  Open", id="open", disabled=not has_revision)
                yield Button("[E]  Export", id="export", disabled=not has_ply)
                yield Button("[D]  Delete", id="delete", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id)

    def action_dismiss_dialog(self) -> None:
        self.dismiss(None)


class ProjectsScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("c", "continue_project", "Continue"),
        Binding("o", "open_project", "Open"),
        Binding("e", "export_project", "Export"),
        Binding("d", "delete_project", "Delete"),
    ]

    def __init__(self, store) -> None:
        super().__init__()
        self.store = store
        self.items: dict[str, dict[str, Any]] = {}
        self.exportError = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="projects-page"):
            yield Label("Projects", classes="page-title")
            yield Static(
                "Select with arrows or mouse · Enter for actions · C/O/E/D for shortcuts",
                classes="page-help",
            )
            yield DataTable(id="projects")
            yield Static("", id="project-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Name", "Updated", "Size", "Keyframes", "Revision", "State")
        self._refresh()
        table.focus()

    def _refresh(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        self.items.clear()
        for item in self.store.list_projects():
            revision = self.store.revision_path(item["project_id"])
            item = {**item, "has_ply": item.get("state") == "saved"}
            self.items[item["project_id"]] = item
            table.add_row(
                item["name"],
                item["updated_at"].replace("T", " ")[:19],
                f"{item['size_bytes'] / 1024 / 1024:.1f} MiB",
                str(item["keyframes"]),
                str(item["revision"]),
                item["state"].upper(),
                key=item["project_id"],
            )
        self.query_one("#project-status", Static).update(
            "No saved projects yet." if not self.items else f"{len(self.items)} project(s)"
        )

    def _selected(self) -> dict[str, Any] | None:
        table = self.query_one(DataTable)
        if table.row_count == 0:
            return None
        key = str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)
        return self.items.get(key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        item = self.items.get(str(event.row_key.value))
        if item:
            self.app.push_screen(ProjectActions(item), self._perform_action)

    async def _perform_action(self, action: str | None) -> None:
        if action:
            await self._execute(action)

    async def _execute(self, action: str) -> None:
        item = self._selected()
        status = self.query_one("#project-status", Static)
        if not item:
            status.update("Select a project first.")
            return
        project_id = item["project_id"]
        revision = self.store.revision_path(project_id)
        try:
            if action == "continue":
                if int(item["revision"]) < 1:
                    raise RuntimeError("This project does not have a checkpoint yet.")
                self.app.push_screen(ScanDashboard(self.store, project_id))
            elif action == "open":
                ply = revision / "reconstruction.ply"
                if not ply.is_file() and not (revision / "checkpoint.npz").is_file():
                    raise RuntimeError("This project has no reconstruction or checkpoint.")
                subprocess.Popen([sys.executable, "-m", "mast3r_slam.desktop_viewer", str(ply)])
                status.update("Opening reconstruction viewer…")
            elif action == "export":
                target = pathlib.Path(f"{project_id}.ply")
                exported = await asyncio.to_thread(self.store.export_ply, project_id, target)
                self.exportError = ""
                status.update(f"Exported to {exported}")
            elif action == "delete":
                self.app.push_screen(
                    ConfirmDialog(
                        "Delete project?",
                        f"{item['name']} and every revision will be permanently removed.",
                        destructive=True,
                    ),
                    lambda confirmed: self._delete_confirmed(project_id, confirmed),
                )
        except (OSError, RuntimeError, ValueError) as error:
            self.exportError = str(error)
            status.update(f"Error: {error}")

    def _delete_confirmed(self, project_id: str, confirmed: bool) -> None:
        if not confirmed:
            return
        try:
            self.store.delete_project(project_id)
            self._refresh()
            self.query_one("#project-status", Static).update("Project deleted.")
        except OSError as error:
            self.query_one("#project-status", Static).update(f"Delete failed: {error}")

    def action_back(self) -> None:
        self.app.pop_screen()

    async def action_continue_project(self) -> None:
        await self._execute("continue")

    async def action_open_project(self) -> None:
        await self._execute("open")

    async def action_export_project(self) -> None:
        await self._execute("export")

    async def action_delete_project(self) -> None:
        await self._execute("delete")


class EngineUiEvent(Message):
    def __init__(self, name: str, data: dict[str, Any]) -> None:
        super().__init__()
        self.name = name
        self.data = data


class ScanDashboard(Screen):
    BINDINGS = [Binding("escape", "request_close", "Back")]

    def __init__(self, store, project_id: str | None = None) -> None:
        super().__init__()
        self.store = store
        self.project_id = project_id
        self.engine = None
        self.server = None
        self.phase = "LOADING"
        self.connected = False
        self.stats: dict[str, Any] = {
            "tracking": "INITIALIZING",
            "keyframes": 0,
            "processed": 0,
            "input": 0,
            "latency": 0,
            "processing": 0,
        }
        self._latest_preview: bytes | None = None
        self._last_preview_at = 0.0
        self._shutdown_started = False
        self.saveProgress = 0.0
        self.operation = ""
        self.alignmentStatus = "SEARCHING"
        self.alignmentGuidance = ""
        self.exportError = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="scan-page"):
            with Horizontal(id="scan-heading"):
                yield Label("Live scan", classes="page-title")
                yield Static("LOADING", id="phase-pill")
            yield Static("Preparing the scanner…", id="scan-detail")
            with Horizontal(id="scan-content"):
                with Vertical(id="preview-panel", classes="panel"):
                    yield Label("SCENE PREVIEW", classes="panel-title")
                    yield Static("Waiting for the first processed frame", id="ascii-preview", markup=False)
                with Vertical(id="side-panel"):
                    with Vertical(id="pair-panel", classes="panel"):
                        yield Label("PAIR PHONE", classes="panel-title")
                        yield Static("Starting secure server…", id="pairing", markup=False)
                    with Vertical(id="stats-panel", classes="panel"):
                        yield Label("SCENE STATISTICS", classes="panel-title")
                        yield Static("", id="stats", markup=False)
            with Horizontal(id="scan-actions"):
                yield Button("Cancel", id="close", variant="error")
        yield Footer()

    async def on_mount(self) -> None:
        await self._start_server()

    async def on_unmount(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        if self.server:
            await self.server.stop()

    async def _start_server(self) -> None:
        try:
            from .mobile.engine import SlamEngine
            from .mobile.mast3r_processor import Mast3rProcessor
            from .mobile.server import MobileServer

            config_path = pathlib.Path("config/base.yaml")

            def build_engine():
                return SlamEngine(
                    Mast3rProcessor(str(config_path)),
                    self.store,
                    model_id="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                    config_fingerprint=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                )

            self.engine = await asyncio.to_thread(build_engine)
            self.engine.on("*", self._receive_engine_event)
            if self.project_id:
                await asyncio.to_thread(self.engine.load_project, self.project_id)
                manifest = self.store.read_manifest(self.project_id)
                if manifest["revisions"]:
                    self.stats["keyframes"] = manifest["revisions"][-1].get("keyframes", 0)
            self.server = MobileServer(self.engine)
            self.server.on_connection_changed(self._connection_changed)
            await self.server.start()
            self.phase = self.engine.session.phase.value
            self.query_one("#pairing", Static).update(self._qr_text(self.server.pairing_uri))
            self.query_one("#scan-detail", Static).update("Scan this QR code from the Android app.")
            self._render_state()
        except Exception as error:
            self.phase = "ERROR"
            self.query_one("#scan-detail", Static).update(f"Could not start scanner: {error}")
            self._render_state()

    def _receive_engine_event(self, event) -> None:
        message = EngineUiEvent(event.name, dict(event.data))
        if threading.current_thread() is threading.main_thread():
            self.post_message(message)
        else:
            self.app.call_from_thread(self.post_message, message)

    def _connection_changed(self, connected: bool, device_id: str | None) -> None:
        if self._shutdown_started or not self.is_running or not self.is_attached:
            return
        try:
            detail = self.query_one("#scan-detail", Static)
        except NoMatches:
            return
        self.connected = connected
        if connected:
            detail.update("Phone connected securely. Ready to scan.")
        else:
            detail.update("Phone disconnected. Scan the QR code to reconnect.")
        self._render_state()

    def on_engine_ui_event(self, event: EngineUiEvent) -> None:
        data = event.data
        if event.name == "state_changed":
            self.phase = data.get("phase", self.phase)
        elif event.name == "map_updated":
            self.stats["keyframes"] = data.get("keyframes", 0)
        elif event.name == "frame_processed":
            self.stats.update(
                tracking=data.get("tracking", "—"),
                input=data.get("input_frame_id", 0),
                processed=data.get("processed_frame_id", 0),
                latency=data.get("latency_ms", 0),
                processing=data.get("processing_ms", 0),
            )
            now = time.monotonic()
            preview = data.get("preview_jpeg")
            if preview and now - self._last_preview_at >= 0.25:
                self._last_preview_at = now
                self._latest_preview = preview
                self._render_ascii()
            guidance = data.get("guidance")
            if guidance:
                self.query_one("#scan-detail", Static).update(guidance)
        elif event.name == "camera_mismatch":
            self.query_one("#scan-detail", Static).update("Camera changed; confirm the warning on Android.")
        elif event.name == "capacity_warning":
            self.query_one("#scan-detail", Static).update(
                f"Map capacity is {data.get('percent', 0)}% full."
            )
        elif event.name == "progress":
            self.operation = data.get("operation", "")
            if self.operation == "save":
                self.saveProgress = float(data.get("fraction", 0.0))
            self.query_one("#scan-detail", Static).update(data.get("detail", self.operation))
        elif event.name == "error":
            if not data.get("recoverable", False):
                self.phase = "ERROR"
            if data.get("code") == "SAVE_FAILED":
                self.exportError = data.get("message", "")
            self.query_one("#scan-detail", Static).update(data.get("message", "Scanner error"))
        elif event.name == "finalized":
            self.phase = "SAVED"
            self.query_one("#scan-detail", Static).update("Project saved successfully.")
        self._render_state()

    def on_resize(self) -> None:
        if self._latest_preview:
            self._render_ascii()

    def _render_ascii(self) -> None:
        if not self._latest_preview:
            return
        try:
            from PIL import Image

            widget = self.query_one("#ascii-preview", Static)
            width = max(20, widget.size.width - 2)
            height = max(6, widget.size.height - 1)
            image = Image.open(io.BytesIO(self._latest_preview)).convert("L")
            image.thumbnail((width, height * 2))
            ramp = " .:-=+*#%@"
            rows = []
            for y in range(image.height):
                rows.append("".join(ramp[value * (len(ramp) - 1) // 255] for value in list(image.crop((0, y, image.width, y + 1)).getdata())))
            widget.update("\n".join(rows))
        except Exception:
            self.query_one("#ascii-preview", Static).update("Preview unavailable")

    def _render_state(self) -> None:
        try:
            pill = self.query_one("#phase-pill", Static)
        except NoMatches:
            return
        pill.update(self.phase.replace("_", " "))
        pill.set_classes(f"phase-{self.phase.lower()}")
        preview_panel = self.query_one("#preview-panel", Vertical)
        pair_panel = self.query_one("#pair-panel", Vertical)
        stats_panel = self.query_one("#stats-panel", Vertical)
        detail = self.query_one("#scan-detail", Static)
        actions = self.query_one("#scan-actions", Horizontal)
        pairing_mode = not self.connected and self.phase != "ERROR"
        if not pairing_mode:
            preview_panel.display = True
            pair_panel.display = False
            stats_panel.display = True
            detail.display = True
            actions.display = True
        else:
            preview_panel.display = False
            pair_panel.display = True
            stats_panel.display = False
            detail.display = False
            actions.display = False
        dropped = self.engine.source.dropped if self.engine else 0
        accepted = self.engine.source.accepted if self.engine else 0
        project = self.engine.project_id if self.engine and self.engine.project_id else "Not created"
        revision = self.engine.revision if self.engine else 0
        self.query_one("#stats", Static).update(
            "\n".join(
                (
                    f"Application   {self.phase}",
                    f"Phone         {'CONNECTED' if self.connected else 'WAITING'}",
                    f"Tracking      {self.stats['tracking']}",
                    "",
                    f"Keyframes     {self.stats['keyframes']}",
                    f"Frames        {self.stats['processed']} / {self.stats['input']}",
                    f"Accepted      {accepted}",
                    f"Dropped       {dropped}",
                    f"Latency       {self.stats['latency']} ms",
                    f"Processing    {self.stats['processing']} ms",
                    "",
                    f"Revision      {revision}",
                    f"Project       {project}",
                )
            )
        )
        button = self.query_one("#close", Button)
        if self.phase in {"SAVED", "ERROR"}:
            button.label = "Back to menu"
            button.variant = "primary"
        elif self.phase in ACTIVE_PHASES:
            button.label = "End scan"
            button.variant = "error"
        else:
            button.label = "Cancel"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.action_request_close()

    def action_request_close(self) -> None:
        has_active_project = bool(self.engine and self.engine.project_id)
        if self.phase in ACTIVE_PHASES and self.phase != "FINALIZING" and has_active_project:
            self.app.push_screen(
                ConfirmDialog("End active scan?", "The current revision will be finalized and saved."),
                self._close_confirmed,
            )
        elif self.phase != "FINALIZING":
            asyncio.create_task(self._stop_and_back(finalize=False))

    def _close_confirmed(self, confirmed: bool) -> None:
        if confirmed:
            asyncio.create_task(self._stop_and_back(finalize=True))

    async def _stop_and_back(self, *, finalize: bool) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        try:
            if finalize and self.engine and self.engine.session.phase.value in ACTIVE_PHASES:
                await asyncio.to_thread(self.engine.stop, str(uuid.uuid4()))
            if self.server:
                await self.server.stop()
        finally:
            self.app.pop_screen()

    @staticmethod
    def _qr_text(uri: str) -> str:
        try:
            import qrcode

            output = io.StringIO()
            qr = qrcode.QRCode(border=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
            qr.add_data(uri)
            qr.make(fit=True)
            qr.print_ascii(out=output, invert=True)
            return output.getvalue() + "\n" + uri
        except ImportError:
            return uri


class ScannerApp(App):
    TITLE = "MASt3R-SLAM"
    CSS = """
    Screen { background: #080d12; color: #e7f4f3; }
    Header { background: #0d171d; color: #72e0ce; }
    Footer { background: #0d171d; color: #b4c9c6; }
    #menu-card { width: 58; height: auto; padding: 2 6; border: tall #1f8076; background: #0d171d; }
    #brand { width: 100%; text-align: center; color: #72e0ce; text-style: bold; }
    #tagline { width: 100%; text-align: center; color: #7faaa4; margin-bottom: 1; }
    #menu-copy { width: 100%; text-align: center; margin-bottom: 1; color: #b4c9c6; }
    .menu-button { width: 100%; margin-top: 1; height: 3; }
    Button { background: #14242c; color: #e7f4f3; border: tall #28505a; }
    Button:hover, Button:focus { background: #1f8076; border: tall #72e0ce; }
    .page-title { color: #72e0ce; text-style: bold; height: 2; }
    .page-help { color: #7faaa4; height: 2; }
    #projects-page, #scan-page { padding: 1 2; }
    #projects { height: 1fr; border: tall #28505a; background: #0b1318; }
    #project-status { height: 2; color: #b4c9c6; padding-left: 1; }
    .dialog { width: 58; height: auto; padding: 1 2; border: tall #72e0ce; background: #0d171d; }
    .project-dialog { width: 48; }
    .dialog-title { width: 100%; text-align: center; color: #72e0ce; text-style: bold; margin-bottom: 1; }
    .dialog-copy, .project-id { width: 100%; text-align: center; color: #b4c9c6; margin-bottom: 1; }
    .dialog-actions { width: 100%; height: 3; align-horizontal: center; }
    .dialog-actions Button { margin: 0 1; min-width: 16; }
    .project-dialog Button { width: 100%; margin-top: 1; }
    ConfirmDialog, ProjectActions { align: center middle; background: #000000 65%; }
    #scan-heading { height: 3; }
    #scan-heading .page-title { width: 1fr; }
    #phase-pill { width: auto; min-width: 16; text-align: center; padding: 0 2; color: #08110f; background: #72e0ce; text-style: bold; }
    .phase-error { background: #ff6b6b !important; }
    .phase-paused, .phase-disconnected { background: #ffc857 !important; }
    .phase-saved { background: #64d98b !important; }
    #scan-detail { height: 3; padding: 0 1; color: #ffc857; }
    #scan-content { height: 1fr; }
    .panel { border: tall #28505a; background: #0b1318; padding: 1; }
    .panel-title { color: #72e0ce; text-style: bold; height: 2; }
    #preview-panel { width: 2fr; margin-right: 1; }
    #ascii-preview { height: 1fr; content-align: center middle; color: #8de8d9; overflow: hidden; }
    #side-panel { width: 1fr; }
    #pair-panel { height: 3fr; margin-bottom: 1; }
    #pairing { height: 1fr; content-align: center middle; overflow: hidden; color: #e7f4f3; }
    #stats-panel { height: 2fr; }
    #stats { height: 1fr; color: #b4c9c6; }
    #scan-actions { height: 4; align-horizontal: right; padding-top: 1; }
    #scan-actions Button { min-width: 20; }
    """

    def __init__(self, store) -> None:
        super().__init__()
        self.store = store

    def on_mount(self) -> None:
        self.push_screen(MainMenuScreen())


def run_tui(store) -> int:
    ScannerApp(store).run()
    return 0
