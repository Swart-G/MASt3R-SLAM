from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
import shutil
import threading
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable, Protocol

from .camera import CameraIdentity, compare_cameras
from .ar_alignment import ar_world_from_slam_map, from_column_major, sim3_matrix, to_column_major
from .frame_source import NetworkFrame, NetworkFrameSource, SubmitResult
from .projects import Checkpoint, ProjectStore
from .state_machine import SessionMachine, SessionPhase


LOGGER = logging.getLogger(__name__)


class Tracking(str, Enum):
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    LOST = "LOST"
    RELOCALIZING = "RELOCALIZING"


@dataclasses.dataclass(slots=True)
class ProcessResult:
    frame_id: int
    tracking: Tracking
    preview_jpeg: bytes | None = None
    map_updated: bool = False
    keyframe_count: int = 0
    guidance: str = ""


class ArAlignmentStatus(str, Enum):
    SEARCHING = "SEARCHING"
    ALIGNED = "ALIGNED"
    LOST = "LOST"


@dataclasses.dataclass(frozen=True, slots=True)
class ArAlignmentFrame:
    frame_id: int
    timestamp_ns: int
    jpeg: bytes
    intrinsics: tuple[float, ...]
    ar_world_from_camera: tuple[float, ...]
    width: int
    height: int


@dataclasses.dataclass(frozen=True, slots=True)
class ArAlignmentResult:
    frame_id: int
    status: ArAlignmentStatus
    quality: float = 0.0
    ar_world_from_slam_map: tuple[float, ...] = ()
    guidance: str = "Наведите камеру на уже отсканированную область"


class SlamProcessor(Protocol):
    def process(self, frame: NetworkFrame, *, relocalizing: bool) -> ProcessResult: ...
    def checkpoint(self) -> Checkpoint: ...
    def load_checkpoint(self, checkpoint: Checkpoint) -> None: ...
    def finalize(self) -> bytes | None: ...
    def mobile_snapshot(self, output_dir: str) -> Any: ...
    def align_ar(self, frame: ArAlignmentFrame) -> tuple[Any, float] | None: ...


@dataclasses.dataclass(frozen=True, slots=True)
class EngineEvent:
    name: str
    data: dict[str, Any]


class SlamEngine:
    """UI- and Dataset-independent lifecycle around a SLAM processor."""

    def __init__(
        self,
        processor: SlamProcessor,
        projects: ProjectStore,
        *,
        model_id: str,
        config_fingerprint: str,
        keyframe_limit: int = 512,
    ) -> None:
        self.processor = processor
        self.projects = projects
        self.model_id = model_id
        self.config_fingerprint = config_fingerprint
        self.keyframe_limit = keyframe_limit
        self.source = NetworkFrameSource()
        self.session = SessionMachine()
        self.session.paired()
        self.project_id: str | None = None
        self.revision = 0
        self.camera: CameraIdentity | None = None
        self._listeners: dict[str, list[Callable[[EngineEvent], None]]] = defaultdict(list)
        self._worker: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._processor_lock = threading.RLock()
        self._needs_relocalization = False
        self._last_good_preview: bytes | None = None
        self._warned_thresholds: set[int] = set()
        self.last_error = ""
        self.snapshot_dir: pathlib.Path | None = None
        self._snapshot_manifest: Any = None
        self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="project-save")
        self._save_lock = threading.RLock()
        self._save_future: Future[None] | None = None
        self._save_error: Exception | None = None
        self._map_dirty = False

    def on(self, event: str, callback: Callable[[EngineEvent], None]) -> None:
        self._listeners[event].append(callback)

    def submit(self, frame: NetworkFrame):
        if self.session.phase != SessionPhase.SCANNING:
            return SubmitResult(False, self.source.dropped)
        return self.source.submit(frame)

    def begin_frame_stream(self) -> None:
        # Android frame IDs are scoped to an app/WebSocket lifetime and restart
        # from one after the process is recreated.
        self.source.reset(reset_sequence=True)

    def start(self, command_id: str, name: str, camera: CameraIdentity) -> str:
        transition = self.session.command(command_id, "start")
        if transition.repeated:
            self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)
            return self.project_id or ""
        self.camera = camera
        self.last_error = ""
        self.project_id = self.projects.create(
            name,
            camera,
            model_id=self.model_id,
            config_fingerprint=self.config_fingerprint,
        )
        self._map_dirty = True
        self._ensure_worker()
        self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)
        return self.project_id

    def pause(self, command_id: str) -> None:
        transition = self.session.command(command_id, "pause")
        if not transition.repeated:
            self.source.reset()
        self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)
        if not transition.repeated:
            self._start_save()

    def resume(
        self,
        command_id: str,
        camera: CameraIdentity,
        *,
        camera_change_confirmed: bool = False,
    ) -> None:
        if self.camera:
            comparison = compare_cameras(self.camera, camera)
            if comparison.requires_confirmation and not camera_change_confirmed:
                self._emit("camera_mismatch", differences=comparison.differences)
                raise ValueError("camera configuration changed; confirmation required")
        self._wait_for_save()
        transition = self.session.command(command_id, "resume")
        if not transition.repeated:
            self._needs_relocalization = True
        self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)

    def stop(self, command_id: str) -> None:
        transition = self.session.command(command_id, "stop")
        if transition.repeated:
            self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)
            return
        self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)
        self.source.reset()
        try:
            try:
                self._wait_for_save()
            except RuntimeError:
                # A failed background pause must be retried by Stop while the
                # last verified revision remains untouched.
                pass
            if transition.previous == SessionPhase.SCANNING or self.revision == 0 or self._save_error:
                self._start_save(force=True)
                self._wait_for_save()
        except Exception:
            self.session.finalization_failed(command_id)
            self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)
            raise
        self.session.finalized()
        self._emit("finalized", project_id=self.project_id, revision=self.revision)

    def enter_ar(self, command_id: str) -> Any:
        self._wait_for_save()
        transition = self.session.command(command_id, "enter_ar")
        if transition.repeated:
            if self._snapshot_manifest is not None:
                self._emit("cloud_ready", manifest=self._snapshot_manifest)
            self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)
            return self._snapshot_manifest
        if not self.project_id:
            raise RuntimeError("no active project")
        self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)
        snapshot_root = self.projects.project_path(self.project_id) / "mobile_snapshots"
        snapshot_root.mkdir(exist_ok=True)
        temporary = snapshot_root / f".{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            with self._processor_lock:
                manifest = self.processor.mobile_snapshot(str(temporary))
            snapshot_dir = snapshot_root / manifest.snapshot_id
            if snapshot_dir.exists():
                raise RuntimeError("snapshot id collision")
            os.replace(temporary, snapshot_dir)
            self.snapshot_dir = snapshot_dir
            self._snapshot_manifest = manifest
        except Exception as error:
            shutil.rmtree(temporary, ignore_errors=True)
            self.session.ar_failed(command_id)
            self.last_error = f"{type(error).__name__}: {error}"
            self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)
            self._emit("error", code="AR_PREPARATION_FAILED", message=str(error), recoverable=True)
            raise
        self.session.ar_ready()
        self._emit("cloud_ready", manifest=manifest)
        self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)
        return manifest

    def leave_ar(self, command_id: str) -> None:
        self.session.command(command_id, "leave_ar")
        self._emit("state_changed", phase=self.session.phase.value, command_id=command_id)

    def align_ar(self, frame: ArAlignmentFrame) -> ArAlignmentResult:
        if self.session.phase != SessionPhase.AR_VIEW:
            raise RuntimeError("AR alignment frames are accepted only in AR_VIEW")
        if len(frame.intrinsics) != 4 or len(frame.ar_world_from_camera) != 16:
            raise ValueError("AR alignment frame has invalid calibration or pose")
        with self._processor_lock:
            match = self.processor.align_ar(frame)
        if match is None:
            return ArAlignmentResult(frame.frame_id, ArAlignmentStatus.SEARCHING)
        slam_pose, quality = match
        transform = ar_world_from_slam_map(
            from_column_major(frame.ar_world_from_camera),
            sim3_matrix(slam_pose),
        )
        return ArAlignmentResult(
            frame.frame_id,
            ArAlignmentStatus.ALIGNED,
            float(quality),
            to_column_major(transform),
            "Совмещение выполнено",
        )

    def load_project(
        self, project_id: str, revision: int | None = None, *, for_continue: bool = True
    ) -> None:
        checkpoint = self.projects.load_revision(
            project_id,
            revision,
            expected_model_id=self.model_id,
            expected_config_fingerprint=self.config_fingerprint,
        )
        with self._processor_lock:
            self.processor.load_checkpoint(checkpoint)
        self.project_id = project_id
        project_manifest = self.projects.read_manifest(project_id)
        saved_camera = project_manifest["camera"]
        self.camera = CameraIdentity(
            saved_camera["device_model"],
            saved_camera["physical_camera_id"],
            saved_camera["width"],
            saved_camera["height"],
            tuple(saved_camera["crop"]),
            tuple(saved_camera["intrinsics"]),
            tuple(saved_camera.get("distortion", ())),
        )
        self.revision = revision or int(project_manifest["latest_revision"])
        self._map_dirty = False
        self._needs_relocalization = True
        self.session.project_loaded(for_continue=for_continue)
        self._ensure_worker()
        self._emit("state_changed", phase=self.session.phase.value)

    def disconnect(self) -> None:
        was_scanning = self.session.phase == SessionPhase.SCANNING
        self.session.disconnect()
        self.source.reset()
        self._emit("state_changed", phase=self.session.phase.value)
        if was_scanning:
            self._start_save(force=True)
        if was_scanning or self._save_future is not None:
            try:
                self._wait_for_save()
            except RuntimeError:
                self._start_save(force=True)
                self._wait_for_save()

    def close(self) -> None:
        if self.project_id and (self._map_dirty or self.session.phase == SessionPhase.SCANNING):
            self.source.reset()
            self._start_save(force=True)
        if self._save_future is not None:
            try:
                self._wait_for_save()
            except RuntimeError:
                try:
                    self._start_save(force=True)
                    self._wait_for_save()
                except RuntimeError:
                    LOGGER.exception("final project save failed during shutdown")
        self._shutdown.set()
        self.source.close()
        if self._worker:
            self._worker.join(timeout=5)
        self._save_executor.shutdown(wait=True, cancel_futures=False)

    def _ensure_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._shutdown.clear()
        self._worker = threading.Thread(target=self._run, name="slam-engine", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while not self._shutdown.is_set():
            if self.session.phase != SessionPhase.SCANNING:
                time.sleep(0.01)
                continue
            frame = self.source.get(timeout=0.1)
            if frame is None:
                continue
            started = time.monotonic_ns()
            try:
                with self._processor_lock:
                    result = self.processor.process(frame, relocalizing=self._needs_relocalization)
            except Exception as error:
                detail = f"{type(error).__name__}: {error}"
                self.last_error = detail
                self._append_journal("PROCESSING_FAILED", traceback.format_exc())
                LOGGER.exception("live frame %s processing failed", frame.frame_id)
                self.session.fail()
                self._emit(
                    "error",
                    code="PROCESSING_FAILED",
                    message=detail,
                    recoverable=False,
                )
                continue
            if self._needs_relocalization:
                if result.tracking == Tracking.TRACKING:
                    self._needs_relocalization = False
                else:
                    result.map_updated = False
                    result.guidance = result.guidance or "Вернитесь к уже отснятой области"
            if result.tracking == Tracking.LOST:
                result.map_updated = False
                result.preview_jpeg = self._last_good_preview
                result.guidance = result.guidance or "Вернитесь к уже отснятой области"
                self._emit("tracking_lost", frame_id=frame.frame_id, guidance=result.guidance)
            elif result.preview_jpeg:
                self._last_good_preview = result.preview_jpeg
            if result.map_updated:
                self._map_dirty = True
                self._emit("map_updated", keyframes=result.keyframe_count)
                self._check_capacity(result.keyframe_count)
            latency_ms = int((time.monotonic_ns() - frame.received_ns) / 1_000_000)
            self._emit(
                "frame_processed",
                input_frame_id=frame.frame_id,
                processed_frame_id=result.frame_id,
                tracking=result.tracking.value,
                latency_ms=latency_ms,
                processing_ms=int((time.monotonic_ns() - started) / 1_000_000),
                preview_jpeg=result.preview_jpeg,
                guidance=result.guidance,
            )

    def _check_capacity(self, count: int) -> None:
        for threshold in (80, 95):
            if count >= self.keyframe_limit * threshold / 100 and threshold not in self._warned_thresholds:
                self._warned_thresholds.add(threshold)
                self._emit("capacity_warning", percent=threshold, keyframes=count, limit=self.keyframe_limit)
        if count >= self.keyframe_limit:
            self._emit("error", code="KEYFRAME_LIMIT", message="keyframe buffer is full", recoverable=True)

    def _start_save(self, *, force: bool = False) -> Future[None] | None:
        if not self.project_id:
            return None
        with self._save_lock:
            if self._save_future is not None and not self._save_future.done():
                return self._save_future
            if not force and not self._map_dirty and self.revision > 0:
                return None
            self._save_error = None
            self._save_future = self._save_executor.submit(self._save_revision_job, self.project_id)
            return self._save_future

    def _save_revision_job(self, project_id: str) -> None:
        self._emit("progress", operation="save", fraction=0.0, detail="Saving project")
        try:
            # Lazy processor mappings must be consumed while processor access is
            # serialized, otherwise Resume could mutate the graph mid-revision.
            with self._processor_lock:
                checkpoint = self.processor.checkpoint()
                info = self.projects.save_revision(
                    project_id,
                    checkpoint,
                    state="saved",
                    progress_callback=lambda fraction, detail: self._emit(
                        "progress", operation="save", fraction=fraction, detail=detail
                    ),
                )
            self.revision = info.number
            self._map_dirty = False
            self._save_error = None
            self._emit("save_completed", project_id=project_id, revision=info.number)
        except Exception as error:
            self._save_error = error
            self.last_error = f"{type(error).__name__}: {error}"
            self._append_journal("SAVE_FAILED", traceback.format_exc())
            self._emit(
                "error",
                code="SAVE_FAILED",
                message=f"Project save failed: {error}",
                recoverable=True,
            )
            raise

    def _wait_for_save(self) -> None:
        with self._save_lock:
            future = self._save_future
        if future is not None:
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(f"project save has not completed successfully: {error}") from error
        if self._save_error is not None:
            raise RuntimeError(f"project save has not completed successfully: {self._save_error}")

    def _append_journal(self, code: str, detail: str) -> None:
        if not self.project_id:
            return
        try:
            journal = self.projects.project_path(self.project_id) / "journal.log"
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with journal.open("a", encoding="utf-8") as stream:
                stream.write(f"[{timestamp}] {code}\n{detail.rstrip()}\n\n")
        except OSError:
            LOGGER.exception("failed to append project journal")

    def _emit(self, name: str, **data: Any) -> None:
        event = EngineEvent(name, data)
        for callback in tuple(self._listeners.get(name, ())) + tuple(self._listeners.get("*", ())):
            callback(event)
