from __future__ import annotations

import dataclasses
import threading
import time
from typing import Optional


@dataclasses.dataclass(frozen=True, slots=True)
class NetworkFrame:
    frame_id: int
    timestamp_ns: int
    jpeg: bytes
    orientation_degrees: int = 0
    physical_camera_id: str = ""
    width: int = 0
    height: int = 0
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)
    intrinsics: tuple[float, ...] = ()
    distortion: tuple[float, ...] = ()
    received_ns: int = dataclasses.field(default_factory=time.monotonic_ns)


@dataclasses.dataclass(frozen=True, slots=True)
class SubmitResult:
    accepted: bool
    dropped_total: int
    replaced_frame_id: Optional[int] = None


class NetworkFrameSource:
    """A one-slot producer/consumer queue that always favors the newest frame.

    The local camera must never wait for SLAM. If processing is slower than the
    sender, `submit` atomically replaces the waiting frame and increments the
    drop counter. The frame currently being processed is not affected.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: Optional[NetworkFrame] = None
        self._closed = False
        self._dropped = 0
        self._accepted = 0
        self._last_submitted_id = -1

    def submit(self, frame: NetworkFrame) -> SubmitResult:
        with self._condition:
            if self._closed or frame.frame_id <= self._last_submitted_id:
                return SubmitResult(False, self._dropped)
            replaced = self._pending.frame_id if self._pending is not None else None
            if replaced is not None:
                self._dropped += 1
            self._pending = frame
            self._last_submitted_id = frame.frame_id
            self._accepted += 1
            self._condition.notify()
            return SubmitResult(True, self._dropped, replaced)

    def get(self, timeout: Optional[float] = None) -> Optional[NetworkFrame]:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._pending is not None or self._closed, timeout=timeout
            )
            if not ready or self._pending is None:
                return None
            frame, self._pending = self._pending, None
            return frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()

    def reset(self, *, reset_sequence: bool = False) -> None:
        """Discard a stale waiting frame and optionally start a new ID stream."""
        with self._condition:
            if self._pending is not None:
                self._pending = None
                self._dropped += 1
            if reset_sequence:
                self._last_submitted_id = -1

    @property
    def dropped(self) -> int:
        with self._condition:
            return self._dropped

    @property
    def accepted(self) -> int:
        with self._condition:
            return self._accepted

    @property
    def pending_frame_id(self) -> Optional[int]:
        with self._condition:
            return self._pending.frame_id if self._pending else None
