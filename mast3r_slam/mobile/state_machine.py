from __future__ import annotations

import dataclasses
from enum import Enum
from threading import RLock


class SessionPhase(str, Enum):
    PAIRING = "PAIRING"
    READY = "READY"
    SCANNING = "SCANNING"
    PAUSED = "PAUSED"
    AR_PREPARING = "AR_PREPARING"
    AR_VIEW = "AR_VIEW"
    FINALIZING = "FINALIZING"
    SAVED = "SAVED"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"


class InvalidTransition(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class Transition:
    command_id: str
    command: str
    previous: SessionPhase
    current: SessionPhase
    repeated: bool = False


class SessionMachine:
    """Thread-safe, idempotent protocol state machine."""

    _allowed = {
        "start": ({SessionPhase.READY}, SessionPhase.SCANNING),
        "pause": ({SessionPhase.SCANNING}, SessionPhase.PAUSED),
        "resume": ({SessionPhase.PAUSED, SessionPhase.AR_VIEW}, SessionPhase.SCANNING),
        "enter_ar": ({SessionPhase.PAUSED, SessionPhase.SAVED}, SessionPhase.AR_PREPARING),
        "leave_ar": ({SessionPhase.AR_VIEW}, SessionPhase.PAUSED),
        "stop": (
            {
                SessionPhase.SCANNING,
                SessionPhase.PAUSED,
                SessionPhase.AR_PREPARING,
                SessionPhase.AR_VIEW,
                SessionPhase.DISCONNECTED,
            },
            SessionPhase.FINALIZING,
        ),
    }

    def __init__(self) -> None:
        self._phase = SessionPhase.PAIRING
        self._history: dict[str, Transition] = {}
        self._lock = RLock()
        self._disconnect_origin: SessionPhase | None = None

    @property
    def phase(self) -> SessionPhase:
        with self._lock:
            return self._phase

    def paired(self) -> SessionPhase:
        with self._lock:
            if self._phase not in {SessionPhase.PAIRING, SessionPhase.DISCONNECTED}:
                raise InvalidTransition(f"cannot pair while {self._phase.value}")
            self._phase = (
                SessionPhase.PAUSED
                if self._disconnect_origin not in {None, SessionPhase.READY}
                else SessionPhase.READY
            )
            self._disconnect_origin = None
            return self._phase

    def project_loaded(self, *, for_continue: bool) -> SessionPhase:
        with self._lock:
            if self._phase not in {SessionPhase.READY, SessionPhase.SAVED}:
                raise InvalidTransition(f"cannot load a project while {self._phase.value}")
            self._phase = SessionPhase.PAUSED if for_continue else SessionPhase.SAVED
            return self._phase

    def command(self, command_id: str, command: str) -> Transition:
        command = command.lower()
        if not command_id:
            raise ValueError("command_id is required")
        with self._lock:
            previous_result = self._history.get(command_id)
            if previous_result:
                if previous_result.command != command:
                    raise InvalidTransition("command_id was already used for another command")
                return dataclasses.replace(previous_result, repeated=True)
            if command not in self._allowed:
                raise ValueError(f"unknown command: {command}")
            sources, target = self._allowed[command]
            if self._phase not in sources:
                raise InvalidTransition(
                    f"{command} is not valid while {self._phase.value}"
                )
            result = Transition(command_id, command, self._phase, target)
            self._phase = target
            self._history[command_id] = result
            return result

    def ar_ready(self) -> None:
        self._advance(SessionPhase.AR_PREPARING, SessionPhase.AR_VIEW)

    def ar_failed(self, command_id: str) -> None:
        """An AR snapshot failure is recoverable and preserves the paused map."""
        self._rollback(command_id, "enter_ar", SessionPhase.AR_PREPARING, SessionPhase.PAUSED)

    def finalized(self) -> None:
        self._advance(SessionPhase.FINALIZING, SessionPhase.SAVED)

    def finalization_failed(self, command_id: str) -> None:
        self._rollback(command_id, "stop", SessionPhase.FINALIZING, SessionPhase.PAUSED)

    def fail(self) -> None:
        with self._lock:
            self._phase = SessionPhase.ERROR

    def disconnect(self) -> None:
        with self._lock:
            if self._phase in {SessionPhase.SAVED, SessionPhase.ERROR}:
                return
            self._disconnect_origin = self._phase
            self._phase = SessionPhase.DISCONNECTED

    def _advance(self, source: SessionPhase, target: SessionPhase) -> None:
        with self._lock:
            if self._phase != source:
                raise InvalidTransition(
                    f"expected {source.value}, currently {self._phase.value}"
                )
            self._phase = target

    def _rollback(
        self,
        command_id: str,
        command: str,
        source: SessionPhase,
        target: SessionPhase,
    ) -> None:
        with self._lock:
            transition = self._history.get(command_id)
            if self._phase != source or transition is None or transition.command != command:
                raise InvalidTransition(f"cannot roll back {command} while {self._phase.value}")
            self._phase = target
            del self._history[command_id]
