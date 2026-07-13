from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Callable


class JobControlError(RuntimeError):
    """Base para interrupciones cooperativas solicitadas por la cola/UI."""


class JobPausedError(JobControlError):
    pass


class JobCancelledError(JobControlError):
    pass


@dataclass(frozen=True)
class ExternalCallReservation:
    """Marcador liviano para identificar una llamada externa activa."""

    kind: str
    provider: str


class ExecutionControl:
    """Control cooperativo de pausa y cancelación por trabajo.

    Las llamadas de traducción usan ``reserve_external_call`` como checkpoint antes de
    contactar un proveedor. La ejecución solo se detiene por pausa o cancelación.
    """

    def __init__(
        self,
        *,
        on_paused: Optional[Callable[[], None]] = None,
        on_resumed: Optional[Callable[[], None]] = None,
    ) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._pause_requested = False
        self._cancel_requested = False
        self._announced_paused = False
        self._on_paused = on_paused
        self._on_resumed = on_resumed

    @property
    def pause_requested(self) -> bool:
        with self._condition:
            return self._pause_requested

    @property
    def pause_triggered(self) -> bool:
        """Indica que un checkpoint aceptó la pausa y está liberando el worker."""
        with self._condition:
            return self._announced_paused

    @property
    def cancel_requested(self) -> bool:
        with self._condition:
            return self._cancel_requested

    def request_pause(self) -> None:
        with self._condition:
            self._pause_requested = True
            self._condition.notify_all()

    def resume(self) -> None:
        callback = None
        with self._condition:
            was_paused = self._pause_requested or self._announced_paused
            self._pause_requested = False
            self._announced_paused = False
            self._condition.notify_all()
            if was_paused:
                callback = self._on_resumed
        if callback:
            callback()

    def request_cancel(self) -> None:
        with self._condition:
            self._cancel_requested = True
            self._pause_requested = False
            self._condition.notify_all()

    def checkpoint(self, poll_seconds: float = 0.25) -> None:
        """Interrumpe en un punto seguro para liberar el worker de la cola."""
        del poll_seconds
        callback = None
        with self._condition:
            if self._cancel_requested:
                raise JobCancelledError("El trabajo fue cancelado por el usuario.")
            if not self._pause_requested:
                return
            if not self._announced_paused:
                self._announced_paused = True
                callback = self._on_paused
        if callback:
            callback()
        raise JobPausedError("El trabajo fue pausado por el usuario.")

    def reserve_external_call(
        self,
        *,
        kind: str,
        provider: str,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
        characters: int = 0,
    ) -> ExternalCallReservation:
        """Ejecuta un checkpoint antes de una llamada externa.

        Los argumentos de tamaño se mantienen para compatibilidad con los proveedores,
        pero no se almacenan ni condicionan la ejecución.
        """
        del estimated_input_tokens, estimated_output_tokens, characters
        self.checkpoint()
        normalized_kind = "llm" if str(kind).lower() == "llm" else "traditional"
        return ExternalCallReservation(
            kind=normalized_kind,
            provider=str(provider or normalized_kind),
        )

    def commit_external_call(
        self,
        reservation: ExternalCallReservation,
        *,
        actual_input_tokens: int | None = None,
        actual_output_tokens: int | None = None,
        actual_characters: int | None = None,
        failed: bool = False,
    ) -> None:
        """Finaliza una llamada y vuelve a comprobar pausa/cancelación."""
        del reservation, actual_input_tokens, actual_output_tokens, actual_characters, failed
        self.checkpoint()


_CURRENT_CONTROL: contextvars.ContextVar[ExecutionControl | None] = contextvars.ContextVar(
    "pmt_execution_control", default=None
)


@contextlib.contextmanager
def execution_control_scope(control: ExecutionControl | None) -> Iterator[ExecutionControl | None]:
    token = _CURRENT_CONTROL.set(control)
    try:
        yield control
    finally:
        _CURRENT_CONTROL.reset(token)


def get_execution_control() -> ExecutionControl | None:
    return _CURRENT_CONTROL.get()


def execution_checkpoint() -> None:
    control = get_execution_control()
    if control is not None:
        control.checkpoint()


def cooperative_sleep(seconds: float, poll_seconds: float = 0.25) -> None:
    """Espera que responde a pausa/cancelación entre intervalos cortos."""
    deadline = time.monotonic() + max(0.0, float(seconds))
    interval = max(0.01, float(poll_seconds))
    while time.monotonic() < deadline:
        execution_checkpoint()
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
    execution_checkpoint()
