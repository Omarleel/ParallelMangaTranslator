from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterator, Optional


class JobControlError(RuntimeError):
    """Base para interrupciones cooperativas solicitadas por la cola/UI."""


class JobPausedError(JobControlError):
    pass


class JobCancelledError(JobControlError):
    pass


class ExternalUsageLimitError(JobControlError):
    pass


@dataclass(frozen=True)
class ExternalUsageLimits:
    """Límites por trabajo. Cero significa sin límite.

    Los precios son configurables porque cambian entre proveedores/modelos. El coste
    se estima antes de cada llamada y se reconcilia con el uso real cuando el SDK lo
    devuelve.
    """

    max_total_calls: int = 0
    max_llm_calls: int = 0
    max_traditional_calls: int = 0
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    max_characters: int = 0
    max_cost_usd: float = 0.0
    llm_input_cost_per_million_tokens: float = 0.0
    llm_output_cost_per_million_tokens: float = 0.0
    traditional_cost_per_million_characters: float = 0.0


@dataclass
class ExternalUsageSnapshot:
    total_calls: int = 0
    llm_calls: int = 0
    traditional_calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    characters: int = 0
    estimated_cost_usd: float = 0.0
    last_provider: str = ""
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["estimated_cost_usd"] = round(float(self.estimated_cost_usd), 8)
        return payload


@dataclass(frozen=True)
class UsageReservation:
    kind: str
    provider: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    characters: int
    estimated_cost_usd: float


class ExecutionControl:
    """Control cooperativo de pausa/cancelación y presupuesto por trabajo."""

    def __init__(
        self,
        *,
        limits: ExternalUsageLimits | None = None,
        initial_usage: Dict[str, Any] | None = None,
        on_paused: Optional[Callable[[], None]] = None,
        on_resumed: Optional[Callable[[], None]] = None,
        on_usage_changed: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.limits = limits or ExternalUsageLimits()
        allowed = ExternalUsageSnapshot.__dataclass_fields__
        safe_usage = {k: v for k, v in dict(initial_usage or {}).items() if k in allowed}
        self.usage = ExternalUsageSnapshot(**safe_usage)
        self._condition = threading.Condition(threading.RLock())
        self._pause_requested = False
        self._cancel_requested = False
        self._announced_paused = False
        self._on_paused = on_paused
        self._on_resumed = on_resumed
        self._on_usage_changed = on_usage_changed

    @property
    def pause_requested(self) -> bool:
        with self._condition:
            return self._pause_requested

    @property
    def pause_triggered(self) -> bool:
        """Indica que un checkpoint ya aceptó la pausa y está liberando el worker."""
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
        """Interrumpe en un punto seguro para liberar el worker de la cola.

        La pausa no deja un hilo bloqueado ocupando la GPU. En su lugar, propaga una
        excepción de control que guarda el manifiesto y devuelve el trabajo a estado
        pausado. Al reanudar se procesa de nuevo únicamente la página interrumpida.
        """
        del poll_seconds  # compatibilidad con llamadas anteriores
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

    @staticmethod
    def _cost(
        limits: ExternalUsageLimits,
        kind: str,
        input_tokens: int,
        output_tokens: int,
        characters: int,
    ) -> float:
        if kind == "llm":
            return (
                max(0, input_tokens) * limits.llm_input_cost_per_million_tokens
                + max(0, output_tokens) * limits.llm_output_cost_per_million_tokens
            ) / 1_000_000.0
        return max(0, characters) * limits.traditional_cost_per_million_characters / 1_000_000.0

    def reserve_external_call(
        self,
        *,
        kind: str,
        provider: str,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
        characters: int = 0,
    ) -> UsageReservation:
        self.checkpoint()
        kind = "llm" if str(kind).lower() == "llm" else "traditional"
        reservation_cost = self._cost(
            self.limits,
            kind,
            int(estimated_input_tokens),
            int(estimated_output_tokens),
            int(characters),
        )
        with self._condition:
            u, limits = self.usage, self.limits
            next_total = u.total_calls + 1
            next_kind = (u.llm_calls if kind == "llm" else u.traditional_calls) + 1
            next_input = u.input_tokens + max(0, int(estimated_input_tokens))
            next_output = u.output_tokens + max(0, int(estimated_output_tokens))
            next_chars = u.characters + max(0, int(characters))
            next_cost = u.estimated_cost_usd + reservation_cost

            failures = []
            if limits.max_total_calls and next_total > limits.max_total_calls:
                failures.append(f"máximo de llamadas ({limits.max_total_calls})")
            if kind == "llm" and limits.max_llm_calls and next_kind > limits.max_llm_calls:
                failures.append(f"máximo de llamadas LLM ({limits.max_llm_calls})")
            if kind == "traditional" and limits.max_traditional_calls and next_kind > limits.max_traditional_calls:
                failures.append(f"máximo de llamadas tradicionales ({limits.max_traditional_calls})")
            if limits.max_input_tokens and next_input > limits.max_input_tokens:
                failures.append(f"máximo de tokens de entrada ({limits.max_input_tokens})")
            if limits.max_output_tokens and next_output > limits.max_output_tokens:
                failures.append(f"máximo de tokens de salida ({limits.max_output_tokens})")
            if limits.max_characters and next_chars > limits.max_characters:
                failures.append(f"máximo de caracteres ({limits.max_characters})")
            if limits.max_cost_usd and next_cost > limits.max_cost_usd + 1e-12:
                failures.append(f"presupuesto USD ({limits.max_cost_usd:.4f})")
            if failures:
                raise ExternalUsageLimitError("Se alcanzó el " + ", ".join(failures) + ".")

            # La reserva se contabiliza de inmediato para evitar carreras/reintentos que
            # sobrepasen el presupuesto. commit_external_call reconcilia la estimación.
            u.total_calls = next_total
            if kind == "llm":
                u.llm_calls = next_kind
            else:
                u.traditional_calls = next_kind
            u.input_tokens = next_input
            u.output_tokens = next_output
            u.characters = next_chars
            u.estimated_cost_usd = next_cost
            u.last_provider = str(provider or kind)
            u.updated_at = time.time()
            callback, snapshot = self._usage_notification_locked()

        if callback:
            callback(snapshot)
        return UsageReservation(
            kind=kind,
            provider=str(provider or kind),
            estimated_input_tokens=max(0, int(estimated_input_tokens)),
            estimated_output_tokens=max(0, int(estimated_output_tokens)),
            characters=max(0, int(characters)),
            estimated_cost_usd=reservation_cost,
        )

    def commit_external_call(
        self,
        reservation: UsageReservation,
        *,
        actual_input_tokens: int | None = None,
        actual_output_tokens: int | None = None,
        actual_characters: int | None = None,
        failed: bool = False,
    ) -> None:
        with self._condition:
            u = self.usage
            actual_in = reservation.estimated_input_tokens if actual_input_tokens is None else max(0, int(actual_input_tokens))
            actual_out = reservation.estimated_output_tokens if actual_output_tokens is None else max(0, int(actual_output_tokens))
            actual_chars = reservation.characters if actual_characters is None else max(0, int(actual_characters))
            actual_cost = self._cost(self.limits, reservation.kind, actual_in, actual_out, actual_chars)
            u.input_tokens += actual_in - reservation.estimated_input_tokens
            u.output_tokens += actual_out - reservation.estimated_output_tokens
            u.characters += actual_chars - reservation.characters
            u.estimated_cost_usd += actual_cost - reservation.estimated_cost_usd
            if failed:
                u.failed_calls += 1
            u.last_provider = reservation.provider
            u.updated_at = time.time()
            callback, snapshot = self._usage_notification_locked()
        if callback:
            callback(snapshot)

    def _usage_notification_locked(self) -> tuple[Optional[Callable[[Dict[str, Any]], None]], Dict[str, Any]]:
        return self._on_usage_changed, self.usage.to_dict()

    def usage_snapshot(self) -> Dict[str, Any]:
        with self._condition:
            return self.usage.to_dict()


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
    while time.monotonic() < deadline:
        execution_checkpoint()
        time.sleep(min(max(0.01, float(poll_seconds)), max(0.0, deadline - time.monotonic())))
