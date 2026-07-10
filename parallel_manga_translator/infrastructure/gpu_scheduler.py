from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator


@dataclass(frozen=True)
class GpuSchedulerSnapshot:
    operations: int
    wait_seconds: float
    hold_seconds: float
    by_operation: Dict[str, Dict[str, float | int]]


class _FifoGpuScheduler:
    """Planificador CUDA reentrante y FIFO para una sola GPU.

    PyTorch, EasyOCR, Ultralytics e inpainting pueden crear streams desde hilos
    distintos. Esta compuerta entrega la GPU a una operación por vez, en orden FIFO,
    y sincroniza antes de cederla. El trabajo CPU situado entre esas operaciones sigue
    ejecutándose en paralelo.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self._operations = 0
        self._wait_seconds = 0.0
        self._hold_seconds = 0.0
        self._by_operation: Dict[str, Dict[str, float | int]] = {}

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _synchronize_cuda() -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def slot(self, operation: str, *, enabled: bool = True) -> Iterator[None]:
        if not enabled or not self._cuda_available():
            yield
            return

        depth = int(getattr(self._local, "depth", 0))
        if depth > 0:
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth = depth
            return

        wait_started = time.perf_counter()
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving_ticket:
                self._condition.wait()
        waited = time.perf_counter() - wait_started

        self._local.depth = 1
        hold_started = time.perf_counter()
        body_failed = False
        sync_error: BaseException | None = None
        try:
            yield
        except BaseException:
            body_failed = True
            raise
        finally:
            try:
                self._synchronize_cuda()
            except BaseException as exc:
                sync_error = exc

            held = time.perf_counter() - hold_started
            self._local.depth = 0
            label = str(operation or "gpu")
            with self._stats_lock:
                self._operations += 1
                self._wait_seconds += waited
                self._hold_seconds += held
                bucket = self._by_operation.setdefault(
                    label,
                    {"operations": 0, "wait_seconds": 0.0, "hold_seconds": 0.0},
                )
                bucket["operations"] = int(bucket["operations"]) + 1
                bucket["wait_seconds"] = float(bucket["wait_seconds"]) + waited
                bucket["hold_seconds"] = float(bucket["hold_seconds"]) + held

            with self._condition:
                self._serving_ticket += 1
                self._condition.notify_all()

            # Una excepción CUDA asíncrona aparece al sincronizar. No debe ocultarse,
            # salvo que ya exista una excepción más informativa en el cuerpo.
            if sync_error is not None and not body_failed and sys.exc_info()[0] is None:
                raise sync_error

    def reset_stats(self) -> None:
        with self._stats_lock:
            self._operations = 0
            self._wait_seconds = 0.0
            self._hold_seconds = 0.0
            self._by_operation = {}

    def snapshot(self) -> GpuSchedulerSnapshot:
        with self._stats_lock:
            details = {
                key: {
                    "operations": int(value["operations"]),
                    "wait_seconds": round(float(value["wait_seconds"]), 4),
                    "hold_seconds": round(float(value["hold_seconds"]), 4),
                }
                for key, value in self._by_operation.items()
            }
            return GpuSchedulerSnapshot(
                operations=int(self._operations),
                wait_seconds=round(float(self._wait_seconds), 4),
                hold_seconds=round(float(self._hold_seconds), 4),
                by_operation=details,
            )


_SCHEDULER = _FifoGpuScheduler()


def gpu_slot(operation: str, *, enabled: bool = True):
    return _SCHEDULER.slot(operation, enabled=enabled)


def reset_gpu_scheduler_stats() -> None:
    _SCHEDULER.reset_stats()


def gpu_scheduler_snapshot() -> GpuSchedulerSnapshot:
    return _SCHEDULER.snapshot()
