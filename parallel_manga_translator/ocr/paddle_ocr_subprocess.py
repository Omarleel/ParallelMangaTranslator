from __future__ import annotations

import atexit
import base64
from collections import deque
import json
import os
import subprocess
import sys
import threading
import uuid
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


class PaddleOcrSubprocess:
    """Cliente persistente para PaddleOCR aislado en otro proceso.

    Incluye un protocolo de arranque explícito para no confundir "el proceso fue
    lanzado" con "PaddleOCR inicializó correctamente". También conserva STDERR
    del worker para mostrar la causa real de fallos de CUDA, paquetes o modelos.
    """

    def __init__(self, lang: str, use_gpu: bool = False) -> None:
        self.lang = lang
        self.use_gpu = bool(use_gpu)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_tail: Deque[str] = deque(maxlen=80)
        self._fatal_error: Optional[str] = None
        self._runtime_info: Dict[str, Any] = {}
        atexit.register(self.close)

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        stream = proc.stderr
        if stream is None:
            return
        try:
            for line in stream:
                clean = line.rstrip()
                if clean:
                    self._stderr_tail.append(clean)
                    logger.debug("PaddleOCR worker STDERR: %s", clean[:500])
        except Exception:
            return

    def _stderr_summary(self) -> str:
        return "\n".join(self._stderr_tail)[-4000:]

    def _process_failure_message(self, prefix: str) -> str:
        proc = self._proc
        return_code = proc.poll() if proc is not None else None
        stderr = self._stderr_summary()
        details = f"{prefix} (código de salida={return_code})"
        if stderr:
            details += f"\nSTDERR del worker:\n{stderr}"
        return details

    def _start(self) -> None:
        if self._fatal_error:
            raise RuntimeError(self._fatal_error)
        if self._proc is not None and self._proc.poll() is None:
            return

        env = os.environ.copy()
        env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
        env.setdefault("FLAGS_allocator_strategy", "auto_growth")
        env.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.45")

        gpu_arg = "1" if self.use_gpu else "0"
        cmd = [
            sys.executable,
            "-m",
            "parallel_manga_translator.ocr.paddle_ocr_worker",
            "--lang",
            self.lang,
            "--gpu",
            gpu_arg,
        ]
        self._stderr_tail.clear()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        proc = self._proc
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(proc,),
            name="pmt-paddle-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

        # Espera a que PaddleOCR termine de importar, validar CUDA y cargar modelos.
        startup = self._read_protocol_message(expected_id=None, startup=True)
        if startup.get("type") == "fatal" or not startup.get("ok"):
            message = startup.get("error", "PaddleOCR worker no pudo iniciar")
            traceback_text = startup.get("traceback") or ""
            if traceback_text:
                message += f"\n{traceback_text[-4000:]}"
            stderr = self._stderr_summary()
            if stderr:
                message += f"\nSTDERR del worker:\n{stderr}"
            self._fatal_error = message
            self.close()
            raise RuntimeError(message)

        self._runtime_info = dict(startup)
        logger.info(
            "PaddleOCR worker listo | lang=%s | device=%s | paddle=%s | paddleocr=%s | api=%s | torch_preload=%s | torch=%s",
            self.lang,
            startup.get("device", "desconocido"),
            startup.get("paddle_version", "desconocida"),
            startup.get("paddleocr_version", "desconocida"),
            startup.get("api_mode", "desconocida"),
            startup.get("torch_preloaded", False),
            startup.get("torch_version", "n/a"),
        )

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("No se pudo codificar imagen para PaddleOCR worker")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _read_protocol_message(self, expected_id: Optional[str], startup: bool = False) -> Dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                raise RuntimeError(
                    self._process_failure_message(
                        "PaddleOCR worker terminó durante el arranque" if startup else "PaddleOCR worker terminó sin responder"
                    )
                )
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # Paddle/PaddleX todavía pueden escribir mensajes informativos en stdout.
                logger.debug("Salida no JSON del worker PaddleOCR ignorada: %s", line[:500])
                continue

            payload_type = payload.get("type")
            if payload_type == "fatal":
                return payload
            if startup:
                if payload_type == "ready":
                    return payload
                continue
            if payload.get("id") in {expected_id, None}:
                return payload

    def ocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if image is None or image.size == 0:
            return []
        with self._lock:
            self._start()
            assert self._proc is not None and self._proc.stdin is not None
            request_id = uuid.uuid4().hex
            request = {"id": request_id, "cmd": "ocr", "image": self._encode_image(image)}
            try:
                self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                message = self._process_failure_message("PaddleOCR worker no acepta más peticiones")
                self._fatal_error = message
                self.close()
                raise RuntimeError(message) from exc

            response = self._read_protocol_message(request_id)
            if not response.get("ok"):
                message = response.get("error", "PaddleOCR worker falló")
                traceback_text = response.get("traceback") or ""
                if traceback_text:
                    message += f"\n{traceback_text[-4000:]}"
                raise RuntimeError(message)
            return list(response.get("lines") or [])

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"id": "shutdown", "cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
