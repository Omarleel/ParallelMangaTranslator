from __future__ import annotations

import atexit
import base64
import json
import os
import subprocess
import sys
import threading
import uuid
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


class PaddleOcrSubprocess:
    """Cliente persistente para PaddleOCR aislado en otro proceso."""

    def __init__(self, lang: str, use_gpu: bool = False) -> None:
        self.lang = lang
        self.use_gpu = bool(use_gpu)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        env = os.environ.copy()
        env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
        env.setdefault("FLAGS_allocator_strategy", "auto_growth")

        gpu_arg = "1" if self.use_gpu else "0"
        cmd = [sys.executable, "-m", "parallel_manga_translator.ocr.paddle_ocr_worker", "--lang", self.lang, "--gpu", gpu_arg]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        logger.info("PaddleOCR worker iniciado en subproceso aislado (lang=%s, gpu=%s)", self.lang, self.use_gpu)

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("No se pudo codificar imagen para PaddleOCR worker")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _read_json_response(self, expected_id: str) -> Dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                stderr = ""
                try:
                    if self._proc.stderr is not None:
                        stderr = self._proc.stderr.read() or ""
                except Exception:
                    pass
                raise RuntimeError(f"PaddleOCR worker terminó sin responder. STDERR: {stderr[-1200:]}")
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Salida no JSON del worker PaddleOCR ignorada: %s", line[:300])
                continue
            if payload.get("type") == "fatal":
                raise RuntimeError(payload.get("error", "PaddleOCR worker no pudo iniciar"))
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
            except BrokenPipeError as exc:
                self._proc = None
                raise RuntimeError("PaddleOCR worker no acepta más peticiones") from exc
            response = self._read_json_response(request_id)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "PaddleOCR worker falló"))
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
