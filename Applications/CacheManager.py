from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


class PersistentJsonCache:
    """Caché simple, persistente y segura para OCR/traducciones.

    Guarda cada entrada como un JSON separado para evitar bloquear todo el archivo si el
    proceso se interrumpe. Es deliberadamente pequeño y sin dependencias externas.
    """

    def __init__(self, namespace: str, base_dir: Optional[str] = None, enabled: Optional[bool] = None) -> None:
        self.enabled = env_flag("PMT_CACHE", True) if enabled is None else bool(enabled)
        root = Path(base_dir or os.getenv("PMT_CACHE_DIR", ".cache"))
        self.path = root / namespace
        self._lock = threading.Lock()
        if self.enabled:
            self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def hash_text(*parts: Any) -> str:
        h = hashlib.sha256()
        for part in parts:
            h.update(str(part).encode("utf-8", errors="ignore"))
            h.update(b"\0")
        return h.hexdigest()

    @staticmethod
    def hash_image(image: np.ndarray, *parts: Any) -> str:
        h = hashlib.sha256()
        for part in parts:
            h.update(str(part).encode("utf-8", errors="ignore"))
            h.update(b"\0")
        if image is not None and image.size:
            ok, buf = cv2.imencode(".png", image)
            if ok:
                h.update(buf.tobytes())
            else:
                h.update(np.ascontiguousarray(image).tobytes())
        return h.hexdigest()

    def _file(self, key: str) -> Path:
        return self.path / f"{key}.json"

    def get(self, key: str, default: Any = None) -> Any:
        if not self.enabled:
            return default
        file_path = self._file(key)
        if not file_path.exists():
            return default
        try:
            with file_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data.get("value", default)
        except Exception:
            return default

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        payload = {"value": value}
        with self._lock:
            fd, tmp_name = tempfile.mkstemp(prefix=f"{key}.", suffix=".tmp", dir=str(self.path))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                os.replace(tmp_name, self._file(key))
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
