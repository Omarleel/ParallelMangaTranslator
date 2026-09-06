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

from parallel_manga_translator.config.app_config import ProcessingConfig


class PersistentJsonCache:
    """Caché simple, persistente y segura para OCR/traducciones.

    La activación y la carpeta base **se reciben**; esta clase no las busca. Vienen de
    `processing.cache` y `processing.cache_dir`, que el composition root propaga. Es una
    capa de infraestructura: leer ella misma la configuración de aplicación era acoplarla
    hacia arriba, y además rompía el aislamiento de caché por trabajo de la UI.
    """

    def __init__(self, namespace: str, base_dir: Optional[str] = None, enabled: Optional[bool] = None) -> None:
        # Los valores por defecto salen del dataclass, no del estado global del proceso.
        # `base_dir` importa de verdad: la UI usa una caché por trabajo, y cuando esa ruta
        # viajaba por `get_active_config()` dos trabajos simultáneos compartían la del
        # último en llamar a `set_active_config`. Quien construya la caché la pasa.
        defaults = ProcessingConfig()
        self.enabled = defaults.cache if enabled is None else bool(enabled)
        root = Path(base_dir or defaults.cache_dir or ".cache")
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
