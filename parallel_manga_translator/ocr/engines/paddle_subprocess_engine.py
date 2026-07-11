from __future__ import annotations

import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ocr.engines.base import OcrEngineBase, OcrEngineSettings
from parallel_manga_translator.ocr.paddle_adapter import PaddleOcrAdapter

logger = get_logger(__name__)


class PaddleSubprocessOcrEngine(OcrEngineBase):
    """Adaptador de PaddleOCR aislado en subproceso."""

    def __init__(self, settings: OcrEngineSettings) -> None:
        super().__init__(settings)
        self._paddle = PaddleOcrAdapter.subprocess(settings)
        self._failure_reported = False

    @property
    def engine_id(self) -> str:
        return "paddle_subprocess"

    def extract_text(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return ""
        image = self.upscale_if_needed(image)
        try:
            lines = self._paddle.read_lines(image)
        except Exception as exc:
            if not self._failure_reported:
                logger.error("PaddleOCR subproceso no está disponible: %s", exc)
                self._failure_reported = True
            else:
                logger.debug("PaddleOCR subproceso continúa deshabilitado: %s", exc)
            return ""
        return self.join_ocr_lines(lines)
