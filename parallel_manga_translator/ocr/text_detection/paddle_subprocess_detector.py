from __future__ import annotations

from typing import List

import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ocr.paddle_adapter import PaddleOcrAdapter
from parallel_manga_translator.ocr.text_detection.base import TextDetection, TextDetectionEngineBase, TextDetectionSettings

logger = get_logger(__name__)


class PaddleSubprocessTextDetector(TextDetectionEngineBase):
    """Localizador de texto con PaddleOCR aislado en un worker."""

    def __init__(self, settings: TextDetectionSettings) -> None:
        super().__init__(settings)
        self._paddle = PaddleOcrAdapter.subprocess(settings.as_ocr_settings())

    @property
    def engine_id(self) -> str:
        return "paddle_subprocess"

    def detect_text_boxes(self, image: np.ndarray) -> List[TextDetection]:
        if image is None or image.size == 0:
            return []
        try:
            lines = self._paddle.read_lines(image)
        except Exception as exc:
            logger.warning("PaddleOCR subproceso falló localizando texto: %s", exc)
            return []
        return self.normalize_paddle_lines(lines)
