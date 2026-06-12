from __future__ import annotations

import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ocr.engines.base import OcrEngineBase, OcrEngineSettings
from parallel_manga_translator.ocr.paddle_adapter import PaddleOcrAdapter

logger = get_logger(__name__)


class PaddleOcrEngine(OcrEngineBase):
    """Adaptador de PaddleOCR en el proceso principal."""

    def __init__(self, settings: OcrEngineSettings) -> None:
        super().__init__(settings)
        self._paddle = PaddleOcrAdapter.direct(settings)

    @property
    def engine_id(self) -> str:
        return "paddleocr"

    def extract_text(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return ""
        image = self.upscale_if_needed(image)
        try:
            lines = self._paddle.read_lines(image)
        except Exception as exc:
            logger.warning("PaddleOCR en proceso principal falló: %s", exc)
            return ""
        return self.join_ocr_lines(lines)
