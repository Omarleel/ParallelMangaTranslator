from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ocr.engines.base import OcrEngineBase, OcrEngineSettings
from parallel_manga_translator.ocr.engines.easyocr_engine import EasyOcrEngine

logger = get_logger(__name__)


class MangaOcrEngine(OcrEngineBase):
    """Adaptador de MangaOCR con fallback opcional a EasyOCR."""

    def __init__(self, settings: OcrEngineSettings, fallback: EasyOcrEngine | None = None) -> None:
        super().__init__(settings)
        self._manga_ocr = None
        self.fallback = fallback

    @property
    def engine_id(self) -> str:
        return "mangaocr" if self.fallback is None else "mangaocr+easyocr"

    def _manga_ocr_instance(self):
        if self._manga_ocr is None:
            from manga_ocr import MangaOcr  # type: ignore

            self._manga_ocr = MangaOcr()
        return self._manga_ocr

    def extract_text(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return ""
        image = self.upscale_if_needed(image)
        area_interes_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        try:
            texto = self._manga_ocr_instance()(area_interes_pil)
            texto = self.normalize_text(texto)
            if texto:
                return texto
        except Exception as exc:
            logger.warning("MangaOCR falló; se intentará fallback si está habilitado: %s", exc)
        return self.fallback.extract_text(image) if self.fallback is not None else ""
