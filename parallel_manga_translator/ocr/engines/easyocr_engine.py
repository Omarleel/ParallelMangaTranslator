from __future__ import annotations

import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ocr.easyocr_adapter import EasyOcrAdapter
from parallel_manga_translator.ocr.engines.base import OcrEngineBase, OcrEngineSettings

logger = get_logger(__name__)


class EasyOcrEngine(OcrEngineBase):
    """Adaptador de EasyOCR."""

    def __init__(self, settings: OcrEngineSettings) -> None:
        super().__init__(settings)
        self._easyocr = EasyOcrAdapter(settings)

    @property
    def engine_id(self) -> str:
        return "easyocr"

    def extract_text(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return ""
        image = self.upscale_if_needed(image)
        try:
            lines = self._easyocr.readtext(
                image,
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                batch_size=6 if self.settings.fast_mode else 4,
                beamWidth=3 if self.settings.fast_mode else 5,
                width_ths=0.35,
                height_ths=0.25,
                canvas_size=1920 if self.settings.fast_mode else 2560,
                mag_ratio=1.2 if self.settings.fast_mode else 1.6,
            )
        except Exception as exc:
            logger.warning("EasyOCR falló en un recorte: %s", exc)
            return ""

        normalized = []
        for line in self.sort_lines(lines):
            try:
                text = line[1]
                conf = float(line[2])
            except Exception:
                text, conf = "", 0.0
            text = self.normalize_text(text)
            if text and not (conf < 0.15 and len(text) <= 2):
                normalized.append(text)
        if self.idioma_entrada in {"Inglés", "Español"}:
            return self.normalize_text(" ".join(normalized))
        return self.normalize_text("".join(t.replace("~", "") for t in normalized))
