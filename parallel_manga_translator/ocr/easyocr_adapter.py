from __future__ import annotations

from typing import Any

import numpy as np

from parallel_manga_translator.ocr.settings import OcrSettings


class EasyOcrAdapter:
    """Inicialización perezosa compartida de EasyOCR."""

    def __init__(self, settings: OcrSettings) -> None:
        self.settings = settings
        self._reader: Any = None

    def reader(self):
        if self._reader is None:
            import easyocr  # type: ignore

            self._reader = easyocr.Reader(self.settings.easyocr_langs, gpu=self.settings.gpu)
        return self._reader

    def readtext(self, image: np.ndarray, **kwargs: Any):
        return self.reader().readtext(image, **kwargs)
