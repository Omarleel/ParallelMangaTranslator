from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

from parallel_manga_translator.ocr.paddle_ocr_subprocess import PaddleOcrSubprocess
from parallel_manga_translator.ocr.paddle_result import PaddleLine, normalize_paddle_result
from parallel_manga_translator.ocr.settings import OcrSettings
from parallel_manga_translator.infrastructure.gpu_scheduler import gpu_slot

PaddleMode = Literal["direct", "subprocess"]


class PaddleOcrAdapter:
    """Cliente único de PaddleOCR para transcripción y detección.

    Encapsula las diferencias entre Paddle en el proceso principal y Paddle aislado en
    worker. Los motores de alto nivel solo piden líneas OCR normalizadas.
    """

    def __init__(self, settings: OcrSettings, mode: PaddleMode = "direct") -> None:
        self.settings = settings
        self.mode = mode
        self._paddle_ocr: Any = None
        self._worker: PaddleOcrSubprocess | None = None

    @classmethod
    def direct(cls, settings: OcrSettings) -> "PaddleOcrAdapter":
        return cls(settings, mode="direct")

    @classmethod
    def subprocess(cls, settings: OcrSettings) -> "PaddleOcrAdapter":
        return cls(settings, mode="subprocess")

    def read_lines(self, image: np.ndarray) -> list[PaddleLine]:
        if image is None or image.size == 0:
            return []
        if self.mode == "subprocess":
            return self._worker_instance().ocr(image)
        return normalize_paddle_result(self._run_direct(image))

    def _worker_instance(self) -> PaddleOcrSubprocess:
        if self._worker is None:
            self._worker = PaddleOcrSubprocess(lang=self.settings.paddle_lang, use_gpu=self.settings.gpu)
        return self._worker

    def _paddle_ocr_instance(self):
        if self._paddle_ocr is None:
            import re
            import paddleocr  # type: ignore
            from paddleocr import PaddleOCR  # type: ignore

            version = str(getattr(paddleocr, "__version__", "0"))
            match = re.match(r"\s*(\d+)", version)
            major = int(match.group(1)) if match else 0
            if major >= 3:
                self._paddle_ocr = PaddleOCR(
                    lang=self.settings.paddle_lang,
                    device="gpu:0" if self.settings.gpu else "cpu",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                self._paddle_api_mode = "3.x"
            else:
                self._paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.settings.paddle_lang,
                    use_gpu=self.settings.gpu,
                    show_log=False,
                )
                self._paddle_api_mode = "2.x"
        return self._paddle_ocr

    def _run_direct(self, image: np.ndarray):
        rgb_image = np.array(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        with gpu_slot("paddleocr.inference", enabled=self.settings.gpu):
            ocr = self._paddle_ocr_instance()
            if getattr(self, "_paddle_api_mode", "2.x") == "3.x":
                return ocr.predict(rgb_image)
            try:
                return ocr.ocr(img=rgb_image, cls=True)
            except TypeError:
                return ocr.ocr(rgb_image)
