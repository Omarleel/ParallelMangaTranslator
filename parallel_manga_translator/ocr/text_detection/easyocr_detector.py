from __future__ import annotations

from typing import List

import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ocr.easyocr_adapter import EasyOcrAdapter
from parallel_manga_translator.ocr.text_detection.base import TextDetection, TextDetectionEngineBase, TextDetectionSettings

logger = get_logger(__name__)


class EasyOcrTextDetector(TextDetectionEngineBase):
    """Localizador de texto basado en EasyOCR.

    EasyOCR se mantiene como opción por defecto para localización porque expone cajas
    de texto confiables y funciona sin entrenamiento adicional en manga.
    """

    def __init__(self, settings: TextDetectionSettings) -> None:
        super().__init__(settings)
        self._easyocr = EasyOcrAdapter(settings)

    @property
    def engine_id(self) -> str:
        return "easyocr"

    def _readtext_once(self, image: np.ndarray):
        if self.detector_settings.fast_mode:
            canvas_size = 1920
            mag_ratio = 1.15
            beam_width = 3
            batch_size = 8
        else:
            canvas_size = 2880
            mag_ratio = 1.65
            beam_width = 5
            batch_size = 4

        return self._easyocr.readtext(
            image,
            paragraph=False,
            decoder="beamsearch",
            batch_size=batch_size,
            beamWidth=beam_width,
            width_ths=0.28,
            height_ths=0.18,
            x_ths=0.22,
            y_ths=0.45,
            min_size=4,
            contrast_ths=0.08,
            adjust_contrast=0.65,
            text_threshold=0.45,
            low_text=0.30,
            link_threshold=0.45,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
            add_margin=0.02,
        )

    def detect_text_boxes(self, image: np.ndarray) -> List[TextDetection]:
        if image is None or image.size == 0:
            return []

        detections: List[TextDetection] = []
        try:
            detections.extend(self._readtext_once(image))
        except Exception as exc:
            logger.warning("EasyOCR falló localizando texto en la imagen original: %s", exc)

        if not self.detector_settings.fast_mode:
            try:
                enhanced = self.enhance_for_detection(image)
                detections.extend(self._readtext_once(enhanced))
            except Exception as exc:
                logger.warning("EasyOCR falló localizando texto en la imagen mejorada: %s", exc)

        normalized: List[TextDetection] = []
        for item in detections:
            try:
                detection = self.normalize_detection(item[0], item[1], item[2])
                if detection is not None:
                    normalized.append(detection)
            except Exception:
                continue
        return normalized
