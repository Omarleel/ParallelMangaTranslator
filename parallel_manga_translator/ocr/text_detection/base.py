from __future__ import annotations

from typing import Any, List, Protocol, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.ocr.engines.base import OcrEngineBase
from parallel_manga_translator.ocr.settings import OcrSettings

TextDetection = Tuple[Sequence[Sequence[float]], str, float]

# Alias público conservado para no romper imports existentes.
TextDetectionSettings = OcrSettings


class TextDetectionEngine(Protocol):
    """Localiza texto y devuelve cajas, texto aproximado y confianza.

    Este contrato es distinto al OCR de transcripción: aquí lo importante son las
    coordenadas. El texto devuelto se usa como pista para filtros, división de globos
    y clasificación de SFX, pero la lectura final la hace `OcrEngine`.
    """

    @property
    def engine_id(self) -> str:
        ...

    def detect_text_boxes(self, image: np.ndarray) -> List[TextDetection]:
        ...


class TextDetectionEngineBase(OcrEngineBase):
    """Utilidades comunes para motores de localización de texto."""

    def __init__(self, settings: TextDetectionSettings) -> None:
        super().__init__(settings.as_ocr_settings())
        self.detector_settings = settings

    @staticmethod
    def enhance_for_detection(image: np.ndarray) -> np.ndarray:
        """Mejora contraste para textos finos o con tramas de fondo."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        smoothed = cv2.bilateralFilter(clahe, d=5, sigmaColor=35, sigmaSpace=35)
        sharpened = cv2.addWeighted(clahe, 1.45, smoothed, -0.45, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def normalize_detection(box: Any, text: Any = "", confidence: Any = 0.0) -> TextDetection | None:
        try:
            points = np.array(box, dtype=np.float32)
            if points.size == 0:
                return None
            points = points.reshape((-1, 2))
            normalized_box = [[float(x), float(y)] for x, y in points[:4]]
            if len(normalized_box) < 4:
                return None
            return normalized_box, str(text or ""), float(confidence or 0.0)
        except Exception:
            return None

    @classmethod
    def normalize_paddle_lines(cls, lines: Sequence[Any]) -> List[TextDetection]:
        detections: List[TextDetection] = []
        for line in lines or []:
            try:
                if isinstance(line, dict):
                    detection = cls.normalize_detection(
                        line.get("box") or line.get("dt_poly") or line.get("points") or [],
                        line.get("text") or "",
                        line.get("confidence") or line.get("score") or 0.0,
                    )
                else:
                    box = line[0]
                    text, confidence = line[-1]
                    detection = cls.normalize_detection(box, text, confidence)
                if detection is not None:
                    detections.append(detection)
            except Exception:
                continue
        return detections
