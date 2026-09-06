from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.geometry.box_geometry import BoxGeometry
from parallel_manga_translator.infrastructure.logging_config import get_logger

logger = get_logger(__name__)
Detection = Tuple[Sequence[Sequence[float]], str, float]


class CleanDetectionPipelineMixin:
    """Detección OCR inicial y deduplicación de cajas."""

    @staticmethod
    def _to_rect(detection) -> Tuple[int, int, int, int]:
        puntos = np.array(detection[0], dtype=np.float32)
        x, y, w, h = cv2.boundingRect(puntos.astype(np.int32))
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _area(rect: Tuple[int, int, int, int]) -> int:
        return BoxGeometry.area(rect)

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        union = CleanDetectionPipelineMixin._area(a) + CleanDetectionPipelineMixin._area(b) - inter
        return inter / union if union else 0.0

    @staticmethod
    def _confidence(detection) -> float:
        try:
            return float(detection[2])
        except Exception:
            return 0.0

    @staticmethod
    def _text_from_detection(detection) -> str:
        try:
            return str(detection[1] or "")
        except Exception:
            return ""

    def _is_sound_effect_detection(self, detection) -> bool:
        texto = self._text_from_detection(detection)
        if self.onomatopoeia_manager.is_onomatopoeia_candidate(texto, self.idioma_entrada):
            return True
        try:
            x, y, w, h = self._to_rect(detection)
        except Exception:
            return False
        aspect = max(w, h) / max(1, min(w, h))
        # Muchas onomatopeyas estilizadas aparecen como letras muy alargadas o grandes,
        # pero una columna vertical japonesa de diálogo también es alargada. El fallback
        # por proporción sólo aplica a líneas horizontales; los SFX verticales reales
        # deben entrar por diccionario/similitud/heurística.
        horizontalish = w >= max(1, h) * 1.35
        return horizontalish and aspect >= 4.2 and len(str(texto or "").strip()) <= 8

    def _dedupe_detections(self, detections: Iterable, image_shape) -> List:
        """Une resultados de pasadas distintas sin duplicar textos detectados."""
        height, width = image_shape[:2]
        min_area = max(8, int(height * width * 0.000008))
        valid = []
        for det in detections:
            try:
                rect = self._to_rect(det)
            except Exception:
                continue
            x, y, w, h = rect
            if w <= 1 or h <= 1 or self._area(rect) < min_area:
                continue
            if x >= width or y >= height:
                continue
            valid.append((det, rect, self._confidence(det)))

        # Conserva primero los cuadros más confiables y/o grandes.
        valid.sort(key=lambda row: (row[2], self._area(row[1])), reverse=True)
        selected = []
        selected_rects: List[Tuple[int, int, int, int]] = []

        for det, rect, _conf in valid:
            duplicate = False
            for chosen in selected_rects:
                if self._iou(rect, chosen) > 0.62:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(det)
                selected_rects.append(rect)

        return selected

    def obtener_cuadros_delimitadores(self, imagen: np.ndarray):
        """Devuelve cajas de texto usando el motor de localización configurado.

        No usa el OCR de transcripción. Esta fase produce bounding boxes para limpieza,
        pistas de texto y separación de globos; luego `OcrManager` transcribe cada región.
        """
        detecciones = self.text_detector.detect_text_boxes(imagen)
        return self._dedupe_detections(detecciones, imagen.shape)

    def fusionar_cuadros_delimitadores(self, imagen: np.ndarray, resultados) -> np.ndarray:
        height, width = imagen.shape[:2]
        mascara = np.zeros((height, width), dtype=np.uint8)
        base_expansion = max(2, int(round(min(height, width) * 0.0035)))

        for detection in resultados:
            caja = detection[0]
            puntos = np.array(caja, dtype=np.int32).reshape((-1, 1, 2))
            x, y, w, h = cv2.boundingRect(puntos)
            if self._is_sound_effect_detection(detection):
                # Las onomatopeyas suelen tener trazos gruesos/sombras y quedan mal si el inpainting
                # solo cubre la caja OCR mínima. Se expande un poco más, pero con límite.
                expansion = int(min(34, max(base_expansion + 2, round(max(w, h) * 0.13))))
            else:
                expansion = int(min(18, max(base_expansion, round(max(w, h) * 0.08))))
            x_margin = max(0, x - expansion)
            y_margin = max(0, y - expansion)
            x2 = min(width, x + w + expansion)
            y2 = min(height, y + h + expansion)
            if x2 <= x_margin or y2 <= y_margin:
                continue

            puntos_margin = np.array(
                [
                    [x_margin, y_margin],
                    [x2, y_margin],
                    [x2, y2],
                    [x_margin, y2],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mascara, [puntos_margin], 255)

        # Cierra huecos entre trazos y cubre bordes de letras para un inpainting más limpio.
        k = max(2, int(round(min(height, width) * 0.0025)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)
        mascara = cv2.dilate(mascara, kernel, iterations=1)
        return mascara
