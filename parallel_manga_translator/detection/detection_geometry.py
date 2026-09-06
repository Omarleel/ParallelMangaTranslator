from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.processing_models import Box
from parallel_manga_translator.geometry.box_geometry import BoxGeometry

logger = get_logger(__name__)


class DetectionGeometry:
    """Geometría de cajas y lectura de detecciones OCR.

    Era `BubbleGeometryMixin`, pero sus 19 métodos no tocan estado de instancia:
    es una biblioteca de funciones puras, no una porción de comportamiento del
    detector. Como mixin obligaba a heredarla para usarla y sus hermanos la
    llamaban por `self`, sin declararla. Como colaborador se inyecta y se prueba
    aparte. Ojo: `processing/` y `layout/` tienen copias propias de varios de estos
    helpers y **no todas son equivalentes** (`_expand_box` tiene otra firma allí),
    así que unificarlas es un paso aparte y hay que comprobarlo caso por caso.
    """

    @staticmethod
    def float_value(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def int_value(value, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def to_rect(detection) -> Box:
        return BoxGeometry.from_detection(detection)

    @staticmethod
    def text(detection) -> str:
        try:
            return str(detection[1] or "")
        except Exception:
            return ""

    @staticmethod
    def confidence(detection) -> float:
        try:
            return float(detection[2])
        except Exception:
            return 0.0

    @staticmethod
    def area(box: Box) -> int:
        return BoxGeometry.area(box)

    @staticmethod
    def union(a: Box, b: Box) -> Box:
        return BoxGeometry.union(a, b)

    @staticmethod
    def intersection_area(a: Box, b: Box) -> int:
        return BoxGeometry.intersection_area(a, b)

    @staticmethod
    def overlap_ratio_1d(a1: int, a2: int, b1: int, b2: int) -> float:
        return BoxGeometry.overlap_ratio_1d(a1, a2, b1, b2)

    @staticmethod
    def center(box: Box) -> Tuple[float, float]:
        return BoxGeometry.center(box)

    @staticmethod
    def point_inside_box(point: Tuple[float, float], box: Box) -> bool:
        return BoxGeometry.point_inside_box(point, box)

    @staticmethod
    def point_inside_mask(point: Tuple[float, float], mask: np.ndarray) -> bool:
        return BoxGeometry.point_inside_mask(point, mask)

    @staticmethod
    def expand_box(box: Box, width: int, height: int, ratio_x: float, ratio_y: float, min_pad: int = 18) -> Box:
        return BoxGeometry.expand(box, width, height, ratio_x, ratio_y, min_pad)

    @staticmethod
    def box_intersection(a: Box, b: Box) -> Optional[Box]:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return None
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    @staticmethod
    def mask_bbox(mask: np.ndarray) -> Optional[Box]:
        pts = cv2.findNonZero(mask)
        if pts is None:
            return None
        x, y, w, h = cv2.boundingRect(pts)
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def clip_box_to_image(box: Box, width_img: int, height_img: int) -> Box:
        return BoxGeometry.clip_box_to_image(box, width_img, height_img)

    def detections_box(self, detections: Sequence) -> Box:
        boxes = [self.to_rect(det) for det in detections]
        merged = boxes[0]
        for box in boxes[1:]:
            merged = self.union(merged, box)
        return merged

    def detections_text(self, detections: Sequence) -> str:
        return " ".join(self.text(det).strip() for det in detections if self.text(det).strip())

    def detections_confidence(self, detections: Sequence) -> float:
        try:
            values = [self.confidence(det) for det in detections]
            return float(np.mean(values)) if values else 0.0
        except Exception:
            return 0.0
