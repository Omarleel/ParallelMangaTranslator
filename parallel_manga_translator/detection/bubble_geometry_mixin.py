from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.detection.yolo_bubble_detector import YoloBubbleCandidate
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.processing_models import Box, TextRegion
from parallel_manga_translator.geometry.box_geometry import BoxGeometry

logger = get_logger(__name__)
BUBBLE_SPLIT_DEBUG_VERSION = "v7_bubble_onomatopoeia_translation_2026_06_11"


class BubbleGeometryMixin:
    """Adaptadores geométricos y utilidades puras usadas por el detector."""

    @staticmethod
    def _float_value(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int_value(value, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_rect(detection) -> Box:
        return BoxGeometry.from_detection(detection)

    @staticmethod
    def _text(detection) -> str:
        try:
            return str(detection[1] or "")
        except Exception:
            return ""

    @staticmethod
    def _confidence(detection) -> float:
        try:
            return float(detection[2])
        except Exception:
            return 0.0

    @staticmethod
    def _area(box: Box) -> int:
        return BoxGeometry.area(box)

    @staticmethod
    def _union(a: Box, b: Box) -> Box:
        return BoxGeometry.union(a, b)

    @staticmethod
    def _intersection_area(a: Box, b: Box) -> int:
        return BoxGeometry.intersection_area(a, b)

    @staticmethod
    def _overlap_ratio_1d(a1: int, a2: int, b1: int, b2: int) -> float:
        return BoxGeometry.overlap_ratio_1d(a1, a2, b1, b2)

    @staticmethod
    def _center(box: Box) -> Tuple[float, float]:
        return BoxGeometry.center(box)

    @staticmethod
    def _point_inside_box(point: Tuple[float, float], box: Box) -> bool:
        return BoxGeometry.point_inside_box(point, box)

    @staticmethod
    def _point_inside_mask(point: Tuple[float, float], mask: np.ndarray) -> bool:
        return BoxGeometry.point_inside_mask(point, mask)

    @staticmethod
    def _expand_box(box: Box, width: int, height: int, ratio_x: float, ratio_y: float, min_pad: int = 18) -> Box:
        return BoxGeometry.expand(box, width, height, ratio_x, ratio_y, min_pad)

    @staticmethod
    def _box_intersection(a: Box, b: Box) -> Optional[Box]:
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
    def _mask_bbox(mask: np.ndarray) -> Optional[Box]:
        pts = cv2.findNonZero(mask)
        if pts is None:
            return None
        x, y, w, h = cv2.boundingRect(pts)
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _clip_box_to_image(box: Box, width_img: int, height_img: int) -> Box:
        x, y, w, h = box
        if width_img <= 0 or height_img <= 0:
            return 0, 0, 1, 1
        x = int(max(0, min(width_img - 1, x)))
        y = int(max(0, min(height_img - 1, y)))
        x2 = int(max(x + 1, min(width_img, x + max(1, w))))
        y2 = int(max(y + 1, min(height_img, y + max(1, h))))
        return x, y, x2 - x, y2 - y

    def _detections_box(self, detections: Sequence) -> Box:
        boxes = [self._to_rect(det) for det in detections]
        merged = boxes[0]
        for box in boxes[1:]:
            merged = self._union(merged, box)
        return merged

    def _detections_text(self, detections: Sequence) -> str:
        return " ".join(self._text(det).strip() for det in detections if self._text(det).strip())

    def _detections_confidence(self, detections: Sequence) -> float:
        try:
            values = [self._confidence(det) for det in detections]
            return float(np.mean(values)) if values else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def compose_mask(regions: Sequence[TextRegion], image_shape) -> np.ndarray:
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            if region.mask is not None and region.mask.size:
                mask = cv2.bitwise_or(mask, region.mask)
        return mask
