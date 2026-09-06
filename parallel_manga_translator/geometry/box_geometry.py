from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from parallel_manga_translator.models.processing_models import Box


class BoxGeometry:
    """Operaciones geométricas pequeñas y reutilizables sobre cajas x, y, w, h."""

    @staticmethod
    def from_detection(detection) -> Box:
        points = np.array(detection[0], dtype=np.float32)
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def area(box: Box) -> int:
        return max(0, box[2]) * max(0, box[3])

    @staticmethod
    def union(a: Box, b: Box) -> Box:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax + aw, bx + bw)
        y2 = max(ay + ah, by + bh)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def intersection_area(a: Box, b: Box) -> int:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def intersection_box(a: Box, b: Box) -> Box:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        return x1, y1, max(0, x2 - x1), max(0, y2 - y1)

    @staticmethod
    def overlap_ratio_1d(a1: int, a2: int, b1: int, b2: int) -> float:
        inter = max(0, min(a2, b2) - max(a1, b1))
        denom = max(1, min(a2 - a1, b2 - b1))
        return inter / denom

    @staticmethod
    def center(box: Box) -> Tuple[float, float]:
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def point_inside_box(point: Tuple[float, float], box: Box) -> bool:
        px, py = point
        x, y, w, h = box
        return x <= px <= x + w and y <= py <= y + h

    @staticmethod
    def point_inside_mask(point: Tuple[float, float], mask: np.ndarray) -> bool:
        if mask is None or mask.size == 0:
            return False
        px, py = int(round(point[0])), int(round(point[1]))
        if py < 0 or px < 0 or py >= mask.shape[0] or px >= mask.shape[1]:
            return False
        return bool(mask[py, px] > 0)

    @staticmethod
    def expand(box: Box, width: int, height: int, ratio_x: float, ratio_y: float, min_pad: int = 18) -> Box:
        x, y, w, h = box
        pad_x = max(min_pad, int(round(w * ratio_x)))
        pad_y = max(min_pad, int(round(h * ratio_y)))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def clip_box_to_image(box: Box, width_img: int, height_img: int) -> Box:
        """Recorta conservando el ancho util cuando la caja empieza fuera de la imagen.

        No es lo mismo que `clip`, aunque lo parezca: `clip` recorta el extremo desde la
        coordenada ya ajustada, asi que una caja que empieza en negativo se le colapsa a
        1 px. Aqui se ajusta primero el origen y se mide el ancho desde ahi. Las dos
        semanticas estaban duplicadas en detection/ y processing/; se conservan ambas
        porque tienen usos distintos, pero cada una vive en un solo sitio.
        """
        x, y, w, h = box
        if width_img <= 0 or height_img <= 0:
            return 0, 0, 1, 1
        x = int(max(0, min(width_img - 1, x)))
        y = int(max(0, min(height_img - 1, y)))
        x2 = int(max(x + 1, min(width_img, x + max(1, w))))
        y2 = int(max(y + 1, min(height_img, y + max(1, h))))
        return x, y, x2 - x, y2 - y

    @staticmethod
    def clip(box: Box, width: int, height: int) -> Box:
        x, y, w, h = box
        x1 = max(0, min(width - 1, x))
        y1 = max(0, min(height - 1, y))
        x2 = max(x1 + 1, min(width, x + w))
        y2 = max(y1 + 1, min(height, y + h))
        return x1, y1, x2 - x1, y2 - y1
