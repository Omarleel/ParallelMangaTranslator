from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.models.processing_models import Box


@dataclass(frozen=True)
class TextMaskRefinementOptions:
    """Parámetros ligeros para refinar máscaras de tinta sin depender de CRF externo."""

    enabled: bool = True
    fine_text_detection: bool = True
    fine_mask_dilate: int = 2
    min_component_area: int = 3
    component_anchor_overlap: float = 0.03
    component_anchor_max_gap_ratio: float = 0.45


class TextInkMaskRefiner:
    """Genera y refina máscaras de tinta usando la señal OCR + componentes conectados.

    - la máscara de globo sigue siendo solo zona segura;
    - la máscara OCR/polígono es una semilla fina de texto;
    - los componentes conectados deciden qué tinta real se borra.
    """

    @staticmethod
    def _clip_rect(rect: Box, image_shape) -> Box:
        x, y, w, h = [int(v) for v in rect]
        height, width = image_shape[:2]
        x = max(0, min(width, x))
        y = max(0, min(height, y))
        x2 = max(x, min(width, x + max(0, w)))
        y2 = max(y, min(height, y + max(0, h)))
        return x, y, x2 - x, y2 - y

    @staticmethod
    def _binary(mask: Optional[np.ndarray], image_shape) -> np.ndarray:
        out = np.zeros(image_shape[:2], dtype=np.uint8)
        if mask is None or getattr(mask, "size", 0) == 0:
            return out
        h = min(out.shape[0], mask.shape[0])
        w = min(out.shape[1], mask.shape[1])
        if h > 0 and w > 0:
            out[:h, :w] = (mask[:h, :w] > 0).astype(np.uint8) * 255
        return out

    @staticmethod
    def _box_from_points(points: Sequence[Sequence[float]]) -> Box:
        arr = np.array(points, dtype=np.float32).reshape((-1, 2))
        x, y, w, h = cv2.boundingRect(arr.astype(np.int32))
        return int(x), int(y), int(w), int(h)

    @classmethod
    def detection_boxes(cls, detections: Sequence) -> List[Box]:
        boxes: List[Box] = []
        for det in detections or []:
            try:
                boxes.append(cls._box_from_points(det[0]))
            except Exception:
                continue
        return boxes

    @classmethod
    def mask_from_detections(
        cls,
        image_shape,
        detections: Sequence,
        *,
        dilate_px: int = 2,
        min_pad: int = 1,
    ) -> np.ndarray:
        """Crea una máscara fina desde polígonos OCR, no desde la bbox rectangular."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for det in detections or []:
            try:
                pts = np.array(det[0], dtype=np.float32).reshape((-1, 2))[:4]
                if pts.shape[0] < 4:
                    continue
                cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
            except Exception:
                continue

        if cv2.countNonZero(mask) == 0:
            for box in cls.detection_boxes(detections):
                x, y, w, h = cls._clip_rect(box, image_shape)
                if w > 0 and h > 0:
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        dilation = max(0, int(dilate_px))
        if dilation > 0 and cv2.countNonZero(mask) > 0:
            k = 2 * dilation + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)

        pad = max(0, int(min_pad))
        if pad > 0 and cv2.countNonZero(mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pad + 1, 2 * pad + 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    @classmethod
    def bounding_rect_from_mask(cls, mask: np.ndarray, image_shape, padding: int = 0) -> Box:
        points = cv2.findNonZero((mask > 0).astype(np.uint8)) if mask is not None and getattr(mask, "size", 0) else None
        if points is None:
            return 0, 0, 0, 0
        x, y, w, h = cv2.boundingRect(points)
        p = max(0, int(padding))
        return cls._clip_rect((x - p, y - p, w + 2 * p, h + 2 * p), image_shape)

    @staticmethod
    def _component_gap_to_anchor(box: Box, anchor_mask: np.ndarray) -> float:
        x, y, w, h = box
        pts = cv2.findNonZero((anchor_mask > 0).astype(np.uint8))
        if pts is None:
            return float("inf")
        ax, ay, aw, ah = cv2.boundingRect(pts)
        left = ax + aw < x
        right = x + w < ax
        above = y + h < ay
        below = ay + ah < y
        dx = max(ax - (x + w), x - (ax + aw), 0)
        dy = max(ay - (y + h), y - (ay + ah), 0)
        if (left or right) and (above or below):
            return float((dx * dx + dy * dy) ** 0.5)
        return float(max(dx, dy))

    @classmethod
    def _estimate_ink_candidates(
        cls,
        image: np.ndarray,
        allowed_mask: np.ndarray,
        safe_mask: np.ndarray,
        anchor_mask: np.ndarray,
    ) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        allowed = allowed_mask > 0
        if not np.any(allowed):
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # Muestra de fondo = zona segura menos el ancla OCR dilatada. Evita confundir tinta con fondo.
        anchor = (anchor_mask > 0).astype(np.uint8) * 255
        if cv2.countNonZero(anchor) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            anchor_dil = cv2.dilate(anchor, kernel, iterations=1)
            bg_mask = cv2.bitwise_and((safe_mask > 0).astype(np.uint8) * 255, cv2.bitwise_not(anchor_dil))
        else:
            bg_mask = (safe_mask > 0).astype(np.uint8) * 255

        if cv2.countNonZero(bg_mask) < 16:
            bg_vals = gray[safe_mask > 0]
        else:
            bg_vals = gray[bg_mask > 0]
        local_vals = gray[allowed]
        if local_vals.size == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        bg_median = float(np.median(bg_vals)) if bg_vals.size else float(np.median(local_vals))
        low = float(np.percentile(local_vals, 25))
        high = float(np.percentile(local_vals, 75))
        dark_contrast = bg_median - low
        bright_contrast = high - bg_median

        if bright_contrast > dark_contrast and bg_median < 170:
            threshold = max(55, min(245, int(bg_median + max(18, bright_contrast * 0.45))))
            ink = ((gray >= threshold) & allowed).astype(np.uint8) * 255
        else:
            threshold = min(245, max(10, int(bg_median - max(18, dark_contrast * 0.45))))
            ink = ((gray <= threshold) & allowed).astype(np.uint8) * 255

        return ink

    @classmethod
    def refine(
        cls,
        image: np.ndarray,
        safe_mask: np.ndarray,
        text_zone: np.ndarray,
        *,
        raw_text_mask: Optional[np.ndarray] = None,
        initial_ink_mask: Optional[np.ndarray] = None,
        options: TextMaskRefinementOptions | None = None,
    ) -> np.ndarray:
        options = options or TextMaskRefinementOptions()
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        safe = cls._binary(safe_mask, image.shape)
        zone = cls._binary(text_zone, image.shape)
        raw = cls._binary(raw_text_mask, image.shape)
        initial = cls._binary(initial_ink_mask, image.shape)
        if cv2.countNonZero(safe) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        anchor = raw if cv2.countNonZero(raw) > 0 else zone
        if cv2.countNonZero(anchor) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # La zona permitida se deriva del ancla OCR/polígono y se recorta por la máscara segura.
        expand_px = max(1, int(options.fine_mask_dilate) + 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand_px + 1, 2 * expand_px + 1))
        allowed = cv2.dilate(anchor, kernel, iterations=1)
        allowed = cv2.bitwise_and(allowed, safe)
        if cv2.countNonZero(allowed) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        candidates = cls._estimate_ink_candidates(image, allowed, safe, anchor)
        if cv2.countNonZero(initial) > 0:
            candidates = cv2.bitwise_or(candidates, cv2.bitwise_and(initial, allowed))

        num, labels, stats, _ = cv2.connectedComponentsWithStats((candidates > 0).astype(np.uint8) * 255, 8)
        refined = np.zeros_like(candidates)
        anchor_area = max(1, cv2.countNonZero(anchor))
        median_anchor_side = 10.0
        pts = cv2.findNonZero(anchor)
        if pts is not None:
            _ax, _ay, aw, ah = cv2.boundingRect(pts)
            median_anchor_side = float(max(6, min(max(aw, ah), max(12, (aw + ah) / 2))))
        max_gap = median_anchor_side * max(0.05, float(options.component_anchor_max_gap_ratio))

        for idx in range(1, num):
            x, y, w, h, area = [int(v) for v in stats[idx]]
            if area < int(options.min_component_area):
                continue
            comp = (labels[y:y + h, x:x + w] == idx)
            anchor_crop = anchor[y:y + h, x:x + w] > 0
            overlap = int(np.count_nonzero(comp & anchor_crop))
            overlap_ratio = overlap / max(1, min(int(area), anchor_area))
            gap = cls._component_gap_to_anchor((x, y, w, h), anchor)
            if overlap_ratio >= float(options.component_anchor_overlap) or gap <= max_gap or area >= 18:
                refined[y:y + h, x:x + w][comp] = 255

        if cv2.countNonZero(refined) == 0:
            refined = cv2.bitwise_and(candidates, allowed)

        dilation = max(0, int(options.fine_mask_dilate))
        if dilation > 0 and cv2.countNonZero(refined) > 0:
            k = 2 * dilation + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
            refined = cv2.dilate(refined, kernel, iterations=1)

        return cv2.bitwise_and(refined, safe)
