from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.config.app_config import QualityConfig
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.layout.reading_order_resolver import ReadingOrderResolver
from parallel_manga_translator.models.processing_models import Box, TextRegion

logger = get_logger(__name__)


@dataclass(frozen=True)
class PanelOrderConfig:
    enabled: bool = True
    min_area_ratio: float = 0.015
    max_area_ratio: float = 0.96
    gutter_px: int = 10

    @classmethod
    def from_quality_config(cls, quality: QualityConfig | None) -> "PanelOrderConfig":
        if quality is None:
            return cls()
        return cls(
            enabled=bool(getattr(quality, "panel_aware_reading_order", True)),
            min_area_ratio=float(getattr(quality, "panel_detection_min_area_ratio", 0.015)),
            max_area_ratio=float(getattr(quality, "panel_detection_max_area_ratio", 0.96)),
            gutter_px=int(getattr(quality, "panel_detection_gutter_px", 10)),
        )


class PanelAwareReadingOrderResolver:
    """Ordena regiones agrupándolas primero por panel.

    Adaptación ligera del enfoque panel-aware,
    pero mantiene el resolver de PMT para ordenar los textos dentro de cada panel.
    """

    def __init__(self, base_resolver: ReadingOrderResolver, config: PanelOrderConfig | None = None) -> None:
        self.base = base_resolver
        self.config = config or PanelOrderConfig()

    @staticmethod
    def _center(box: Box) -> Tuple[float, float]:
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def _distance_to_box(point: Tuple[float, float], box: Box) -> float:
        px, py = point
        x, y, w, h = box
        x2, y2 = x + w, y + h
        dx = max(x - px, 0.0, px - x2)
        dy = max(y - py, 0.0, py - y2)
        return float((dx * dx + dy * dy) ** 0.5)

    def detect_panels(self, image: np.ndarray) -> List[Box]:
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        page_area = max(1, height * width)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        # Detecta bordes/gutters sin depender de modelos externos.
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 120)
        gutter = max(3, int(self.config.gutter_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gutter, gutter))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        panels: List[Box] = []
        for cnt in contours:
            x, y, w, h = [int(v) for v in cv2.boundingRect(cnt)]
            if w < width * 0.08 or h < height * 0.08:
                continue
            area_ratio = (w * h) / page_area
            if area_ratio < self.config.min_area_ratio or area_ratio > self.config.max_area_ratio:
                continue
            # Rechaza cajas muy alargadas que suelen ser líneas de borde/gutter.
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 7.5:
                continue
            panels.append((x, y, w, h))

        if len(panels) <= 1:
            return []
        return self._dedupe_panels(panels)

    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_w = max(0, min(ax2, bx2) - max(ax, bx))
        inter_h = max(0, min(ay2, by2) - max(ay, by))
        inter = inter_w * inter_h
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    def _dedupe_panels(self, panels: Sequence[Box]) -> List[Box]:
        ordered = sorted(panels, key=lambda b: b[2] * b[3], reverse=True)
        selected: List[Box] = []
        for panel in ordered:
            if any(self._iou(panel, other) > 0.70 for other in selected):
                continue
            selected.append(panel)
        return self.sort_panels(selected)

    def sort_panels(self, panels: Sequence[Box]) -> List[Box]:
        if not panels:
            return []
        heights = [p[3] for p in panels]
        row_threshold = max(16.0, float(np.median(heights)) * 0.30)
        remaining = sorted(list(panels), key=lambda p: p[1])
        rows: List[List[Box]] = []
        for panel in remaining:
            placed = False
            for row in rows:
                row_y = float(np.mean([p[1] for p in row]))
                if abs(panel[1] - row_y) <= row_threshold:
                    row.append(panel)
                    placed = True
                    break
            if not placed:
                rows.append([panel])
        rows.sort(key=lambda row: min(p[1] for p in row))
        ordered: List[Box] = []
        for row in rows:
            row.sort(key=lambda p: p[0], reverse=self.base.page_reads_right_to_left)
            ordered.extend(row)
        return ordered

    def sort_regions(self, image: np.ndarray, regions: Sequence[TextRegion]) -> List[TextRegion]:
        if not self.config.enabled or not regions:
            return self.base.sort_regions(regions)
        try:
            panels = self.detect_panels(image)
        except Exception as exc:
            logger.debug("No se pudieron detectar paneles; usando orden base: %s", exc)
            panels = []
        if not panels:
            ordered = self.base.sort_regions(regions)
            for index, region in enumerate(ordered):
                region.metadata["panel_order_source"] = "base_no_panels_detected"
                region.metadata["panel_index"] = 0
                region.metadata["reading_order_index"] = index
            return ordered

        grouped: dict[int, List[TextRegion]] = {idx: [] for idx in range(len(panels))}
        floating: List[TextRegion] = []
        for region in regions:
            center = self._center(region.bbox)
            assigned = None
            for idx, panel in enumerate(panels):
                x, y, w, h = panel
                if x <= center[0] <= x + w and y <= center[1] <= y + h:
                    assigned = idx
                    break
            if assigned is None:
                distances = [(self._distance_to_box(center, panel), idx) for idx, panel in enumerate(panels)]
                if distances and min(distances)[0] <= max(image.shape[:2]) * 0.12:
                    assigned = min(distances)[1]
            if assigned is None:
                floating.append(region)
            else:
                grouped[assigned].append(region)
                region.metadata["panel_index"] = assigned
                region.metadata["panel_bbox"] = list(map(int, panels[assigned]))
                region.metadata["panel_order_source"] = "opencv_panel_edges"

        ordered: List[TextRegion] = []
        for idx in range(len(panels)):
            ordered.extend(self.base.sort_regions(grouped.get(idx, [])))
        if floating:
            for region in floating:
                region.metadata["panel_index"] = -1
                region.metadata["panel_order_source"] = "floating_text_base_order"
            ordered.extend(self.base.sort_regions(floating))
        for index, region in enumerate(ordered):
            region.metadata["reading_order_index"] = index
        return ordered
