from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence, Tuple, TypeVar

import numpy as np

from Applications.ProcessingModels import Box, TextRegion

T = TypeVar("T")


@dataclass(frozen=True)
class ReadingOrderConfig:
    """Configuración de orden de lectura para páginas y líneas OCR.

    - Japonés/Chino vertical: columnas de derecha a izquierda y dentro de cada columna
      de arriba hacia abajo.
    - Idiomas occidentales: filas de arriba hacia abajo y dentro de cada fila de
      izquierda a derecha.
    """

    language: str = "Japonés"
    min_band_px: float = 24.0
    page_row_ratio: float = 0.65
    ocr_row_ratio: float = 0.70
    vertical_aspect_ratio: float = 1.15
    horizontal_aspect_ratio: float = 1.20


class ReadingOrderResolver:
    """Ordenador reutilizable para globos, regiones, cajas y líneas OCR.

    El proyecto antes tenía pequeñas claves de orden en varios módulos. Esta clase
    centraliza la lógica para evitar que el OCR y los globos se ordenen con reglas
    distintas.
    """

    RTL_VERTICAL_LANGS = {"Japonés", "Chino"}
    CJK_LANGS = {"Japonés", "Chino", "Coreano"}
    WESTERN_LANGS = {"Inglés", "Español"}

    def __init__(self, language: str = "Japonés", config: ReadingOrderConfig | None = None) -> None:
        base = config or ReadingOrderConfig(language=language)
        object.__setattr__(base, "language", language or base.language)
        self.config = base
        self.language = base.language

    @property
    def is_cjk(self) -> bool:
        return self.language in self.CJK_LANGS

    @property
    def page_reads_right_to_left(self) -> bool:
        return self.language in self.RTL_VERTICAL_LANGS

    @staticmethod
    def normalize_box(box: Sequence[float] | Box) -> Box:
        if len(box) != 4:
            raise ValueError(f"Se esperaba box de 4 valores, recibido: {box!r}")
        x, y, w, h = box
        return int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h)))

    @staticmethod
    def box_from_points(points: Sequence[Sequence[float]]) -> Box:
        arr = np.array(points, dtype=np.float32)
        if arr.size == 0:
            return 0, 0, 1, 1
        xs = arr[:, 0]
        ys = arr[:, 1]
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        return int(round(x1)), int(round(y1)), max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))

    @classmethod
    def box_from_ocr_item(cls, item: Any) -> Box:
        """Acepta formatos comunes de EasyOCR/PaddleOCR/dicts internos."""
        if isinstance(item, TextRegion):
            return item.bbox
        if isinstance(item, dict):
            if "bbox" in item:
                return cls.normalize_box(item["bbox"])
            if "box" in item:
                return cls.box_from_points(item.get("box") or [])
            if "points" in item:
                return cls.box_from_points(item.get("points") or [])
        if isinstance(item, (tuple, list)):
            # Caja directa (x, y, w, h)
            if len(item) == 4 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in item):
                return cls.normalize_box(item)  # type: ignore[arg-type]
            # EasyOCR/Paddle: ([[x,y]...], text, conf) o [[x,y]...], (text, conf)
            if item and isinstance(item[0], (tuple, list, np.ndarray)):
                return cls.box_from_points(item[0])
        raise ValueError(f"No se pudo extraer bbox del item OCR: {item!r}")

    @staticmethod
    def center(box: Box) -> Tuple[float, float]:
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def _median(values: Sequence[float], default: float) -> float:
        return float(np.median(list(values))) if values else float(default)

    @staticmethod
    def _overlap_ratio_1d(a1: float, a2: float, b1: float, b2: float) -> float:
        inter = max(0.0, min(a2, b2) - max(a1, b1))
        denom = max(1.0, min(a2 - a1, b2 - b1))
        return inter / denom

    def _horizontal_row_key(self, box: Box, *, right_to_left: bool = False, ratio: float | None = None) -> Tuple[int, float, float]:
        x, y, w, h = box
        cx, cy = self.center(box)
        band = max(self.config.min_band_px, h * (self.config.ocr_row_ratio if ratio is None else ratio))
        row = int(round(cy / band))
        x_key = -cx if right_to_left else cx
        return row, x_key, y

    def key_for_page_box(self, box: Box) -> Tuple[int, float, float]:
        """Clave simple para compatibilidad con sorted(..., key=...)."""
        x, y, w, h = box
        cx, cy = self.center(box)
        band = max(self.config.min_band_px, h * self.config.page_row_ratio)
        row = int(round(cy / band))
        return row, -cx if self.page_reads_right_to_left else cx, y

    def sort_boxes(self, boxes: Sequence[Box]) -> List[Box]:
        return [self.normalize_box(box) for box in sorted(boxes, key=lambda b: self.key_for_page_box(self.normalize_box(b)))]

    def sort_regions(self, regions: Sequence[TextRegion]) -> List[TextRegion]:
        return list(sorted(regions, key=lambda r: self.key_for_page_box(r.bbox)))

    def sort_by_box(self, items: Sequence[T], box_getter: Callable[[T], Box]) -> List[T]:
        return list(sorted(items, key=lambda item: self.key_for_page_box(box_getter(item))))

    def _looks_vertical_layout(self, boxes: Sequence[Box]) -> bool:
        if not boxes:
            return False
        vertical = 0
        horizontal = 0
        for _x, _y, w, h in boxes:
            if h >= w * self.config.vertical_aspect_ratio:
                vertical += 1
            if w >= h * self.config.horizontal_aspect_ratio:
                horizontal += 1
        total = max(1, len(boxes))
        vertical_ratio = vertical / total
        horizontal_ratio = horizontal / total
        if not self.is_cjk:
            return False
        return vertical_ratio >= 0.45 or (vertical_ratio >= 0.25 and horizontal_ratio < 0.60)

    def _cluster_columns(self, indexed_boxes: Sequence[Tuple[int, Box]]) -> List[List[Tuple[int, Box]]]:
        if not indexed_boxes:
            return []
        widths = [box[2] for _idx, box in indexed_boxes]
        threshold = max(10.0, self._median(widths, 16.0) * 0.90)
        ordered = sorted(indexed_boxes, key=lambda item: self.center(item[1])[0], reverse=self.page_reads_right_to_left)
        columns: List[List[Tuple[int, Box]]] = []
        centers: List[float] = []

        for idx, box in ordered:
            cx, _cy = self.center(box)
            placed = False
            for col_idx, column in enumerate(columns):
                representative = centers[col_idx]
                # Misma columna si el centro X es cercano o si las cajas se solapan en X.
                overlaps = any(
                    self._overlap_ratio_1d(box[0], box[0] + box[2], other[0], other[0] + other[2]) >= 0.22
                    for _other_idx, other in column
                )
                if abs(cx - representative) <= threshold or overlaps:
                    column.append((idx, box))
                    centers[col_idx] = float(np.mean([self.center(b)[0] for _i, b in column]))
                    placed = True
                    break
            if not placed:
                columns.append([(idx, box)])
                centers.append(cx)

        columns = sorted(columns, key=lambda col: np.mean([self.center(box)[0] for _idx, box in col]), reverse=self.page_reads_right_to_left)
        for column in columns:
            column.sort(key=lambda item: (item[1][1], self.center(item[1])[1], item[0]))
        return columns

    def _sort_ocr_indices_vertical(self, boxes: Sequence[Box]) -> List[int]:
        columns = self._cluster_columns(list(enumerate(boxes)))
        return [idx for column in columns for idx, _box in column]

    def _sort_ocr_indices_horizontal(self, boxes: Sequence[Box]) -> List[int]:
        if not boxes:
            return []
        heights = [box[3] for box in boxes]
        band = max(self.config.min_band_px / 2.0, self._median(heights, 12.0) * self.config.ocr_row_ratio)

        enriched = []
        for idx, box in enumerate(boxes):
            x, y, w, h = box
            cx, cy = self.center(box)
            # En texto horizontal japonés, el orden dentro de una línea también es LTR.
            enriched.append((idx, int(round(cy / band)), x, y, cx))
        enriched.sort(key=lambda row: (row[1], row[3], row[2], row[0]))
        return [idx for idx, *_ in enriched]

    def sort_ocr_items(self, items: Sequence[T], box_getter: Callable[[T], Box] | None = None) -> List[T]:
        if not items:
            return []
        boxes: List[Box] = []
        valid_items: List[T] = []
        getter = box_getter or self.box_from_ocr_item
        for item in items:
            try:
                boxes.append(getter(item))
                valid_items.append(item)
            except Exception:
                # Mantén items problemáticos al final, en su orden original.
                pass
        if not valid_items:
            return list(items)

        if self._looks_vertical_layout(boxes):
            ordered_indices = self._sort_ocr_indices_vertical(boxes)
        else:
            ordered_indices = self._sort_ocr_indices_horizontal(boxes)
        ordered = [valid_items[idx] for idx in ordered_indices]

        # Preserva al final cualquier item que no se pudo interpretar.
        interpreted_ids = {id(item) for item in valid_items}
        ordered.extend([item for item in items if id(item) not in interpreted_ids])
        return ordered

    def sort_detection_groups(self, groups: Sequence[Sequence[T]], box_getter: Callable[[T], Box] | None = None) -> List[List[T]]:
        """Ordena grupos OCR por el bbox unido de cada grupo."""
        getter = box_getter or self.box_from_ocr_item

        def group_box(group: Sequence[T]) -> Box:
            boxes = [getter(item) for item in group]
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[0] + b[2] for b in boxes)
            y2 = max(b[1] + b[3] for b in boxes)
            return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

        valid: List[Tuple[List[T], Box]] = []
        invalid: List[List[T]] = []
        for group in groups:
            group_list = list(group)
            if not group_list:
                continue
            try:
                valid.append((group_list, group_box(group_list)))
            except Exception:
                invalid.append(group_list)
        valid.sort(key=lambda row: self.key_for_page_box(row[1]))
        return [group for group, _box in valid] + invalid
