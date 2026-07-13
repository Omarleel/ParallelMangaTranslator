from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from parallel_manga_translator.config.constants import COLOR_BLANCO, COLOR_NEGRO, FACTOR_ESPACIO, RUTA_FUENTE, TAMANIO_MINIMO_FUENTE


class MaskTextAreaMixin:
    """Cálculo de áreas seguras de escritura dentro de máscaras."""

    @staticmethod
    def _normalize_clip_mask(clip_mask: np.ndarray, width: int, height: int) -> np.ndarray:
        """Devuelve la máscara local del globo como uint8 binaria del tamaño de render."""
        local = np.asarray(clip_mask, dtype=np.uint8)
        if local.ndim == 3:
            local = cv2.cvtColor(local, cv2.COLOR_BGR2GRAY)
        if local.shape[:2] != (height, width):
            local = cv2.resize(local, (width, height), interpolation=cv2.INTER_NEAREST)
        _, local = cv2.threshold(local, 31, 255, cv2.THRESH_BINARY)
        return local

    @staticmethod
    def _largest_rect_from_row_spans(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calcula un rectángulo interior estable para máscaras convexas/semiconvexas.

        Para globos ovalados, usar toda la bbox permite que las líneas superiores o inferiores
        queden dentro del rectángulo pero fuera de la curva real del globo, y luego la máscara
        las recorta. Este método busca un rectángulo cuyas filas estén dentro de la máscara.
        """
        rows = binary_mask > 0
        height, width = rows.shape[:2]
        lefts = np.full(height, width, dtype=np.int32)
        rights = np.full(height, -1, dtype=np.int32)
        valid_rows = []

        for row_idx in range(height):
            cols = np.flatnonzero(rows[row_idx])
            if cols.size == 0:
                continue
            lefts[row_idx] = int(cols[0])
            rights[row_idx] = int(cols[-1])
            valid_rows.append(row_idx)

        if not valid_rows:
            return 0, 0, width, height

        # Reduce carga en máscaras muy altas sin perder estabilidad visual.
        step = max(1, height // 420)
        candidate_tops = range(valid_rows[0], valid_rows[-1] + 1, step)
        best = (0, int(valid_rows[0]), max(1, width), max(1, valid_rows[-1] - valid_rows[0] + 1))
        best_score = -1.0
        center_x = width / 2.0
        center_y = height / 2.0

        for top in candidate_tops:
            if rights[top] < lefts[top]:
                continue
            left = int(lefts[top])
            right = int(rights[top])
            for bottom in range(top, valid_rows[-1] + 1, step):
                if rights[bottom] < lefts[bottom]:
                    break
                left = max(left, int(lefts[bottom]))
                right = min(right, int(rights[bottom]))
                rect_w = right - left + 1
                rect_h = bottom - top + 1
                if rect_w <= 1 or rect_h <= 1:
                    continue
                area = rect_w * rect_h
                # Cuando hay varias áreas parecidas, preferimos la que queda más centrada
                # en el globo para que el bloque de texto no se pegue a la cola/borde.
                rect_cx = left + rect_w / 2.0
                rect_cy = top + rect_h / 2.0
                center_penalty = 1.0 - min(0.35, (abs(rect_cx - center_x) / max(1.0, width) + abs(rect_cy - center_y) / max(1.0, height)))
                score = area * center_penalty
                if score > best_score:
                    best_score = score
                    best = (left, top, rect_w, rect_h)

        return best

    def _safe_text_area_from_mask(self, local_mask: Optional[np.ndarray], width: int, height: int, style: str) -> Tuple[int, int, int, int]:
        """
        Devuelve la caja interior donde debe ajustarse el texto antes de aplicar clip_mask.

        El clip se mantiene como protección final, pero el ajuste de fuente/líneas usa esta
        caja interior. Así el texto se encoge/redistribuye y no queda comido por la máscara
        curva del globo.
        """
        if local_mask is None or style.startswith("onomatopeya"):
            return 0, 0, width, height

        mask = self._normalize_clip_mask(local_mask, width, height)
        if cv2.countNonZero(mask) < max(12, int(width * height * 0.02)):
            return 0, 0, width, height

        # Protege contra bordes curvos y contra el ancho del stroke. En globos grandes
        # el valor crece poco para no desperdiciar demasiado espacio.
        min_side = max(1, min(width, height))
        if style == "narracion":
            pad_ratio = 0.035
        else:
            pad_ratio = 0.055
        pad = max(2, min(18, int(round(min_side * pad_ratio))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
        inner = cv2.erode(mask, kernel, iterations=1)
        if cv2.countNonZero(inner) < max(12, int(cv2.countNonZero(mask) * 0.25)):
            inner = mask

        sx, sy, sw, sh = self._largest_rect_from_row_spans(inner)
        if sw < max(8, width * 0.15) or sh < max(8, height * 0.15):
            x, y, w, h = cv2.boundingRect(inner)
            sx, sy, sw, sh = int(x), int(y), int(w), int(h)

        # Un último margen mínimo evita que el stroke caiga justo sobre el límite del rectángulo seguro.
        extra = max(1, min(4, int(round(min(sw, sh) * 0.025))))
        sx = min(width - 1, max(0, sx + extra))
        sy = min(height - 1, max(0, sy + extra))
        sw = max(1, min(width - sx, sw - 2 * extra))
        sh = max(1, min(height - sy, sh - 2 * extra))
        return sx, sy, sw, sh
