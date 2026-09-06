from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from parallel_manga_translator.language.onomatopoeia_manager import OnomatopoeiaManager
from parallel_manga_translator.language.source_language_filter import SourceLanguageFilter
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.translation.text_normalization import OcrTextNormalizer
from parallel_manga_translator.infrastructure.logging_config import get_logger

Box = Tuple[int, int, int, int]
logger = get_logger(__name__)


class TranslationGeometry:
    """Geometría de cajas y máscaras de la etapa de traducción.

    Era `TranslationGeometryMixin`. Siete de sus nueve métodos son funciones puras y
    los otros dos sólo necesitan el idioma de origen, así que heredarla no aportaba
    nada: obligaba a `TranslateManga` a llevarla en sus bases para que un mixin
    hermano la llamara por `self`. Ahora el idioma es una dependencia declarada.

    Ojo: `detection/detection_geometry.py` tiene helpers con los mismos nombres y
    **no son equivalentes** —`expand_box` recibe otros argumentos—, así que no se
    pueden unificar sin comprobarlo caso por caso.
    """

    def __init__(self, idioma_entrada: str) -> None:
        self.idioma_entrada = idioma_entrada

    @staticmethod
    def rect_from_contour(contour) -> Box:
        x, y, w, h = cv2.boundingRect(contour)
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def box_area(box: Box) -> int:
        return max(0, box[2]) * max(0, box[3])

    @staticmethod
    def union(a: Box, b: Box) -> Box:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax + aw, bx + bw)
        y2 = max(ay + ah, by + bh)
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def overlap_ratio_1d(a1: int, a2: int, b1: int, b2: int) -> float:
        inter = max(0, min(a2, b2) - max(a1, b1))
        denom = max(1, min(a2 - a1, b2 - b1))
        return inter / denom

    def should_merge_boxes(self, a: Box, b: Box) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        gap_x = max(0, max(bx - ax2, ax - bx2))
        gap_y = max(0, max(by - ay2, ay - by2))
        avg_h = max(1, (ah + bh) / 2)
        avg_w = max(1, (aw + bw) / 2)
        x_overlap = self.overlap_ratio_1d(ax, ax2, bx, bx2)
        y_overlap = self.overlap_ratio_1d(ay, ay2, by, by2)

        # Líneas de un mismo globo suelen estar una debajo de otra y comparten rango X.
        if x_overlap >= 0.22 and gap_y <= max(12, avg_h * 1.45):
            return True

        # Texto japonés vertical: columnas cercanas con bastante solape vertical.
        if self.idioma_entrada == "Japonés" and y_overlap >= 0.22 and gap_x <= max(10, avg_w * 1.15):
            return True

        # Fragmentos rotos de una misma palabra/línea.
        if y_overlap >= 0.45 and gap_x <= max(10, avg_h * 0.80):
            return True

        # Onomatopeyas estilizadas: EasyOCR/inpainting puede separar letras enormes o trazos
        # decorativos. Estas reglas unen componentes próximos sin exigir tanto solape.
        if y_overlap >= 0.18 and gap_x <= max(18, avg_h * 1.35, avg_w * 0.80):
            return True
        if x_overlap >= 0.18 and gap_y <= max(18, avg_w * 1.35, avg_h * 0.80):
            return True

        return False

    def merge_boxes(self, boxes: Sequence[Box]) -> List[Box]:
        merged = list(boxes)
        changed = True
        while changed:
            changed = False
            result: List[Box] = []
            consumed = [False] * len(merged)
            for i, box in enumerate(merged):
                if consumed[i]:
                    continue
                current = box
                consumed[i] = True
                for j in range(i + 1, len(merged)):
                    if consumed[j]:
                        continue
                    if self.should_merge_boxes(current, merged[j]):
                        current = self.union(current, merged[j])
                        consumed[j] = True
                        changed = True
                result.append(current)
            merged = result
        return merged

    def mask_to_boxes(self, mascara_capa: np.ndarray) -> List[Box]:
        height, width = mascara_capa.shape[:2]
        _, mascara_binaria = cv2.threshold(mascara_capa, 127, 255, cv2.THRESH_BINARY)
        mascara_binaria = np.uint8(mascara_binaria)

        if self.idioma_entrada == "Japonés":
            kernel_h = max(3, round(height * 0.011))
            kernel_w = max(3, round(width * 0.006))
        else:
            kernel_h = max(3, round(height * 0.008))
            kernel_w = max(5, round(width * 0.020))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        mascara_agrupada = cv2.morphologyEx(mascara_binaria, cv2.MORPH_CLOSE, kernel, iterations=1)
        mascara_agrupada = cv2.dilate(mascara_agrupada, kernel, iterations=1)

        contours, _ = cv2.findContours(mascara_agrupada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_img = height * width
        min_area = max(20, int(area_img * 0.00003))
        boxes = []
        for contour in contours:
            x, y, w, h = self.rect_from_contour(contour)
            if w < 3 or h < 3 or self.box_area((x, y, w, h)) < min_area:
                continue
            boxes.append((x, y, w, h))

        return self.merge_boxes(boxes)

    @staticmethod
    def expand_box(box: Box, width_img: int, height_img: int) -> Box:
        x, y, w, h = box
        # Margen proporcional: más grande en globos complejos, pero acotado para no invadir viñetas vecinas.
        aspect = max(w, h) / max(1, min(w, h))
        if aspect >= 3.2:
            # SFX/onomatopeyas largas necesitan algo más de aire para no cortar contornos.
            pad_x = int(min(42, max(5, round(w * 0.10))))
            pad_y = int(min(38, max(5, round(h * 0.14))))
        else:
            pad_x = int(min(30, max(6, round(w * 0.18))))
            pad_y = int(min(26, max(6, round(h * 0.22))))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width_img, x + w + pad_x)
        y2 = min(height_img, y + h + pad_y)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def clip_box_to_image(box: Box, width_img: int, height_img: int) -> Box:
        x, y, w, h = box
        x = int(max(0, min(width_img - 1, x)))
        y = int(max(0, min(height_img - 1, y)))
        x2 = int(max(x + 1, min(width_img, x + max(1, w))))
        y2 = int(max(y + 1, min(height_img, y + max(1, h))))
        return x, y, x2 - x, y2 - y
