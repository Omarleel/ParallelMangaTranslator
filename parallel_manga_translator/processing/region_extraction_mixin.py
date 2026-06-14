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


class RegionExtractionMixin:
    """Extracción y ordenamiento de áreas de interés para OCR."""

    @staticmethod
    def _prepare_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return crop

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h=8)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        # Binarización adaptativa + corrección de texto blanco sobre fondo oscuro.
        block_size = max(15, (min(gray.shape[:2]) // 8) * 2 + 1)
        binaria = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            9,
        )

        pixeles_blancos = cv2.countNonZero(binaria)
        pixeles_totales = binaria.size
        pixeles_negros = pixeles_totales - pixeles_blancos
        if pixeles_negros > pixeles_blancos:
            binaria = cv2.bitwise_not(binaria)

        # Borde blanco para que OCR no pierda caracteres pegados a la caja.
        border = max(6, min(18, int(round(min(binaria.shape[:2]) * 0.06))))
        binaria = cv2.copyMakeBorder(binaria, border, border, border, border, cv2.BORDER_CONSTANT, value=255)

        h, w = binaria.shape[:2]
        min_side = min(h, w)
        max_side = max(h, w)
        scale = 1.0
        if min_side < 96:
            scale = max(scale, min(3.0, 96 / max(1, min_side)))
        if max_side < 420:
            scale = max(scale, min(2.2, 420 / max(1, max_side)))
        if scale > 1.01:
            binaria = cv2.resize(binaria, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

    def _reading_order_key(self, item):
        box = item.bbox if isinstance(item, TextRegion) else item
        return self.reading_order_resolver.key_for_page_box(box)

    def _sort_regions_for_reading(self, regiones: Sequence[TextRegion]) -> List[TextRegion]:
        try:
            return self.reading_order_resolver.sort_regions(regiones)
        except Exception as exc:
            logger.warning("No se pudo ordenar regiones por lectura; usando fallback: %s", exc)
            return sorted(list(regiones), key=self._reading_order_key)

    def _sort_boxes_for_reading(self, boxes: Sequence[Box]) -> List[Box]:
        try:
            return self.reading_order_resolver.sort_boxes(boxes)
        except Exception as exc:
            logger.warning("No se pudo ordenar cajas por lectura; usando fallback: %s", exc)
            return sorted(list(boxes), key=self._reading_order_key)

    def _masked_region_crop_for_ocr(self, imagen: np.ndarray, region: TextRegion) -> np.ndarray:
        """Prepara un recorte de OCR desde la región detectada.

        Para globos usa la máscara YOLO completa: OCR dentro del globo, no dentro de la
        caja OCR antigua. Para texto libre/SFX se conserva su máscara local expandida.
        """
        height_img, width_img = imagen.shape[:2]
        use_text_hint = (
            self.ocr_region_mode in {"text", "text_hint", "tight"}
            and region.detections_count > 0
            and region.kind in {"dialogue", "narration", "unknown"}
        )
        box = region.text_bbox if use_text_hint else region.ocr_bbox
        x, y, w, h = self._clip_box_to_image(box, width_img, height_img)
        crop = imagen[y:y + h, x:x + w]
        if crop.size == 0:
            return crop

        local_mask = region.mask[y:y + h, x:x + w]
        if local_mask.size and cv2.countNonZero(local_mask) > 0:
            if local_mask.shape[:2] != crop.shape[:2]:
                local_mask = cv2.resize(local_mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Deja todo lo que esté fuera del globo en blanco para que OCR no lea arte cercano.
            canvas = np.full_like(crop, 255)
            canvas[local_mask > 0] = crop[local_mask > 0]
            crop = canvas

        return self._prepare_crop_for_ocr(crop)

    def obtener_areas_interes_desde_regiones(self, imagen, regiones):
        cuadros_delimitadores: List[Box] = []
        imagenes_interes = []
        regiones_ordenadas = self._sort_regions_for_reading(list(regiones))
        height_img, width_img = imagen.shape[:2]

        for region in regiones_ordenadas:
            area_limpia = self._masked_region_crop_for_ocr(imagen, region)
            cuadros_delimitadores.append(region.render_bbox)
            imagenes_interes.append(area_limpia)

        return cuadros_delimitadores, imagenes_interes, regiones_ordenadas

    def obtener_areas_interes(self, imagen, mascara_capa):
        cuadros_delimitadores: List[Box] = []
        imagenes_interes = []
        height_img, width_img = imagen.shape[:2]

        boxes = self._mask_to_boxes(mascara_capa)

        for box in self._sort_boxes_for_reading(boxes):
            x, y, w, h = self._expand_box(box, width_img, height_img)
            area_interes = imagen[y:y + h, x:x + w]
            area_limpia = self._prepare_crop_for_ocr(area_interes)

            cuadros_delimitadores.append((x, y, w, h))
            imagenes_interes.append(area_limpia)

        return cuadros_delimitadores, imagenes_interes
