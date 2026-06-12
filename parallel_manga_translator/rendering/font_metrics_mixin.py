from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from parallel_manga_translator.config.constants import COLOR_BLANCO, COLOR_NEGRO, FACTOR_ESPACIO, RUTA_FUENTE, TAMANIO_MINIMO_FUENTE


class FontMetricsMixin:
    """Carga de fuentes y métricas tipográficas."""

    @lru_cache(maxsize=192)
    def _get_font(self, size: int):
        safe_size = max(self.absolute_min_font_size, int(size))
        try:
            return ImageFont.truetype(self.font_path, safe_size)
        except OSError:
            # Evita que el renderizado falle si la fuente aún no fue descargada.
            return ImageFont.load_default()

    @staticmethod
    def _text_width(texto: str, fuente) -> int:
        bbox = fuente.getbbox(texto or " ")
        return max(0, bbox[2] - bbox[0])

    @staticmethod
    def _text_height(texto: str, fuente) -> int:
        bbox = fuente.getbbox(texto or " ")
        return max(1, bbox[3] - bbox[1])

    @staticmethod
    def _line_spacing(fuente) -> float:
        size = getattr(fuente, "size", TAMANIO_MINIMO_FUENTE)
        return max(1.0, min(size * 0.22, size * FACTOR_ESPACIO))
