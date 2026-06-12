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


class TranslationOrchestratorMixin:
    """Orquestación de alto nivel para traducir una página."""

    def insertar_json_queue(self, indice_imagen, transcripcion_queue, traduccion_queue):
        self.indice_imagen = indice_imagen
        self.transcripcion_queue = transcripcion_queue
        self.traduccion_queue = traduccion_queue

    def traducir_manga(self, imagen, imagen_limpia, mascara_capa, text_regions=None):
        if text_regions:
            cuadros_delimitadores, imagenes_interes, regiones_ordenadas = self.obtener_areas_interes_desde_regiones(imagen, text_regions)
            self.ultimas_regiones = regiones_ordenadas
        else:
            cuadros_delimitadores, imagenes_interes = self.obtener_areas_interes(imagen, mascara_capa)
            self.ultimas_regiones = []
        textos = self.obtener_textos(imagenes_interes)
        return self.incrustar_textos(imagen_limpia, cuadros_delimitadores, textos)
