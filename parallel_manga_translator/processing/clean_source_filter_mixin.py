from __future__ import annotations

import asyncio
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import nest_asyncio
import numpy as np
import torch
from PIL import Image

from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.language.source_language_filter import SourceLanguageFilter
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.ocr.ocr_manager import OcrManager

logger = get_logger(__name__)
Detection = Tuple[Sequence[Sequence[float]], str, float]


class CleanSourceFilterMixin:
    """Reglas para filtrar regiones por idioma de origen."""

    def _source_filter(self) -> SourceLanguageFilter:
        filtro = getattr(self, "source_language_filter", None)
        if filtro is None:
            filtro = SourceLanguageFilter(getattr(self, "idioma_entrada", ""))
            self.source_language_filter = filtro
        return filtro

    def _region_matches_source_language(self, region: TextRegion) -> bool:
        filtro = self._source_filter()
        metadata = getattr(region, "metadata", None)

        if filtro.should_preserve_region_without_processing(region):
            reason = filtro.explain_preserved_region(region)
            if isinstance(metadata, dict):
                metadata["processing_skipped"] = True
                metadata["processing_skip_reason"] = reason
                metadata["source_language_filter"] = reason
                metadata["source_language_allowed"] = False
                metadata["source_language"] = getattr(self, "idioma_entrada", "")
            return False

        allowed = filtro.should_process_region(region, allow_unknown=True)
        if isinstance(metadata, dict):
            metadata["source_language_filter"] = filtro.explain_region(region)
            metadata["source_language_allowed"] = bool(allowed)
            metadata["source_language"] = getattr(self, "idioma_entrada", "")
        return bool(allowed)

    def _filter_regions_by_source_language(self, regiones: Sequence[TextRegion]) -> List[TextRegion]:
        filtradas: List[TextRegion] = []
        for region in regiones or []:
            if self._region_matches_source_language(region):
                filtradas.append(region)
            else:
                logger.debug(
                    "Región omitida por idioma de origen: idioma=%s bbox=%s hint=%r",
                    getattr(self, "idioma_entrada", ""),
                    getattr(region, "bbox", None),
                    getattr(region, "source_text_hint", ""),
                )
        return filtradas
