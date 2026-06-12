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


class TranslationSourceFilterMixin:
    """Filtro de idioma de origen aplicado antes de OCR/traducción."""

    def _source_filter(self) -> SourceLanguageFilter:
        filtro = getattr(self, "source_language_filter", None)
        if filtro is None:
            filtro = SourceLanguageFilter(getattr(self, "idioma_entrada", ""))
            self.source_language_filter = filtro
        return filtro

    def _mark_region_source_language(self, region: Optional[TextRegion], allowed: bool, reason: str) -> None:
        if region is None:
            return
        metadata = getattr(region, "metadata", None)
        if isinstance(metadata, dict):
            metadata["source_language_allowed"] = bool(allowed)
            metadata["source_language_filter"] = reason
            metadata["source_language"] = getattr(self, "idioma_entrada", "")

    def _region_allows_source_language(self, region: Optional[TextRegion]) -> bool:
        if region is None:
            return True
        filtro = self._source_filter()
        if filtro.should_preserve_region_without_processing(region):
            reason = filtro.explain_preserved_region(region)
            self._mark_region_source_language(region, False, reason)
            metadata = getattr(region, "metadata", None)
            if isinstance(metadata, dict):
                metadata["processing_skipped"] = True
                metadata["processing_skip_reason"] = reason
            return False
        allowed = filtro.should_process_region(region, allow_unknown=True)
        self._mark_region_source_language(region, allowed, filtro.explain_region(region))
        return bool(allowed)

    def _text_is_source_language(self, texto: str, region: Optional[TextRegion] = None) -> bool:
        filtro = self._source_filter()
        if not self._region_allows_source_language(region):
            return False
        allowed = filtro.should_process_text(texto, allow_empty=False)
        if region is not None:
            self._mark_region_source_language(region, allowed, filtro.explain_text(texto))
        return bool(allowed)

    def _source_language_flags_for_texts(self, textos: Sequence[str]) -> List[bool]:
        flags: List[bool] = []
        for idx, texto in enumerate(textos):
            region = self.ultimas_regiones[idx] if self.ultimas_regiones and idx < len(self.ultimas_regiones) else None
            flags.append(self._text_is_source_language(str(texto or ""), region))
        return flags
