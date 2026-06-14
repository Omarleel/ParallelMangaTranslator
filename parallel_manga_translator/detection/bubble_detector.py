from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.detection.bubble_detector_config import BubbleDetectorSettings
from parallel_manga_translator.geometry.box_geometry import BoxGeometry
from parallel_manga_translator.language.onomatopoeia_manager import OnomatopoeiaManager
from parallel_manga_translator.models.processing_models import Box, TextRegion
from parallel_manga_translator.layout.reading_order_resolver import ReadingOrderResolver
from parallel_manga_translator.layout.panel_order_resolver import PanelAwareReadingOrderResolver, PanelOrderConfig
from parallel_manga_translator.detection.yolo_bubble_detector import YoloBubbleCandidate, YoloBubbleDetector
from parallel_manga_translator.config.app_config import ProcessingConfig, QualityConfig
from parallel_manga_translator.infrastructure.logging_config import get_logger

from parallel_manga_translator.detection.bubble_geometry_mixin import BubbleGeometryMixin
from parallel_manga_translator.detection.bubble_text_rules_mixin import BubbleTextRulesMixin
from parallel_manga_translator.detection.bubble_region_builder_mixin import BubbleRegionBuilderMixin
from parallel_manga_translator.detection.bubble_splitter_mixin import BubbleSplitterMixin
from parallel_manga_translator.detection.bubble_debug_mixin import BubbleDebugMixin
from parallel_manga_translator.detection.free_text_recovery_mixin import FreeTextRecoveryMixin
logger = get_logger(__name__)

BUBBLE_SPLIT_DEBUG_VERSION = "v7_bubble_onomatopoeia_translation_2026_06_11"


class BubbleDetector(BubbleGeometryMixin, BubbleTextRulesMixin, BubbleRegionBuilderMixin, BubbleSplitterMixin, BubbleDebugMixin, FreeTextRecoveryMixin):
    """Detector de regiones basado SOLO en modelos preentrenados para globos.

    Ya no existe fallback heurístico para globos de texto. El flujo YOLO es:

        modelo preentrenado detecta globos -> OCR dentro de cada globo -> OCR global
        solo aporta pistas y textos libres/SFX fuera de globos.

    Si el modelo no puede cargarse, se lanza error. Las onomatopeyas/textos libres fuera
    de globo siguen usando cajas OCR como máscara propia, pero eso no se usa para inventar
    globos de diálogo.
    """

    def __init__(
        self,
        idioma_entrada: str = "Japonés",
        quality_config: QualityConfig | None = None,
        processing_config: ProcessingConfig | None = None,
    ) -> None:
        self.idioma_entrada = idioma_entrada
        self.quality_config = quality_config
        self.processing_config = processing_config
        self.onomatopoeia_manager = OnomatopoeiaManager()
        self.reading_order_resolver = ReadingOrderResolver(idioma_entrada)
        self._apply_settings(BubbleDetectorSettings.from_config(quality_config, processing_config))
        q = quality_config
        self.fine_text_detection = bool(getattr(q, "fine_text_detection", True))
        self.fine_text_mask_dilate = int(getattr(q, "fine_text_mask_dilate", 2))
        self.panel_order_resolver = PanelAwareReadingOrderResolver(
            self.reading_order_resolver,
            PanelOrderConfig.from_quality_config(q),
        )
        self._debug_page_index = 0
        if self.merge_debug:
            logger.info("Bubble split debug activo: %s", BUBBLE_SPLIT_DEBUG_VERSION)
        if not self.enabled:
            raise RuntimeError(
                "quality.bubble_detection=false no está permitido en esta versión: la detección de globos "
                "debe hacerse con un modelo preentrenado."
            )
        self.yolo_detector = YoloBubbleDetector(quality_config=quality_config)

    def _apply_settings(self, settings: BubbleDetectorSettings) -> None:
        """Aplica la configuración del detector de globos."""
        self.enabled = settings.enabled

        self.split_merged_bubbles = settings.split.enabled
        self.split_min_ocr_groups = settings.split.min_ocr_groups
        self.split_min_gap_px = settings.split.min_gap_px
        self.split_gap_ratio = settings.split.gap_ratio
        self.split_cluster_min_gap_px = settings.split.cluster_min_gap_px
        self.split_cluster_gap_ratio = settings.split.cluster_gap_ratio
        self.split_group_pad_x = settings.split.group_pad_x
        self.split_group_pad_y = settings.split.group_pad_y
        self.split_group_min_pad = settings.split.group_min_pad

        self.ocr_merge_x_overlap = settings.ocr_merge.x_overlap
        self.ocr_merge_y_gap_ratio = settings.ocr_merge.y_gap_ratio
        self.ocr_merge_cjk_y_overlap = settings.ocr_merge.cjk_y_overlap
        self.ocr_merge_cjk_x_gap_ratio = settings.ocr_merge.cjk_x_gap_ratio
        self.ocr_merge_cjk_columns = settings.ocr_merge.cjk_columns
        self.ocr_merge_line_y_overlap = settings.ocr_merge.line_y_overlap
        self.ocr_merge_line_x_gap_ratio = settings.ocr_merge.line_x_gap_ratio
        self.ocr_merge_line_horizontal_only = settings.ocr_merge.line_horizontal_only

        self.free_text_max_area_ratio = settings.free_text.max_area_ratio
        self.free_text_hard_max_area_ratio = settings.free_text.hard_max_area_ratio
        self.free_text_max_width_ratio = settings.free_text.max_width_ratio
        self.free_text_max_height_ratio = settings.free_text.max_height_ratio
        self.free_text_min_confidence = settings.free_text.min_confidence
        self.free_text_large_min_confidence = settings.free_text.large_min_confidence
        self.free_text_gap_recovery = settings.free_text.gap_recovery
        self.free_text_gap_max_px = settings.free_text.gap_max_px
        self.free_text_gap_min_y_overlap = settings.free_text.gap_min_y_overlap
        self.free_text_gap_min_density = settings.free_text.gap_min_density
        self.free_text_gap_max_density = settings.free_text.gap_max_density

        self.merge_debug = settings.merge_debug.enabled
        self.merge_debug_pair_limit = settings.merge_debug.pair_limit
        self.merge_debug_dir = settings.merge_debug.directory


























































    


