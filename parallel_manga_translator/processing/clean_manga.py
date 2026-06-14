from __future__ import annotations

from typing import List

import nest_asyncio
import numpy as np
import torch

from parallel_manga_translator.inpainting import AOTInpainter, BNInpainter, LamaInpainterMPE, LamaLarge, OpenCVInpainter
from parallel_manga_translator.language.onomatopoeia_manager import OnomatopoeiaManager
from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.language.source_language_filter import SourceLanguageFilter
from parallel_manga_translator.config.app_config import OcrConfig, OnomatopoeiaConfig, ProcessingConfig, QualityConfig
from parallel_manga_translator.config.runtime_config import get_active_config
from parallel_manga_translator.ocr.text_detection import TextDetectionFactory

nest_asyncio.apply()
from parallel_manga_translator.processing.clean_source_filter_mixin import CleanSourceFilterMixin
from parallel_manga_translator.processing.clean_onomatopoeia_guard_mixin import CleanOnomatopoeiaGuardMixin
from parallel_manga_translator.processing.clean_mask_strategy_mixin import CleanMaskStrategyMixin
from parallel_manga_translator.processing.clean_inpainting_pipeline_mixin import CleanInpaintingPipelineMixin
from parallel_manga_translator.processing.clean_detection_pipeline_mixin import CleanDetectionPipelineMixin

class CleanManga(CleanSourceFilterMixin, CleanOnomatopoeiaGuardMixin, CleanMaskStrategyMixin, CleanInpaintingPipelineMixin, CleanDetectionPipelineMixin):
    INPAINTER_FACTORIES = {
        "opencv-tela": OpenCVInpainter,
        "lama_mpe": LamaInpainterMPE,
        "lama_large_512px": LamaLarge,
        "aot": AOTInpainter,
        "B/N": BNInpainter,
    }

    def __init__(
        self,
        modelo_inpaint: str,
        idioma_entrada: str = "Japonés",
        quality_config: QualityConfig | None = None,
        onomatopoeia_config: OnomatopoeiaConfig | None = None,
        processing_config: ProcessingConfig | None = None,
        ocr_config: OcrConfig | None = None,
    ) -> None:
        active_config = get_active_config()
        quality_config = quality_config or active_config.quality
        onomatopoeia_config = onomatopoeia_config or active_config.onomatopoeia
        processing_config = processing_config or active_config.processing
        ocr_config = ocr_config or active_config.ocr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inpaint_model = modelo_inpaint
        self.idioma_entrada = idioma_entrada
        self.fast_mode = bool(ocr_config.fast_mode)
        self.ocr_config = ocr_config
        self.inpaint_mode = str(quality_config.inpaint_mode or "auto").strip().lower()
        self.bubble_fill = bool(quality_config.bubble_fill)
        # Por defecto NO rellenamos plano todo el interior del globo: en páginas reales
        # algunas máscaras YOLO vienen casi rectangulares y eso genera parches cuadrados
        # que cruzan bordes. quality.bubble_fill significa "limpiar el texto dentro
        # del globo" usando tinta + recorte por máscara del globo. Si se quiere el
        # comportamiento antiguo se puede activar explícitamente.
        self.bubble_fill_whole_interior = bool(quality_config.bubble_fill_whole_interior)
        self.bubble_fill_edge_margin = int(quality_config.bubble_fill_edge_margin)
        self.bubble_fill_text_dilate = int(quality_config.bubble_fill_text_dilate)
        self.bubble_fill_feather = float(quality_config.bubble_fill_feather)
        self.bubble_fill_flat_max_rectangularity = float(quality_config.bubble_fill_flat_max_rectangularity)
        self.bubble_fill_strategy = str(quality_config.bubble_fill_strategy or "inpaint").strip().lower()
        self.bubble_fill_background_std_threshold = float(quality_config.bubble_fill_background_std_threshold)
        self.bubble_fill_inpaint_padding = int(quality_config.bubble_fill_inpaint_padding)
        self.fine_text_detection = bool(quality_config.fine_text_detection)
        self.fine_text_mask_dilate = int(quality_config.fine_text_mask_dilate)
        self.ink_mask_refinement = bool(quality_config.ink_mask_refinement)
        self.ink_mask_min_component_area = int(quality_config.ink_mask_min_component_area)
        self.ink_mask_component_anchor_overlap = float(quality_config.ink_mask_component_anchor_overlap)
        self.ink_mask_component_anchor_max_gap_ratio = float(quality_config.ink_mask_component_anchor_max_gap_ratio)
        self.onomatopoeia_manager = OnomatopoeiaManager()
        self.onomatopoeia_mode = str(onomatopoeia_config.mode or "translate").strip().lower()
        self.translate_onomatopoeia = bool(onomatopoeia_config.translate)
        if not self.translate_onomatopoeia:
            self.onomatopoeia_mode = "keep"
        self.clean_onomatopoeia = bool(onomatopoeia_config.clean)
        self.bubble_detector = BubbleDetector(idioma_entrada=idioma_entrada, quality_config=quality_config, processing_config=processing_config)
        self.source_language_filter = SourceLanguageFilter(idioma_entrada)
        if modelo_inpaint != "auto":
            self.inpainter = self._build_inpainter(modelo_inpaint)
        else:
            self.inpainter = None
        self.text_detector = TextDetectionFactory.create(idioma_entrada, ocr_config)
        self.last_regions: List[TextRegion] = []