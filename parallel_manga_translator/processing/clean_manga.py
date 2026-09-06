from __future__ import annotations

from typing import List

import nest_asyncio
import torch

from parallel_manga_translator.language.onomatopoeia_manager import OnomatopoeiaManager
from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.language.source_language_filter import SourceLanguageFilter
from parallel_manga_translator.architecture.ports import InpainterPort, RegionDetectorPort, TextDetectionPort
from parallel_manga_translator.config.app_config import OcrConfig, OnomatopoeiaConfig, ProcessingConfig, QualityConfig
from parallel_manga_translator.quality.visual_inpaint_verifier import VisualInpaintVerifier
from parallel_manga_translator.ocr.text_detection import TextDetectionFactory

nest_asyncio.apply()

#: Distingue "no me han inyectado inpainter" de "inyéctame None", que ya significa
#: resolverlo por página según el contenido.
_SIN_INYECTAR = object()
from parallel_manga_translator.processing.clean_source_filter_mixin import CleanSourceFilterMixin
from parallel_manga_translator.processing.clean_onomatopoeia_guard_mixin import CleanOnomatopoeiaGuardMixin
from parallel_manga_translator.processing.bubble_fill_policy import BubbleFillPolicy
from parallel_manga_translator.processing.inpainter_runner import InpainterRunner
from parallel_manga_translator.processing.clean_mask_strategy import CleanMaskStrategy
from parallel_manga_translator.processing.clean_inpainting_pipeline_mixin import CleanInpaintingPipelineMixin
from parallel_manga_translator.processing.clean_detection_pipeline_mixin import CleanDetectionPipelineMixin

class CleanManga(CleanSourceFilterMixin, CleanOnomatopoeiaGuardMixin, CleanInpaintingPipelineMixin, CleanDetectionPipelineMixin):
    def __init__(
        self,
        modelo_inpaint: str,
        idioma_entrada: str = "Japonés",
        quality_config: QualityConfig | None = None,
        onomatopoeia_config: OnomatopoeiaConfig | None = None,
        processing_config: ProcessingConfig | None = None,
        ocr_config: OcrConfig | None = None,
        bubble_detector: RegionDetectorPort | None = None,
        text_detector: TextDetectionPort | None = None,
        inpainter: InpainterPort | None = _SIN_INYECTAR,
        mask_strategy: CleanMaskStrategy | None = None,
        fill_policy: BubbleFillPolicy | None = None,
        visual_inpaint_verifier: VisualInpaintVerifier | None = None,
        inpainter_runner: InpainterRunner | None = None,
    ) -> None:
        # Por defecto, los valores del dataclass; nunca el estado global del proceso.
        quality_config = quality_config if quality_config is not None else QualityConfig()
        onomatopoeia_config = onomatopoeia_config if onomatopoeia_config is not None else OnomatopoeiaConfig()
        processing_config = processing_config if processing_config is not None else ProcessingConfig()
        ocr_config = ocr_config if ocr_config is not None else OcrConfig()
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
        self.bubble_ink_without_ocr_max_ratio = float(quality_config.bubble_ink_without_ocr_max_ratio)
        self.bubble_fill_feather = float(quality_config.bubble_fill_feather)
        self.bubble_fill_flat_max_rectangularity = float(quality_config.bubble_fill_flat_max_rectangularity)
        # Colaborador explícito: 12 de sus 13 métodos son puros y el otro sólo necesita
        # estos cuatro ajustes, que antes leía del `self` de quien la heredara.
        self.mask_strategy = mask_strategy if mask_strategy is not None else CleanMaskStrategy(
            bubble_fill_whole_interior=self.bubble_fill_whole_interior,
            bubble_fill_edge_margin=self.bubble_fill_edge_margin,
            bubble_fill_text_dilate=self.bubble_fill_text_dilate,
            bubble_fill_flat_max_rectangularity=self.bubble_fill_flat_max_rectangularity,
        )
        self.bubble_fill_background_std_threshold = float(quality_config.bubble_fill_background_std_threshold)
        self.bubble_fill_inpaint_padding = int(quality_config.bubble_fill_inpaint_padding)
        self.visual_inpaint_verifier_enabled = bool(quality_config.visual_inpaint_verifier)
        self.visual_inpaint_retry = bool(quality_config.visual_inpaint_retry)
        self.visual_inpaint_retry_models = str(quality_config.visual_inpaint_retry_models or "solid,opencv-tela,lama_mpe,aot")
        self.visual_inpaint_max_retries = int(quality_config.visual_inpaint_max_retries)
        self.visual_inpaint_debug = bool(quality_config.visual_inpaint_debug)
        self.visual_inpaint_best_of_textured = bool(quality_config.visual_inpaint_best_of_textured)
        # Qué candidato se intenta primero y con cuáles se reintenta. Aplicar el
        # relleno y verificarlo siguen en el pipeline; aquí sólo está la elección.
        self.fill_policy = fill_policy if fill_policy is not None else BubbleFillPolicy(
            bubble_fill_strategy=str(quality_config.bubble_fill_strategy or "inpaint").strip().lower(),
            visual_inpaint_retry=self.visual_inpaint_retry,
            visual_inpaint_retry_models=self.visual_inpaint_retry_models,
            visual_inpaint_max_retries=self.visual_inpaint_max_retries,
        )
        # Inyectable: los tests lo sustituían mutando el atributo después de construir,
        # que es la versión frágil de lo mismo.
        self.visual_inpaint_verifier = (
            visual_inpaint_verifier if visual_inpaint_verifier is not None
            else VisualInpaintVerifier(accept_score=float(quality_config.visual_inpaint_accept_score))
        )
        self.fine_text_detection = bool(quality_config.fine_text_detection)
        self.fine_text_mask_dilate = int(quality_config.fine_text_mask_dilate)
        self.ink_mask_refinement = bool(quality_config.ink_mask_refinement)
        self.ink_mask_min_component_area = int(quality_config.ink_mask_min_component_area)
        self.ink_mask_component_anchor_overlap = float(quality_config.ink_mask_component_anchor_overlap)
        self.ink_mask_component_anchor_max_gap_ratio = float(quality_config.ink_mask_component_anchor_max_gap_ratio)
        self.text_halo_growth_px = int(quality_config.text_halo_growth_px)
        self.onomatopoeia_manager = OnomatopoeiaManager()
        self.onomatopoeia_mode = str(onomatopoeia_config.mode or "translate").strip().lower()
        self.translate_onomatopoeia = bool(onomatopoeia_config.translate)
        if not self.translate_onomatopoeia:
            self.onomatopoeia_mode = "keep"
        self.clean_onomatopoeia = bool(onomatopoeia_config.clean)
        # Las tres etapas pesadas se pueden inyectar. Sin esto, construir un CleanManga
        # cargaba YOLO, el detector de texto y un inpainter, así que probarlo obligaba a
        # `object.__new__` y a rellenar los atributos a mano: un test que no pasa por el
        # constructor no prueba lo que hace el constructor.
        self.bubble_detector = (
            bubble_detector if bubble_detector is not None
            else BubbleDetector(idioma_entrada=idioma_entrada, quality_config=quality_config, processing_config=processing_config)
        )
        self.source_language_filter = SourceLanguageFilter(idioma_entrada)
        # Ejecutar el inpainting (resolver modelo, cachear instancias, aplicarlas) es del
        # runner. Elegir el candidato es de `fill_policy`, y juzgar el resultado del
        # verificador: tres responsabilidades que estaban en el mismo mixin.
        self.inpainter_runner = inpainter_runner if inpainter_runner is not None else InpainterRunner(
            inpaint_model=modelo_inpaint,
            fill_policy=self.fill_policy,
            bubble_fill_inpaint_padding=self.bubble_fill_inpaint_padding,
        )
        if inpainter is not _SIN_INYECTAR:
            # `None` ya significa "resolver por página", así que hace falta un centinela
            # para distinguir "no me han inyectado nada" de "inyéctame None".
            self.inpainter_runner.inpainter = inpainter
        elif modelo_inpaint != "auto":
            self.inpainter_runner.inpainter = self.inpainter_runner.build_inpainter(modelo_inpaint)
        self.text_detector = (
            text_detector if text_detector is not None
            else TextDetectionFactory.create(idioma_entrada, ocr_config)
        )
        self.last_regions: List[TextRegion] = []
    @property
    def inpainter(self):
        """Delega en el runner: una sola fuente de verdad, como con la estrategia."""
        return self.inpainter_runner.inpainter

    @inpainter.setter
    def inpainter(self, value) -> None:
        self.inpainter_runner.inpainter = value

    @property
    def bubble_fill_strategy(self) -> str:
        """Delega en la política: una sola fuente de verdad.

        Guardar aquí una copia además de en `fill_policy` era una trampa silenciosa —
        mutar el atributo no habría surtido efecto y nadie se enteraría.
        """
        return self.fill_policy.bubble_fill_strategy

    @bubble_fill_strategy.setter
    def bubble_fill_strategy(self, value: str) -> None:
        self.fill_policy.bubble_fill_strategy = str(value or "inpaint").strip().lower()

    def set_debug_page_context(
        self,
        page_index: int,
        *,
        source_filename: str | None = None,
        output_filename: str | None = None,
    ) -> None:
        """Propaga al detector el índice global de la página que se está procesando."""
        self.bubble_detector.set_debug_page_context(
            page_index,
            source_filename=source_filename,
            output_filename=output_filename,
        )

    def clear_debug_page_context(self) -> None:
        self.bubble_detector.clear_debug_page_context()
