from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple

import torch

from parallel_manga_translator.ocr.ocr_manager import OcrManager
from parallel_manga_translator.language.onomatopoeia_manager import OnomatopoeiaManager
from parallel_manga_translator.translation.text_normalization import OcrTextNormalizer
from parallel_manga_translator.language.source_language_filter import SourceLanguageFilter
from parallel_manga_translator.rendering.text_renderer import TextRenderer
from parallel_manga_translator.translation.translator_manager import TranslatorManager
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.layout.reading_order_resolver import ReadingOrderResolver
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.config.constants import RUTA_FUENTE, TAMANIO_MINIMO_FUENTE
from parallel_manga_translator.config.app_config import CharacterMemoryConfig, OcrConfig, OnomatopoeiaConfig, ProcessingConfig, QualityConfig, TranslationConfig


Box = Tuple[int, int, int, int]

logger = get_logger(__name__)


from parallel_manga_translator.processing.translation_source_filter_mixin import TranslationSourceFilterMixin
from parallel_manga_translator.processing.translation_orchestrator_mixin import TranslationOrchestratorMixin
from parallel_manga_translator.processing.translation_geometry import TranslationGeometry
from parallel_manga_translator.processing.region_extraction_mixin import RegionExtractionMixin
from parallel_manga_translator.processing.ocr_text_pipeline_mixin import OcrTextPipelineMixin
from parallel_manga_translator.processing.translation_pipeline_mixin import TranslationPipelineMixin
from parallel_manga_translator.processing.rendering_pipeline_mixin import RenderingPipelineMixin
class TranslateManga(TranslationSourceFilterMixin, TranslationOrchestratorMixin, RegionExtractionMixin, OcrTextPipelineMixin, TranslationPipelineMixin, RenderingPipelineMixin):
    def __init__(
        self,
        idioma_entrada,
        idioma_salida,
        metodo_traduccion="Tradicional",
        groq_api_key="",
        lore_manga="",
        ocr_config: OcrConfig | None = None,
        translation_config: TranslationConfig | None = None,
        quality_config: QualityConfig | None = None,
        onomatopoeia_config: OnomatopoeiaConfig | None = None,
        character_memory_config: CharacterMemoryConfig | None = None,
        processing_config: ProcessingConfig | None = None,
        geometry: TranslationGeometry | None = None,
        ocr_manager: OcrManager | None = None,
        translator_manager: TranslatorManager | None = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.idioma_entrada = idioma_entrada
        self.idioma_salida = idioma_salida
        self.metodo_traduccion = metodo_traduccion

        # Por defecto, los valores del dataclass; nunca el estado global del proceso.
        if quality_config is None:
            quality_config = QualityConfig()
        if onomatopoeia_config is None:
            onomatopoeia_config = OnomatopoeiaConfig()
        if processing_config is None:
            processing_config = ProcessingConfig()
        self.processing_config = processing_config

        # Los dos pesados se pueden inyectar: sin esto no se puede componer un modo
        # solo-OCR (no hace falta traductor) ni probar los pasos por separado sin
        # cargar el motor OCR. Es el mismo arreglo que ya tiene CleanManga.
        self.translator_manager = translator_manager if translator_manager is not None else TranslatorManager(
            idioma_entrada,
            idioma_salida,
            metodo=metodo_traduccion,
            groq_api_key=groq_api_key,
            lore_manga=lore_manga,
            translation_config=translation_config,
            character_memory_config=character_memory_config,
            cache_dir=processing_config.cache_dir,
            cache_enabled=processing_config.cache,
        )
        self.ocr_manager = ocr_manager if ocr_manager is not None else OcrManager(
            idioma_entrada=idioma_entrada, config=ocr_config, cache_dir=processing_config.cache_dir
        )
        # Colaborador explícito: siete de sus métodos son puros y los otros dos sólo
        # necesitan el idioma, que ahora recibe en vez de tomarlo del `self` ajeno.
        self.geometry = geometry if geometry is not None else TranslationGeometry(idioma_entrada)
        self.reading_order_resolver = ReadingOrderResolver(idioma_entrada)
        self.text_renderer = TextRenderer(
            font_path=RUTA_FUENTE,
            min_font_size=TAMANIO_MINIMO_FUENTE,
            smart_typography=bool(getattr(quality_config, "typography_smart_wrap", True)),
            hyphenation=bool(getattr(quality_config, "typography_hyphenation", True)),
            balance_lines=bool(getattr(quality_config, "typography_balance_lines", True)),
            line_spacing_factor=float(getattr(quality_config, "typography_line_spacing_factor", 1.0)),
        )
        self.text_normalizer = OcrTextNormalizer()
        self.source_language_filter = SourceLanguageFilter(idioma_entrada)
        self.onomatopoeia_manager = OnomatopoeiaManager()
        self.historial_contexto = deque(maxlen=3)
        self.ultimo_estilos_texto = []
        self.ultimas_regiones: List[TextRegion] = []
        self.ultimos_textos_originales: List[str] = []
        self.ultimos_textos_traducidos: List[str] = []
        self.ultimos_source_language_flags: List[bool] = []
        self.ultimas_asignaciones_hablante: List[Dict[str, Any]] = []
        self.onomatopoeia_mode = str(onomatopoeia_config.mode or "translate").strip().lower()
        self.translate_onomatopoeia = bool(onomatopoeia_config.translate)
        if not self.translate_onomatopoeia:
            self.onomatopoeia_mode = "keep"
        # bubble: OCR sobre el globo completo segmentado; text_hint: recorte más ajustado si hubo OCR global.
        self.ocr_region_mode = str(quality_config.ocr_region_mode or "bubble").strip().lower()
        self.indice_imagen = 0
        self.transcripcion_queue = None
        self.traduccion_queue = None


