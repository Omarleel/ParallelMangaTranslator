from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from parallel_manga_translator.detection.bubble_detector_config import BUBBLE_SPLIT_DEBUG_VERSION
from parallel_manga_translator.detection.bubble_detector_config import BubbleDetectorSettings
from parallel_manga_translator.language.onomatopoeia_manager import OnomatopoeiaManager
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.layout.reading_order_resolver import ReadingOrderResolver
from parallel_manga_translator.layout.panel_order_resolver import PanelAwareReadingOrderResolver, PanelOrderConfig
from parallel_manga_translator.detection.yolo_bubble_detector import YoloBubbleDetector
from parallel_manga_translator.config.app_config import ProcessingConfig, QualityConfig
from parallel_manga_translator.infrastructure.logging_config import get_logger

from parallel_manga_translator.detection.detection_geometry import DetectionGeometry
from parallel_manga_translator.detection.bubble_text_rules_mixin import BubbleTextRulesMixin
from parallel_manga_translator.detection.bubble_region_builder_mixin import BubbleRegionBuilderMixin
from parallel_manga_translator.detection.bubble_splitter_mixin import BubbleSplitterMixin
from parallel_manga_translator.detection.bubble_debug_mixin import BubbleDebugMixin
from parallel_manga_translator.detection.free_text_recovery_mixin import FreeTextRecoveryMixin
logger = get_logger(__name__)



class BubbleDetector(BubbleTextRulesMixin, BubbleRegionBuilderMixin, BubbleSplitterMixin, BubbleDebugMixin, FreeTextRecoveryMixin):
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
        geometry: DetectionGeometry | None = None,
    ) -> None:
        # Los valores por defecto salen del dataclass, no del estado global del proceso.
        # Antes se caía a `get_active_config()`, así que construir un detector sin
        # configuración leía el config.yaml del usuario: los tests dejaban de ser
        # deterministas y dos trabajos de la UI compartían ajustes sin saberlo.
        quality_config = quality_config if quality_config is not None else QualityConfig()
        processing_config = processing_config if processing_config is not None else ProcessingConfig()
        # Colaborador explícito en vez de clase base: sus métodos son funciones puras,
        # así que heredarlas sólo servía para que los mixins hermanos las llamaran por
        # `self` sin declarar la dependencia.
        self.geometry = geometry if geometry is not None else DetectionGeometry()
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
        # Contador de respaldo para usos aislados del detector. En el flujo normal,
        # ImageProcessor establece el índice global real de la página antes de detectar.
        self._debug_page_index = 0
        self._debug_page_number: Optional[int] = None
        self._debug_source_filename: Optional[str] = None
        self._debug_output_filename: Optional[str] = None
        if self.merge_debug:
            logger.info("Bubble split debug activo: %s", BUBBLE_SPLIT_DEBUG_VERSION)
        if not self.enabled:
            raise RuntimeError(
                "quality.bubble_detection=false no está permitido en esta versión: la detección de globos "
                "debe hacerse con un modelo preentrenado."
            )
        self.yolo_detector = YoloBubbleDetector(quality_config=quality_config)

    def set_debug_page_context(
        self,
        page_index: int,
        *,
        source_filename: str | None = None,
        output_filename: str | None = None,
    ) -> None:
        """Asocia los artefactos debug con el índice global de la página.

        ``page_index`` es base cero, como el resto del pipeline. Los nombres de
        archivos debug son base uno (pagina_0001, pagina_0002, ...).
        """
        self._debug_page_number = max(1, int(page_index) + 1)
        self._debug_source_filename = source_filename
        self._debug_output_filename = output_filename

    def clear_debug_page_context(self) -> None:
        self._debug_page_number = None
        self._debug_source_filename = None
        self._debug_output_filename = None

    def _apply_settings(self, settings: BubbleDetectorSettings) -> None:
        """Aplica la configuración del detector de globos."""
        self.enabled = settings.enabled

        # Se guardan los grupos de ajustes, no sus campos sueltos. Aplanarlos
        # destruía el contrato que estos dataclasses ya expresan y obligaba a los
        # mixins a leer 28 atributos que nadie declaraba.
        self.split = settings.split
        self.ocr_merge = settings.ocr_merge
        self.free_text = settings.free_text



        self.merge_debug = settings.merge_debug.enabled
        self.merge_debug_pair_limit = settings.merge_debug.pair_limit
        self.merge_debug_dir = settings.merge_debug.directory

    @staticmethod
    def compose_mask(regions: Sequence[TextRegion], image_shape) -> np.ndarray:
        """Compone la máscara de región segura.

        En globos esta máscara representa el interior/área permitida para OCR y
        renderizado. No debe asumirse que es la máscara de limpieza.
        """
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            if region.mask is not None and region.mask.size:
                mask = cv2.bitwise_or(mask, region.mask)
        return mask

    @staticmethod
    def compose_clean_mask(regions: Sequence[TextRegion], image_shape) -> np.ndarray:
        """Compone únicamente la máscara de tinta/texto a borrar.

        Para globos se usa ``region.clean_mask``. Si una región de globo no tiene
        clean_mask, se considera vacía para evitar limpiar todo el globo por
        accidente. Para texto libre/SFX se conserva ``region.mask`` como fallback
        porque esa máscara ya representa el texto expandido, no un globo completo.
        """
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        bubble_kinds = {"dialogue", "narration", "unknown"}
        for region in regions:
            region_clean = getattr(region, "clean_mask", None)
            if region_clean is not None and getattr(region_clean, "size", 0):
                mask = cv2.bitwise_or(mask, (region_clean > 0).astype(np.uint8) * 255)
                continue
            if region.kind not in bubble_kinds and region.mask is not None and region.mask.size:
                mask = cv2.bitwise_or(mask, region.mask)
        return mask
