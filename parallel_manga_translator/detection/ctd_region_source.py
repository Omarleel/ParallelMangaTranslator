"""Fuente de regiones basada en comic-text-detector.

Sustituye a `BubbleDetector` cuando `quality.region_source: comic_text_detector`. Ocupa
su sitio exacto —`CleanManga` lo recibe inyectado— y por eso expone los mismos cuatro
métodos que el orquestador consume, ni uno más.

La diferencia de fondo con la vía YOLO no es el modelo, es de dónde sale cada máscara:

    YOLO       globo segmentado -> zona segura; la tinta se deriva dentro del globo.
    CTD        bloque de texto  -> zona segura = la caja; la tinta la da el modelo.

Por eso aquí no hay recuperación de texto libre por OCR: el detector ya ve el texto
fuera de globo, que es justo la fuente que en la vía YOLO aporta la mayoría de falsos
positivos. Emitir ambas cosas duplicaría regiones y reintroduciría ese ruido.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

from parallel_manga_translator.architecture.ports import RegionSourcePort  # noqa: F401  (contrato documentado)
from parallel_manga_translator.config.app_config import ProcessingConfig, QualityConfig
from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.detection.comic_text_detector import ComicTextDetection, ComicTextDetector
from parallel_manga_translator.detection.region_reading_order import order_and_number
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.layout.panel_order_resolver import PanelAwareReadingOrderResolver, PanelOrderConfig
from parallel_manga_translator.layout.reading_order_resolver import ReadingOrderResolver
from parallel_manga_translator.models.processing_models import TextRegion

logger = get_logger(__name__)

#: Clases del modelo, medidas sobre paginas reales: la 1 es texto dentro de globo y la
#: 0 texto fuera (rotulos, gritos sobre el arte). Es un *prior* estructural; la
#: clasificacion semantica fina (dialogue/thought/narration/sfx) la hace luego el VLM.
CTD_CLASS_KINDS = {0: "free_text", 1: "dialogue"}


class CtdRegionSource:
    """Adapta la salida del detector de texto al dominio del pipeline."""

    def __init__(
        self,
        idioma_entrada: str = "Japonés",
        quality_config: QualityConfig | None = None,
        processing_config: ProcessingConfig | None = None,
        detector: ComicTextDetector | None = None,
    ) -> None:
        quality_config = quality_config if quality_config is not None else QualityConfig()
        processing_config = processing_config if processing_config is not None else ProcessingConfig()
        self.idioma_entrada = idioma_entrada
        self.quality_config = quality_config
        self.processing_config = processing_config
        # Inyectable: los tests no deberían necesitar 94 MB de pesos para comprobar cómo
        # se construyen las regiones.
        self.detector = detector if detector is not None else ComicTextDetector(
            conf_threshold=float(getattr(quality_config, "comic_text_detector_conf", 0.40)),
            mask_threshold=float(getattr(quality_config, "comic_text_detector_mask_threshold", 0.30)),
        )
        self.reading_order_resolver = ReadingOrderResolver(idioma_entrada)
        self.panel_order_resolver = PanelAwareReadingOrderResolver(
            self.reading_order_resolver,
            PanelOrderConfig.from_quality_config(quality_config),
        )
        self.text_mask_dilate = max(0, int(getattr(quality_config, "fine_text_mask_dilate", 2)))
        self._debug_page_number: Optional[int] = None
        self._debug_source_filename: Optional[str] = None
        self._debug_output_filename: Optional[str] = None

    # -- contexto de depuración (mismo contrato que BubbleDetector) -------------

    def set_debug_page_context(
        self,
        page_index: int,
        *,
        source_filename: str | None = None,
        output_filename: str | None = None,
    ) -> None:
        self._debug_page_number = max(1, int(page_index) + 1)
        self._debug_source_filename = source_filename
        self._debug_output_filename = output_filename

    def clear_debug_page_context(self) -> None:
        self._debug_page_number = None
        self._debug_source_filename = None
        self._debug_output_filename = None

    # -- composición de máscaras -----------------------------------------------
    # Son funciones puras sobre regiones y no dependen del detector: reusar las de
    # `BubbleDetector` evita que las dos vías compongan la página de formas distintas.

    compose_mask = staticmethod(BubbleDetector.compose_mask)
    compose_clean_mask = staticmethod(BubbleDetector.compose_clean_mask)

    # -- detección --------------------------------------------------------------

    def detect_primary_bubble_regions(self, image: np.ndarray) -> List[TextRegion]:
        detection = self.detector.predict(image)
        regions = self._regions_from_detection(image, detection)
        logger.info(
            "comic-text-detector: %s bloques (%s en globo, %s fuera)",
            len(regions),
            sum(1 for r in regions if r.kind == "dialogue"),
            sum(1 for r in regions if r.kind != "dialogue"),
        )
        return regions

    def build_regions_from_bubbles_and_text(
        self,
        image: np.ndarray,
        bubble_regions: Sequence[TextRegion],
        detections: Sequence,
    ) -> List[TextRegion]:
        """Ordena y numera las regiones ya detectadas.

        `detections` son las cajas del OCR global y aquí se ignoran **a propósito**: en
        esta vía el detector es la única fuente de regiones. Se recibe el parámetro
        porque el orquestador llama igual a las dos vías, y se registra cuántas se
        descartan para que el silencio no parezca un olvido.
        """
        if detections:
            logger.debug(
                "comic-text-detector es fuente única: %s cajas del OCR global no crean regiones.",
                len(detections),
            )
        return self._order_and_number(image, list(bubble_regions))

    # -- construcción interna ---------------------------------------------------

    def _regions_from_detection(self, image: np.ndarray, detection: ComicTextDetection) -> List[TextRegion]:
        height, width = image.shape[:2]
        ink = detection.text_mask
        if ink is None or ink.size == 0:
            ink = np.zeros((height, width), dtype=np.uint8)

        regions: List[TextRegion] = []
        for box, score, class_id in zip(detection.boxes, detection.scores, detection.class_ids, strict=True):
            x, y, w, h = box
            if w <= 1 or h <= 1:
                continue
            safe_zone = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(safe_zone, (x, y), (x + w, y + h), 255, -1)
            # La tinta del modelo, acotada a la caja: es lo que se borra y lo que sirve
            # de semilla fina de texto. Sin acotar, una caja heredaría tinta vecina.
            region_ink = cv2.bitwise_and(ink, safe_zone)
            if self.text_mask_dilate > 0 and cv2.countNonZero(region_ink):
                kernel = np.ones((self.text_mask_dilate * 2 + 1,) * 2, dtype=np.uint8)
                region_ink = cv2.dilate(region_ink, kernel, iterations=1)
                region_ink = cv2.bitwise_and(region_ink, safe_zone)

            kind = CTD_CLASS_KINDS.get(int(class_id), "free_text")
            regions.append(
                TextRegion(
                    bbox=(x, y, w, h),
                    text_bbox=(x, y, w, h),
                    mask=safe_zone,
                    kind=kind,
                    confidence=float(score),
                    source_text_hint="",
                    detections_count=1,
                    metadata={
                        "region_source": "comic_text_detector",
                        "ctd_class_id": int(class_id),
                        # Clase estructural del modelo, antes de que el VLM afine el tipo.
                        "structural_kind": "speech_bubble" if kind == "dialogue" else "out_of_bubble",
                        "ctd_score": float(score),
                        "ink_pixels": int(cv2.countNonZero(region_ink)),
                    },
                    clean_mask=region_ink,
                    text_boxes=[(x, y, w, h)],
                    text_mask=region_ink,
                )
            )
        return regions

    def _order_and_number(self, image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
        return order_and_number(
            image,
            regions,
            panel_resolver=self.panel_order_resolver,
            reading_resolver=self.reading_order_resolver,
            idioma_entrada=self.idioma_entrada,
        )
