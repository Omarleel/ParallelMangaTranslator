"""De dónde sale la tinta que se borra, además del método derivado.

El método actual deriva la tinta de la imagen (umbral + componentes conectados anclados
al OCR). Un detector de texto entrenado ve otra cosa, y medido sobre el banco ve **sitios
distintos**, no más ni menos: IoU entre lo que el pipeline borró de verdad y la tinta de
`comic-text-detector` es 0.27–0.42 con áreas casi iguales (razón 0.90–0.96).

Por eso esto **une**, nunca sustituye. Las dos máscaras aciertan cosas distintas y quedarse
con una sola perdería lo que la otra ve. Y por eso hay fallback obligatorio: el detector no
encuentra tinta en el 7 %, 31 % y 1 % de las regiones de los tres casos del banco; sin
fallback, esas regiones dejarían de limpiarse.

Lo que esto puede ganar, medido antes de implementarlo: en `ja_01` la región típica
conserva un 18 % de tinta que el detector ve y la limpieza dejó oscura (33 de 59 regiones).
En los dos casos ingleses la mediana es 0.0000, así que ahí lo esperable es quedar igual.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from parallel_manga_translator.config.app_config import QualityConfig
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.processing_models import TextRegion

logger = get_logger(__name__)


class CtdInkMaskSource:
    """Añade a la máscara derivada la tinta que ve `comic-text-detector`.

    El detector corre **una vez por página**, no una por región: son ~1 s de CPU y hay
    entre 4 y 12 regiones por página.
    """

    def __init__(self, detector=None, *, restrict_to_safe_zone: bool = True) -> None:
        if detector is None:
            from parallel_manga_translator.detection.comic_text_detector import ComicTextDetector

            detector = ComicTextDetector()
        self.detector = detector
        self.restrict_to_safe_zone = bool(restrict_to_safe_zone)
        self._page_ink: Optional[np.ndarray] = None

    def prepare(self, image: np.ndarray) -> None:
        """Calcula la tinta de la página. Lo llama la estrategia una vez por página."""
        self._page_ink = None
        if image is None or getattr(image, "size", 0) == 0:
            return
        try:
            self._page_ink = self.detector.predict(image).text_mask
        except Exception as exc:  # pragma: no cover - depende de los pesos del usuario
            # Un fallo del detector no puede dejar la página sin limpiar: se sigue con
            # el método derivado, que es el que ya funcionaba.
            logger.warning("comic-text-detector no disponible para la máscara de tinta (%s).", exc)
            self._page_ink = None

    def release(self) -> None:
        self._page_ink = None

    def augment(self, region: TextRegion, derived: np.ndarray) -> Tuple[np.ndarray, str]:
        """Devuelve (máscara, etiqueta de origen). Si no hay aporte, deja la derivada."""
        page_ink = self._page_ink
        if page_ink is None or page_ink.size == 0 or derived is None:
            return derived, ""
        if page_ink.shape[:2] != derived.shape[:2]:
            return derived, ""

        zona = self._region_zone(region, derived.shape)
        aporte = cv2.bitwise_and((page_ink > 0).astype(np.uint8) * 255, zona)
        if cv2.countNonZero(aporte) == 0:
            # El 31 % de las regiones de en_01 caen aquí: es el caso normal, no un error.
            return derived, "fallback_derivada"

        unido = cv2.bitwise_or((derived > 0).astype(np.uint8) * 255, aporte)
        return unido, "union_ctd"

    def _region_zone(self, region: TextRegion, shape) -> np.ndarray:
        """Dónde se le permite aportar tinta: la zona segura, o su caja si no la hay."""
        height, width = shape[:2]
        if self.restrict_to_safe_zone:
            mask = getattr(region, "mask", None)
            if mask is not None and getattr(mask, "size", 0) and mask.shape[:2] == (height, width):
                return (mask > 0).astype(np.uint8) * 255
        zona = np.zeros((height, width), dtype=np.uint8)
        x, y, w, h = [int(v) for v in region.bbox]
        cv2.rectangle(zona, (x, y), (x + w, y + h), 255, -1)
        return zona


#: Valores admitidos por `quality.ink_mask_source`.
INK_MASK_SOURCES = ("derivada", "derivada+ctd")


def create_ink_mask_source(quality_config: QualityConfig | None = None, detector=None):
    """Factory: devuelve `None` para el método derivado de siempre."""
    quality = quality_config if quality_config is not None else QualityConfig()
    requested = str(getattr(quality, "ink_mask_source", "derivada") or "derivada").strip().lower()
    if requested in {"", "derivada", "derived", "none"}:
        return None
    if requested in {"derivada+ctd", "derived+ctd", "ctd"}:
        logger.info("Máscara de tinta: derivada + comic-text-detector (unión con fallback).")
        return CtdInkMaskSource(detector)
    raise ValueError(
        f"quality.ink_mask_source no soportado: {requested!r}. Soportados: {', '.join(INK_MASK_SOURCES)}"
    )
