"""Orden de lectura y numeración de regiones, común a las fuentes que no son YOLO.

Existe por el mismo motivo por el que `CtdRegionSource` reutiliza `compose_mask` de
`BubbleDetector`: si cada fuente ordena y numera por su cuenta, las dos vías acaban
componiendo la página de formas distintas sin que nada lo note. `region_id` se dibuja en
la página anotada y es lo que el VLM devuelve para cada bloque, así que una divergencia
aquí no da error, da bloques cruzados.
"""

from __future__ import annotations

from typing import List

import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.layout.panel_order_resolver import PanelAwareReadingOrderResolver
from parallel_manga_translator.layout.reading_order_resolver import ReadingOrderResolver
from parallel_manga_translator.models.processing_models import TextRegion

logger = get_logger(__name__)


def order_and_number(
    image: np.ndarray,
    regions: List[TextRegion],
    *,
    panel_resolver: PanelAwareReadingOrderResolver,
    reading_resolver: ReadingOrderResolver,
    idioma_entrada: str,
) -> List[TextRegion]:
    """Ordena por viñetas y anota el índice de lectura en cada región."""
    if not regions:
        return []
    try:
        ordered = panel_resolver.sort_regions(image, regions)
    except Exception as exc:  # pragma: no cover - depende de la detección de viñetas
        logger.warning("Orden por viñetas no disponible (%s); se usa el orden de página.", exc)
        ordered = reading_resolver.sort_regions(regions)

    flow = "rtl_vertical" if reading_resolver.page_reads_right_to_left else "ltr_horizontal"
    for index, region in enumerate(ordered):
        region.metadata["reading_order_index"] = index
        region.metadata["reading_order_language"] = idioma_entrada
        region.metadata["reading_order_flow"] = flow
        # Identificador estable y legible por humanos: es el que se dibuja sobre la
        # página anotada y el que el VLM devuelve para cada bloque.
        region.metadata["region_id"] = index + 1
    return ordered
