"""Elige la fuente de regiones a partir de la configuración.

Existe para que el `if` viva aquí y no dentro del pipeline: `CleanManga` recibe una
fuente ya construida y sólo conoce su contrato (`RegionSourcePort`). Añadir una tercera
fuente es registrar un constructor más, sin tocar el orquestador.
"""

from __future__ import annotations

from typing import Callable, Dict

from parallel_manga_translator.config.app_config import ProcessingConfig, QualityConfig
from parallel_manga_translator.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


def _build_yolo_source(idioma_entrada: str, quality: QualityConfig, processing: ProcessingConfig):
    from parallel_manga_translator.detection.bubble_detector import BubbleDetector

    return BubbleDetector(
        idioma_entrada=idioma_entrada,
        quality_config=quality,
        processing_config=processing,
    )


def _build_comic_text_detector_source(idioma_entrada: str, quality: QualityConfig, processing: ProcessingConfig):
    from parallel_manga_translator.detection.comic_text_detector import (
        ComicTextDetector,
        ComicTextDetectorWeights,
    )
    from parallel_manga_translator.detection.ctd_region_source import CtdRegionSource

    detector = ComicTextDetector(
        weights=ComicTextDetectorWeights(model_path=quality.comic_text_detector_model_path),
        conf_threshold=quality.comic_text_detector_conf,
        mask_threshold=quality.comic_text_detector_mask_threshold,
    )
    return CtdRegionSource(
        idioma_entrada=idioma_entrada,
        quality_config=quality,
        processing_config=processing,
        detector=detector,
    )


#: Los constructores se importan perezosamente: elegir CTD no debe cargar ultralytics,
#: y elegir YOLO no debe tocar el .onnx de 95 MB.
REGION_SOURCES: Dict[str, Callable[..., object]] = {
    "yolo": _build_yolo_source,
    "comic_text_detector": _build_comic_text_detector_source,
}


def create_region_source(
    idioma_entrada: str,
    quality_config: QualityConfig | None = None,
    processing_config: ProcessingConfig | None = None,
):
    """Construye la fuente indicada por `quality.region_source`."""
    quality = quality_config if quality_config is not None else QualityConfig()
    processing = processing_config if processing_config is not None else ProcessingConfig()
    requested = str(getattr(quality, "region_source", "yolo") or "yolo").strip().lower()
    builder = REGION_SOURCES.get(requested)
    if builder is None:
        supported = ", ".join(sorted(REGION_SOURCES))
        raise ValueError(f"quality.region_source no soportado: {requested!r}. Soportados: {supported}")
    logger.info("Fuente de regiones: %s", requested)
    return builder(idioma_entrada, quality, processing)
