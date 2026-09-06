"""Adaptador de entrada por línea de comandos.

Sólo orquesta. Qué se construye y cómo se prepara el proceso vive en
`parallel_manga_translator.bootstrap`, del que también dependen la UI y el banco de
pruebas. Este módulo no debe volver a ser el sitio del que cuelgan los demás.
"""

from __future__ import annotations

from parallel_manga_translator.bootstrap import (
    build_default_config,
    build_image_processor,
    ensure_output_directories,
    prepare_assets,
    prepare_runtime,
)
from parallel_manga_translator.config.app_config import ApplicationConfig
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.processing.parallel_processor import ParallelProcessor

logger = get_logger(__name__)


def run_pipeline(config: ApplicationConfig) -> bool:
    """Run the full manga cleaning, OCR, translation and export pipeline."""
    ensure_output_directories(config)

    if config.translation.metodo_traduccion == "LLM" and not config.translation.groq_api_key:
        logger.warning("No se encontró GROQ_API_KEY en el archivo .env. La traducción LLM podría fallar.")

    image_processor = build_image_processor(config)
    logger.info(
        "Iniciando procesamiento | método=%s | paralelismo=%s | inpaint=%s",
        config.translation.metodo_traduccion,
        config.processing.usar_paralelismo,
        config.translation.modelo_inpaint,
    )

    return ParallelProcessor(config).procesar(
        ruta_carpeta_entrada=config.processing.ruta_carpeta_entrada,
        ruta_carpeta_salida=config.processing.ruta_carpeta_salida,
        process_func=image_processor.procesar,
        batch_size=config.processing.batch_size,
        parallel=config.processing.usar_paralelismo,
    )


def main() -> None:
    """Console entry point."""
    prepare_runtime()
    config = build_default_config()
    prepare_assets()
    run_pipeline(config)
