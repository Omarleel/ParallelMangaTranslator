from __future__ import annotations

import os
import warnings

import torch
from dotenv import load_dotenv
from loguru import logger as loguru_logger

from parallel_manga_translator.config.app_config import ApplicationConfig
from parallel_manga_translator.config.config_manager import ConfigManager
from parallel_manga_translator.config.runtime_config import set_active_config
from parallel_manga_translator.infrastructure.logging_config import configure_logging, get_logger
from parallel_manga_translator.io.utilities import Utilities
from parallel_manga_translator.processing.image_processor import ImageProcessor
from parallel_manga_translator.processing.parallel_processor import ParallelProcessor

loguru_logger.remove()
configure_logging()
logger = get_logger(__name__)

warnings.filterwarnings("ignore", message="The class ViTFeatureExtractor is deprecated", category=FutureWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def build_default_config(config_path: str = "config.yaml") -> ApplicationConfig:
    """Build the application configuration from YAML."""
    config = ConfigManager(config_path).build_application_config()
    set_active_config(config)
    configure_logging(log_file=config.logging.file, level=config.logging.level)
    return config


def prepare_runtime() -> None:
    """Load secret variables and prepare GPU memory state."""
    load_dotenv()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_assets(utilities: Utilities | None = None) -> None:
    """Ensure required fonts and model files are available before processing."""
    utilities = utilities or Utilities()
    utilities.download_fonts()
    utilities.download_models()


def build_image_processor(config: ApplicationConfig) -> ImageProcessor:
    """Create the processing pipeline for a configured translation run."""
    return ImageProcessor(
        idioma_entrada=config.translation.idioma_entrada,
        idioma_salida=config.translation.idioma_salida,
        modelo_inpaint=config.translation.modelo_inpaint,
        metodo_traduccion=config.translation.metodo_traduccion,
        groq_api_key=config.translation.groq_api_key,
        lore_manga=config.translation.lore_manga,
        ocr_config=config.ocr,
        translation_config=config.translation,
        processing_config=config.processing,
        quality_config=config.quality,
        onomatopoeia_config=config.onomatopoeia,
        character_memory_config=config.character_memory,
    )


def ensure_output_directories(config: ApplicationConfig) -> None:
    """Create output folders used by the pipeline."""
    os.makedirs(config.processing.ruta_carpeta_salida, exist_ok=True)
    os.makedirs(config.processing.ruta_carpeta_limpieza, exist_ok=True)
    os.makedirs(config.processing.ruta_carpeta_traduccion, exist_ok=True)


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

    return ParallelProcessor().procesar(
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
