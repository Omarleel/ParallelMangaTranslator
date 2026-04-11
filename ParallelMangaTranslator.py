from __future__ import annotations

import os
import warnings

import torch
from dotenv import load_dotenv
from loguru import logger as loguru_logger

from Applications.AppConfig import ApplicationConfig, ProcessingConfig, TranslationConfig
from Applications.ImageProcessor import ImageProcessor
from Applications.LoggingConfig import configure_logging, get_logger
from Applications.ParallelProcessor import ParallelProcessor
from Applications.Utilities import Utilities
from Utils.Constantes import MODELOS_INPAINT

loguru_logger.remove()
configure_logging()
logger = get_logger(__name__)

warnings.filterwarnings("ignore", message="The class ViTFeatureExtractor is deprecated", category=FutureWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def build_default_config() -> ApplicationConfig:
    translation_method = "LLM"
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    usar_paralelismo = translation_method != "LLM"

    translation = TranslationConfig(
        idioma_entrada="Japonés",
        idioma_salida="Español",
        metodo_traduccion=translation_method,
        modelo_inpaint=MODELOS_INPAINT[1],
        lore_manga="",
        groq_api_key=groq_api_key,
    )
    processing = ProcessingConfig(
        ruta_carpeta_entrada="Dataset",
        batch_size=8,
        usar_paralelismo=usar_paralelismo,
    )
    return ApplicationConfig(translation=translation, processing=processing)


def main() -> None:
    load_dotenv()
    config = build_default_config()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    utilities = Utilities()
    utilities.download_fonts()
    utilities.download_models()

    if config.translation.metodo_traduccion == "LLM" and not config.translation.groq_api_key:
        logger.warning("No se encontró GROQ_API_KEY en el archivo .env. La traducción LLM podría fallar.")

    os.makedirs(config.processing.ruta_carpeta_salida, exist_ok=True)
    os.makedirs(config.processing.ruta_carpeta_limpieza, exist_ok=True)
    os.makedirs(config.processing.ruta_carpeta_traduccion, exist_ok=True)

    image_processor = ImageProcessor(
        idioma_entrada=config.translation.idioma_entrada,
        idioma_salida=config.translation.idioma_salida,
        modelo_inpaint=config.translation.modelo_inpaint,
        metodo_traduccion=config.translation.metodo_traduccion,
        groq_api_key=config.translation.groq_api_key,
        lore_manga=config.translation.lore_manga,
    )

    logger.info(
        "Iniciando procesamiento | método=%s | paralelismo=%s | inpaint=%s",
        config.translation.metodo_traduccion,
        config.processing.usar_paralelismo,
        config.translation.modelo_inpaint,
    )

    parallel_processor = ParallelProcessor()
    parallel_processor.procesar(
        ruta_carpeta_entrada=config.processing.ruta_carpeta_entrada,
        ruta_carpeta_salida=config.processing.ruta_carpeta_salida,
        process_func=image_processor.procesar,
        batch_size=config.processing.batch_size,
        parallel=config.processing.usar_paralelismo,
    )


if __name__ == "__main__":
    main()