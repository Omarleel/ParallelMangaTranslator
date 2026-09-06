"""Composition root del proceso: qué se construye y con qué motores concretos.

Por qué existe
--------------
Esto vivía en `cli.py`, así que la UI tenía que importar el CLI para arrancar un trabajo:
`ui/job_manager.py` importaba `build_image_processor`, `build_default_config`,
`prepare_assets` y `prepare_runtime`. Dos adaptadores de entrada al mismo dominio, y uno
colgando del otro; cualquier cosa que se tocara en el CLI podía romper la web.

Y no era sólo conceptual. `cli.py` configuraba el proceso **al importarse** —quitaba el
sink de loguru, llamaba a `configure_logging()`, filtraba un warning y fijaba
`PYTORCH_CUDA_ALLOC_CONF`—, de modo que la UI heredaba todo eso de rebote, por el mero
hecho de que `ui/app.py` construye un `JobManager` en tiempo de import. Un cambio inocente
en los imports del CLI habría dejado la UI sin configurar, y nada habría fallado.

Ahora CLI, UI y el banco de pruebas dependen de este módulo, y ninguno del otro.

Qué NO va aquí
--------------
Nada de leer argumentos, servir HTTP ni recorrer carpetas. Este módulo construye objetos y
prepara el proceso; quién los usa y cuándo es cosa de cada adaptador (`cli.py`,
`ui/app.py`, `quality/eval_runner.py`).
"""

from __future__ import annotations

import os
import warnings

import torch
from dotenv import load_dotenv
from loguru import logger as loguru_logger

from parallel_manga_translator.config.app_config import ApplicationConfig
from parallel_manga_translator.config.config_manager import ConfigManager
from parallel_manga_translator.infrastructure.logging_config import configure_logging, get_logger
from parallel_manga_translator.io.utilities import Utilities
from parallel_manga_translator.processing.clean_manga import CleanManga
from parallel_manga_translator.processing.image_processor import ImageProcessor
from parallel_manga_translator.processing.translate_manga import TranslateManga

_proceso_configurado = False


def configure_process() -> None:
    """Ajustes de proceso que deben ocurrir una vez, antes de tocar modelos.

    Es idempotente y se invoca al importar este módulo. Esa invocación al importar no es
    elegante, pero es exactamente lo que hacía `cli.py`, y de ella dependen dos cosas
    sensibles al momento: el sink de loguru (si no se quita antes de configurar, la salida
    se duplica) y `PYTORCH_CUDA_ALLOC_CONF`, que el asignador de CUDA lee la primera vez
    que se usa la GPU. Convertirlo en una llamada explícita de cada entrypoint habría
    cambiado ese orden sin forma de validarlo.
    """
    global _proceso_configurado
    if _proceso_configurado:
        return
    loguru_logger.remove()
    configure_logging()
    warnings.filterwarnings(
        "ignore", message="The class ViTFeatureExtractor is deprecated", category=FutureWarning
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _proceso_configurado = True


configure_process()

logger = get_logger(__name__)


def build_default_config(config_path: str = "config.yaml") -> ApplicationConfig:
    """Build the application configuration from YAML."""
    config = ConfigManager(config_path).build_application_config()
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
    """El único sitio que sabe qué motores concretos se usan.

    `ImageProcessor` recibe las etapas ya construidas y sólo conoce sus contratos
    (`PageCleanerPort`, `PageTranslatorPort`). Para sustituir una etapa —o inyectar una
    falsa en un test— se cambia aquí, no dentro del orquestador.
    """
    cleaner = CleanManga(
        config.translation.modelo_inpaint,
        idioma_entrada=config.translation.idioma_entrada,
        quality_config=config.quality,
        onomatopoeia_config=config.onomatopoeia,
        processing_config=config.processing,
        ocr_config=config.ocr,
    )
    translator = TranslateManga(
        config.translation.idioma_entrada,
        config.translation.idioma_salida,
        metodo_traduccion=config.translation.metodo_traduccion,
        groq_api_key=config.translation.groq_api_key,
        lore_manga=config.translation.lore_manga,
        ocr_config=config.ocr,
        translation_config=config.translation,
        quality_config=config.quality,
        onomatopoeia_config=config.onomatopoeia,
        character_memory_config=config.character_memory,
        processing_config=config.processing,
    )
    return ImageProcessor(cleaner, translator)


def ensure_output_directories(config: ApplicationConfig) -> None:
    """Create output folders used by the pipeline."""
    os.makedirs(config.processing.ruta_carpeta_salida, exist_ok=True)
    os.makedirs(config.processing.ruta_carpeta_limpieza, exist_ok=True)
    os.makedirs(config.processing.ruta_carpeta_traduccion, exist_ok=True)


__all__ = [
    "configure_process",
    "build_default_config",
    "prepare_runtime",
    "prepare_assets",
    "build_image_processor",
    "ensure_output_directories",
]
