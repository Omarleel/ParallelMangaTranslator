from __future__ import annotations

from parallel_manga_translator.config.app_config import OcrConfig
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ocr.engine_registry import OcrRegistry
from parallel_manga_translator.ocr.engines.base import OcrEngine, OcrEngineSettings
from parallel_manga_translator.ocr.settings import PADDLE_PREFERRED_LANGUAGES, paddle_disponible
from parallel_manga_translator.ocr.engines.easyocr_engine import EasyOcrEngine
from parallel_manga_translator.ocr.engines.manga_ocr_engine import MangaOcrEngine
from parallel_manga_translator.ocr.engines.paddle_ocr_engine import PaddleOcrEngine
from parallel_manga_translator.ocr.engines.paddle_subprocess_engine import PaddleSubprocessOcrEngine

logger = get_logger(__name__)

class OcrFactory:
    """Factory de motores de transcripción OCR.

    El registro real se delega a `OcrRegistry`, compartido con el factory de detección
    para evitar alias y reglas de Paddle duplicadas.
    """

    _registry = OcrRegistry[OcrEngine]()

    @classmethod
    def register(cls, name: str, builder) -> None:
        cls._registry.register(name, builder)

    @classmethod
    def create(cls, language: str, config: OcrConfig) -> OcrEngine:
        settings = OcrEngineSettings.from_config(language, config)
        requested = cls._requested_engine(config)
        engine_name = cls._resolve_engine_name(requested, language, settings)
        return cls._registry.create(engine_name, settings, requested_label=requested, kind="transcripción")

    @staticmethod
    def _requested_engine(config: OcrConfig) -> str:
        transcription_engine = str(config.transcription_engine or "auto").strip().lower()
        return transcription_engine

    @classmethod
    def _resolve_engine_name(cls, requested: str, language: str, settings: OcrEngineSettings) -> str:
        engine_name = cls._registry.normalize_name(requested)
        if engine_name == "auto":
            engine_name = cls._auto_engine_for(language)
        return cls._registry.resolve_paddle_mode(engine_name, settings)

    @classmethod
    def _auto_engine_for(cls, language: str) -> str:
        """Qué motor elige `auto`, que es lo que usa casi todo el mundo.

        Japonés lo lee MangaOCR, que es lo suyo. Para chino y coreano manda Paddle: son los
        dos casos donde EasyOCR falla de forma que no se ve como "transcripción peor" sino
        como "la página no tiene texto", porque el guardián de OCR borra lo que no se lee.
        El resto se queda en EasyOCR.

        Si Paddle no está instalado —es un extra opcional— se vuelve a EasyOCR en vez de
        reventar: un defecto no puede depender de un extra.
        """
        if language == "Japonés":
            return "mangaocr_auto"
        if language in PADDLE_PREFERRED_LANGUAGES:
            if paddle_disponible():
                return "paddleocr"
            logger.warning(
                "%s se lee mucho mejor con PaddleOCR, que no está instalado; se usa EasyOCR. "
                "Instálalo con: pip install parallel-manga-translator[paddle]",
                language,
            )
        return "easyocr"

    @classmethod
    def supported_engines(cls) -> set[str]:
        return cls._registry.supported_engines()


def _manga_with_easyocr(settings: OcrEngineSettings) -> MangaOcrEngine:
    return MangaOcrEngine(settings, fallback=EasyOcrEngine(settings))


OcrFactory.register("easyocr", EasyOcrEngine)
OcrFactory.register("mangaocr", MangaOcrEngine)
OcrFactory.register("mangaocr_auto", _manga_with_easyocr)
OcrFactory.register("paddleocr", PaddleOcrEngine)
OcrFactory.register("paddle_subprocess", PaddleSubprocessOcrEngine)
