from __future__ import annotations

from parallel_manga_translator.config.app_config import OcrConfig
from parallel_manga_translator.ocr.engine_registry import OcrRegistry
from parallel_manga_translator.ocr.engines.base import OcrEngine, OcrEngineSettings
from parallel_manga_translator.ocr.engines.easyocr_engine import EasyOcrEngine
from parallel_manga_translator.ocr.engines.manga_ocr_engine import MangaOcrEngine
from parallel_manga_translator.ocr.engines.paddle_ocr_engine import PaddleOcrEngine
from parallel_manga_translator.ocr.engines.paddle_subprocess_engine import PaddleSubprocessOcrEngine


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
        legacy_engine = str(config.engine or "auto").strip().lower()
        return legacy_engine if transcription_engine == "auto" and legacy_engine != "auto" else transcription_engine

    @classmethod
    def _resolve_engine_name(cls, requested: str, language: str, settings: OcrEngineSettings) -> str:
        engine_name = cls._registry.normalize_name(requested)
        if engine_name == "auto":
            engine_name = "mangaocr_auto" if language == "Japonés" else "easyocr"
        return cls._registry.resolve_paddle_mode(engine_name, settings)

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
