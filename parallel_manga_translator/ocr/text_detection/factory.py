from __future__ import annotations

from parallel_manga_translator.config.app_config import OcrConfig
from parallel_manga_translator.ocr.engine_registry import OcrRegistry
from parallel_manga_translator.ocr.text_detection.base import TextDetectionEngine, TextDetectionSettings
from parallel_manga_translator.ocr.text_detection.easyocr_detector import EasyOcrTextDetector
from parallel_manga_translator.ocr.text_detection.paddle_detector import PaddleTextDetector
from parallel_manga_translator.ocr.text_detection.paddle_subprocess_detector import PaddleSubprocessTextDetector


class TextDetectionFactory:
    """Factory para OCR de localización de texto.

    Comparte registry con la transcripción. Mantiene su propio default porque MangaOCR
    no entrega bounding boxes y, por tanto, no puede usarse como detector.
    """

    _registry = OcrRegistry[TextDetectionEngine](
        aliases={
            "manga": "easyocr",
            "mangaocr": "easyocr",
            "manga_ocr": "easyocr",
        }
    )

    @classmethod
    def register(cls, name: str, builder) -> None:
        cls._registry.register(name, builder)

    @classmethod
    def create(cls, language: str, config: OcrConfig) -> TextDetectionEngine:
        settings = TextDetectionSettings.from_config(language, config)
        requested = str(config.detection_engine or "auto").strip().lower()
        engine_name = cls._resolve_engine_name(requested, settings)
        return cls._registry.create(engine_name, settings, requested_label=requested, kind="localización")

    @classmethod
    def _resolve_engine_name(cls, requested: str, settings: TextDetectionSettings) -> str:
        engine_name = cls._registry.normalize_name(requested)
        if engine_name == "auto":
            engine_name = "easyocr"
        return cls._registry.resolve_paddle_mode(engine_name, settings)

    @classmethod
    def supported_engines(cls) -> set[str]:
        return cls._registry.supported_engines()


TextDetectionFactory.register("easyocr", EasyOcrTextDetector)
TextDetectionFactory.register("paddleocr", PaddleTextDetector)
TextDetectionFactory.register("paddle_subprocess", PaddleSubprocessTextDetector)
