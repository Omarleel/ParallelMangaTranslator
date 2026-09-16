from __future__ import annotations

from parallel_manga_translator.config.app_config import OcrConfig
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ocr.engine_registry import OcrRegistry
from parallel_manga_translator.ocr.settings import PADDLE_PREFERRED_LANGUAGES, paddle_disponible
from parallel_manga_translator.ocr.text_detection.base import TextDetectionEngine, TextDetectionSettings
from parallel_manga_translator.ocr.text_detection.easyocr_detector import EasyOcrTextDetector
from parallel_manga_translator.ocr.text_detection.paddle_detector import PaddleTextDetector
from parallel_manga_translator.ocr.text_detection.paddle_subprocess_detector import PaddleSubprocessTextDetector
from parallel_manga_translator.ocr.text_detection.rtdetr_detector import RtDetrTextDetectionEngine

logger = get_logger(__name__)


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
            engine_name = cls._auto_engine_for(settings.language)
        return cls._registry.resolve_paddle_mode(engine_name, settings)

    @classmethod
    def _auto_engine_for(cls, language: str) -> str:
        """Qué localizador elige `auto`.

        Para chino y coreano manda Paddle. No es sólo que EasyOCR transcriba peor: aquí
        localiza peor de una forma que se propaga. Medido sobre una página china real,
        EasyOCR propone 21 cajas y las reglas de texto libre tiran 12 por "ruido OCR"
        —porque esas reglas deciden a partir del texto que el localizador cree leer, y no
        sabe leer chino—, mientras Paddle propone 5 y no se descarta ninguna. Resultado en
        esa página: 0 regiones de texto libre frente a 3.

        Si Paddle no está instalado se vuelve a EasyOCR: es un extra opcional y el valor por
        defecto no puede exigirlo.
        """
        if language in PADDLE_PREFERRED_LANGUAGES:
            if paddle_disponible():
                return "paddleocr"
            logger.warning(
                "%s se localiza mucho mejor con PaddleOCR, que no está instalado; se usa "
                "EasyOCR. Instálalo con: pip install parallel-manga-translator[paddle]",
                language,
            )
        return "easyocr"

    @classmethod
    def supported_engines(cls) -> set[str]:
        return cls._registry.supported_engines()


TextDetectionFactory.register("easyocr", EasyOcrTextDetector)
TextDetectionFactory.register("paddleocr", PaddleTextDetector)
TextDetectionFactory.register("paddle_subprocess", PaddleSubprocessTextDetector)
TextDetectionFactory.register("rtdetr", RtDetrTextDetectionEngine)
