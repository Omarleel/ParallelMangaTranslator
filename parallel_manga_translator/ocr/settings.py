from __future__ import annotations

from dataclasses import dataclass

from parallel_manga_translator.config.app_config import OcrConfig

PADDLE_LANGS = {
    "Inglés": "en",
    "Coreano": "korean",
    "Chino": "ch",
    "Español": "es",
    "Japonés": "japan",
}

EASY_OCR_LANGS = {
    "Japonés": ["ja", "en"],
    "Inglés": ["en"],
    "Coreano": ["ko", "en"],
    "Chino": ["ch_sim", "en"],
    "Español": ["es", "en"],
}

_TRUE_VALUES = {"1", "true", "yes", "on", "always"}
_FALSE_VALUES = {"0", "false", "no", "off", "never"}


@dataclass(frozen=True)
class OcrSettings:
    """Settings compartidos por transcripción OCR y detección de cajas.

    Antes existían dos dataclasses casi iguales (`OcrEngineSettings` y
    `TextDetectionSettings`). Mantener una sola fuente de verdad evita que un ajuste
    nuevo del OCR tenga que duplicarse en dos jerarquías distintas.
    """

    language: str
    gpu: bool = False
    paddle_subprocess: str = "auto"
    fast_mode: bool = False

    @classmethod
    def from_config(cls, language: str, config: OcrConfig) -> "OcrSettings":
        return cls(
            language=language,
            gpu=bool(config.gpu),
            paddle_subprocess=str(config.paddle_subprocess or "auto"),
            fast_mode=bool(config.fast_mode),
        )

    def as_ocr_settings(self) -> "OcrSettings":
        """Compatibilidad con el código de detección anterior."""
        return self

    @property
    def paddle_lang(self) -> str:
        return PADDLE_LANGS.get(self.language, "en")

    @property
    def easyocr_langs(self) -> list[str]:
        return list(EASY_OCR_LANGS.get(self.language, ["en"]))

    def use_paddle_subprocess(self) -> bool:
        value = str(self.paddle_subprocess or "auto").strip().lower()
        if value in _TRUE_VALUES:
            return True
        if value in _FALSE_VALUES:
            return False
        return bool(self.gpu)
