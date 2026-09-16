from __future__ import annotations

import importlib.util
from functools import lru_cache
from dataclasses import dataclass

from parallel_manga_translator.config.app_config import OcrConfig

PADDLE_LANGS = {
    "Inglés": "en",
    "Coreano": "korean",
    "Chino": "ch",
    "Español": "es",
    "Japonés": "japan",
}

#: CJK que no es japones. MangaOCR solo lee japones y EasyOCR lee mal estos dos, tanto
#: localizando como transcribiendo. Medido sobre un tomo chino real: al transcribir, de 31
#: regiones EasyOCR leyo 8 y Paddle 20; al localizar, EasyOCR propone 21 cajas de las que
#: las reglas de texto libre tiran 12 por "ruido OCR" -no porque no haya texto, sino porque
#: no sabe leerlo- y Paddle propone 5 sin que se descarte ninguna.
PADDLE_PREFERRED_LANGUAGES = frozenset({"Chino", "Coreano"})


@lru_cache(maxsize=1)
def paddle_disponible() -> bool:
    """Esta instalado PaddleOCR? Es un extra opcional, y el defecto no puede exigirlo."""
    return importlib.util.find_spec("paddleocr") is not None


#: Idiomas cuyo rotulado de comic se escribe convencionalmente en MAYUSCULAS. El OCR los
#: devuelve en minuscula o mezclado, y eso es la mayor parte de su error medido.
LATIN_SCRIPT_LANGUAGES = frozenset({"Inglés", "Español", "Portugués", "Francés", "Italiano"})

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
