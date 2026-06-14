from __future__ import annotations

from typing import Callable, Generic, TypeVar

from parallel_manga_translator.ocr.settings import OcrSettings

T = TypeVar("T")
Builder = Callable[[OcrSettings], T]

ENGINE_ALIASES = {
    "easy": "easyocr",
    "manga": "mangaocr",
    "manga_ocr": "mangaocr",
    "paddle": "paddleocr",
    "paddle_ocr": "paddleocr",
    "paddle-worker": "paddle_subprocess",
    "paddle_worker": "paddle_subprocess",
}


class OcrRegistry(Generic[T]):
    """Registry pequeño reutilizable para motores OCR.

    La detección de cajas y la transcripción siguen teniendo factories separadas porque
    resuelven defaults distintos, pero comparten alias, registro y selección del worker
    de PaddleOCR.
    """

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self._registry: dict[str, Builder[T]] = {}
        self._aliases = dict(ENGINE_ALIASES)
        if aliases:
            self._aliases.update({key.strip().lower(): value for key, value in aliases.items()})

    def register(self, name: str, builder: Builder[T]) -> None:
        self._registry[name.strip().lower()] = builder

    def normalize_name(self, requested: str) -> str:
        normalized = str(requested or "auto").strip().lower()
        if normalized in {"", "none", "null", "nil", "default"}:
            normalized = "auto"
        return self._aliases.get(normalized, normalized)

    def resolve_paddle_mode(self, engine_name: str, settings: OcrSettings) -> str:
        if engine_name == "paddleocr" and settings.use_paddle_subprocess():
            return "paddle_subprocess"
        return engine_name

    def create(self, engine_name: str, settings: OcrSettings, requested_label: str, kind: str) -> T:
        try:
            return self._registry[engine_name](settings)
        except KeyError as exc:
            supported = ", ".join(sorted(self.supported_engines()))
            raise ValueError(f"Motor OCR de {kind} no soportado: {requested_label!r}. Soportados: {supported}") from exc

    def supported_engines(self) -> set[str]:
        return set(self._registry) | set(self._aliases) | {"auto"}
