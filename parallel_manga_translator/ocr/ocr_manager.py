from __future__ import annotations

import os
from typing import Sequence

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")

import numpy as np

from parallel_manga_translator.config.app_config import OcrConfig
from parallel_manga_translator.infrastructure.cache_manager import PersistentJsonCache
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.language.source_language_filter import SourceLanguageFilter
from parallel_manga_translator.ocr.engines import OcrEngine, OcrFactory

logger = get_logger(__name__)


class OcrManager:
    """Fachada batch de OCR.

    La selección de motores está aislada en `OcrFactory`, por lo que agregar OCRs nuevos
    no requiere cambiar el pipeline de procesamiento ni esta fachada.
    """

    def __init__(
        self,
        idioma_entrada: str,
        config: OcrConfig | None = None,
        engine: OcrEngine | None = None,
        cache_dir: str = "",
        cache: PersistentJsonCache | None = None,
    ) -> None:
        self.idioma_entrada = idioma_entrada
        # Por defecto, los valores del dataclass; nunca el estado global del proceso.
        self.config = config if config is not None else OcrConfig()
        self.engine = engine or OcrFactory.create(idioma_entrada, self.config)
        self.cache = cache if cache is not None else PersistentJsonCache("ocr", base_dir=cache_dir or None)
        self.source_language_filter = SourceLanguageFilter(idioma_entrada)

    @property
    def engine_id(self) -> str:
        return self.engine.engine_id

    def extract_texts(self, imagenes_interes: Sequence[np.ndarray]) -> list[str]:
        resultados: list[str] = []
        for imagen in imagenes_interes:
            key = self.cache.hash_image(imagen, self.idioma_entrada, self.engine_id)
            cached = self.cache.get(key)
            if cached is not None:
                resultados.append(str(cached))
                continue

            texto = self.engine.extract_text(imagen)
            if texto and not self.source_language_filter.should_process_text(texto, allow_empty=False):
                logger.debug(
                    "OCR descartado por idioma de origen: idioma=%s texto=%r",
                    self.idioma_entrada,
                    texto[:40],
                )
                texto = ""
            self.cache.set(key, texto)
            resultados.append(texto)
        return resultados
