from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

try:
    from deep_translator import DeeplTranslator, GoogleTranslator
    from deep_translator.exceptions import (
        InvalidSourceOrTargetLanguage,
        LanguageNotSupportedException,
        NotValidLength,
        NotValidPayload,
        RequestError,
        ServerException,
        TooManyRequests,
        TranslationNotFound,
    )
except ImportError:  # pragma: no cover
    DeeplTranslator = None  # type: ignore
    GoogleTranslator = None  # type: ignore

    class TranslationNotFound(Exception): pass
    class TooManyRequests(Exception): pass
    class RequestError(Exception): pass
    class ServerException(Exception): pass
    class NotValidPayload(Exception): pass
    class NotValidLength(Exception): pass
    class InvalidSourceOrTargetLanguage(Exception): pass
    class LanguageNotSupportedException(Exception): pass

from parallel_manga_translator.infrastructure.cache_manager import PersistentJsonCache
from parallel_manga_translator.translation.character_memory_manager import CharacterMemoryManager
from parallel_manga_translator.translation.glossary_manager import GlossaryManager
from parallel_manga_translator.translation.providers.base import TranslationProviderConfig
from parallel_manga_translator.translation.traditional_translation_mixin import TraditionalTranslationMixin


class TraditionalTranslationProvider(TraditionalTranslationMixin):
    """Proveedor tradicional basado en DeepL/Google.

    Si quieres agregar otro proveedor clásico, crea otra clase con el mismo contrato y
    regístrala en `TranslatorFactory`, sin modificar el pipeline.
    """

    UI_LANGS = {
        "Auto": "auto",
        "Español": "es",
        "Inglés": "en",
        "Portugués": "pt",
        "Francés": "fr",
        "Italiano": "it",
        "Japonés": "ja",
        "Coreano": "ko",
        "Chino": "zh",
    }

    TRADITIONAL_EXCEPTIONS = (
        TranslationNotFound,
        TooManyRequests,
        RequestError,
        ServerException,
        NotValidPayload,
        NotValidLength,
        InvalidSourceOrTargetLanguage,
        LanguageNotSupportedException,
    )

    def __init__(self, config: TranslationProviderConfig) -> None:
        if config.source_language not in self.UI_LANGS:
            raise ValueError(f"Idioma de entrada no soportado: {config.source_language}")
        if config.target_language not in self.UI_LANGS:
            raise ValueError(f"Idioma de salida no soportado: {config.target_language}")

        self.metodo = "Tradicional"
        self.idioma_entrada = config.source_language
        self.idioma_salida = config.target_language
        self.traditional_provider = (config.traditional_provider or "auto").strip().lower()
        self.traditional_min_interval = max(0.0, float(config.traditional_min_interval))
        self.traditional_block_cooldown = max(0.0, float(config.traditional_block_cooldown))
        self.traditional_block_max_wait = max(0.0, float(config.traditional_block_max_wait))
        self.deepl_api_key = config.deepl_api_key
        self.max_retries = max(1, int(config.max_retries))
        self._translation_cache = {}
        self.cache = PersistentJsonCache("translations")
        self.glossary = GlossaryManager(glossary_path=config.glossary_path, project_dir=config.project_dir)
        self.character_memory = CharacterMemoryManager(
            project_dir=config.project_dir,
            memory_path=config.character_memory_path,
            enabled=config.character_memory_enabled,
            max_context_pages=config.character_memory_max_context_pages,
        )
        self.provider = None
        self.translator = self._build_traditional_translator()
        self.provider_name = self.provider or "traditional"

    def traducir_textos(
        self,
        textos_actuales: Sequence[str],
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
        items_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        character_memory: Optional[Mapping[str, Any]] = None,
    ) -> list[str]:
        return self.traducir_textos_tradicional(textos_actuales)
