from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore

from parallel_manga_translator.infrastructure.cache_manager import PersistentJsonCache
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.translation.character_memory_manager import CharacterMemoryManager
from parallel_manga_translator.translation.glossary_manager import GlossaryManager
from parallel_manga_translator.translation.llm_translation_mixin import LlmTranslationMixin
from parallel_manga_translator.translation.providers.base import TranslationProviderConfig
from parallel_manga_translator.translation.providers.traditional_provider import TraditionalTranslationProvider
from parallel_manga_translator.translation.traditional_translation_mixin import TraditionalTranslationMixin

logger = get_logger(__name__)


class GroqTranslationProvider(TraditionalTranslationMixin, LlmTranslationMixin):
    """Proveedor LLM Groq con fallback tradicional inyectado."""

    UI_LANGS = TraditionalTranslationProvider.UI_LANGS
    TRADITIONAL_EXCEPTIONS = TraditionalTranslationProvider.TRADITIONAL_EXCEPTIONS

    def __init__(self, config: TranslationProviderConfig) -> None:
        if config.source_language not in self.UI_LANGS:
            raise ValueError(f"Idioma de entrada no soportado: {config.source_language}")
        if config.target_language not in self.UI_LANGS:
            raise ValueError(f"Idioma de salida no soportado: {config.target_language}")

        self.metodo = "LLM"
        self.idioma_entrada = config.source_language
        self.idioma_salida = config.target_language
        self.modelo = config.llm_model
        self.seed = int(config.seed)
        self.max_retries = max(1, int(config.max_retries))
        self.traditional_provider = (config.traditional_provider or "auto").strip().lower()
        self.traditional_min_interval = max(0.0, float(config.traditional_min_interval))
        self.traditional_block_cooldown = max(0.0, float(config.traditional_block_cooldown))
        self.traditional_block_max_wait = max(0.0, float(config.traditional_block_max_wait))
        self.deepl_api_key = config.deepl_api_key
        self.groq_api_key = config.groq_api_key
        self.lore_manga = (config.lore or "").strip()
        self.llm_strict_json_schema = bool(config.strict_json_schema)
        self._translation_cache = {}
        self.cache = PersistentJsonCache("translations", base_dir=config.cache_dir or None, enabled=config.cache_enabled)
        self.glossary = GlossaryManager(glossary_path=config.glossary_path, project_dir=config.project_dir)
        self.character_memory = CharacterMemoryManager(
            project_dir=config.project_dir,
            memory_path=config.character_memory_path,
            enabled=config.character_memory_enabled,
            max_context_pages=config.character_memory_max_context_pages,
        )

        self.provider = None
        self.translator = self._build_traditional_translator()
        self.client = None
        if self.groq_api_key:
            if Groq is None:
                logger.warning("El paquete groq no está instalado; se usará fallback tradicional.")
            else:
                self.client = Groq(api_key=self.groq_api_key)
        self.provider_name = "groq"

    def traducir_textos(
        self,
        textos_actuales: Sequence[str],
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
        items_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        character_memory: Optional[Mapping[str, Any]] = None,
    ) -> list[str]:
        return self.traducir_textos_llm(
            textos_actuales,
            contexto_previo=contexto_previo,
            items_metadata=items_metadata,
            character_memory=character_memory,
        )
