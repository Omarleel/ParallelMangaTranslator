from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence

from dotenv import load_dotenv

from parallel_manga_translator.config.app_config import CharacterMemoryConfig, LlmConfig, TranslationConfig
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.translation.providers import TranslationProviderConfig, TranslatorFactory
from parallel_manga_translator.translation.translation_response_schema import validate_translation_response

load_dotenv()
logger = get_logger(__name__)


class TranslatorManager:
    """Fachada de traducción basada en proveedores intercambiables.

    La lógica específica vive en providers (`translation/providers`). Para agregar OpenAI,
    Gemini, Azure, LibreTranslate, etc., crea un provider y regístralo en `TranslatorFactory`.
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

    def __init__(
        self,
        idioma_entrada: str,
        idioma_salida: str,
        metodo: str = "Tradicional",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        seed: int = 7,
        max_retries: int = 3,
        lore_manga: str = "",
        traditional_provider: str = "auto",
        llm_provider: str = "groq",
        strict_json_schema: bool = True,
        translation_config: TranslationConfig | None = None,
        character_memory_config: CharacterMemoryConfig | None = None,
        cache_dir: str = "",
        cache_enabled: bool | None = None,
    ) -> None:
        if translation_config is not None:
            idioma_entrada = translation_config.idioma_entrada
            idioma_salida = translation_config.idioma_salida
            metodo = translation_config.metodo_traduccion
            groq_api_key = translation_config.groq_api_key
            groq_model = translation_config.llm.model
            seed = translation_config.llm.seed
            max_retries = translation_config.llm.max_retries
            lore_manga = translation_config.lore_manga
            traditional_provider = translation_config.traditional_provider
            traditional_min_interval = translation_config.traditional_min_interval
            traditional_block_cooldown = translation_config.traditional_block_cooldown
            traditional_block_max_wait = translation_config.traditional_block_max_wait
            llm_provider = translation_config.llm.provider
            strict_json_schema = translation_config.llm.strict_json_schema
            project_dir = translation_config.project_dir
            deepl_api_key = translation_config.deepl_api_key
            glossary_path = translation_config.glossary_path
        else:
            project_dir = None
            deepl_api_key = os.getenv("DEEPL_API_KEY", "")
            glossary_path = ""
            traditional_min_interval = TranslationConfig.traditional_min_interval
            traditional_block_cooldown = TranslationConfig.traditional_block_cooldown
            traditional_block_max_wait = TranslationConfig.traditional_block_max_wait

        # Por defecto, los valores del dataclass; nunca el estado global del proceso.
        if character_memory_config is None:
            character_memory_config = CharacterMemoryConfig()

        self.metodo = metodo.strip()
        self.idioma_entrada = idioma_entrada
        self.idioma_salida = idioma_salida
        self.config = TranslationProviderConfig(
            source_language=idioma_entrada,
            target_language=idioma_salida,
            method=self.metodo,
            traditional_provider=traditional_provider,
            traditional_min_interval=float(traditional_min_interval),
            traditional_block_cooldown=float(traditional_block_cooldown),
            traditional_block_max_wait=float(traditional_block_max_wait),
            llm_provider=llm_provider,
            llm_model=groq_model,
            strict_json_schema=bool(strict_json_schema),
            seed=int(seed),
            max_retries=max(1, int(max_retries)),
            lore=lore_manga,
            groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY", ""),
            deepl_api_key=deepl_api_key,
            project_dir=project_dir,
            glossary_path=glossary_path,
            character_memory_enabled=character_memory_config.enabled,
            character_memory_path=character_memory_config.path,
            character_memory_max_context_pages=character_memory_config.max_context_pages,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
        self.provider = TranslatorFactory.create(self.config)

        # Fallback explícito para mantener métodos públicos existentes del pipeline.
        traditional_config = TranslationProviderConfig(
            **{**self.config.__dict__, "method": "Tradicional"}
        )
        self.traditional_provider = TranslatorFactory.create(traditional_config)

    @classmethod
    def from_config(cls, config: TranslationConfig, character_memory_config: CharacterMemoryConfig | None = None) -> "TranslatorManager":
        return cls(config.idioma_entrada, config.idioma_salida, translation_config=config, character_memory_config=character_memory_config)

    def traducir_textos(
        self,
        textos_actuales: Sequence[str],
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
        items_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        character_memory: Optional[Mapping[str, Any]] = None,
    ) -> list[str]:
        return self.provider.traducir_textos(
            textos_actuales,
            contexto_previo=contexto_previo,
            items_metadata=items_metadata,
            character_memory=character_memory,
        )

    def traducir_textos_tradicional(self, textos: Sequence[str]) -> list[str]:
        return self.traditional_provider.traducir_textos(textos)

    def traducir_texto(self, texto: str) -> str:
        translator = getattr(self.traditional_provider, "traducir_texto", None)
        if translator is None:
            return self.traducir_textos_tradicional([texto])[0]
        return translator(texto)

    def traducir_textos_llm(
        self,
        textos_actuales: Sequence[str],
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
        items_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        character_memory: Optional[Mapping[str, Any]] = None,
    ) -> list[str]:
        llm_translator = getattr(self.provider, "traducir_textos_llm", None)
        if llm_translator is None:
            logger.warning("El proveedor activo no implementa LLM; usando proveedor activo genérico.")
            return self.traducir_textos(
                textos_actuales,
                contexto_previo=contexto_previo,
                items_metadata=items_metadata,
                character_memory=character_memory,
            )
        return llm_translator(
            textos_actuales,
            contexto_previo=contexto_previo,
            items_metadata=items_metadata,
            character_memory=character_memory,
        )

    def character_memory_snapshot(self) -> Dict[str, Any]:
        snapshot = getattr(self.provider, "character_memory_snapshot", None)
        if snapshot is None:
            return {"characters": []}
        return snapshot()

    def analyze_character_memory(
        self,
        textos: Sequence[str],
        page_index: Optional[int] = None,
        region_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
    ) -> list[dict[str, Any]]:
        analyzer = getattr(self.provider, "analyze_character_memory", None)
        if analyzer is None:
            return []
        return analyzer(
            textos,
            page_index=page_index,
            region_metadata=region_metadata,
            contexto_previo=contexto_previo,
        )
