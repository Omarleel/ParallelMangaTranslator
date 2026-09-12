from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class TranslationProviderConfig:
    source_language: str
    target_language: str
    method: str = "Tradicional"
    traditional_provider: str = "auto"
    # Segundos mínimos entre peticiones al proveedor tradicional y pausa base
    # al detectar que nos está limitando. Ver `traditional_translation_mixin`.
    traditional_min_interval: float = 0.5
    traditional_block_cooldown: float = 6.0
    traditional_block_max_wait: float = 180.0
    llm_provider: str = "groq"
    llm_model: str = "qwen/qwen3.8-27b"
    strict_json_schema: bool = True
    seed: int = 7
    max_retries: int = 5
    retry_max_wait_seconds: float = 90.0
    retry_base_seconds: float = 1.0
    retry_max_backoff_seconds: float = 12.0
    retry_jitter_seconds: float = 0.35
    fallback_to_traditional_on_error: bool = False
    lore: str = ""
    groq_api_key: str = ""
    deepl_api_key: str = ""
    project_dir: Optional[str] = None
    glossary_path: str = ""
    # Caché de traducciones. La UI usa una carpeta por trabajo, así que esto tiene que
    # viajar explícitamente en vez de leerse del estado global del proceso.
    cache_dir: str = ""
    cache_enabled: Optional[bool] = None
    character_memory_enabled: bool = True
    character_memory_path: str = ""
    character_memory_max_context_pages: int = 8


class TranslationProvider(Protocol):
    """Contrato mínimo para motores/proveedores de traducción."""

    provider_name: str

    def traducir_textos(
        self,
        textos_actuales: Sequence[str],
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
        items_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        character_memory: Optional[Mapping[str, Any]] = None,
    ) -> list[str]:
        ...
