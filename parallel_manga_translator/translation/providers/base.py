from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class TranslationProviderConfig:
    source_language: str
    target_language: str
    method: str = "Tradicional"
    traditional_provider: str = "auto"
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    strict_json_schema: bool = True
    seed: int = 7
    max_retries: int = 3
    lore: str = ""
    groq_api_key: str = ""
    deepl_api_key: str = ""
    project_dir: Optional[str] = None
    glossary_path: str = ""
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
