from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from parallel_manga_translator.translation.character_memory_schema import CHARACTER_MEMORY_RESPONSE_SCHEMA, _clamp_confidence, _now, _parse_json_object, validate_character_memory_response

logger = logging.getLogger(__name__)

SPECIAL_SPEAKERS = {"narrator", "unknown", "sfx"}


from parallel_manga_translator.translation.character_memory_store_mixin import CharacterMemoryStoreMixin
from parallel_manga_translator.translation.character_memory_prompt_mixin import CharacterMemoryPromptMixin
from parallel_manga_translator.translation.character_memory_merge_mixin import CharacterMemoryMergeMixin
class CharacterMemoryManager(CharacterMemoryStoreMixin, CharacterMemoryPromptMixin, CharacterMemoryMergeMixin):
    """Memoria persistente de personajes/hablantes construida automáticamente con LLM.

    La memoria no intenta reconocer rostros: infiere hablantes, estilos de habla y relaciones desde
    OCR, orden de lectura, tipo de región y contexto acumulado. Si no hay cliente LLM disponible,
    devuelve asignaciones conservadoras sin crear personajes inventados.
    """

    def __init__(
        self,
        project_dir: Optional[str] = None,
        memory_path: Optional[str] = None,
        enabled: bool = True,
        max_context_pages: int = 8,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_context_pages = int(max_context_pages)
        project = Path(project_dir or "dataset")
        explicit_path = (memory_path or "").strip()
        self.memory_path = Path(explicit_path) if explicit_path else project / "character_memory.json"
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load()


