from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional, Sequence

try:
    from deep_translator import DeeplTranslator, GoogleTranslator
    from deep_translator.exceptions import AuthorizationException
except ImportError:  # pragma: no cover
    DeeplTranslator = None  # type: ignore
    GoogleTranslator = None  # type: ignore
    class AuthorizationException(Exception):
        pass


logger = logging.getLogger(__name__)


class TranslationDispatchMixin:
    """Punto único de despacho entre traducción tradicional y LLM."""

    def traducir_textos(
        self,
        textos_actuales: Sequence[str],
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
        items_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        character_memory: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        if self.metodo == "LLM":
            return self.traducir_textos_llm(
                textos_actuales,
                contexto_previo=contexto_previo,
                items_metadata=items_metadata,
                character_memory=character_memory,
            )
        return self.traducir_textos_tradicional(textos_actuales)
