from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence

from parallel_manga_translator.translation.character_memory_schema import CHARACTER_MEMORY_RESPONSE_SCHEMA

logger = logging.getLogger(__name__)
SPECIAL_SPEAKERS = {"narrator", "unknown", "sfx"}


class CharacterMemoryPromptMixin:
    """Construcción de prompts y payloads para inferencia de hablantes."""

    @staticmethod
    def _region_to_payload(region: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "kind": region.get("kind", "dialogue"),
            "bbox": region.get("bbox"),
            "confidence": region.get("confidence", 0.0),
            "reading_order_index": region.get("reading_order_index"),
        }

    def _build_system_prompt(self, source_language: str, target_language: str) -> str:
        return (
            "Eres un editor de manga que construye una memoria automática de personajes y hablantes.\n"
            f"Analiza textos OCR del {source_language} al {target_language}, el orden de lectura y el tipo de región.\n"
            "No inventes nombres, género ni relaciones si no hay evidencia. Si un personaje no tiene nombre, usa display_name genérico como 'Personaje 1'.\n"
            "Reutiliza IDs existentes cuando el estilo, apodo o continuidad lo sugieran. Para personajes nuevos usa IDs temporales new_1, new_2, etc.\n"
            "Usa speaker_id='narrator' para narración, 'sfx' para onomatopeyas/efectos y 'unknown' cuando no haya pistas suficientes.\n"
            "Responde únicamente con JSON válido según el esquema solicitado."
        )

    def _build_user_payload(
        self,
        page_index: Optional[int],
        texts: Sequence[str],
        region_metadata: Optional[Sequence[Mapping[str, Any]]],
        contexto_previo: Optional[Sequence[Sequence[str]]],
    ) -> Dict[str, Any]:
        items = []
        for idx, text in enumerate(texts):
            if not str(text or "").strip():
                continue
            region = region_metadata[idx] if region_metadata and idx < len(region_metadata) and isinstance(region_metadata[idx], Mapping) else {}
            items.append({
                "text_id": idx,
                "text": str(text),
                "region": self._region_to_payload(region),
            })
        previous_context = []
        for page in list(contexto_previo or [])[-self.max_context_pages:]:
            previous_context.append([str(x) for x in page if str(x).strip()])
        return {
            "page_index": page_index,
            "existing_characters": self._compact_existing_characters(),
            "recent_page_summaries": list(self.data.get("page_summaries") or [])[-self.max_context_pages:],
            "contexto_bilingue_previo": previous_context,
            "items": items,
            "next_new_id_format": "new_1, new_2, ...",
        }

    @staticmethod
    def _schema_response_format() -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "character_memory_page",
                "schema": CHARACTER_MEMORY_RESPONSE_SCHEMA,
                "strict": True,
            },
        }
