from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from parallel_manga_translator.translation.character_memory_schema import _now

logger = logging.getLogger(__name__)
SPECIAL_SPEAKERS = {"narrator", "unknown", "sfx"}


class CharacterMemoryStoreMixin:
    """Persistencia y serialización de memoria de personajes."""

    def _load(self) -> Dict[str, Any]:
        if self.memory_path.exists():
            try:
                data = json.loads(self.memory_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("version", 1)
                    data.setdefault("next_character_index", 1)
                    data.setdefault("characters", {})
                    data.setdefault("page_summaries", [])
                    return data
            except Exception as exc:
                logger.warning("No se pudo cargar memoria de personajes %s: %s", self.memory_path, exc)
        return {
            "version": 1,
            "created_at": _now(),
            "updated_at": _now(),
            "next_character_index": 1,
            "characters": {},
            "page_summaries": [],
        }

    def save(self) -> None:
        self.data["updated_at"] = _now()
        tmp_path = self.memory_path.with_suffix(self.memory_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.memory_path)

    def _new_character_id(self) -> str:
        index = int(self.data.get("next_character_index") or 1)
        self.data["next_character_index"] = index + 1
        return f"char_{index:03d}"

    def snapshot(self, max_characters: int = 24) -> Dict[str, Any]:
        characters = list((self.data.get("characters") or {}).values())
        characters.sort(key=lambda row: (-int(row.get("utterance_count") or 0), str(row.get("id") or "")))
        return {
            "characters": characters[:max_characters],
            "recent_page_summaries": list(self.data.get("page_summaries") or [])[-self.max_context_pages:],
        }

    def as_prompt_text(self, max_characters: int = 16) -> str:
        snap = self.snapshot(max_characters=max_characters)
        lines = []
        for char in snap["characters"]:
            aliases = ", ".join(char.get("aliases") or [])
            style = char.get("speech_style") or ""
            notes = char.get("personality_notes") or ""
            display = char.get("display_name") or char.get("id")
            lines.append(
                f"- {char.get('id')}: {display}; alias={aliases or 'sin alias'}; "
                f"estilo={style or 'desconocido'}; notas={notes or 'sin notas'}"
            )
        if not lines:
            return ""
        return "Memoria automática de personajes:\n" + "\n".join(lines)

    def _compact_existing_characters(self) -> List[Dict[str, Any]]:
        result = []
        for char in list((self.data.get("characters") or {}).values()):
            result.append({
                "id": char.get("id"),
                "display_name": char.get("display_name", ""),
                "aliases": char.get("aliases", []),
                "role": char.get("role", ""),
                "speech_style": char.get("speech_style", ""),
                "personality_notes": char.get("personality_notes", ""),
                "utterance_count": char.get("utterance_count", 0),
            })
        result.sort(key=lambda row: str(row.get("id") or ""))
        return result
