from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Mapping, Sequence


CHARACTER_MEMORY_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "character_id": {"type": "string"},
                    "display_name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "role": {"type": "string"},
                    "speech_style": {"type": "string"},
                    "personality_notes": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "character_id",
                    "display_name",
                    "aliases",
                    "role",
                    "speech_style",
                    "personality_notes",
                    "confidence",
                ],
            },
        },
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text_id": {"type": "integer"},
                    "speaker_id": {"type": "string"},
                    "confidence": {"type": "number"},
                    "is_narration": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["text_id", "speaker_id", "confidence", "is_narration", "evidence"],
            },
        },
        "page_summary": {"type": "string"},
    },
    "required": ["characters", "assignments", "page_summary"],
}


def _now() -> float:
    return round(time.time(), 3)


def _clamp_confidence(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _parse_json_object(content: str) -> Dict[str, Any]:
    content = (content or "").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(content[start:end + 1])
    if not isinstance(data, dict):
        raise TypeError("La respuesta de memoria de personajes debe ser un objeto JSON.")
    return data


def validate_character_memory_response(data: Mapping[str, Any], expected_text_ids: Sequence[int]) -> Dict[str, Any]:
    """Valida la inferencia de memoria sin depender de jsonschema/pydantic."""
    if not isinstance(data, Mapping):
        raise TypeError("La inferencia de personajes debe ser un objeto JSON.")
    allowed = {"characters", "assignments", "page_summary"}
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(f"Campos no permitidos en memoria de personajes: {sorted(unknown)}")

    characters = data.get("characters")
    assignments = data.get("assignments")
    page_summary = data.get("page_summary")
    if not isinstance(characters, list):
        raise TypeError("characters debe ser una lista.")
    if not isinstance(assignments, list):
        raise TypeError("assignments debe ser una lista.")
    if not isinstance(page_summary, str):
        raise TypeError("page_summary debe ser texto.")

    normalized_characters: List[Dict[str, Any]] = []
    for row in characters:
        if not isinstance(row, Mapping):
            raise TypeError("Cada character debe ser un objeto.")
        required = {"character_id", "display_name", "aliases", "role", "speech_style", "personality_notes", "confidence"}
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Character sin campos requeridos: {sorted(missing)}")
        unknown = set(row.keys()) - required
        if unknown:
            raise ValueError(f"Campos no permitidos en character: {sorted(unknown)}")
        aliases = row.get("aliases")
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise TypeError("aliases debe ser lista de texto.")
        normalized_characters.append({
            "character_id": str(row.get("character_id") or "").strip(),
            "display_name": str(row.get("display_name") or "").strip(),
            "aliases": [a.strip() for a in aliases if a.strip()],
            "role": str(row.get("role") or "").strip(),
            "speech_style": str(row.get("speech_style") or "").strip(),
            "personality_notes": str(row.get("personality_notes") or "").strip(),
            "confidence": _clamp_confidence(row.get("confidence")),
        })

    expected = set(int(x) for x in expected_text_ids)
    seen = set()
    normalized_assignments: List[Dict[str, Any]] = []
    for row in assignments:
        if not isinstance(row, Mapping):
            raise TypeError("Cada assignment debe ser un objeto.")
        required = {"text_id", "speaker_id", "confidence", "is_narration", "evidence"}
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Assignment sin campos requeridos: {sorted(missing)}")
        unknown = set(row.keys()) - required
        if unknown:
            raise ValueError(f"Campos no permitidos en assignment: {sorted(unknown)}")
        text_id = row.get("text_id")
        if not isinstance(text_id, int):
            raise TypeError(f"text_id inválido: {text_id!r}")
        if text_id not in expected:
            raise ValueError(f"text_id fuera del payload esperado: {text_id}")
        if text_id in seen:
            raise ValueError(f"text_id duplicado en assignments: {text_id}")
        speaker_id = str(row.get("speaker_id") or "unknown").strip() or "unknown"
        normalized_assignments.append({
            "text_id": text_id,
            "speaker_id": speaker_id,
            "confidence": _clamp_confidence(row.get("confidence")),
            "is_narration": bool(row.get("is_narration")),
            "evidence": str(row.get("evidence") or "").strip(),
        })
        seen.add(text_id)

    # No exigimos assignment para textos vacíos; el llamador controla expected_text_ids.
    return {
        "characters": normalized_characters,
        "assignments": normalized_assignments,
        "page_summary": page_summary.strip(),
    }
