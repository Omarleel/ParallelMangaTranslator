from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from Applications.Environment import env_bool, env_int

logger = logging.getLogger(__name__)

SPECIAL_SPEAKERS = {"narrator", "unknown", "sfx"}

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


class CharacterMemoryManager:
    """Memoria persistente de personajes/hablantes construida automáticamente con LLM.

    La memoria no intenta reconocer rostros: infiere hablantes, estilos de habla y relaciones desde
    OCR, orden de lectura, tipo de región y contexto acumulado. Si no hay cliente LLM disponible,
    devuelve asignaciones conservadoras sin crear personajes inventados.
    """

    def __init__(self, project_dir: Optional[str] = None, memory_path: Optional[str] = None) -> None:
        self.enabled = env_bool("PMT_CHARACTER_MEMORY", True)
        self.max_context_pages = env_int("PMT_CHARACTER_MEMORY_MAX_CONTEXT_PAGES", 8)
        project = Path(project_dir or os.getenv("PMT_PROJECT_DIR") or "Dataset")
        explicit_path = memory_path or os.getenv("PMT_CHARACTER_MEMORY_PATH", "").strip()
        self.memory_path = Path(explicit_path) if explicit_path else project / "character_memory.json"
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load()

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

    def _fallback_assignments(
        self,
        texts: Sequence[str],
        region_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        assignments = []
        for idx, text in enumerate(texts):
            kind = "dialogue"
            if region_metadata and idx < len(region_metadata) and isinstance(region_metadata[idx], Mapping):
                kind = str(region_metadata[idx].get("kind") or "dialogue")
            if not str(text or "").strip():
                speaker_id = "unknown"
                is_narration = False
                confidence = 0.0
            elif kind == "narration":
                speaker_id = "narrator"
                is_narration = True
                confidence = 0.95
            elif kind in {"sfx", "onomatopoeia", "free_text"}:
                speaker_id = "sfx"
                is_narration = False
                confidence = 0.75
            else:
                speaker_id = "unknown"
                is_narration = False
                confidence = 0.2
            assignments.append({
                "text_id": idx,
                "speaker_id": speaker_id,
                "confidence": confidence,
                "is_narration": is_narration,
                "evidence": "fallback_sin_llm",
            })
        return assignments

    @staticmethod
    def _merge_text_value(old: str, new: str) -> str:
        old = str(old or "").strip()
        new = str(new or "").strip()
        if not old:
            return new
        if not new:
            return old
        if len(new) > len(old) and new.lower() not in old.lower():
            return new
        return old

    def _resolve_candidate_id(self, candidate_id: str, temp_map: Dict[str, str]) -> str:
        candidate_id = str(candidate_id or "").strip()
        if candidate_id in SPECIAL_SPEAKERS:
            return candidate_id
        if candidate_id in (self.data.get("characters") or {}):
            return candidate_id
        if candidate_id.startswith("new_"):
            if candidate_id not in temp_map:
                temp_map[candidate_id] = self._new_character_id()
            return temp_map[candidate_id]
        if re.fullmatch(r"char_\d{3,}", candidate_id):
            # Si el modelo propuso un char_### que no existe, lo aceptamos para no romper la página,
            # pero mantenemos next_character_index por encima de ese número.
            number = int(candidate_id.split("_", 1)[1])
            self.data["next_character_index"] = max(int(self.data.get("next_character_index") or 1), number + 1)
            return candidate_id
        if not candidate_id:
            return "unknown"
        # ID descriptivo inesperado: convertirlo en nuevo personaje real.
        if candidate_id not in temp_map:
            temp_map[candidate_id] = self._new_character_id()
        return temp_map[candidate_id]

    def _merge_inference(self, inference: Mapping[str, Any], page_index: Optional[int], texts: Sequence[str]) -> List[Dict[str, Any]]:
        temp_map: Dict[str, str] = {}
        characters = self.data.setdefault("characters", {})

        for candidate in inference.get("characters", []):
            resolved_id = self._resolve_candidate_id(candidate.get("character_id", ""), temp_map)
            if resolved_id in SPECIAL_SPEAKERS:
                continue
            current = characters.setdefault(resolved_id, {
                "id": resolved_id,
                "display_name": candidate.get("display_name") or f"Personaje {resolved_id.split('_')[-1]}",
                "aliases": [],
                "role": "",
                "speech_style": "",
                "personality_notes": "",
                "first_seen_page": page_index,
                "last_seen_page": page_index,
                "utterance_count": 0,
                "confidence": 0.0,
                "evidence_samples": [],
            })
            current["display_name"] = self._merge_text_value(current.get("display_name"), candidate.get("display_name")) or current.get("display_name")
            aliases = set(str(a).strip() for a in current.get("aliases", []) if str(a).strip())
            aliases.update(str(a).strip() for a in candidate.get("aliases", []) if str(a).strip())
            current["aliases"] = sorted(aliases)
            current["role"] = self._merge_text_value(current.get("role"), candidate.get("role"))
            current["speech_style"] = self._merge_text_value(current.get("speech_style"), candidate.get("speech_style"))
            current["personality_notes"] = self._merge_text_value(current.get("personality_notes"), candidate.get("personality_notes"))
            current["confidence"] = round(max(float(current.get("confidence") or 0), _clamp_confidence(candidate.get("confidence"))), 4)
            if current.get("first_seen_page") is None:
                current["first_seen_page"] = page_index
            current["last_seen_page"] = page_index

        normalized_assignments: List[Dict[str, Any]] = []
        for assignment in inference.get("assignments", []):
            text_id = int(assignment.get("text_id"))
            resolved_id = self._resolve_candidate_id(assignment.get("speaker_id", "unknown"), temp_map)
            confidence = _clamp_confidence(assignment.get("confidence"))
            item = {
                "text_id": text_id,
                "speaker_id": resolved_id,
                "confidence": confidence,
                "is_narration": bool(assignment.get("is_narration")),
                "evidence": str(assignment.get("evidence") or "").strip(),
            }
            normalized_assignments.append(item)
            if resolved_id not in SPECIAL_SPEAKERS:
                current = characters.setdefault(resolved_id, {
                    "id": resolved_id,
                    "display_name": f"Personaje {resolved_id.split('_')[-1]}",
                    "aliases": [],
                    "role": "",
                    "speech_style": "",
                    "personality_notes": "",
                    "first_seen_page": page_index,
                    "last_seen_page": page_index,
                    "utterance_count": 0,
                    "confidence": 0.0,
                    "evidence_samples": [],
                })
                current["utterance_count"] = int(current.get("utterance_count") or 0) + 1
                current["last_seen_page"] = page_index
                current["confidence"] = round(max(float(current.get("confidence") or 0), confidence), 4)
                sample = str(texts[text_id]) if 0 <= text_id < len(texts) else ""
                samples = list(current.get("evidence_samples") or [])
                if sample and sample not in samples:
                    samples.append(sample[:120])
                current["evidence_samples"] = samples[-8:]

        summary = str(inference.get("page_summary") or "").strip()
        if summary:
            summaries = list(self.data.get("page_summaries") or [])
            summaries.append({"page_index": page_index, "summary": summary, "updated_at": _now()})
            self.data["page_summaries"] = summaries[-self.max_context_pages:]

        self.save()
        return normalized_assignments

    def analyze_page(
        self,
        client: Any,
        model: str,
        texts: Sequence[str],
        page_index: Optional[int] = None,
        region_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
        source_language: str = "Japonés",
        target_language: str = "Español",
        seed: int = 7,
        max_retries: int = 2,
    ) -> List[Dict[str, Any]]:
        texts = list(texts)
        if not self.enabled:
            return self._fallback_assignments(texts, region_metadata)
        expected_ids = [i for i, text in enumerate(texts) if str(text or "").strip()]
        if not expected_ids:
            return self._fallback_assignments(texts, region_metadata)
        if client is None:
            logger.info("Memoria de personajes activada, pero no hay cliente LLM; usando asignación conservadora.")
            return self._fallback_assignments(texts, region_metadata)

        system_prompt = self._build_system_prompt(source_language, target_language)
        user_payload = self._build_user_payload(page_index, texts, region_metadata, contexto_previo)

        last_error: Optional[Exception] = None
        formats = [self._schema_response_format(), {"type": "json_object"}]
        for attempt in range(1, max(1, int(max_retries)) + 1):
            for response_format in formats:
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                        ],
                        response_format=response_format,
                        temperature=0.15,
                        seed=seed,
                        max_completion_tokens=max(512, len(expected_ids) * 160),
                    )
                    content = resp.choices[0].message.content
                    data = _parse_json_object(content)
                    validated = validate_character_memory_response(data, expected_ids)
                    llm_assignments = self._merge_inference(validated, page_index, texts)

                    # Completa textos vacíos o ids omitidos con fallback conservador.
                    fallback = {row["text_id"]: row for row in self._fallback_assignments(texts, region_metadata)}
                    merged = {row["text_id"]: row for row in fallback.values()}
                    merged.update({row["text_id"]: row for row in llm_assignments})
                    return [merged[i] for i in range(len(texts))]
                except Exception as exc:
                    last_error = exc
                    # Si json_schema no es soportado por el proveedor, probamos json_object de inmediato.
                    if response_format.get("type") == "json_schema":
                        logger.debug("json_schema no disponible o falló para memoria de personajes: %s", exc)
                        continue
                    logger.warning("Fallo memoria de personajes intento %s/%s: %s", attempt, max_retries, exc)
                    break
        logger.error("Memoria de personajes agotó reintentos. Último error: %s", last_error)
        return self._fallback_assignments(texts, region_metadata)
