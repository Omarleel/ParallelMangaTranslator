from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from parallel_manga_translator.translation.character_memory_schema import CHARACTER_MEMORY_RESPONSE_SCHEMA, _clamp_confidence, _now, _parse_json_object, validate_character_memory_response

logger = logging.getLogger(__name__)
SPECIAL_SPEAKERS = {"narrator", "unknown", "sfx"}


class CharacterMemoryMergeMixin:
    """Fallbacks, resolución de IDs y fusión de inferencias."""

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
