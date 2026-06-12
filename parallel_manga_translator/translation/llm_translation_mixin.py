from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from deep_translator import DeeplTranslator, GoogleTranslator
    from deep_translator.exceptions import AuthorizationException
except ImportError:  # pragma: no cover
    DeeplTranslator = None  # type: ignore
    GoogleTranslator = None  # type: ignore
    class AuthorizationException(Exception):
        pass

from parallel_manga_translator.translation.translation_response_schema import LLM_TRANSLATION_RESPONSE_SCHEMA, validate_translation_response

logger = logging.getLogger(__name__)


class LlmTranslationMixin:
    """Prompting, validación JSON estricta y traducción vía LLM."""

    def _build_llm_system_prompt(self, character_memory_text: str = "") -> str:
        lore_str = f"Contexto general de la obra: {self.lore_manga}\n" if self.lore_manga else ""
        glossary_text = self.glossary.as_prompt_text()
        glossary_str = f"Glosario obligatorio:\n{glossary_text}\n" if glossary_text else ""
        memory_str = f"{character_memory_text.strip()}\n" if character_memory_text and character_memory_text.strip() else ""
        return (
            f"Eres un traductor profesional de manga del {self.idioma_entrada} al {self.idioma_salida}.\n"
            f"{lore_str}"
            f"{glossary_str}"
            f"{memory_str}"
            "Objetivo: entregar diálogos naturales, breves y fáciles de insertar en globos de texto.\n"
            "Reglas estrictas:\n"
            "1. Corrige errores evidentes de OCR solo cuando el contexto lo permita; no inventes contenido que no esté sugerido por el texto.\n"
            "2. Mantén nombres propios, tratamientos, apodos y consistencia de una página a otra usando contexto_previo y memoria_personajes.\n"
            "3. Respeta speaker_id, estilo de habla y tono cuando estén disponibles; si el hablante es desconocido, no inventes identidad.\n"
            "4. Adapta modismos, partículas, interjecciones y onomatopeyas de forma natural en el idioma destino.\n"
            "5. Si un elemento parece efecto de sonido u onomatopeya, devuelve un equivalente breve de cómic, no una explicación. Ejemplo: 'ドン' -> '¡BUM!'.\n"
            "6. Si el OCR trae basura visual incomprensible, devuelve '...'. Si el diálogo está entrecortado, conserva esa sensación con puntos suspensivos.\n"
            "7. Evita traducciones innecesariamente largas: prioriza frases compactas que quepan en un globo sin perder sentido.\n"
            "8. Responde única y exclusivamente con JSON válido usando esta estructura exacta y sin campos extra: "
            '{"traducciones": [{"id": 0, "traduccion": "texto traducido"}]}'
        )

    @staticmethod
    def _schema_response_format() -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "manga_translation_batch",
                "schema": LLM_TRANSLATION_RESPONSE_SCHEMA,
                "strict": True,
            },
        }

    def _response_formats_for_llm(self) -> List[Dict[str, Any]]:
        if self.llm_strict_json_schema:
            return [self._schema_response_format(), {"type": "json_object"}]
        return [{"type": "json_object"}]

    def character_memory_snapshot(self) -> Dict[str, Any]:
        return self.character_memory.snapshot()

    def analyze_character_memory(
        self,
        textos: Sequence[str],
        page_index: Optional[int] = None,
        region_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
    ) -> List[Dict[str, Any]]:
        return self.character_memory.analyze_page(
            client=self.client,
            model=self.modelo,
            texts=textos,
            page_index=page_index,
            region_metadata=region_metadata,
            contexto_previo=contexto_previo,
            source_language=self.idioma_entrada,
            target_language=self.idioma_salida,
            seed=self.seed,
            max_retries=min(2, self.max_retries),
        )

    @staticmethod
    def _parse_llm_json(content: str):
        content = (content or "").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Algunos modelos envuelven el JSON con texto extra pese a la instrucción.
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(content[start:end + 1])

    def _enrich_item_for_llm(self, item_id: int, texto: str, metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        item = {"id": item_id, "texto": texto}
        if not metadata:
            return item
        allowed_fields = {
            "source_index",
            "kind",
            "bbox",
            "confidence",
            "speaker_id",
            "speaker_confidence",
            "speech_style",
            "is_narration",
            "max_chars",
        }
        for key in allowed_fields:
            if key in metadata and metadata[key] is not None:
                item[key] = metadata[key]
        return item

    def traducir_textos_llm(
        self,
        textos_actuales: Sequence[str],
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
        items_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        character_memory: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        textos_actuales = list(textos_actuales)

        if not textos_actuales:
            return []

        if self._same_language():
            return textos_actuales[:]

        if self.client is None:
            logger.warning("Groq no está configurado; usando fallback tradicional.")
            return self.traducir_textos_tradicional(textos_actuales)

        salida = textos_actuales[:]
        items = []
        for i, t in enumerate(textos_actuales):
            if self._is_blank(t):
                continue
            persistent = self.cache.get(self._persistent_key(t, "llm"))
            if persistent is not None:
                salida[i] = self._normalize_translation(persistent)
            else:
                metadata = items_metadata[i] if items_metadata and i < len(items_metadata) and isinstance(items_metadata[i], Mapping) else None
                items.append(self._enrich_item_for_llm(i, t, metadata))

        if not items:
            return salida

        contexto = []
        for page in list(contexto_previo or [])[-3:]:
            contexto.append([str(x) for x in page if str(x).strip()])

        memory_payload = dict(character_memory or self.character_memory_snapshot())
        system_prompt = self._build_llm_system_prompt(self.character_memory.as_prompt_text())
        user_payload = {
            "contexto_previo": contexto,
            "memoria_personajes": memory_payload,
            "textos_a_traducir": items,
            "idioma_destino": self.idioma_salida,
            "schema_esperado": {"traducciones": [{"id": "int", "traduccion": "str"}]},
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            for response_format in self._response_formats_for_llm():
                try:
                    resp = self.client.chat.completions.create(
                        model=self.modelo,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                        ],
                        response_format=response_format,
                        temperature=0.25,
                        seed=self.seed,
                        max_completion_tokens=max(384, len(items) * 150),
                    )

                    content = resp.choices[0].message.content
                    if not content:
                        raise ValueError("Respuesta vacía del LLM.")

                    data = self._parse_llm_json(content)
                    expected_ids = [item["id"] for item in items]
                    traducciones = validate_translation_response(data, expected_ids)

                    for row in traducciones:
                        idx = row["id"]
                        traducido = row["traduccion"]
                        normalized = self._normalize_translation(traducido)
                        normalized = self.glossary.apply_to_translation(textos_actuales[idx], normalized)
                        salida[idx] = normalized
                        self.cache.set(self._persistent_key(textos_actuales[idx], "llm"), normalized)

                    return salida

                except Exception as exc:
                    last_error = exc
                    err_str = str(exc).lower()

                    if "429" in err_str and "rate_limit_exceeded" in err_str and "tokens" in err_str:
                        logger.error("Límite de tokens de Groq agotado. Cambiando a traductor tradicional de forma definitiva.")
                        self.metodo = "Tradicional"
                        return self.traducir_textos_tradicional(textos_actuales)

                    if response_format.get("type") == "json_schema":
                        logger.debug("json_schema no disponible o falló en traducción LLM; probando json_object: %s", exc)
                        continue

                    logger.warning("Fallo LLM intento %s/%s: %s", attempt, self.max_retries, exc)
                    break

            if attempt < self.max_retries:
                time.sleep(2 ** (attempt - 1))

        logger.error("LLM agotó reintentos. Último error: %s", last_error)
        return self.traducir_textos_tradicional(textos_actuales)
