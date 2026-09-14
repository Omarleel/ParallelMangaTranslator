from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    from deep_translator import DeeplTranslator, GoogleTranslator
    from deep_translator.exceptions import AuthorizationException
except ImportError:  # pragma: no cover
    DeeplTranslator = None  # type: ignore
    GoogleTranslator = None  # type: ignore
    class AuthorizationException(Exception):
        pass

from parallel_manga_translator.translation.translation_response_schema import LLM_TRANSLATION_RESPONSE_SCHEMA, validate_translation_response
from parallel_manga_translator.translation.groq_retry import inspect_groq_error, retry_delay_seconds

from parallel_manga_translator.infrastructure.execution_control import JobControlError, get_execution_control, cooperative_sleep

logger = logging.getLogger(__name__)


from parallel_manga_translator.translation.provider_rules import (
    is_blank,
    normalize_translation,
    persistent_key,
    same_language,
)


class LlmTranslationMixin:
    """Prompting, validación JSON estricta y traducción vía LLM."""

    # Queda anotado cuando una traducción "LLM" acabó resolviéndose con el traductor
    # tradicional. El pipeline lo ignora —el fallback es deliberado y silencioso—, pero
    # quien pidió LLM explícitamente necesita poder enterarse en vez de recibir una
    # traducción de otro motor sin avisar.
    llm_fallback_reason: str = ""

    def _build_llm_system_prompt(self, character_memory_text: str = "") -> str:
        lore_str = f"Contexto general de la obra: {self.lore_manga}\n" if self.lore_manga else ""
        glossary_text = self.glossary.as_prompt_text()
        glossary_str = f"Glosario obligatorio:\n{glossary_text}\n" if glossary_text else ""
        memory_str = f"{character_memory_text.strip()}\n" if character_memory_text and character_memory_text.strip() else ""
        return (
            f"Eres un traductor YOLO de manga del {self.idioma_entrada} al {self.idioma_salida}.\n"
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
            "8. Responde única y exclusivamente con JSON válido, sin campos extra. "
            "Devuelve EXACTAMENTE una traducción por cada elemento recibido en textos_a_traducir; "
            "conserva literalmente su id, no renumeres, no omitas ids y no inventes ids. Estructura: "
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
            return [self._schema_response_format()]

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

    @staticmethod
    def _adaptive_max_completion_tokens(items: Sequence[Mapping[str, Any]]) -> int:
        """Reserva una salida razonable sin penalizar TPM en páginas pequeñas.

        El texto de manga suele ser corto. Un mínimo fijo de 1024 tokens hace que una
        página con uno o dos globos solicite un margen desproporcionado. Este cálculo
        deja espacio para JSON y expansión al español, con un techo conservador.
        """
        source_chars = sum(len(str(item.get("texto") or "")) for item in items)
        estimated = 192 + (96 * len(items)) + source_chars
        return max(384, min(1536, estimated))

    def _same_language(self) -> bool:
        return same_language(self.UI_LANGS, self.provider, self.idioma_entrada, self.idioma_salida)

    def _persistent_key(self, texto: str, method: Optional[str] = None) -> str:
        return persistent_key(
            self.cache, texto, method or self.metodo, self.provider, self.idioma_entrada, self.idioma_salida
        )

    def _finish_llm_failure(self, textos_actuales: Sequence[str], reason: str) -> List[str]:
        """Falla de forma explícita o usa fallback solo si el usuario lo habilitó."""
        reason = str(reason).strip()
        if getattr(self, "llm_fallback_to_traditional_on_error", False):
            self.llm_fallback_reason = f"{reason} Se usó el traductor tradicional por configuración."
            logger.warning("%s", self.llm_fallback_reason)
            return self.traditional.traducir_textos(textos_actuales)

        self.llm_fallback_reason = ""
        raise RuntimeError(reason)

    def _emit_llm_event(
        self,
        kind: str,
        message: str,
        *,
        level: str = "info",
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        callback = getattr(self, "translation_event_callback", None)
        if not callable(callback):
            return
        try:
            callback(kind, message, level=level, details=dict(details or {}))
        except Exception:
            # El historial de UI nunca debe romper una traducción que sí puede continuar.
            logger.debug("No se pudo registrar un evento LLM en la UI.", exc_info=True)

    @staticmethod
    def _rate_limit_detail(info) -> str:
        parts = []
        if info.limit_scope != "unknown":
            parts.append(info.limit_scope.upper())
        if info.remaining_tokens:
            parts.append(f"tokens_restantes={info.remaining_tokens}")
        if info.token_reset:
            parts.append(f"reset_tokens={info.token_reset}")
        if info.remaining_requests:
            parts.append(f"requests_restantes={info.remaining_requests}")
        return ", ".join(parts) or "límite no identificado"

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
            return self._finish_llm_failure(
                textos_actuales,
                "Groq no está configurado (falta GROQ_API_KEY o el paquete groq).",
            )

        # Se limpia al comenzar cada lote. Solo se rellena si de verdad se usa el
        # traductor tradicional como fallback configurado.
        self.llm_fallback_reason = ""
        salida = textos_actuales[:]
        items = []
        for i, t in enumerate(textos_actuales):
            if is_blank(t):
                continue
            persistent = self.cache.get(self._persistent_key(t, "llm"))
            if persistent is not None:
                salida[i] = normalize_translation(persistent)
            else:
                metadata = items_metadata[i] if items_metadata and i < len(items_metadata) and isinstance(items_metadata[i], Mapping) else None
                items.append(self._enrich_item_for_llm(i, t, metadata))

        if not items:
            return salida

        contexto = []
        for page in list(contexto_previo or [])[-3:]:
            contexto.append([str(x) for x in page if str(x).strip()])

        # Si la memoria está desactivada, no cargamos ni inyectamos el JSON persistido.
        # Una memoria pasada explícitamente por el llamador sí se respeta.
        memory_enabled = bool(getattr(self.character_memory, "enabled", False))
        if character_memory is not None:
            memory_payload = dict(character_memory)
            memory_prompt = self.character_memory.as_prompt_text() if memory_payload else ""
        elif memory_enabled:
            memory_payload = dict(self.character_memory_snapshot())
            memory_prompt = self.character_memory.as_prompt_text()
        else:
            memory_payload = {}
            memory_prompt = ""

        system_prompt = self._build_llm_system_prompt(memory_prompt)
        response_format = self._response_formats_for_llm()[0]
        last_error: Optional[BaseException] = None

        # Conservamos cualquier traducción válida recibida. Si el modelo omite algunos
        # ids, los siguientes intentos piden únicamente esos ids en vez de regenerar
        # toda la página (menos OTPM y menos riesgo de volver a perder filas válidas).
        original_items_by_id = {int(item["id"]): item for item in items}
        pending_ids = list(original_items_by_id)
        collected: Dict[int, str] = {}

        for attempt in range(1, self.max_retries + 1):
            try:
                pending_items = [original_items_by_id[idx] for idx in pending_ids]
                user_payload = {
                    "contexto_previo": contexto,
                    "memoria_personajes": memory_payload,
                    "textos_a_traducir": pending_items,
                    "ids_obligatorios": pending_ids,
                    "instruccion_salida": (
                        "Devuelve exactamente una entrada por cada id de ids_obligatorios. "
                        "Conserva esos ids sin renumerarlos, no omitas ninguno y no agregues otros."
                    ),
                    "idioma_destino": self.idioma_salida,
                }
                user_content = json.dumps(user_payload, ensure_ascii=False)
                max_completion_tokens = self._adaptive_max_completion_tokens(pending_items)
                control = get_execution_control()
                reservation = None
                usage_committed = False
                if control is not None:
                    reservation = control.reserve_external_call(
                        kind="llm",
                        provider=getattr(self, "provider_name", "groq"),
                        # Esto es solo telemetría/checkpoint; no intenta contar tokens
                        # exactamente a partir de bytes.
                        estimated_input_tokens=max(1, len(system_prompt + user_content) // 3),
                        estimated_output_tokens=max_completion_tokens,
                    )
                try:
                    request_kwargs = {
                        "model": self.modelo,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "response_format": response_format,
                        "temperature": 0.20,
                        "seed": self.seed,
                        "max_completion_tokens": max_completion_tokens,
                    }
                    if self.modelo in {"qwen/qwen3.6-27b", "qwen/qwen3.8-27b"}:
                        request_kwargs["reasoning_effort"] = "none"

                    resp = self.client.chat.completions.create(**request_kwargs)
                    if control is not None and reservation is not None:
                        usage = getattr(resp, "usage", None)
                        control.commit_external_call(
                            reservation,
                            actual_input_tokens=getattr(usage, "prompt_tokens", None),
                            actual_output_tokens=getattr(usage, "completion_tokens", None),
                        )
                        usage_committed = True
                except Exception:
                    if control is not None and reservation is not None and not usage_committed:
                        control.commit_external_call(reservation, failed=True)
                    raise

                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("Respuesta vacía del LLM.")

                data = self._parse_llm_json(content)
                expected_ids = list(pending_ids)
                traducciones = validate_translation_response(data, expected_ids, allow_missing=True)

                returned_ids = set()
                for row in traducciones:
                    idx = int(row["id"])
                    collected[idx] = str(row["traduccion"])
                    returned_ids.add(idx)

                usage = getattr(resp, "usage", None)
                if usage is not None:
                    logger.info(
                        "Groq respuesta intento %s/%s: prompt=%s completion=%s total=%s max_completion=%s recibidos=%s/%s",
                        attempt,
                        self.max_retries,
                        getattr(usage, "prompt_tokens", "?"),
                        getattr(usage, "completion_tokens", "?"),
                        getattr(usage, "total_tokens", "?"),
                        max_completion_tokens,
                        len(returned_ids),
                        len(expected_ids),
                    )
                    self._emit_llm_event(
                        "llm_response",
                        f"Groq respondió en el intento {attempt}/{self.max_retries}: {len(returned_ids)}/{len(expected_ids)} traducciones.",
                        details={
                            "attempt": attempt,
                            "max_retries": self.max_retries,
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                            "max_completion_tokens": max_completion_tokens,
                            "received": len(returned_ids),
                            "expected": len(expected_ids),
                        },
                    )

                missing_ids = [idx for idx in expected_ids if idx not in returned_ids]
                if missing_ids:
                    pending_ids = missing_ids
                    logger.warning(
                        "Respuesta parcial de Groq: se conservaron %s traducciones y faltan ids %s. "
                        "El siguiente intento pedirá solo los faltantes.",
                        len(returned_ids),
                        missing_ids,
                    )
                    self._emit_llm_event(
                        "partial_response",
                        f"Respuesta parcial: faltan ids {missing_ids}; el siguiente intento pedirá solo esos textos.",
                        level="warning",
                        details={"missing_ids": missing_ids, "attempt": attempt},
                    )
                    raise ValueError(f"Faltan traducciones para ids: {missing_ids}")

                # Todos los ids pendientes llegaron. Aplicamos el lote completo de forma
                # atómica para no modificar/cachar media página si finalmente se agotan
                # los reintentos.
                if set(collected) == set(original_items_by_id):
                    for idx, traducido in collected.items():
                        normalized = normalize_translation(traducido)
                        normalized = self.glossary.apply_to_translation(textos_actuales[idx], normalized)
                        salida[idx] = normalized
                    for idx in collected:
                        self.cache.set(self._persistent_key(textos_actuales[idx], "llm"), salida[idx])
                    logger.info(
                        "Groq lote completo: %s/%s traducciones resueltas en %s intento(s).",
                        len(collected),
                        len(original_items_by_id),
                        attempt,
                    )
                    self._emit_llm_event(
                        "batch_completed",
                        f"Lote LLM completo: {len(collected)}/{len(original_items_by_id)} textos en {attempt} intento(s).",
                        details={"attempts": attempt, "items": len(collected)},
                    )
                    return salida

                # Caso defensivo: debería ser imposible porque pending_ids representa
                # exactamente lo que falta, pero evita devolver una página incompleta.
                unresolved = sorted(set(original_items_by_id) - set(collected))
                pending_ids = unresolved
                raise ValueError(f"Faltan traducciones para ids: {unresolved}")

            except JobControlError:
                raise
            except Exception as exc:
                last_error = exc
                info = inspect_groq_error(exc)
                local_output_error = isinstance(exc, (json.JSONDecodeError, ValueError)) and info.status_code is None

                if info.status_code == 429 and info.daily_limit:
                    reason = (
                        f"Groq alcanzó un límite diario ({info.limit_scope.upper() if info.limit_scope != 'unknown' else '429'}). "
                        "No conviene reintentar automáticamente hasta que se renueve la cuota. "
                        f"Detalle: {info.message}"
                    )
                    logger.error("%s", reason)
                    self._emit_llm_event(
                        "daily_limit", reason, level="error",
                        details={"scope": info.limit_scope, "status": info.status_code},
                    )
                    return self._finish_llm_failure(textos_actuales, reason)

                retryable = info.retryable or local_output_error
                if not retryable:
                    reason = (
                        f"Groq rechazó la petición y no es un error temporal reintentable. "
                        f"status={info.status_code or 'n/a'} code={info.error_code or 'n/a'}: {info.message}"
                    )
                    logger.error("%s", reason)
                    self._emit_llm_event(
                        "request_rejected", reason, level="error",
                        details={"status": info.status_code, "code": info.error_code},
                    )
                    return self._finish_llm_failure(textos_actuales, reason)

                if attempt >= self.max_retries:
                    break

                delay = retry_delay_seconds(
                    info,
                    attempt=attempt,
                    base_seconds=getattr(self, "llm_retry_base_seconds", 1.0),
                    max_backoff_seconds=getattr(self, "llm_retry_max_backoff_seconds", 12.0),
                    jitter_seconds=getattr(self, "llm_retry_jitter_seconds", 0.35),
                )
                max_wait = max(0.0, float(getattr(self, "llm_retry_max_wait_seconds", 90.0)))
                if delay > max_wait:
                    reason = (
                        f"Groq pidió esperar {delay:.1f}s antes de reintentar, por encima del máximo configurado "
                        f"de {max_wait:.1f}s. No se modificó la página. Detalle: {info.message}"
                    )
                    logger.error("%s", reason)
                    self._emit_llm_event(
                        "retry_wait_too_long", reason, level="error",
                        details={"delay_seconds": delay, "max_wait_seconds": max_wait, "scope": info.limit_scope},
                    )
                    return self._finish_llm_failure(textos_actuales, reason)

                if info.status_code == 429:
                    logger.warning(
                        "Rate limit Groq (%s), intento %s/%s. Esperando %.2fs según el servidor. %s",
                        info.limit_scope.upper() if info.limit_scope != "unknown" else "429",
                        attempt,
                        self.max_retries,
                        delay,
                        self._rate_limit_detail(info),
                    )
                    self._emit_llm_event(
                        "rate_limit",
                        f"Rate limit {info.limit_scope.upper() if info.limit_scope != 'unknown' else '429'}; reintento {attempt + 1}/{self.max_retries} en {delay:.2f}s.",
                        level="warning",
                        details={
                            "attempt": attempt,
                            "delay_seconds": delay,
                            "scope": info.limit_scope,
                            "remaining_tokens": info.remaining_tokens,
                            "token_reset": info.token_reset,
                            "remaining_requests": info.remaining_requests,
                        },
                    )
                else:
                    logger.warning(
                        "Fallo temporal Groq status=%s code=%s, intento %s/%s: %s. Reintentando en %.2fs.",
                        info.status_code or "n/a",
                        info.error_code or "n/a",
                        attempt,
                        self.max_retries,
                        info.message,
                        delay,
                    )
                    self._emit_llm_event(
                        "retry",
                        f"Reintento {attempt + 1}/{self.max_retries} en {delay:.2f}s: {info.message}",
                        level="warning",
                        details={"attempt": attempt, "delay_seconds": delay, "status": info.status_code, "code": info.error_code},
                    )
                cooperative_sleep(delay)

        reason = f"Groq agotó {self.max_retries} intentos. Último error: {last_error}"
        logger.error("%s", reason)
        self._emit_llm_event(
            "retries_exhausted", reason, level="error", details={"max_retries": self.max_retries}
        )
        return self._finish_llm_failure(textos_actuales, reason)
