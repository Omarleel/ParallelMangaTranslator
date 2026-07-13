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

from parallel_manga_translator.infrastructure.execution_control import JobControlError, get_execution_control, cooperative_sleep

logger = logging.getLogger(__name__)


class TraditionalTranslationMixin:
    """Proveedor tradicional, caché y fallback batch/item."""

    def _provider_lang_code(self, ui_lang: str, provider: str) -> str:
        code = self.UI_LANGS[ui_lang]

        if provider == "google":
            if code == "zh":
                return "zh-CN"
            return code

        if provider == "deepl":
            if code == "zh":
                return "ZH"
            if code == "pt":
                return "PT-PT"
            if code == "auto":
                return "auto"
            return code.upper()

        raise ValueError(f"Proveedor no soportado: {provider}")

    def _build_traditional_translator(self):
        if DeeplTranslator is None or GoogleTranslator is None:
            self.provider = "unavailable"
            return None

        configured_provider = str(getattr(self, "traditional_provider", "auto") or "auto").strip().lower()
        if configured_provider in {"google", "deepl"}:
            preferred_order = [configured_provider]
        elif configured_provider in {"auto", "traditional", "tradicional", ""}:
            preferred_order = ["deepl", "google"] if self.deepl_api_key else ["google"]
        else:
            raise ValueError(f"Proveedor tradicional no soportado: {configured_provider}")

        last_error = None
        for provider in preferred_order:
            try:
                source = self._provider_lang_code(self.idioma_entrada, provider)
                target = self._provider_lang_code(self.idioma_salida, provider)

                if provider == "deepl":
                    self.provider = "deepl"
                    return DeeplTranslator(
                        api_key=self.deepl_api_key,
                        source=source,
                        target=target,
                        use_free_api=True,
                    )

                self.provider = "google"
                return GoogleTranslator(source=source, target=target)

            except AuthorizationException as exc:
                last_error = exc
                logger.warning("DEEPL_API_KEY inválida; cambiando a Google.")
                continue
            except Exception as exc:
                last_error = exc
                logger.warning("No se pudo inicializar %s: %s", provider, exc)
                continue

        raise RuntimeError(f"No se pudo inicializar ningún traductor: {last_error}")

    def _same_language(self) -> bool:
        """Evita traducir cuando origen y destino son efectivamente iguales."""
        if self.provider not in {"google", "deepl"}:
            return False

        src = self._provider_lang_code(self.idioma_entrada, self.provider)
        tgt = self._provider_lang_code(self.idioma_salida, self.provider)

        if src.lower() == "auto":
            return False

        return src.split("-")[0].lower() == tgt.split("-")[0].lower()

    @staticmethod
    def _is_blank(texto: Optional[str]) -> bool:
        return texto is None or not str(texto).strip()

    @staticmethod
    def _normalize_translation(texto: str) -> str:
        texto = str(texto or "")
        texto = texto.replace("\u3000", " ")
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    @staticmethod
    def _chunk_payload(indices: Sequence[int], textos: Sequence[str], max_chars: int = 4200, max_items: int = 35):
        chunk_indices = []
        chunk_texts = []
        total_chars = 0
        for idx in indices:
            texto = str(textos[idx])
            if chunk_texts and (len(chunk_texts) >= max_items or total_chars + len(texto) > max_chars):
                yield chunk_indices, chunk_texts
                chunk_indices = []
                chunk_texts = []
                total_chars = 0
            chunk_indices.append(idx)
            chunk_texts.append(texto)
            total_chars += len(texto)
        if chunk_texts:
            yield chunk_indices, chunk_texts

    def _cache_key(self, texto: str) -> Tuple[str, str, str, str]:
        return (self.provider or "unknown", self.idioma_entrada, self.idioma_salida, str(texto or ""))

    def _persistent_key(self, texto: str, method: Optional[str] = None) -> str:
        return self.cache.hash_text(method or self.metodo, self.provider or "unknown", self.idioma_entrada, self.idioma_salida, texto)

    def traducir_texto(self, texto: str) -> str:
        if self._is_blank(texto):
            return texto or ""

        if self._same_language():
            return texto

        cache_key = self._cache_key(texto)
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]
        persistent = self.cache.get(self._persistent_key(texto, "traditional"))
        if persistent is not None:
            persistent = self._normalize_translation(persistent)
            self._translation_cache[cache_key] = persistent
            return persistent

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.translator is None:
                    raise RuntimeError("deep_translator no está instalado; instala requirements.txt o usa LLM con proveedor configurado.")
                control = get_execution_control()
                reservation = None
                if control is not None:
                    reservation = control.reserve_external_call(
                        kind="traditional",
                        provider=str(self.provider or "traditional"),
                        characters=len(str(texto or "")),
                    )
                try:
                    traducido = self.translator.translate(texto)
                    if control is not None and reservation is not None:
                        control.commit_external_call(reservation)
                except Exception:
                    if control is not None and reservation is not None:
                        control.commit_external_call(reservation, failed=True)
                    raise
                salida = self._normalize_translation(traducido) if isinstance(traducido, str) and traducido else texto
                salida = self.glossary.apply_to_translation(texto, salida)
                self._translation_cache[cache_key] = salida
                self.cache.set(self._persistent_key(texto, "traditional"), salida)
                return salida
            except JobControlError:
                raise
            except self.TRADITIONAL_EXCEPTIONS as exc:
                logger.warning("Fallo en traducción tradicional intento %s/%s: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    cooperative_sleep(0.5 * attempt)
        return texto

    def traducir_textos_tradicional(self, textos: Sequence[str]) -> List[str]:
        textos = list(textos)
        if not textos:
            return []

        if self._same_language():
            return textos[:]

        indices = [i for i, t in enumerate(textos) if not self._is_blank(t)]
        if not indices:
            return textos[:]

        salida = textos[:]
        pendientes_por_texto: Dict[str, List[int]] = {}

        for idx in indices:
            texto = textos[idx]
            cache_key = self._cache_key(texto)
            if cache_key in self._translation_cache:
                salida[idx] = self._translation_cache[cache_key]
            else:
                persistent = self.cache.get(self._persistent_key(texto, "traditional"))
                if persistent is not None:
                    persistent = self._normalize_translation(persistent)
                    self._translation_cache[cache_key] = persistent
                    salida[idx] = persistent
                else:
                    pendientes_por_texto.setdefault(texto, []).append(idx)

        textos_unicos = list(pendientes_por_texto.keys())
        if not textos_unicos:
            return salida

        indices_unicos = list(range(len(textos_unicos)))
        for chunk_indices, payload in self._chunk_payload(indices_unicos, textos_unicos):
            try:
                if self.translator is None:
                    raise RuntimeError("deep_translator no está instalado; instala requirements.txt o usa LLM con proveedor configurado.")
                control = get_execution_control()
                reservation = None
                if control is not None:
                    reservation = control.reserve_external_call(
                        kind="traditional",
                        provider=str(self.provider or "traditional"),
                        characters=sum(len(str(item or "")) for item in payload),
                    )
                try:
                    traducidos = self.translator.translate_batch(payload)
                    if control is not None and reservation is not None:
                        control.commit_external_call(reservation)
                except Exception:
                    if control is not None and reservation is not None:
                        control.commit_external_call(reservation, failed=True)
                    raise
                if not isinstance(traducidos, list) or len(traducidos) != len(payload):
                    raise ValueError("translate_batch devolvió un tamaño inesperado.")

                for local_idx, traducido in zip(chunk_indices, traducidos):
                    original = textos_unicos[local_idx]
                    salida_normalizada = self._normalize_translation(traducido) if isinstance(traducido, str) and traducido else original
                    salida_normalizada = self.glossary.apply_to_translation(original, salida_normalizada)
                    self._translation_cache[self._cache_key(original)] = salida_normalizada
                    self.cache.set(self._persistent_key(original, "traditional"), salida_normalizada)
                    for original_idx in pendientes_por_texto[original]:
                        salida[original_idx] = salida_normalizada

            except JobControlError:
                raise
            except self.TRADITIONAL_EXCEPTIONS as exc:
                logger.warning("Fallo batch tradicional, usando fallback item por item: %s", exc)
                for local_idx in chunk_indices:
                    original = textos_unicos[local_idx]
                    traducido = self.traducir_texto(original)
                    for original_idx in pendientes_por_texto[original]:
                        salida[original_idx] = traducido
            except Exception as exc:
                logger.warning("Fallo inesperado en batch tradicional, usando fallback item por item: %s", exc)
                for local_idx in chunk_indices:
                    original = textos_unicos[local_idx]
                    traducido = self.traducir_texto(original)
                    for original_idx in pendientes_por_texto[original]:
                        salida[original_idx] = traducido

        return salida
