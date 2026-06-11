import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dotenv import load_dotenv
try:
    from groq import Groq
except ImportError:  # pragma: no cover - entorno de tests sin proveedor LLM
    Groq = None  # type: ignore

try:
    from deep_translator import DeeplTranslator, GoogleTranslator
    from deep_translator.exceptions import (
        AuthorizationException,
        InvalidSourceOrTargetLanguage,
        LanguageNotSupportedException,
        NotValidLength,
        NotValidPayload,
        RequestError,
        ServerException,
        TooManyRequests,
        TranslationNotFound,
    )
except ImportError:  # pragma: no cover - permite importar utilidades en entornos mínimos de test
    DeeplTranslator = None  # type: ignore
    GoogleTranslator = None  # type: ignore

    class TranslationNotFound(Exception): pass
    class TooManyRequests(Exception): pass
    class RequestError(Exception): pass
    class ServerException(Exception): pass
    class NotValidPayload(Exception): pass
    class NotValidLength(Exception): pass
    class InvalidSourceOrTargetLanguage(Exception): pass
    class LanguageNotSupportedException(Exception): pass
    class AuthorizationException(Exception): pass

from Applications.CacheManager import PersistentJsonCache
from Applications.GlossaryManager import GlossaryManager
from Applications.CharacterMemoryManager import CharacterMemoryManager

load_dotenv()
logger = logging.getLogger(__name__)


LLM_TRANSLATION_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "traducciones": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer"},
                    "traduccion": {"type": "string"},
                },
                "required": ["id", "traduccion"],
            },
        }
    },
    "required": ["traducciones"],
}


def validate_translation_response(data: Mapping[str, Any], expected_ids: Sequence[int]) -> List[Dict[str, Any]]:
    """Valida la salida del LLM de forma estricta sin dependencia externa.

    El proveedor puede garantizar JSON, pero esta validación evita respuestas con ids
    duplicados, campos extra, listas incompletas o tipos incorrectos antes de renderizar.
    """
    if not isinstance(data, Mapping):
        raise TypeError("La respuesta de traducción debe ser un objeto JSON.")
    allowed_root = {"traducciones"}
    extra_root = set(data.keys()) - allowed_root
    if extra_root:
        raise ValueError(f"Campos raíz no permitidos: {sorted(extra_root)}")
    traducciones = data.get("traducciones")
    if not isinstance(traducciones, list):
        raise TypeError("'traducciones' debe ser una lista.")

    expected = set(int(x) for x in expected_ids)
    seen = set()
    validated: List[Dict[str, Any]] = []
    for row in traducciones:
        if not isinstance(row, Mapping):
            raise TypeError("Cada traducción debe ser un objeto.")
        allowed_row = {"id", "traduccion"}
        extra_row = set(row.keys()) - allowed_row
        if extra_row:
            raise ValueError(f"Campos no permitidos en traducción: {sorted(extra_row)}")
        if "id" not in row or "traduccion" not in row:
            raise ValueError("Cada traducción requiere 'id' y 'traduccion'.")
        idx = row["id"]
        if not isinstance(idx, int):
            raise TypeError(f"id inválido: {idx!r}")
        if idx not in expected:
            raise ValueError(f"id fuera de rango o inesperado: {idx}")
        if idx in seen:
            raise ValueError(f"id duplicado: {idx}")
        traducido = row["traduccion"]
        if not isinstance(traducido, str):
            raise TypeError(f"traduccion inválida para id={idx}")
        validated.append({"id": idx, "traduccion": traducido})
        seen.add(idx)

    if seen != expected:
        raise ValueError(f"Faltan traducciones para ids: {sorted(expected - seen)}")
    return validated


class TranslatorManager:
    """
    - Tradicional: usa DeepL si hay DEEPL_API_KEY válida; si no, Google.
    - LLM: usa Groq con salida JSON estricta y fallback al modo tradicional.
    """

    UI_LANGS = {
        "Auto": "auto",
        "Español": "es",
        "Inglés": "en",
        "Portugués": "pt",
        "Francés": "fr",
        "Italiano": "it",
        "Japonés": "ja",
        "Coreano": "ko",
        "Chino": "zh",
    }

    TRADITIONAL_EXCEPTIONS = (
        TranslationNotFound,
        TooManyRequests,
        RequestError,
        ServerException,
        NotValidPayload,
        NotValidLength,
        InvalidSourceOrTargetLanguage,
        LanguageNotSupportedException,
    )

    def __init__(
        self,
        idioma_entrada: str,
        idioma_salida: str,
        metodo: str = "Tradicional",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        seed: int = 7,
        max_retries: int = 3,
        lore_manga: str = "",
    ):
        if idioma_entrada not in self.UI_LANGS:
            raise ValueError(f"Idioma de entrada no soportado: {idioma_entrada}")
        if idioma_salida not in self.UI_LANGS:
            raise ValueError(f"Idioma de salida no soportado: {idioma_salida}")

        self.metodo = metodo.strip()
        self.idioma_entrada = idioma_entrada
        self.idioma_salida = idioma_salida
        self.modelo = groq_model
        self.seed = int(seed)
        self.max_retries = max(1, int(max_retries))

        self.deepl_api_key = os.getenv("DEEPL_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.lore_manga = (lore_manga or "").strip()
        self._translation_cache: Dict[Tuple[str, str, str, str], str] = {}
        self.cache = PersistentJsonCache("translations")
        self.glossary = GlossaryManager(project_dir=os.getenv("PMT_PROJECT_DIR"))
        self.character_memory = CharacterMemoryManager(project_dir=os.getenv("PMT_PROJECT_DIR"))
        self.llm_strict_json_schema = os.getenv("PMT_LLM_STRICT_JSON_SCHEMA", "1").strip().lower() not in {"0", "false", "no", "off"}

        self.provider = None
        self.translator = self._build_traditional_translator()

        self.client = None
        if self.metodo == "LLM" and self.groq_api_key:
            if Groq is None:
                logger.warning("El paquete groq no está instalado; se usará fallback tradicional si se solicita LLM.")
            else:
                self.client = Groq(api_key=self.groq_api_key)

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

        preferred_order = ["deepl", "google"] if self.deepl_api_key else ["google"]

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
                traducido = self.translator.translate(texto)
                salida = self._normalize_translation(traducido) if isinstance(traducido, str) and traducido else texto
                salida = self.glossary.apply_to_translation(texto, salida)
                self._translation_cache[cache_key] = salida
                self.cache.set(self._persistent_key(texto, "traditional"), salida)
                return salida
            except self.TRADITIONAL_EXCEPTIONS as exc:
                logger.warning("Fallo en traducción tradicional intento %s/%s: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)
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
                traducidos = self.translator.translate_batch(payload)
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
