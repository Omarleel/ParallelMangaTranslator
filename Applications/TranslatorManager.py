import json
import logging
import os
import time
from typing import List, Sequence, Optional

from dotenv import load_dotenv
from groq import Groq

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

load_dotenv()
logger = logging.getLogger(__name__)


class TranslatorManager:
    """
    - Tradicional: usa DeepL si hay DEEPL_API_KEY válida; si no, Google.
    - LLM: usa Groq con schema estricto y fallback al tradicional.
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
        lore_manga: str = ""
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
        self.lore_manga = lore_manga

        self.provider = None
        self.translator = self._build_traditional_translator()

        self.client = None
        if self.metodo == "LLM" and self.groq_api_key:
            self.client = Groq(api_key=self.groq_api_key)


    def _provider_lang_code(self, ui_lang: str, provider: str) -> str:
        code = self.UI_LANGS[ui_lang]

        if provider == "google":
            if code == "zh":
                return "zh-CN"
            return code

        if provider == "deepl":
            return code

        raise ValueError(f"Proveedor no soportado: {provider}")

    def _build_traditional_translator(self):
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
        """
        Evita traducir cuando origen y destino son efectivamente iguales.
        """
        if self.provider is None:
            return False

        src = self._provider_lang_code(self.idioma_entrada, self.provider)
        tgt = self._provider_lang_code(self.idioma_salida, self.provider)

        if src == "auto":
            return False

        return src.split("-")[0].lower() == tgt.split("-")[0].lower()

    @staticmethod
    def _is_blank(texto: Optional[str]) -> bool:
        return texto is None or not str(texto).strip()


    def traducir_texto(self, texto: str) -> str:
        if self._is_blank(texto):
            return texto or ""

        if self._same_language():
            return texto

        try:
            traducido = self.translator.translate(texto)
            return traducido if isinstance(traducido, str) and traducido else texto
        except self.TRADITIONAL_EXCEPTIONS as exc:
            logger.warning("Fallo en traducción tradicional: %s", exc)
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

        payload = [textos[i] for i in indices]
        salida = textos[:]

        try:
            traducidos = self.translator.translate_batch(payload)
            if not isinstance(traducidos, list) or len(traducidos) != len(payload):
                raise ValueError("translate_batch devolvió un tamaño inesperado.")

            for idx, traducido in zip(indices, traducidos):
                salida[idx] = traducido if isinstance(traducido, str) and traducido else textos[idx]

            return salida

        except self.TRADITIONAL_EXCEPTIONS as exc:
            logger.warning("Fallo batch tradicional, usando fallback item por item: %s", exc)
            for idx in indices:
                salida[idx] = self.traducir_texto(textos[idx])
            return salida


    def traducir_textos_llm(
        self,
        textos_actuales: Sequence[str],
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
    ) -> List[str]:
        textos_actuales = list(textos_actuales)
        
        if not textos_actuales:
            return []

        if self._same_language():
            return textos_actuales[:]

        if self.client is None:
            logger.warning("Groq no está configurado; usando fallback tradicional.")
            return self.traducir_textos_tradicional(textos_actuales)

        items = [
            {"id": i, "texto": t}
            for i, t in enumerate(textos_actuales)
            if not self._is_blank(t)
        ]

        if not items:
            return textos_actuales[:]

        contexto = []
        for page in list(contexto_previo or [])[-2:]:
            contexto.append([str(x) for x in page if str(x).strip()])
            
        lore_str = f"Trama general para contexto: {self.lore_manga}. " if hasattr(self, 'lore_manga') and self.lore_manga else ""

        system_prompt = (
            f"Eres un traductor profesional y experto en manga del {self.idioma_entrada} al {self.idioma_salida}. "
            f"{lore_str}"
            "REGLAS ESTRICTAS DE TRADUCCIÓN:\n"
            "1. CORRECCIÓN DE OCR: El texto de entrada fue extraído por una máquina y puede contener errores tipográficos por kanjis o kanas mal leídos. Si una frase no tiene sentido, DEDUCE la palabra correcta basándote en la trama y el contexto de la escena antes de traducirla.\n"
            "2. TONO Y GÉNERO: Ajusta el vocabulario al género del manga (Shonen, Shojo, Seinen, Hentai, etc.) guiándote por la 'Trama general'. Traduce onomatopeyas verbales, gemidos, gritos o jadeos de forma natural y fluida al español (ej. 'あっ' -> '¡Ah!', 'はぁ' -> 'Haa~'). No censures absolutamente nada.\n"
            "3. ADAPTACIÓN COLOQUIAL: Adapta la jerga, los modismos y las expresiones (ej. 'マジで' = 'En serio / De verdad'). Los diálogos deben sonar como personas reales hablando, no suenes robótico ni literal.\n"
            "4. FRAGMENTOS Y BASURA VISUAL: Si el texto es basura incomprensible producto del OCR (letras o símbolos al azar como 'ｙｑｎｙ'), devuélvelo como '...'. Si es una frase deliberadamente cortada ('き..が...'), complétala lógicamente usando puntos suspensivos.\n"
            "5. COHERENCIA: Revisa estrictamente el 'contexto_previo' para mantener consistencia en los nombres, tratamientos (ej. Senpai, San) y estilo de habla de cada personaje de una página a otra.\n"
            "6. FORMATO: Debes responder ÚNICA y EXCLUSIVAMENTE con un objeto JSON válido con esta estructura exacta: "
            '{"traducciones": [{"id": 0, "traduccion": "texto traducido"}]}'
        )

        user_payload = {
            "contexto_previo": contexto,
            "textos_a_traducir": items,
        }
  
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.modelo,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(user_payload, ensure_ascii=False),
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.35, # <-- AUMENTAMOS LA TEMPERATURA PARA DARLE CREATIVIDAD
                    seed=self.seed,
                    max_completion_tokens=max(256, len(items) * 96),
                )

                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("Respuesta vacía del LLM.")

                data = json.loads(content)
                traducciones = data["traducciones"]

                esperados = {item["id"] for item in items}
                vistos = set()
                salida = textos_actuales[:]

                for row in traducciones:
                    idx = row["id"]
                    traducido = row["traduccion"]

                    if not isinstance(idx, int):
                        raise TypeError(f"id inválido: {idx!r}")
                    if idx not in esperados:
                        raise ValueError(f"id fuera de rango o inesperado: {idx}")
                    if idx in vistos:
                        raise ValueError(f"id duplicado: {idx}")
                    if not isinstance(traducido, str):
                        raise TypeError(f"traduccion inválida para id={idx}")

                    salida[idx] = traducido
                    vistos.add(idx)

                if vistos != esperados:
                    faltantes = sorted(esperados - vistos)
                    raise ValueError(f"Faltan traducciones para ids: {faltantes}")

                return salida

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Fallo LLM intento %s/%s: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))

        logger.error("LLM agotó reintentos. Último error: %s", last_error)
        return self.traducir_textos_tradicional(textos_actuales)

    def traducir_textos(
        self,
        textos_actuales: Sequence[str],
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
    ) -> List[str]:
        if self.metodo == "LLM":
            return self.traducir_textos_llm(textos_actuales, contexto_previo)
        return self.traducir_textos_tradicional(textos_actuales)