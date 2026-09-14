from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from deep_translator import DeeplTranslator, GoogleTranslator
    from deep_translator.exceptions import AuthorizationException
except ImportError:  # pragma: no cover
    DeeplTranslator = None  # type: ignore
    GoogleTranslator = None  # type: ignore
    class AuthorizationException(Exception):
        pass


from parallel_manga_translator.infrastructure.execution_control import JobControlError, get_execution_control, cooperative_sleep

logger = logging.getLogger(__name__)


# --- Ritmo de salida hacia los proveedores tradicionales -----------------------
#
# `GoogleTranslator` de deep_translator no usa API oficial: raspa
# translate.google.com/m. Cuando Google limita el trafico NO responde 429, responde
# 200 con una pagina sin el div de resultado, y deep_translator lo convierte en
# `TranslationNotFound`. Es decir, el bloqueo llega disfrazado de "este texto no se
# puede traducir". Sin freno propio el pipeline acelera hasta que lo bloquean y luego
# insiste, alargando el bloqueo.
#
# El estado es de modulo, no de instancia, porque el limite lo impone el proveedor
# por IP de origen: `TranslatorManager` crea a la vez el provider principal y el
# fallback tradicional, y ambos salen por la misma IP.

_MAX_NIVELES_BLOQUEO = 4
# Tope de reintentos por bloqueo. Normalmente manda `traditional_block_max_wait`;
# este contador es la red de seguridad si la pausa configurada es 0.
_MAX_REINTENTOS_BLOQUEO = 8

_THROTTLE_LOCK = threading.Lock()
_THROTTLE_STATE: Dict[str, Dict[str, float]] = {}

_EXCEPCIONES_DE_BLOQUEO = {"TooManyRequests", "RequestError", "ServerException"}


def _estado_throttle(provider: str) -> Dict[str, float]:
    return _THROTTLE_STATE.setdefault(provider, {"proximo_hueco": 0.0, "bloqueos": 0.0})


def _reservar_hueco(provider: str, intervalo: float) -> float:
    """Reserva el siguiente hueco de salida y devuelve cuantos segundos hay que esperar."""
    with _THROTTLE_LOCK:
        estado = _estado_throttle(provider)
        ahora = time.monotonic()
        salida = max(ahora, estado["proximo_hueco"])
        estado["proximo_hueco"] = salida + max(0.0, float(intervalo))
        return max(0.0, salida - ahora)


def _registrar_bloqueo(provider: str, castigo_base: float) -> float:
    """Aplaza el hueco global tras un bloqueo y devuelve la pausa aplicada.

    La pausa es exponencial y afecta a todas las peticiones siguientes del mismo
    proveedor, no solo al texto que fallo: si nos limitaron, el problema no es ese
    texto concreto.
    """
    with _THROTTLE_LOCK:
        estado = _estado_throttle(provider)
        estado["bloqueos"] = min(estado["bloqueos"] + 1.0, float(_MAX_NIVELES_BLOQUEO))
        castigo = max(0.0, float(castigo_base)) * (2.0 ** (estado["bloqueos"] - 1.0))
        # Jitter: evita que varios reintentos vuelvan a salir sincronizados.
        castigo *= 1.0 + random.uniform(0.0, 0.25)
        estado["proximo_hueco"] = max(estado["proximo_hueco"], time.monotonic() + castigo)
        return castigo


def _registrar_exito(provider: str) -> None:
    with _THROTTLE_LOCK:
        _estado_throttle(provider)["bloqueos"] = 0.0


def _reiniciar_throttle() -> None:
    """Limpia el estado global. Pensado para tests."""
    with _THROTTLE_LOCK:
        _THROTTLE_STATE.clear()


def _es_bloqueo_de_proveedor(exc: Exception, texto: str) -> bool:
    """Distingue "me limitaron" de "este texto no se pudo traducir".

    Se compara por nombre de clase para no depender de que deep_translator este
    instalado: sin el, las excepciones son stubs locales de cada modulo.
    """
    nombre = type(exc).__name__
    if nombre in _EXCEPCIONES_DE_BLOQUEO:
        return True
    return nombre == "TranslationNotFound" and bool(str(texto or "").strip())


def _resumen_texto(texto: str, limite: int = 60) -> str:
    plano = str(texto or "").replace("\n", " ").strip()
    return plano if len(plano) <= limite else plano[: limite - 1] + "..."


from parallel_manga_translator.translation.provider_rules import (
    is_blank,
    normalize_translation,
    persistent_key,
    provider_lang_code,
    same_language,
)


class TraditionalTranslationMixin:
    """Proveedor tradicional, cache y control de ritmo hacia el proveedor."""

    # Los proveedores los sobrescriben desde config.yaml (`translation.traditional_*`).
    traditional_min_interval: float = 0.5
    traditional_block_cooldown: float = 6.0
    traditional_block_max_wait: float = 60.0

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
                source = provider_lang_code(self.UI_LANGS, self.idioma_entrada, provider)
                target = provider_lang_code(self.UI_LANGS, self.idioma_salida, provider)

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
        return same_language(self.UI_LANGS, self.provider, self.idioma_entrada, self.idioma_salida)

    def _cache_key(self, texto: str) -> Tuple[str, str, str, str]:
        return (self.provider or "unknown", self.idioma_entrada, self.idioma_salida, str(texto or ""))

    def _persistent_key(self, texto: str, method: Optional[str] = None) -> str:
        return persistent_key(
            self.cache, texto, method or self.metodo, self.provider, self.idioma_entrada, self.idioma_salida
        )

    def _buscar_en_cache(self, texto: str) -> Optional[str]:
        cache_key = self._cache_key(texto)
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]
        persistent = self.cache.get(self._persistent_key(texto, "traditional"))
        if persistent is None:
            return None
        persistent = normalize_translation(persistent)
        self._translation_cache[cache_key] = persistent
        return persistent

    def _guardar_traduccion(self, original: str, traducido: Any) -> str:
        """Normaliza, aplica glosario y cachea una traducción recién obtenida.

        Cachear aquí, texto a texto, es lo que impide que el fallo de un texto obligue
        a repetir los que ya habían salido bien.
        """
        salida = normalize_translation(traducido) if isinstance(traducido, str) and traducido else original
        salida = self.glossary.apply_to_translation(original, salida)
        self._translation_cache[self._cache_key(original)] = salida
        self.cache.set(self._persistent_key(original, "traditional"), salida)
        return salida

    def _llamar_proveedor(self, texto: str) -> Any:
        """Una petición al proveedor, respetando el ritmo global de salida."""
        if self.translator is None:
            raise RuntimeError("deep_translator no está instalado; instala requirements.txt o usa LLM con proveedor configurado.")

        provider = str(self.provider or "traditional")
        espera = _reservar_hueco(provider, self.traditional_min_interval)
        if espera > 0:
            cooperative_sleep(espera)

        control = get_execution_control()
        reservation = None
        if control is not None:
            reservation = control.reserve_external_call(
                kind="traditional",
                provider=provider,
                characters=len(str(texto or "")),
            )
        try:
            traducido = self.translator.translate(texto)
        except Exception:
            if control is not None and reservation is not None:
                control.commit_external_call(reservation, failed=True)
            raise
        if control is not None and reservation is not None:
            control.commit_external_call(reservation)
        _registrar_exito(provider)
        return traducido

    def _registrar_fallo(self, texto: str, exc: Exception, intento: int) -> Optional[float]:
        """Anota el fallo y devuelve la pausa impuesta si fue un bloqueo del proveedor.

        Devuelve `None` cuando el fallo es propio del texto, que es lo que distingue
        "gastar un reintento" de "esperar a que nos dejen pasar".
        """
        provider = str(self.provider or "traditional")
        if _es_bloqueo_de_proveedor(exc, texto):
            # Aquí no se avisa de la pausa: quien decide si de verdad habrá reintento es
            # `traducir_texto`, y anunciar "pausando Xs antes de reintentar" para acto
            # seguido rendirse era exactamente lo que se leía en los logs.
            return _registrar_bloqueo(provider, self.traditional_block_cooldown)

        logger.warning(
            "Fallo en traducción tradicional intento %s/%s: %s --> %s",
            intento,
            self.max_retries,
            _resumen_texto(texto),
            exc,
        )
        return None

    def traducir_texto(self, texto: str) -> str:
        if is_blank(texto):
            return texto or ""

        if self._same_language():
            return texto

        cacheado = self._buscar_en_cache(texto)
        if cacheado is not None:
            return cacheado

        fallos = 0
        bloqueos = 0
        espera_por_bloqueo = 0.0
        while True:
            try:
                return self._guardar_traduccion(texto, self._llamar_proveedor(texto))
            except JobControlError:
                raise
            except self.TRADITIONAL_EXCEPTIONS as exc:
                # La espera del siguiente intento la impone el hueco global:
                # `_registrar_fallo` aplaza el turno cuando detecta un bloqueo.
                castigo = self._registrar_fallo(texto, exc, fallos + 1)

                if castigo is None:
                    fallos += 1
                    if fallos >= self.max_retries:
                        return texto
                    continue

                # Un bloqueo no dice nada sobre este texto: dice que el proveedor no nos
                # deja pasar ahora mismo. Gastar aquí los reintentos dejaría sin traducir
                # a los textos que caen dentro de la ventana de bloqueo, elegidos al azar.
                # Se espera a que pase, con un techo total para no colgar el trabajo.
                bloqueos += 1
                # El margen se comprueba contando la pausa que viene, no después de
                # haberla sumado: comprobarlo después dejaba pasar una espera entera por
                # encima del techo (180s configurados, 225s esperados de verdad).
                if (espera_por_bloqueo + castigo >= self.traditional_block_max_wait
                        or bloqueos >= _MAX_REINTENTOS_BLOQUEO):
                    logger.warning(
                        "%s sigue limitando y se agota el margen de %.0fs (esperados %.0fs); "
                        "el texto queda sin traducir: %s",
                        self.provider,
                        self.traditional_block_max_wait,
                        espera_por_bloqueo,
                        _resumen_texto(texto),
                    )
                    return texto

                espera_por_bloqueo += castigo
                logger.warning(
                    "%s está limitando las peticiones; pausando %.1fs antes de reintentar "
                    "(%.0fs de %.0fs de margen): %s",
                    self.provider,
                    castigo,
                    espera_por_bloqueo,
                    self.traditional_block_max_wait,
                    _resumen_texto(texto),
                )

    def traducir_textos_tradicional(self, textos: Sequence[str]) -> List[str]:
        """Traduce una lista deduplicando y cacheando texto a texto.

        No se usa `translate_batch` de deep_translator: internamente es un bucle de
        peticiones individuales sin pausa (ningún proveedor de la librería agrupa de
        verdad), así que agrupar no ahorraba ni una petición y, en cambio, perdía todo
        el chunk cuando fallaba un único texto.
        """
        textos = list(textos)
        if not textos:
            return []

        if self._same_language():
            return textos[:]

        indices = [i for i, t in enumerate(textos) if not is_blank(t)]
        if not indices:
            return textos[:]

        salida = textos[:]
        pendientes_por_texto: Dict[str, List[int]] = {}

        for idx in indices:
            texto = textos[idx]
            cacheado = self._buscar_en_cache(texto)
            if cacheado is not None:
                salida[idx] = cacheado
            else:
                pendientes_por_texto.setdefault(texto, []).append(idx)

        for texto, posiciones in pendientes_por_texto.items():
            traducido = self.traducir_texto(texto)
            for idx in posiciones:
                salida[idx] = traducido

        return salida
