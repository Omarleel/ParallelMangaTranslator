"""Ritmo de salida y caché texto a texto del traductor tradicional.

Google no expone API oficial ni responde 429 cuando limita, así que estos tests
cubren las tres defensas propias: no salir más rápido de la cuenta, no repetir
peticiones que ya habían salido bien, y no confundir "me limitaron" con "este texto
no se puede traducir".
"""

import time
import unittest

from parallel_manga_translator.translation.traditional_translation_mixin import (
    TraditionalTranslationMixin,
    _MAX_REINTENTOS_BLOQUEO,
    _es_bloqueo_de_proveedor,
    _registrar_bloqueo,
    _registrar_exito,
    _reiniciar_throttle,
    _reservar_hueco,
)


class TranslationNotFound(Exception):
    """Stub con el mismo nombre que la excepción de deep_translator.

    La clasificación de bloqueos es por nombre de clase, precisamente para no
    depender de que deep_translator esté instalado.
    """


class NotValidPayload(Exception):
    """Fallo atribuible al texto, no al proveedor."""


class _FakeCache:
    def __init__(self):
        self.data = {}

    def hash_text(self, *partes):
        return "|".join(str(p) for p in partes)

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


class _FakeGlossary:
    def apply_to_translation(self, original, salida):
        return salida


class _FakeTranslator:
    def __init__(self, fallos=None, excepcion=TranslationNotFound, fallos_maximos=None):
        self.llamadas = []
        self.fallos = set(fallos or ())
        self.excepcion = excepcion
        # None = falla siempre; N = falla las N primeras veces y luego responde.
        self.fallos_maximos = fallos_maximos

    def translate(self, texto):
        self.llamadas.append(texto)
        if texto in self.fallos:
            intentos = self.llamadas.count(texto)
            if self.fallos_maximos is None or intentos <= self.fallos_maximos:
                raise self.excepcion(texto)
        return f"[{texto}]"


class _Provider(TraditionalTranslationMixin):
    UI_LANGS = {"Japonés": "ja", "Español": "es"}
    TRADITIONAL_EXCEPTIONS = (TranslationNotFound, NotValidPayload)

    def __init__(self, translator, min_interval=0.0, cooldown=0.0, max_wait=180.0, max_retries=3):
        self.metodo = "Tradicional"
        self.idioma_entrada = "Japonés"
        self.idioma_salida = "Español"
        self.provider = "google"
        self.translator = translator
        self.max_retries = max_retries
        self.traditional_min_interval = min_interval
        self.traditional_block_cooldown = cooldown
        self.traditional_block_max_wait = max_wait
        self._translation_cache = {}
        self.cache = _FakeCache()
        self.glossary = _FakeGlossary()


class TraditionalThrottleTests(unittest.TestCase):
    def setUp(self):
        _reiniciar_throttle()

    def tearDown(self):
        _reiniciar_throttle()

    def test_un_fallo_no_repite_los_textos_ya_traducidos(self):
        """Antes, el fallo de un texto tiraba el chunk entero y lo repetía todo."""
        translator = _FakeTranslator(fallos={"B"}, excepcion=NotValidPayload)
        provider = _Provider(translator)

        salida = provider.traducir_textos_tradicional(["A", "B", "C"])

        self.assertEqual(salida, ["[A]", "B", "[C]"])
        self.assertEqual(translator.llamadas.count("A"), 1)
        self.assertEqual(translator.llamadas.count("C"), 1)
        self.assertEqual(translator.llamadas.count("B"), provider.max_retries)

    def test_texto_repetido_no_repite_peticion(self):
        translator = _FakeTranslator()
        provider = _Provider(translator)

        salida = provider.traducir_textos_tradicional(["A", "A", "  ", "B"])

        self.assertEqual(salida, ["[A]", "[A]", "  ", "[B]"])
        self.assertEqual(translator.llamadas, ["A", "B"])

    def test_respeta_el_intervalo_minimo_entre_peticiones(self):
        translator = _FakeTranslator()
        provider = _Provider(translator, min_interval=0.05)

        inicio = time.monotonic()
        provider.traducir_textos_tradicional(["A", "B", "C"])
        transcurrido = time.monotonic() - inicio

        # La primera petición sale sin espera; las otras dos pagan el intervalo.
        self.assertGreaterEqual(transcurrido, 0.10)

    def test_translation_not_found_sobre_texto_real_es_bloqueo(self):
        self.assertTrue(_es_bloqueo_de_proveedor(TranslationNotFound("x"), "うわっ時田さん"))
        # Sin texto no hay nada que interpretar como límite del proveedor.
        self.assertFalse(_es_bloqueo_de_proveedor(TranslationNotFound(""), "   "))
        self.assertFalse(_es_bloqueo_de_proveedor(NotValidPayload("x"), "texto"))

    def test_un_bloqueo_no_gasta_los_reintentos_del_texto(self):
        """El texto que cae dentro de la ventana de bloqueo se elige al azar.

        Rendirse a los `max_retries` dejaba sin traducir a esos textos aunque el
        bloqueo se levantara un segundo después.
        """
        translator = _FakeTranslator(fallos={"B"}, fallos_maximos=5)
        provider = _Provider(translator, max_retries=3)

        self.assertEqual(provider.traducir_texto("B"), "[B]")
        self.assertEqual(translator.llamadas.count("B"), 6)

    def test_un_bloqueo_persistente_acaba_rindiendose(self):
        translator = _FakeTranslator(fallos={"B"})
        provider = _Provider(translator)

        self.assertEqual(provider.traducir_texto("B"), "B")
        self.assertLessEqual(translator.llamadas.count("B"), _MAX_REINTENTOS_BLOQUEO + 1)

    def test_el_techo_de_espera_corta_el_bloqueo(self):
        translator = _FakeTranslator(fallos={"B"})
        # La primera pausa (>= 1.0s) ya alcanza el techo, así que no se reintenta.
        provider = _Provider(translator, cooldown=1.0, max_wait=1.0)

        self.assertEqual(provider.traducir_texto("B"), "B")
        self.assertEqual(translator.llamadas.count("B"), 1)

    def test_un_bloqueo_aplaza_las_peticiones_siguientes(self):
        castigo = _registrar_bloqueo("google", 1.0)

        self.assertGreaterEqual(castigo, 1.0)
        # El castigo es global: la siguiente petición espera aunque sea otro texto.
        self.assertGreater(_reservar_hueco("google", 0.0), 0.5)
        # Y no afecta a otro proveedor.
        self.assertEqual(_reservar_hueco("deepl", 0.0), 0.0)

    def test_el_backoff_crece_y_una_traduccion_correcta_lo_reinicia(self):
        primero = _registrar_bloqueo("google", 1.0)
        segundo = _registrar_bloqueo("google", 1.0)
        self.assertGreater(segundo, primero)

        _registrar_exito("google")
        tercero = _registrar_bloqueo("google", 1.0)
        self.assertLess(tercero, segundo)


if __name__ == "__main__":
    unittest.main()
