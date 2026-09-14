"""Cómo despachan los dos proveedores de traducción, fijado antes de reordenarlos.

`GroqTranslationProvider` heredaba de `TraditionalTranslationMixin` y `LlmTranslationMixin`
para conseguir dos cosas: el motor tradicional al que cae cuando el LLM falla, y un puñado de
ayudantes de texto. Era composición disfrazada de herencia. Estos tests se escribieron
**antes** de deshacerlo y no ha hecho falta cambiar ninguno: esa es la prueba de que estaban
puestos al nivel correcto.

`eval_runner` **no cubre traducción** —mide limpieza, detección y OCR—, así que antes de
tocar nada hace falta esta red. Fija el comportamiento observable por la API pública
(`traducir_textos`), con el backend falseado en su punto más bajo (`provider.translator`),
para que siga valiendo cuando la herencia se convierta en composición: si un test de aquí
tuviera que cambiar con el refactor, no estaría protegiendo nada.

Ninguno de estos tests toca la red: sin `groq_api_key` el cliente queda a `None`, y el
traductor tradicional se sustituye por un doble que cuenta llamadas.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from parallel_manga_translator.translation.providers.base import TranslationProviderConfig
from parallel_manga_translator.translation.providers.groq_provider import GroqTranslationProvider
from parallel_manga_translator.translation.providers.traditional_provider import (
    TraditionalTranslationProvider,
)


class _TraductorFalso:
    """Sustituye al backend de deep-translator. Cuenta lo que se le pide."""

    def __init__(self, prefijo: str = "es:"):
        self.prefijo = prefijo
        self.recibidos: list[str] = []

    def translate(self, texto: str) -> str:
        self.recibidos.append(texto)
        return f"{self.prefijo}{texto}"


class _ClienteLlmFalso:
    """Sustituye al cliente de Groq. Devuelve el JSON que el proveedor sabe leer."""

    def __init__(self, traducciones):
        self._traducciones = list(traducciones)
        self.peticiones: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.peticiones.append(kwargs)
        contenido = json.dumps({"traducciones": self._traducciones}, ensure_ascii=False)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=contenido))],
            usage=None,
        )


class _ProveedorDeTest(unittest.TestCase):
    """Construye proveedores reales sin red, sin caché en disco y sin memoria de personajes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _config(self, **kwargs) -> TranslationProviderConfig:
        base = {
            "source_language": "Japonés",
            "target_language": "Español",
            "method": "Tradicional",
            "groq_api_key": "",
            "cache_enabled": False,
            "character_memory_enabled": False,
            "cache_dir": str(self.tmp / "cache"),
            "project_dir": str(self.tmp),
            "traditional_min_interval": 0.0,
            "traditional_block_cooldown": 0.0,
        }
        base.update(kwargs)
        return TranslationProviderConfig(**base)

    def _tradicional(self, **kwargs):
        proveedor = TraditionalTranslationProvider(self._config(**kwargs))
        backend = _TraductorFalso()
        proveedor.translator = backend
        return proveedor, backend

    def _groq(self, **kwargs):
        proveedor = GroqTranslationProvider(self._config(method="LLM", **kwargs))
        backend = _TraductorFalso()
        proveedor.translator = backend
        return proveedor, backend


class ProveedorTradicionalTests(_ProveedorDeTest):
    def test_traduce_texto_a_texto(self):
        proveedor, backend = self._tradicional()

        salida = proveedor.traducir_textos(["こんにちは", "さようなら"])

        self.assertEqual(salida, ["es:こんにちは", "es:さようなら"])
        self.assertEqual(backend.recibidos, ["こんにちは", "さようなら"])

    def test_conserva_los_vacios_sin_preguntar_al_backend(self):
        proveedor, backend = self._tradicional()

        salida = proveedor.traducir_textos(["", "   ", "ドン"])

        self.assertEqual(salida[0], "")
        self.assertEqual(salida[1], "   ")
        self.assertEqual(salida[2], "es:ドン")
        self.assertEqual(backend.recibidos, ["ドン"], "los vacíos no pueden gastar una petición")

    def test_mismo_idioma_devuelve_el_original_sin_tocar_el_backend(self):
        proveedor, backend = self._tradicional(source_language="Español", target_language="Español")

        salida = proveedor.traducir_textos(["hola", "adiós"])

        self.assertEqual(salida, ["hola", "adiós"])
        self.assertEqual(backend.recibidos, [])

    def test_una_lista_vacia_no_hace_nada(self):
        proveedor, backend = self._tradicional()

        self.assertEqual(proveedor.traducir_textos([]), [])
        self.assertEqual(backend.recibidos, [])

    def test_repetidos_se_piden_una_sola_vez(self):
        proveedor, backend = self._tradicional()

        salida = proveedor.traducir_textos(["ドン", "ドン", "バン"])

        self.assertEqual(salida, ["es:ドン", "es:ドン", "es:バン"])
        self.assertEqual(set(backend.recibidos), {"ドン", "バン"}, "un texto repetido no se pide dos veces")


class ProveedorGroqTests(_ProveedorDeTest):
    def test_sin_cliente_y_sin_respaldo_falla_diciendo_por_que(self):
        """El fallo explícito es la opción por defecto: traducir mal en silencio es peor."""
        proveedor, backend = self._groq(fallback_to_traditional_on_error=False)

        with self.assertRaises(RuntimeError):
            proveedor.traducir_textos(["こんにちは"])

        self.assertEqual(backend.recibidos, [], "sin respaldo no puede colarse el tradicional")
        self.assertEqual(proveedor.llm_fallback_reason, "")

    def test_sin_cliente_y_con_respaldo_cae_al_tradicional(self):
        """Es la razón por la que hoy hereda el motor tradicional entero."""
        proveedor, backend = self._groq(fallback_to_traditional_on_error=True)

        salida = proveedor.traducir_textos(["こんにちは", "ドン"])

        self.assertEqual(salida, ["es:こんにちは", "es:ドン"])
        self.assertEqual(backend.recibidos, ["こんにちは", "ドン"])
        self.assertIn("tradicional", proveedor.llm_fallback_reason)

    def test_con_cliente_traduce_por_el_llm_y_no_toca_el_tradicional(self):
        """El camino que el refactor moverá: con LLM disponible, el tradicional ni se roza."""
        proveedor, backend = self._groq(fallback_to_traditional_on_error=True)
        proveedor.client = _ClienteLlmFalso(
            [{"id": 0, "traduccion": "Hola"}, {"id": 1, "traduccion": "Adiós"}]
        )

        salida = proveedor.traducir_textos(["こんにちは", "さようなら"])

        self.assertEqual(salida, ["Hola", "Adiós"])
        self.assertEqual(backend.recibidos, [], "con LLM disponible no puede caer al tradicional")
        self.assertEqual(proveedor.llm_fallback_reason, "")
        self.assertEqual(len(proveedor.client.peticiones), 1, "una página, una petición")

    def test_el_respaldo_respeta_las_reglas_del_tradicional(self):
        """Caer al respaldo no puede saltarse el filtro de vacíos ni el de mismo idioma."""
        proveedor, backend = self._groq(fallback_to_traditional_on_error=True)

        salida = proveedor.traducir_textos(["", "ドン"])

        self.assertEqual(salida[0], "")
        self.assertEqual(salida[1], "es:ドン")
        self.assertEqual(backend.recibidos, ["ドン"])


if __name__ == "__main__":
    unittest.main()
