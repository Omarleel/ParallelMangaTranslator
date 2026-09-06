"""Los cuatro pasos de `TranslateManga` son invocables por separado.

`traducir_manga` era un bloque único: extraer regiones, transcribir, traducir y rotular
en el mismo método. Quien quisiera detenerse a mitad —un modo solo-OCR, o `eval_runner`,
que mide sin traducir— tenía que reconstruir la secuencia a mano y quedaba desincronizado
del pipeline real.

Estos tests fijan las dos condiciones que hacen posible ese uso parcial:
sus dos colaboradores pesados se inyectan (se puede construir sin traductor ni motor OCR
reales), y encadenar los pasos a mano da exactamente lo mismo que llamar a `traducir_manga`.
"""

import unittest

import numpy as np

from parallel_manga_translator.config.app_config import ProcessingConfig, QualityConfig
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.processing.translate_manga import TranslateManga


class _OcrFalso:
    """Sustituto de `OcrManager`: devuelve un texto por recorte y cuenta las llamadas."""

    def __init__(self, textos):
        self.textos = list(textos)
        self.llamadas = 0

    def extract_texts(self, imagenes):
        self.llamadas += 1
        return [self.textos[i] if i < len(self.textos) else "" for i in range(len(imagenes))]


class _TraductorFalso:
    """Sustituto de `TranslatorManager` que registra si alguien lo usó."""

    def __init__(self):
        self.textos_recibidos = []

    def traducir_textos(self, textos, **kwargs):
        self.textos_recibidos.append(list(textos))
        return [f"[{t}]" for t in textos]

    def traducir_textos_tradicional(self, textos):
        return self.traducir_textos(textos)

    def traducir_texto(self, texto):
        return self.traducir_textos([texto])[0]

    def character_memory_snapshot(self):
        return {"characters": []}

    def analyze_character_memory(self, *args, **kwargs):
        return []


class _TraductorProhibido(_TraductorFalso):
    """El modo solo-OCR no debe rozar el traductor; si lo hace, que falle aquí."""

    def traducir_textos(self, textos, **kwargs):
        raise AssertionError("el modo solo-OCR no debe traducir")


def _pagina(alto=200, ancho=160):
    return np.full((alto, ancho, 3), 255, dtype=np.uint8)


def _region(bbox, alto=200, ancho=160):
    x, y, w, h = bbox
    mask = np.zeros((alto, ancho), dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    return TextRegion(bbox=bbox, text_bbox=bbox, mask=mask, kind="dialogue", confidence=0.9)


def _traductor_de_prueba(ocr, traductor):
    """Un `TranslateManga` real, sin motor OCR ni traductor de verdad detrás."""
    return TranslateManga(
        "en",
        "es",
        metodo_traduccion="Tradicional",
        quality_config=QualityConfig(),
        processing_config=ProcessingConfig(),
        ocr_manager=ocr,
        translator_manager=traductor,
    )


class PasosDelPipelineTests(unittest.TestCase):
    def setUp(self):
        self.imagen = _pagina()
        self.limpia = _pagina()
        self.mascara = np.zeros((200, 160), dtype=np.uint8)
        self.regiones = [_region((10, 10, 60, 40)), _region((10, 90, 60, 40))]

    def test_extraer_regiones_devuelve_cajas_y_recortes_alineados(self):
        tm = _traductor_de_prueba(_OcrFalso(["hello", "world"]), _TraductorFalso())

        cuadros, recortes = tm.extraer_regiones(self.imagen, self.mascara, self.regiones)

        self.assertEqual(len(cuadros), len(recortes))
        self.assertEqual(len(cuadros), len(tm.ultimas_regiones))
        self.assertEqual(len(cuadros), 2)

    def test_extraer_regiones_sin_regiones_no_arrastra_las_de_la_pagina_anterior(self):
        """`ultimas_regiones` es estado compartido entre pasos: rotular lo usa para recortar."""
        tm = _traductor_de_prueba(_OcrFalso(["hello", "world"]), _TraductorFalso())
        tm.extraer_regiones(self.imagen, self.mascara, self.regiones)
        self.assertEqual(len(tm.ultimas_regiones), 2)

        tm.extraer_regiones(self.imagen, self.mascara, None)

        self.assertEqual(tm.ultimas_regiones, [])

    def test_modo_solo_ocr_transcribe_sin_tocar_el_traductor(self):
        ocr = _OcrFalso(["hello", "world"])
        tm = _traductor_de_prueba(ocr, _TraductorProhibido())

        _, recortes = tm.extraer_regiones(self.imagen, self.mascara, self.regiones)
        textos = tm.obtener_textos(recortes)

        self.assertEqual(ocr.llamadas, 1)
        self.assertEqual([t.lower() for t in textos], ["hello", "world"])

    def test_rotular_no_traduce(self):
        """Paso 4 recibe textos ya resueltos: quien ya tradujo no debe pagarlo dos veces."""
        tm = _traductor_de_prueba(_OcrFalso(["hello", "world"]), _TraductorProhibido())
        cuadros, _ = tm.extraer_regiones(self.imagen, self.mascara, self.regiones)

        salida = tm.rotular(self.limpia, cuadros, ["hola", "mundo"])

        self.assertEqual(salida.shape, self.limpia.shape)

    def test_encadenar_los_cuatro_pasos_equivale_a_traducir_manga(self):
        """Si divergen, el modo por pasos mide una cosa y producción hace otra."""
        completo = _traductor_de_prueba(_OcrFalso(["hello", "world"]), _TraductorFalso())
        salida_completa = completo.traducir_manga(self.imagen, self.limpia.copy(), self.mascara, self.regiones)

        por_pasos = _traductor_de_prueba(_OcrFalso(["hello", "world"]), _TraductorFalso())
        cuadros, recortes = por_pasos.extraer_regiones(
            self.imagen, self.mascara, [_region((10, 10, 60, 40)), _region((10, 90, 60, 40))]
        )
        textos = por_pasos.obtener_textos(recortes)
        para_render = por_pasos.traducir_textos_de_regiones(cuadros, textos)
        salida_por_pasos = por_pasos.rotular(self.limpia.copy(), cuadros, para_render)

        self.assertEqual(completo.ultimos_textos_originales, por_pasos.ultimos_textos_originales)
        self.assertEqual(completo.ultimos_textos_traducidos, por_pasos.ultimos_textos_traducidos)
        np.testing.assert_array_equal(salida_completa, salida_por_pasos)


if __name__ == "__main__":
    unittest.main()
