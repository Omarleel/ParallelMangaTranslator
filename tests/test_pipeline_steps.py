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
from parallel_manga_translator.models.page_context import PageContext
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

    def _contexto(self, regiones=None):
        return PageContext(
            imagen=self.imagen,
            imagen_limpia=self.limpia.copy(),
            mascara_capa=self.mascara,
            regiones=list(regiones or []),
        )

    def test_extraer_regiones_devuelve_cajas_y_recortes_alineados(self):
        tm = _traductor_de_prueba(_OcrFalso(["hello", "world"]), _TraductorFalso())
        ctx = self._contexto(self.regiones)

        tm.extraer_regiones(ctx)

        self.assertEqual(len(ctx.cuadros), len(ctx.recortes))
        self.assertEqual(len(ctx.cuadros), len(ctx.regiones_ordenadas))
        self.assertEqual(len(ctx.cuadros), 2)

    def test_cada_pagina_trae_su_contexto_y_no_hereda_el_de_la_anterior(self):
        """El traductor ya no recuerda la última página: el estado es del contexto.

        Antes esto era un atributo del traductor, así que una página sin regiones se
        quedaba con las de la anterior si alguien olvidaba limpiarlo.
        """
        tm = _traductor_de_prueba(_OcrFalso(["hello", "world"]), _TraductorFalso())
        primera = self._contexto(self.regiones)
        tm.extraer_regiones(primera)
        self.assertEqual(len(primera.regiones_ordenadas), 2)

        segunda = self._contexto()
        tm.extraer_regiones(segunda)

        self.assertEqual(segunda.regiones_ordenadas, [])
        self.assertEqual(len(primera.regiones_ordenadas), 2, "procesar otra página no puede tocar la anterior")

    def test_modo_solo_ocr_transcribe_sin_tocar_el_traductor(self):
        ocr = _OcrFalso(["hello", "world"])
        tm = _traductor_de_prueba(ocr, _TraductorProhibido())
        ctx = self._contexto(self.regiones)

        tm.extraer_regiones(ctx)
        tm.obtener_textos(ctx)

        self.assertEqual(ocr.llamadas, 1)
        self.assertEqual([t.lower() for t in ctx.textos], ["hello", "world"])

    def test_rotular_no_traduce(self):
        """Paso 4 recibe textos ya resueltos: quien ya tradujo no debe pagarlo dos veces."""
        tm = _traductor_de_prueba(_OcrFalso(["hello", "world"]), _TraductorProhibido())
        ctx = self._contexto(self.regiones)
        tm.extraer_regiones(ctx)
        ctx.textos_para_render = ["hola", "mundo"]

        tm.rotular(ctx)

        self.assertEqual(ctx.imagen_final.shape, self.limpia.shape)

    def test_encadenar_los_cuatro_pasos_equivale_a_traducir_manga(self):
        """Si divergen, el modo por pasos mide una cosa y producción hace otra."""
        completo = _traductor_de_prueba(_OcrFalso(["hello", "world"]), _TraductorFalso())
        salida_completa = completo.traducir_manga(self.imagen, self.limpia.copy(), self.mascara, self.regiones)

        por_pasos = _traductor_de_prueba(_OcrFalso(["hello", "world"]), _TraductorFalso())
        ctx = PageContext(
            imagen=self.imagen,
            imagen_limpia=self.limpia.copy(),
            mascara_capa=self.mascara,
            regiones=[_region((10, 10, 60, 40)), _region((10, 90, 60, 40))],
        )
        por_pasos.extraer_regiones(ctx)
        por_pasos.obtener_textos(ctx)
        por_pasos.traducir_textos_de_regiones(ctx)
        por_pasos.rotular(ctx)

        self.assertEqual([t.lower() for t in ctx.textos_originales], ["hello", "world"])
        self.assertEqual(len(ctx.textos_traducidos), 2)
        self.assertEqual(len(ctx.textos_para_render), 2)
        # La igualdad de las dos páginas rotuladas es la equivalencia de verdad: cubre
        # texto, estilo y geometría de una vez.
        np.testing.assert_array_equal(salida_completa, ctx.imagen_final)

    def test_el_traductor_no_guarda_estado_de_la_pagina(self):
        """La razón de todo esto: un objeto de vida larga no puede ser el cuaderno de notas."""
        tm = _traductor_de_prueba(_OcrFalso(["hello", "world"]), _TraductorFalso())
        ctx = self._contexto(self.regiones)
        tm.extraer_regiones(ctx)
        tm.obtener_textos(ctx)
        tm.traducir_textos_de_regiones(ctx)

        for atributo in (
            "ultimas_regiones",
            "ultima_pagina",
            "ultimo_estilos_texto",
            "ultimos_textos_originales",
            "ultimos_textos_traducidos",
            "ultimos_source_language_flags",
            "ultimas_asignaciones_hablante",
            "indice_imagen",
        ):
            self.assertFalse(hasattr(tm, atributo), f"{atributo} volvió a vivir en el traductor")


if __name__ == "__main__":
    unittest.main()
