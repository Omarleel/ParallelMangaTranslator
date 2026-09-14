"""El pipeline como lista de etapas: componer, no reescribir.

Antes la secuencia estaba escrita por dentro de `limpiar_manga` y `traducir_manga`, así
que ejecutar sólo una parte obligaba a reimplementarla. `eval_runner` lo hacía, y al
hacerlo tocaba el estado interno del traductor. Estos tests fijan que las composiciones
por defecto son las que ya se ejecutaban y que una parcial funciona de verdad de punta a
punta, que es lo único que prueba que la composición sirve para algo.
"""

import queue
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from parallel_manga_translator.processing.image_processor import ImageProcessor
from parallel_manga_translator.processing.pipeline import (
    ExtraerRegiones,
    LimpiarPagina,
    PageContext,
    Pipeline,
    TranscribirTextos,
    pipeline_limpieza_y_ocr,
    pipeline_solo_ocr,
)


class _CleanerFalso:
    def __init__(self, regiones=()):
        self.regiones = list(regiones)
        self.paginas = 0

    def limpiar_manga(self, ctx):
        self.paginas += 1
        ctx.mascara_capa = np.zeros(ctx.imagen.shape[:2], dtype=np.uint8)
        ctx.imagen_limpia = ctx.imagen.copy()
        ctx.regiones = list(self.regiones)


class _TranslatorFalso:
    """Registra qué pasos se ejecutaron, que es lo que distingue una composición de otra.

    Cada paso escribe en el contexto, como los de verdad: el doble de prueba también
    sirve para fijar que ningún paso devuelve el estado por la puerta de atrás.
    """

    def __init__(self, ordenadas=()):
        self.ordenadas = list(ordenadas)
        self.pasos = []

    def insertar_json_queue(self, transcripcion_queue, traduccion_queue):
        self.pasos.append("json_queue")

    def extraer_regiones(self, ctx):
        self.pasos.append("extraer")
        ctx.cuadros = [(0, 0, 4, 4)]
        ctx.recortes = [ctx.imagen[:4, :4]]
        ctx.regiones_ordenadas = list(self.ordenadas)

    def obtener_textos(self, ctx):
        self.pasos.append("transcribir")
        ctx.textos = ["hello"]

    def traducir_textos_de_regiones(self, ctx):
        self.pasos.append("traducir")
        ctx.textos_originales = list(ctx.textos)
        ctx.textos_traducidos = ["hola"]
        ctx.textos_para_render = ["hola"]

    def rotular(self, ctx):
        self.pasos.append("rotular")
        ctx.imagen_final = ctx.imagen_limpia

    def traducir_manga(self, imagen, imagen_limpia, mascara_capa, text_regions=None, indice_pagina=0):
        self.pasos.append("traducir_manga")
        return imagen_limpia


def _pagina():
    return np.full((32, 24, 3), 255, dtype=np.uint8)


class ComposicionTests(unittest.TestCase):
    def test_las_composiciones_con_nombre_dicen_sus_etapas(self):
        cleaner, translator = _CleanerFalso(), _TranslatorFalso()

        self.assertEqual(pipeline_solo_ocr(translator).nombres, ["extraer_regiones", "transcribir"])
        self.assertEqual(
            pipeline_limpieza_y_ocr(cleaner, translator).nombres,
            ["limpieza", "extraer_regiones", "transcribir"],
        )

    def test_solo_ocr_no_traduce_ni_rotula(self):
        translator = _TranslatorFalso()

        pipeline_solo_ocr(translator).run(PageContext(imagen=_pagina()))

        self.assertEqual(translator.pasos, ["extraer", "transcribir"])

    def test_una_composicion_parcial_deja_vacio_lo_que_no_ejecuta(self):
        """`imagen_final` a None significa que nadie rotuló, no que el rotulado fallara."""
        ctx = pipeline_limpieza_y_ocr(_CleanerFalso(), _TranslatorFalso()).run(PageContext(imagen=_pagina()))

        self.assertIsNotNone(ctx.imagen_limpia)
        self.assertEqual(ctx.textos, ["hello"])
        self.assertIsNone(ctx.imagen_final)
        self.assertEqual(ctx.textos_para_render, [])

    def test_las_regiones_ordenadas_no_pisan_a_las_crudas(self):
        """eval_runner mide limpieza sobre las crudas y el JSON sobre las ordenadas.

        La extracción puede descartar regiones, así que confundirlas cambiaría en
        silencio lo que se mide.
        """
        crudas = ["cruda_1", "cruda_2"]
        ordenadas = ["ordenada_1"]
        cleaner = _CleanerFalso(regiones=crudas)
        translator = _TranslatorFalso(ordenadas=ordenadas)

        ctx = pipeline_limpieza_y_ocr(cleaner, translator).run(PageContext(imagen=_pagina()))

        self.assertEqual(ctx.regiones, crudas)
        self.assertEqual(ctx.regiones_ordenadas, ordenadas)

    def test_rechaza_una_etapa_que_no_cumple_el_contrato(self):
        with self.assertRaises(TypeError) as ctx:
            Pipeline([LimpiarPagina(_CleanerFalso()), object()])

        self.assertIn("PageStage", str(ctx.exception))

    def test_las_etapas_son_reordenables_porque_son_datos(self):
        """La prueba de que es composición: la lista se construye, no se hereda."""
        cleaner, translator = _CleanerFalso(), _TranslatorFalso()

        propia = Pipeline([LimpiarPagina(cleaner), ExtraerRegiones(translator), TranscribirTextos(translator)])

        self.assertEqual(propia.nombres, pipeline_limpieza_y_ocr(cleaner, translator).nombres)


class OrquestadorConComposicionParcialTests(unittest.TestCase):
    """Un modo solo-OCR ejecutado de punta a punta por el orquestador real."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="pmt-comp-"))
        self.entrada = self.root / "entrada"
        self.limpieza = self.root / "salida" / "limpieza"
        self.traduccion = self.root / "salida" / "traduccion"
        for carpeta in (self.entrada, self.limpieza, self.traduccion):
            carpeta.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.entrada / "p001.png"), _pagina())

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _procesar(self, **kwargs):
        procesador = ImageProcessor(_CleanerFalso(), kwargs.pop("translator"), **kwargs)
        procesador.procesar(
            str(self.entrada),
            str(self.limpieza),
            str(self.traduccion),
            {0: "p001.png"},
            queue.Queue(),
            queue.Queue(),
        )
        return procesador

    def test_sin_rotular_guarda_la_limpieza_y_no_la_traduccion(self):
        translator = _TranslatorFalso()

        self._procesar(translator=translator, pipeline_traduccion=pipeline_solo_ocr(translator))

        self.assertEqual(translator.pasos, ["json_queue", "extraer", "transcribir"])
        # El orquestador renombra la salida por indice de pagina, no conserva el origen.
        self.assertEqual([p.name for p in self.limpieza.iterdir()], ["0001.png"])
        self.assertEqual(list(self.traduccion.iterdir()), [])

    def test_la_composicion_por_defecto_si_guarda_la_traduccion(self):
        """El contraste: sin esto, el test de arriba pasaría aunque nada funcionara."""
        translator = _TranslatorFalso()

        self._procesar(translator=translator)

        self.assertEqual(translator.pasos, ["json_queue", "extraer", "transcribir", "traducir", "rotular"])
        self.assertEqual([p.name for p in self.traduccion.iterdir()], ["0001.png"])


if __name__ == "__main__":
    unittest.main()
