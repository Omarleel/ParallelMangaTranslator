"""Los puertos como contratos reales, no como documentación.

Antes de este cambio `architecture/ports.py` tenía **cero** referencias fuera de sí mismo:
los Protocol existían pero nadie los usaba, así que no impedían nada. Estas pruebas fijan
las dos mitades del arreglo: que `ImageProcessor` recibe abstracciones y rechaza lo que no
cumple el contrato, y que los motores reales siguen cumpliéndolo.
"""

import unittest

import numpy as np

from parallel_manga_translator.architecture.ports import (
    InpainterPort,
    PageCleanerPort,
    PageTranslatorPort,
)
from parallel_manga_translator.processing.clean_manga import CleanManga
from parallel_manga_translator.processing.image_processor import ImageProcessor
from parallel_manga_translator.processing.translate_manga import TranslateManga

#: Los métodos que el orquestador necesita de cada etapa. Escritos a mano a propósito:
#: si alguien amplía un puerto, este test obliga a decidir si la etapa real lo cumple.
METODOS_CLEANER = (
    "limpiar_manga",
    "set_debug_page_context",
    "clear_debug_page_context",
    "set_visual_inpaint_debug_context",
    "clear_visual_inpaint_debug_context",
)
METODOS_TRANSLATOR = (
    "insertar_json_queue",
    "traducir_manga",
    "extraer_regiones",
    "obtener_textos",
    "traducir_textos_de_regiones",
    "rotular",
)


class _CleanerFalso:
    def __init__(self):
        self.paginas = []

    def limpiar_manga(self, imagen):
        self.paginas.append(imagen)
        return imagen, np.zeros(imagen.shape[:2], dtype=np.uint8), []

    def set_debug_page_context(self, page_index, *, source_filename=None, output_filename=None):
        pass

    def clear_debug_page_context(self):
        pass

    def set_visual_inpaint_debug_context(self, output_root, page_index, filename):
        pass

    def clear_visual_inpaint_debug_context(self):
        pass


class _TranslatorFalso:
    ultimas_regiones = ()

    def insertar_json_queue(self, indice_imagen, transcripcion_queue, traduccion_queue):
        pass

    def traducir_manga(self, imagen, imagen_limpia, mascara_capa, text_regions=None):
        return imagen_limpia

    def extraer_regiones(self, imagen, mascara_capa, text_regions=None):
        return [], []

    def obtener_textos(self, imagenes_interes):
        return []

    def traducir_textos_de_regiones(self, cuadros_delimitadores, textos):
        return []

    def rotular(self, imagen_limpia, cuadros_delimitadores, textos_para_render):
        return imagen_limpia


class _CleanerIncompleto:
    """Le falta el contexto de depuración de inpaint."""

    def limpiar_manga(self, imagen):
        return imagen, None, []

    def set_debug_page_context(self, page_index, *, source_filename=None, output_filename=None):
        pass

    def clear_debug_page_context(self):
        pass


class PuertosDelPipelineTests(unittest.TestCase):
    def test_el_orquestador_acepta_etapas_inyectadas(self):
        """Se puede construir sin cargar YOLO, OCR ni traductor: esa es la ganancia."""
        procesador = ImageProcessor(_CleanerFalso(), _TranslatorFalso())

        self.assertIsInstance(procesador.clean_manga, _CleanerFalso)
        self.assertIsInstance(procesador.translate_manga, _TranslatorFalso)

    def test_rechaza_una_etapa_que_no_cumple_el_contrato(self):
        with self.assertRaises(TypeError) as ctx:
            ImageProcessor(_CleanerIncompleto(), _TranslatorFalso())

        self.assertIn("PageCleanerPort", str(ctx.exception))

    def test_rechaza_un_traductor_que_no_cumple_el_contrato(self):
        with self.assertRaises(TypeError) as ctx:
            ImageProcessor(_CleanerFalso(), object())

        self.assertIn("PageTranslatorPort", str(ctx.exception))

    def test_el_orquestador_ya_no_construye_implementaciones(self):
        """Su firma es la frontera: abstracciones, ninguna configuración.

        Posicionales siguen siendo sólo las dos etapas. Las dos composiciones son de
        palabra clave y opcionales: son `Pipeline`, no valores de configuración, y su
        defecto reproduce la secuencia que este orquestador ya ejecutaba. Si alguna vez
        aparece aquí un parámetro con un valor de config, este test debe fallar.
        """
        import inspect

        firma = inspect.signature(ImageProcessor.__init__)
        params = firma.parameters

        posicionales = [
            nombre
            for nombre, p in params.items()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        self.assertEqual(posicionales, ["self", "cleaner", "translator"])

        por_clave = {nombre: p for nombre, p in params.items() if p.kind is p.KEYWORD_ONLY}
        self.assertEqual(sorted(por_clave), ["pipeline_limpieza", "pipeline_traduccion"])
        for nombre, p in por_clave.items():
            self.assertIsNone(p.default, f"{nombre} trae un valor por defecto que no es None")

    def test_las_composiciones_por_defecto_son_las_que_ya_se_ejecutaban(self):
        """El defecto no puede ser una secuencia nueva: sería un cambio de comportamiento."""
        procesador = ImageProcessor(_CleanerFalso(), _TranslatorFalso())

        self.assertEqual(procesador.pipeline_limpieza.nombres, ["limpieza"])
        self.assertEqual(
            procesador.pipeline_traduccion.nombres,
            ["extraer_regiones", "transcribir", "traducir", "rotular"],
        )

    def test_las_etapas_reales_cumplen_su_puerto(self):
        """Comprobado sobre la clase: instanciarlas cargaría modelos."""
        for metodo in METODOS_CLEANER:
            self.assertTrue(hasattr(CleanManga, metodo), f"CleanManga no expone {metodo}")
        for metodo in METODOS_TRANSLATOR:
            self.assertTrue(hasattr(TranslateManga, metodo), f"TranslateManga no expone {metodo}")

    def test_el_puerto_de_inpaint_describe_el_metodo_real(self):
        """Declaraba `async _load`/`async _inpaint`, que ninguna implementación tenía."""
        from parallel_manga_translator.inpainting import OpenCVInpainter

        self.assertIsInstance(OpenCVInpainter(), InpainterPort)

    def test_los_falsos_satisfacen_los_puertos_estructuralmente(self):
        self.assertIsInstance(_CleanerFalso(), PageCleanerPort)
        self.assertIsInstance(_TranslatorFalso(), PageTranslatorPort)
        self.assertNotIsInstance(_CleanerIncompleto(), PageCleanerPort)


if __name__ == "__main__":
    unittest.main()
