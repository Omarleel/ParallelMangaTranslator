"""`DetectionGeometry` como colaborador, no como clase base.

Era `BubbleGeometryMixin`: 19 métodos que no tocaban estado de instancia, heredados por
`BubbleDetector` sólo para que sus mixins hermanos los llamaran por `self`. Esa dependencia
no estaba declarada en ninguna parte y no se podía probar sin construir el detector entero
—que carga YOLO—. Estas pruebas existen porque ahora sí se puede.
"""

import ast
import inspect
import textwrap
import unittest

import numpy as np

from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.detection.detection_geometry import DetectionGeometry
from parallel_manga_translator.processing.clean_manga import CleanManga
from parallel_manga_translator.processing.translate_manga import TranslateManga

#: Una detección con la forma que produce EasyOCR: puntos, texto, confianza.
DETECCION = ([[10, 20], [50, 20], [50, 60], [10, 60]], "こんにちは", 0.87)


class GeometriaPuraTests(unittest.TestCase):
    def setUp(self):
        self.g = DetectionGeometry()

    def test_no_necesita_estado_de_instancia(self):
        """El motivo de que fuera un mixin era falso: no comparte nada con el detector."""
        self.assertEqual(vars(self.g), {})

    def test_area_union_y_centro(self):
        self.assertEqual(self.g.area((0, 0, 10, 10)), 100)
        self.assertEqual(self.g.union((0, 0, 10, 10), (5, 5, 10, 10)), (0, 0, 15, 15))
        self.assertEqual(self.g.center((0, 0, 10, 10)), (5.0, 5.0))

    def test_interseccion(self):
        self.assertEqual(self.g.intersection_area((0, 0, 10, 10), (5, 5, 10, 10)), 25)
        self.assertEqual(self.g.intersection_area((0, 0, 10, 10), (50, 50, 10, 10)), 0)
        self.assertIsNone(self.g.box_intersection((0, 0, 10, 10), (50, 50, 10, 10)))

    def test_solape_en_un_eje(self):
        self.assertAlmostEqual(self.g.overlap_ratio_1d(0, 10, 5, 15), 0.5)
        self.assertAlmostEqual(self.g.overlap_ratio_1d(0, 10, 20, 30), 0.0)

    def test_punto_dentro_de_caja(self):
        self.assertTrue(self.g.point_inside_box((5, 5), (0, 0, 10, 10)))
        self.assertFalse(self.g.point_inside_box((50, 5), (0, 0, 10, 10)))

    def test_recorte_a_la_imagen(self):
        x, y, w, h = self.g.clip_box_to_image((-5, -5, 200, 200), 100, 80)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, 100)
        self.assertLessEqual(y + h, 80)

    def test_caja_de_una_mascara(self):
        mascara = np.zeros((50, 50), dtype=np.uint8)
        mascara[10:20, 5:15] = 255

        self.assertEqual(self.g.mask_bbox(mascara), (5, 10, 10, 10))
        self.assertIsNone(self.g.mask_bbox(np.zeros((50, 50), dtype=np.uint8)))


class LecturaDeDeteccionesTests(unittest.TestCase):
    def setUp(self):
        self.g = DetectionGeometry()

    def test_lee_texto_y_confianza(self):
        self.assertEqual(self.g.text(DETECCION), "こんにちは")
        self.assertAlmostEqual(self.g.confidence(DETECCION), 0.87)

    def test_una_deteccion_malformada_no_revienta(self):
        """El OCR devuelve basura de vez en cuando; devolver un default es lo correcto."""
        self.assertEqual(self.g.text(("solo la caja",)), "")
        self.assertEqual(self.g.confidence(("solo la caja",)), 0.0)

    def test_agrega_varias_detecciones(self):
        otra = ([[100, 20], [140, 20], [140, 60], [100, 60]], "さようなら", 0.63)

        caja = self.g.detections_box([DETECCION, otra])
        self.assertEqual(caja[0], 10)
        self.assertGreaterEqual(caja[2], 130)
        self.assertEqual(self.g.detections_text([DETECCION, otra]), "こんにちは さようなら")
        self.assertAlmostEqual(self.g.detections_confidence([DETECCION, otra]), 0.75, places=2)

    def test_coercion_con_valor_por_defecto(self):
        self.assertEqual(self.g.float_value("1.5", 0.0), 1.5)
        self.assertEqual(self.g.float_value("no es un número", 0.25), 0.25)
        self.assertEqual(self.g.int_value("3.9", 0), 3)
        self.assertEqual(self.g.int_value(None, 7), 7)


class ComposicionEnElDetectorTests(unittest.TestCase):
    def test_el_detector_dejo_de_heredar_la_geometria(self):
        bases = [b.__name__ for b in BubbleDetector.__bases__]

        self.assertNotIn("BubbleGeometryMixin", bases)
        self.assertEqual(len(bases), 5)

    def test_la_geometria_es_inyectable(self):
        """La prueba de que es composición y no sólo un cambio de sitio."""

        class _GeometriaFalsa(DetectionGeometry):
            @staticmethod
            def area(box):
                return -1

        detector = BubbleDetector("Japonés", geometry=_GeometriaFalsa())

        self.assertEqual(detector.geometry.area((0, 0, 10, 10)), -1)

    def test_componer_mascaras_sigue_siendo_api_del_detector(self):
        """`clean_inpainting_pipeline_mixin` y los tests las llaman sobre la clase."""
        self.assertTrue(hasattr(BubbleDetector, "compose_mask"))
        self.assertTrue(hasattr(BubbleDetector, "compose_clean_mask"))


class ReferenciasHuerfanasTests(unittest.TestCase):
    """Métodos que se pasan como valor y dejaron de existir tras mover código.

    Al convertir los mixins de geometría en `DetectionGeometry` se renombraron todas
    las **llamadas** `self._to_rect(...)` a `self.geometry.to_rect(...)`, pero quedó una
    referencia sin llamar: `sort_detection_groups(grupos, self._to_rect)`, pasada como
    callback. Como estaba dentro de un `try/except Exception`, el fallo salía por el log
    como un aviso y el orden de lectura del texto libre dejó de aplicarse en silencio.
    Las métricas del banco de pruebas no lo vieron porque miden detección y OCR, no
    orden de lectura.

    Este test generaliza el caso: recorre las clases compuestas de verdad y comprueba
    que todo `self._x` usado **como valor** resuelve en la clase o se asigna en algún
    punto de su jerarquía.
    """

    CLASES = (BubbleDetector, CleanManga, TranslateManga)

    @staticmethod
    def _referencias_y_asignaciones(clase):
        referencias, asignados, llamados = {}, set(), set()
        for base in clase.__mro__:
            if base is object:
                continue
            try:
                arbol = ast.parse(textwrap.dedent(inspect.getsource(base)))
            except (OSError, TypeError, SyntaxError):
                continue
            for nodo in ast.walk(arbol):
                if (
                    isinstance(nodo, ast.Call)
                    and isinstance(nodo.func, ast.Attribute)
                    and isinstance(nodo.func.value, ast.Name)
                    and nodo.func.value.id == "self"
                ):
                    llamados.add(nodo.func.attr)
                if isinstance(nodo, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    destinos = nodo.targets if isinstance(nodo, ast.Assign) else [nodo.target]
                    for destino in destinos:
                        if isinstance(destino, ast.Attribute) and isinstance(destino.value, ast.Name) and destino.value.id == "self":
                            asignados.add(destino.attr)
                if isinstance(nodo, ast.Attribute) and isinstance(nodo.value, ast.Name) and nodo.value.id == "self":
                    referencias.setdefault(nodo.attr, base.__name__)
        return referencias, asignados, llamados

    def test_ninguna_referencia_a_metodo_inexistente(self):
        huerfanas = []
        for clase in self.CLASES:
            referencias, asignados, _ = self._referencias_y_asignaciones(clase)
            for nombre, definida_en in referencias.items():
                if nombre.startswith("__") or nombre in asignados:
                    continue
                if not hasattr(clase, nombre):
                    huerfanas.append(f"{clase.__name__}.{nombre} (en {definida_en})")

        self.assertEqual(huerfanas, [], f"self.<x> sin resolver: {huerfanas}")


if __name__ == "__main__":
    unittest.main()
