"""El maquetado no puede cambiar al mover código: valores dorados de `TextRenderer`.

`build_layout` y `resolve_manual_layout` son 315 líneas que deciden dónde y a qué tamaño
va cada texto. Alimentan dos cosas que se ven: la página que rotula el pipeline y el editor
manual de la UI, que guarda ese layout en `Layout UI` y lo reutiliza al recomponer.

`eval_runner` **no puede vigilarlas**: mide limpieza, detección y OCR, y nunca rotula. Así
que antes de partir esa clase hacía falta una red propia. Esto lo es: se congeló la salida
del código vigente sobre una matriz de casos —horizontal y vertical, uno y dos lóbulos,
tamaño fijo, CJK, acotado por la imagen, y el escalado de un layout guardado, que es el
camino real de la UI— y cualquier diferencia posterior es una regresión, no una mejora.

Si un cambio de verdad pretende alterar el maquetado, hay que regenerar el fichero **a
propósito** y mirar el diff caso por caso. Que cueste es la idea.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from parallel_manga_translator.rendering.text_renderer import TextRenderer

DORADOS = Path(__file__).parent / "fixtures" / "text_layout_golden.json"

TEXTO = "Hola mundo, esto es una prueba de maquetado con varias palabras."
CORTO = "Vale"
CJK = "これはテストです。もう一度読んでください。"


def _mascara_dos_lobulos(ancho=240, alto=120):
    """Dos globos unidos por un cuello: el caso que produce dos bloques."""
    m = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.ellipse(m, (60, 60), (48, 45), 0, 0, 360, 255, -1)
    cv2.ellipse(m, (180, 60), (48, 45), 0, 0, 360, 255, -1)
    cv2.rectangle(m, (108, 55), (132, 65), 255, -1)
    return m


def _mascara_ovalo(ancho=200, alto=120):
    m = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.ellipse(m, (100, 60), (95, 55), 0, 0, 360, 255, -1)
    return m


def casos_build(renderer: TextRenderer) -> dict:
    dos = _mascara_dos_lobulos()
    ovalo = _mascara_ovalo()
    return {
        "dialogo_corto": renderer.build_layout((10, 10, 180, 90), CORTO, "dialogo"),
        "dialogo_largo": renderer.build_layout((10, 10, 180, 90), TEXTO, "dialogo"),
        "dialogo_estrecho": renderer.build_layout((0, 0, 70, 140), TEXTO, "dialogo"),
        "narracion": renderer.build_layout((10, 10, 180, 90), TEXTO, "narracion"),
        "onomatopeya": renderer.build_layout((10, 10, 120, 60), "BOOM", "onomatopeya"),
        "tamano_fijo": renderer.build_layout((10, 10, 180, 90), TEXTO, "dialogo", requested_font_size=14),
        "vertical_por_rotacion": renderer.build_layout((70, 20, 80, 150), "ANGLE", "dialogo", rotation_angle=19),
        "rotado": renderer.build_layout((10, 10, 180, 90), TEXTO, "dialogo", rotation_angle=19),
        "clip_ovalo": renderer.build_layout((0, 0, 200, 120), TEXTO, "dialogo", clip_mask=ovalo),
        "clip_dos_lobulos": renderer.build_layout((0, 0, 240, 120), TEXTO, "dialogo", clip_mask=dos),
        "clip_dos_lobulos_rtl": renderer.build_layout(
            (0, 0, 240, 120), TEXTO, "dialogo", clip_mask=dos, reading_order_right_to_left=True
        ),
        "alineado_izq_arriba": renderer.build_layout(
            (10, 10, 180, 90), TEXTO, "dialogo", text_align="left", vertical_align="top"
        ),
        "interlineado_y_offset": renderer.build_layout(
            (10, 10, 180, 90), TEXTO, "dialogo", line_spacing_factor=1.4, text_offset_x=6, text_offset_y=-4
        ),
        "cjk": renderer.build_layout((10, 10, 180, 90), CJK, "dialogo"),
        "acotado_por_imagen": renderer.build_layout(
            (150, 80, 180, 90), TEXTO, "dialogo", image_shape=(140, 260)
        ),
    }


def casos_resolve(renderer: TextRenderer, previos: dict) -> dict:
    guardado = previos["dialogo_largo"]
    return {
        "sin_layout_previo": renderer.resolve_manual_layout((10, 10, 180, 90), TEXTO, "dialogo"),
        "con_layout_previo": renderer.resolve_manual_layout(
            (10, 10, 180, 90), TEXTO, "dialogo", ui_layout=guardado
        ),
        "layout_previo_escalado": renderer.resolve_manual_layout(
            (10, 10, 240, 120), TEXTO, "dialogo", ui_layout=guardado
        ),
        "layout_previo_texto_nuevo": renderer.resolve_manual_layout(
            (10, 10, 180, 90), "Texto distinto, mas corto.", "dialogo", ui_layout=guardado
        ),
        "con_offsets": renderer.resolve_manual_layout(
            (10, 10, 180, 90), TEXTO, "dialogo",
            text_align="right", vertical_align="bottom",
            line_spacing_factor=1.25, text_offset_x=-5, text_offset_y=7,
        ),
        "tamano_fijo": renderer.resolve_manual_layout(
            (10, 10, 180, 90), TEXTO, "dialogo", requested_font_size=16
        ),
        "rotado": renderer.resolve_manual_layout(
            (10, 10, 180, 90), TEXTO, "dialogo", rotation_angle=12.5
        ),
        "dos_lobulos_previo": renderer.resolve_manual_layout(
            (0, 0, 240, 120), TEXTO, "dialogo", ui_layout=previos["clip_dos_lobulos"]
        ),
    }


def _normalizado(valor):
    """Tuplas y listas son lo mismo una vez el layout viaja a JSON, que es su destino."""
    return json.loads(json.dumps(valor, ensure_ascii=False))


class MaquetadoDoradoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dorados = json.loads(DORADOS.read_text(encoding="utf-8"))
        cls.renderer = TextRenderer(absolute_min_font_size=7, inner_margin_ratio=0.03)
        cls.build = casos_build(cls.renderer)
        cls.resolve = casos_resolve(cls.renderer, cls.build)

    def test_build_layout_no_cambia(self):
        esperados = self.dorados["build_layout"]
        self.assertEqual(sorted(esperados), sorted(self.build), "la matriz de casos no coincide con el fichero")
        for nombre, esperado in esperados.items():
            with self.subTest(caso=nombre):
                self.assertEqual(_normalizado(self.build[nombre]), esperado)

    def test_resolve_manual_layout_no_cambia(self):
        esperados = self.dorados["resolve_manual_layout"]
        self.assertEqual(sorted(esperados), sorted(self.resolve), "la matriz de casos no coincide con el fichero")
        for nombre, esperado in esperados.items():
            with self.subTest(caso=nombre):
                self.assertEqual(_normalizado(self.resolve[nombre]), esperado)

    def test_la_matriz_cubre_las_ramas_que_importan(self):
        """Una red que solo pasara por el camino fácil no protegería nada."""
        modos = {lay.get("writing_mode") for lay in self.build.values()}
        bloques = {len(lay.get("blocks") or []) for lay in self.build.values()}

        self.assertIn("horizontal", modos)
        self.assertIn("vertical_chars", modos, "falta el modo de caracteres verticales")
        self.assertIn(2, bloques, "falta un globo partido en dos lóbulos")
        self.assertTrue(
            any(lay.get("uses_clip_mask") for lay in self.build.values()),
            "ningún caso usa máscara de globo",
        )


if __name__ == "__main__":
    unittest.main()
