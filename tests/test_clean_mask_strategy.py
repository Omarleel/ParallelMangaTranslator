"""`CleanMaskStrategy` como colaborador, no como clase base.

Era `CleanMaskStrategyMixin`: 12 de sus 13 métodos no tocaban estado y el decimotercero
sólo necesitaba los cuatro ajustes de relleno. Heredarla servía únicamente para que
`CleanInpaintingPipelineMixin` la llamara por `self` sin declarar la dependencia.
"""

import unittest

import cv2
import numpy as np

from parallel_manga_translator.processing.clean_mask_strategy import CleanMaskStrategy

AJUSTES = dict(
    bubble_fill_whole_interior=False,
    bubble_fill_edge_margin=6,
    bubble_fill_text_dilate=2,
    bubble_fill_flat_max_rectangularity=0.82,
)


def _mascara(alto=80, ancho=100, rect=None):
    m = np.zeros((alto, ancho), dtype=np.uint8)
    if rect:
        x, y, w, h = rect
        m[y:y + h, x:x + w] = 255
    return m


class CleanMaskStrategyTests(unittest.TestCase):
    def setUp(self):
        self.e = CleanMaskStrategy(**AJUSTES)

    def test_los_ajustes_de_relleno_son_dependencias_declaradas(self):
        """Antes se leían del `self` del `CleanManga` que la heredara."""
        self.assertEqual(self.e.bubble_fill_edge_margin, 6)
        self.assertEqual(self.e.bubble_fill_text_dilate, 2)
        self.assertFalse(self.e.bubble_fill_whole_interior)
        self.assertAlmostEqual(self.e.bubble_fill_flat_max_rectangularity, 0.82)

    def test_recorta_a_los_limites_de_la_imagen(self):
        self.assertEqual(self.e.clip_rect((-5, -5, 200, 200), (80, 100)), (0, 0, 100, 80))
        self.assertEqual(self.e.clip_rect((10, 10, 20, 20), (80, 100)), (10, 10, 20, 20))

    def test_expandir_no_se_sale_de_la_imagen(self):
        x, y, w, h = self.e.expand_rect((50, 40, 20, 20), 40, 40, (80, 100))

        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, 100)
        self.assertLessEqual(y + h, 80)

    def test_mascara_desde_rectangulo(self):
        """Cubre el fallo del primer intento: llamaba a la clase vieja por su nombre."""
        m = self.e.rect_mask((10, 10, 20, 15), (80, 100))

        self.assertEqual(m.shape, (80, 100))
        self.assertEqual(int(cv2.countNonZero(m)), 20 * 15)

    def test_binarizar_deja_solo_ceros_y_255(self):
        gris = np.full((80, 100), 128, dtype=np.uint8)

        m = self.e.binary_mask(gris, (80, 100))

        self.assertTrue(set(np.unique(m)).issubset({0, 255}))

    def test_rectangularidad(self):
        rect = _mascara(rect=(10, 10, 40, 30))
        elipse = _mascara()
        cv2.ellipse(elipse, (50, 40), (20, 15), 0, 0, 360, 255, -1)

        self.assertAlmostEqual(self.e.mask_rectangularity(rect), 1.0, places=2)
        self.assertLess(self.e.mask_rectangularity(elipse), 0.85)

    def test_la_mascara_segura_se_encoge_hacia_dentro(self):
        """El margen existe para no comerse el borde negro del globo."""
        globo = _mascara(rect=(20, 20, 40, 30))

        segura = self.e.safe_bubble_mask(globo, (80, 100), margin=6)

        self.assertLess(int(cv2.countNonZero(segura)), int(cv2.countNonZero(globo)))

    def test_una_mascara_vacia_cuenta_como_rectangular(self):
        """Deliberado, no un descuido: sin tinta devuelve 1.0.

        Importa porque `build_clean_mask_for_region` compara ese valor contra
        `bubble_fill_flat_max_rectangularity`, así que una máscara vacía cae siempre en
        la rama de "no es plano".
        """
        vacia = _mascara()

        self.assertEqual(self.e.mask_rectangularity(vacia), 1.0)
        self.assertEqual(int(cv2.countNonZero(self.e.binary_mask(vacia, (80, 100)))), 0)


if __name__ == "__main__":
    unittest.main()
