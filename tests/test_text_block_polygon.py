"""Derivación del polígono del bloque de texto dentro de una caja.

Medido sobre `dataset_eval`, de las regiones que el detector de globos no ve **casi
ninguna es un globo**: son columnas de texto vertical dibujadas sobre el arte. Por eso
aquí no se busca el interior de un globo sino la tinta, y la garantía que protegen estas
pruebas es que el resultado **nunca sea un rectángulo**: uno llena el recorte de OCR de
arte vecino.
"""

import unittest

import cv2
import numpy as np

from parallel_manga_translator.quality.text_block_polygon import text_block_polygon

#: Caja de trabajo en formato x, y, w, h.
CAJA = (140, 120, 120, 200)


def _pagina(fondo=235):
    return np.full((400, 400, 3), fondo, dtype=np.uint8)


def _con_columna_de_texto(fondo=235, tinta=30):
    """Página clara con una columna estrecha de glifos oscuros dentro de la caja."""
    img = _pagina(fondo)
    for i in range(6):
        y = 130 + i * 30
        cv2.rectangle(img, (170, y), (210, y + 22), (tinta, tinta, tinta), -1)
    return img


class TextBlockPolygonTests(unittest.TestCase):
    def test_devuelve_la_envolvente_del_texto_no_la_caja(self):
        puntos, motivo = text_block_polygon(_con_columna_de_texto(), CAJA)

        self.assertEqual(motivo, "ok")
        self.assertIsNotNone(puntos)
        area = cv2.contourArea(puntos.astype(np.int32).reshape(-1, 1, 2))
        area_caja = CAJA[2] * CAJA[3]
        # La columna ocupa una franja estrecha: si saliera la caja entera, sería un
        # rectángulo y estaríamos en el fallo que este módulo existe para evitar.
        self.assertLess(area, area_caja * 0.60)
        self.assertGreater(area, 0)

    def test_el_poligono_queda_contenido_en_la_caja(self):
        puntos, _ = text_block_polygon(_con_columna_de_texto(), CAJA)
        x, y, w, h = CAJA

        self.assertTrue((puntos[:, 0] >= x).all() and (puntos[:, 0] <= x + w).all())
        self.assertTrue((puntos[:, 1] >= y).all() and (puntos[:, 1] <= y + h).all())

    def test_funciona_con_texto_claro_sobre_fondo_oscuro(self):
        """La tinta se mide por distancia de color, así que no hay que elegir polaridad."""
        puntos, motivo = text_block_polygon(_con_columna_de_texto(fondo=25, tinta=240), CAJA)

        self.assertEqual(motivo, "ok")
        self.assertIsNotNone(puntos)

    def test_rechaza_una_zona_plana(self):
        puntos, motivo = text_block_polygon(_pagina(), CAJA)

        self.assertIsNone(puntos)
        self.assertIn(motivo, {"sin tinta", "tinta insuficiente", "solo motas"})

    def test_rechaza_una_mancha_que_ocupa_la_caja(self):
        img = _pagina()
        cv2.rectangle(img, (140, 120), (260, 320), (20, 20, 20), -1)

        puntos, motivo = text_block_polygon(img, CAJA)

        self.assertIsNone(puntos)

    def test_una_mota_suelta_no_infla_la_envolvente(self):
        """Una salpicadura del arte en la esquina no debe arrastrar el polígono."""
        img = _con_columna_de_texto()
        cv2.circle(img, (250, 310), 2, (0, 0, 0), -1)

        puntos, motivo = text_block_polygon(img, CAJA)

        self.assertEqual(motivo, "ok")
        self.assertLess(puntos[:, 0].max(), 240)

    def test_caja_degenerada(self):
        puntos, motivo = text_block_polygon(_con_columna_de_texto(), (10, 10, 2, 2))

        self.assertIsNone(puntos)
        self.assertEqual(motivo, "caja degenerada")


if __name__ == "__main__":
    unittest.main()
