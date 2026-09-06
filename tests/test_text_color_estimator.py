"""Estimación del color original del texto y de su contorno.

Lo que había antes miraba la imagen **ya limpia** —de la que el texto original fue
borrado— y elegía entre negro y blanco según el brillo medio. Esto mira la imagen original
a través de la máscara de tinta que el pipeline ya calcula.

Aviso que vale más que los tests: **el banco de pruebas no cubre esto**. `eval_runner` se
detiene antes de traducir y rotular, así que un color mal estimado no mueve ninguna métrica
y sólo se ve en la página. De ahí que el estimador devuelva `None` en cuanto duda, y que
estos tests insistan tanto en cuándo NO debe afirmar un color.
"""

import unittest

import cv2
import numpy as np

from parallel_manga_translator.rendering.text_color_estimator import estimar_colores

LADO = 120

#: En BGR, que es como trabaja OpenCV. El estimador debe devolverlos en RGB.
BGR_ROJO = (0, 0, 255)
BGR_BLANCO = (255, 255, 255)
BGR_NEGRO = (0, 0, 0)
BGR_GRIS = (128, 128, 128)


def _lienzo(color_bgr) -> np.ndarray:
    return np.full((LADO, LADO, 3), color_bgr, np.uint8)


def _trazo(mascara_o_imagen, color=None, grosor=7):
    """Dibuja una barra gruesa central: hace de 'texto' con área suficiente."""
    cv2.line(mascara_o_imagen, (25, 60), (95, 60), color if color is not None else 255, grosor)


def _mascara_del_trazo(grosor=7) -> np.ndarray:
    mascara = np.zeros((LADO, LADO), np.uint8)
    _trazo(mascara, 255, grosor)
    return mascara


class RellenoTests(unittest.TestCase):
    def test_tinta_negra_sobre_papel_blanco(self):
        imagen = _lienzo(BGR_BLANCO)
        _trazo(imagen, BGR_NEGRO)

        colores = estimar_colores(imagen, _mascara_del_trazo())

        self.assertIsNotNone(colores)
        self.assertEqual(colores.relleno, (0, 0, 0))

    def test_el_color_sale_en_rgb_no_en_bgr(self):
        """El fallo más fácil de colar: OpenCV es BGR y PIL dibuja en RGB.

        Con rojo puro, invertir canales da azul puro y nada falla ruidosamente: sólo salen
        los rótulos del color equivocado.
        """
        imagen = _lienzo(BGR_BLANCO)
        _trazo(imagen, BGR_ROJO)

        colores = estimar_colores(imagen, _mascara_del_trazo())

        self.assertEqual(colores.relleno, (255, 0, 0), "el rojo salió como azul: canales invertidos")

    def test_informa_de_cuantos_pixeles_lo_sostienen(self):
        colores = estimar_colores(_lienzo(BGR_BLANCO), _mascara_del_trazo())

        self.assertGreater(colores.pixeles_tinta, 0)


class ContornoTests(unittest.TestCase):
    def test_detecta_un_contorno_realmente_distinto(self):
        """Texto blanco perfilado en negro sobre fondo gris: el caso clásico de un SFX."""
        imagen = _lienzo(BGR_GRIS)
        _trazo(imagen, BGR_NEGRO, grosor=15)   # el contorno, más ancho
        _trazo(imagen, BGR_BLANCO, grosor=7)   # la tinta, dentro

        colores = estimar_colores(imagen, _mascara_del_trazo(grosor=7))

        self.assertEqual(colores.relleno, (255, 255, 255))
        self.assertIsNotNone(colores.contorno)
        self.assertEqual(colores.contorno, (0, 0, 0))

    def test_no_inventa_contorno_cuando_solo_hay_papel(self):
        """Sin esto, todo texto normal saldría 'perfilado' del color del fondo."""
        imagen = _lienzo(BGR_BLANCO)
        _trazo(imagen, BGR_NEGRO)

        colores = estimar_colores(imagen, _mascara_del_trazo())

        self.assertIsNone(colores.contorno)

    def test_la_zona_segura_acota_el_muestreo(self):
        """Fuera del globo hay arte; medir el contorno allí da un color inventado."""
        imagen = _lienzo(BGR_BLANCO)
        _trazo(imagen, BGR_NEGRO)
        zona = np.zeros((LADO, LADO), np.uint8)
        cv2.rectangle(zona, (20, 50), (100, 70), 255, -1)

        colores = estimar_colores(imagen, _mascara_del_trazo(), zona_segura=zona)

        self.assertIsNotNone(colores)
        self.assertEqual(colores.relleno, (0, 0, 0))


class CuandoNoDebeAfirmarNadaTests(unittest.TestCase):
    def test_sin_tinta_suficiente_no_estima(self):
        mascara = np.zeros((LADO, LADO), np.uint8)
        cv2.circle(mascara, (60, 60), 2, 255, -1)   # un puñado de píxeles

        self.assertIsNone(estimar_colores(_lienzo(BGR_BLANCO), mascara))

    def test_una_mascara_de_otro_tamano_no_estima(self):
        """Prefiere no afirmar a alinear mal la máscara con la imagen."""
        self.assertIsNone(estimar_colores(_lienzo(BGR_BLANCO), np.zeros((10, 10), np.uint8)))

    def test_una_imagen_en_escala_de_grises_no_estima(self):
        self.assertIsNone(estimar_colores(np.zeros((LADO, LADO), np.uint8), _mascara_del_trazo()))

    def test_sin_mascara_no_estima(self):
        self.assertIsNone(estimar_colores(_lienzo(BGR_BLANCO), None))


class CableadoEnElPipelineTests(unittest.TestCase):
    """Que la estimación llegue del paso 1 al rotulado, y sólo si se pide."""

    def _traductor(self, **quality):
        from parallel_manga_translator.config.app_config import QualityConfig
        from parallel_manga_translator.processing.translate_manga import TranslateManga

        class _OcrFalso:
            def extract_texts(self, imagenes):
                return [""] * len(imagenes)

        class _TraductorFalso:
            ultimas_regiones = ()

            def traducir_textos(self, textos, **kwargs):
                return list(textos)

            traducir_textos_tradicional = traducir_textos

            def traducir_texto(self, texto):
                return texto

            def character_memory_snapshot(self):
                return {"characters": []}

            def analyze_character_memory(self, *a, **k):
                return []

        return TranslateManga(
            "en", "es",
            quality_config=QualityConfig(**quality),
            ocr_manager=_OcrFalso(),
            translator_manager=_TraductorFalso(),
        )

    def _region(self):
        from parallel_manga_translator.models.processing_models import TextRegion

        imagen = _lienzo(BGR_BLANCO)
        _trazo(imagen, BGR_ROJO)
        return imagen, TextRegion(
            bbox=(20, 50, 80, 20),
            text_bbox=(20, 50, 80, 20),
            mask=np.full((LADO, LADO), 255, np.uint8),
            clean_mask=_mascara_del_trazo(),
            kind="dialogue",
            confidence=0.9,
        )

    def test_apagado_no_anota_nada(self):
        """Cambia la salida visible y el banco no lo cubre: no puede ser el defecto."""
        from parallel_manga_translator.config.app_config import QualityConfig

        self.assertFalse(QualityConfig().estimate_text_colors)
        traductor = self._traductor()
        imagen, region = self._region()
        traductor.ultimas_regiones = [region]

        traductor._estimar_colores_de_regiones(imagen)

        self.assertNotIn("text_fill_color", region.metadata)

    def test_encendido_anota_el_color_de_la_tinta(self):
        traductor = self._traductor(estimate_text_colors=True)
        imagen, region = self._region()
        traductor.ultimas_regiones = [region]

        traductor._estimar_colores_de_regiones(imagen)

        self.assertEqual(region.metadata["text_fill_color"], [255, 0, 0])

    def test_rotular_traslada_los_colores_al_renderizador(self):
        traductor = self._traductor(estimate_text_colors=True)
        _imagen, region = self._region()
        region.metadata["text_fill_color"] = [255, 0, 0]
        traductor.ultimas_regiones = [region]
        recogido = {}

        def _render(imagen_limpia, cuadros, textos, **kwargs):
            recogido.update(kwargs)
            return imagen_limpia

        traductor.text_renderer.render = _render
        traductor.rotular(_lienzo(BGR_BLANCO), [region.bbox], ["hola"])

        self.assertEqual(recogido["text_colors"], [[255, 0, 0]])
        self.assertEqual(recogido["stroke_colors"], [None])


if __name__ == "__main__":
    unittest.main()
