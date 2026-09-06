"""El contrato del relleno de globos, hecho explícito y comprobable.

`quality.bubble_fill_strategy` elige el primer candidato, no el resultado: el verificador
visual puede sustituirlo. Eso no estaba escrito en ninguna parte —la documentación decía
"usa siempre"— y por eso un test llegó a medir una configuración que nadie ejecuta.
"""

import unittest

import numpy as np

from parallel_manga_translator.processing.bubble_fill_policy import strategy_honored


class ContratoDeRellenoTests(unittest.TestCase):
    def test_solid_se_respeta_solo_si_el_relleno_fue_solido(self):
        self.assertTrue(strategy_honored(requested="solid", method="solid_color"))
        self.assertTrue(
            strategy_honored(requested="solid", method="solid_color:best_failed_visual_score"),
            "sigue siendo relleno sólido aunque el verificador lo marcara",
        )
        self.assertFalse(strategy_honored(requested="solid", method="configured_inpaint:aot"))

    def test_inpaint_se_respeta_solo_si_hubo_inpaint(self):
        self.assertTrue(strategy_honored(requested="inpaint", method="configured_inpaint"))
        self.assertTrue(strategy_honored(requested="inpaint", method="configured_inpaint:lama_mpe"))
        self.assertFalse(strategy_honored(requested="inpaint", method="solid_color"))

    def test_auto_nunca_se_incumple(self):
        """Delega en el contenido del fondo por definición: no hay nada que incumplir."""
        self.assertTrue(strategy_honored(requested="auto", method="solid_color"))
        self.assertTrue(strategy_honored(requested="auto", method="configured_inpaint:lama_mpe"))

    def test_una_estrategia_desconocida_no_se_da_por_incumplida(self):
        self.assertTrue(strategy_honored(requested="", method="solid_color"))
        self.assertTrue(strategy_honored(requested="lo-que-sea", method="solid_color"))


class ConstanciaEnLaMetadataTests(unittest.TestCase):
    """La sustitución tiene que verse en la salida, no sólo en el código."""

    @staticmethod
    def _globo_sobre_trama():
        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        for y in range(18, 102):
            for x in range(18, 102):
                v = 170 + ((x * 7 + y * 5) % 70)
                image[y, x] = (v, v, v)
        image[48:62, 44:70] = 0
        mask = np.zeros((120, 120), dtype=np.uint8)
        mask[18:102, 18:102] = 255
        return image, mask

    def test_la_region_registra_que_la_estrategia_no_se_respeto(self):
        import sys

        sys.path.insert(0, "tests")
        from test_quality_systems import COMO_PRODUCCION, _cleaner_de_prueba
        from parallel_manga_translator.models.processing_models import TextRegion

        image, mask = self._globo_sobre_trama()
        region = TextRegion(
            bbox=(18, 18, 84, 84),
            text_bbox=(44, 48, 26, 14),
            mask=mask,
            kind="dialogue",
            detections_count=1,
            metadata={},
        )
        cleaner = _cleaner_de_prueba(
            verificador_visual=COMO_PRODUCCION,
            bubble_fill_strategy="solid",
            bubble_fill_background_std_threshold=4.0,
        )

        [prepared] = cleaner.mask_strategy.attach_clean_masks(image, [region])
        cleaner._fill_bubble_interiors(image, [prepared])

        self.assertEqual(prepared.metadata["bubble_fill_strategy"], "solid")
        self.assertFalse(
            prepared.metadata["bubble_fill_strategy_honored"],
            "pediste solid y el verificador puso un inpaint: tiene que constar",
        )


if __name__ == "__main__":
    unittest.main()
