"""Taxonomía de errores: en qué etapa falló y de qué clase fue el fallo.

En un pipeline de IA la mayoría de los fallos son de recursos o de proveedor externo
—transitorios— pero llegan todos como `Exception` y se confunden con errores de código.
La etapa dice *dónde*; `FailureKind` dice *si merece la pena reintentar*.
"""

import logging
import unittest

from parallel_manga_translator.infrastructure.error_handling import (
    FailureKind,
    PageFailureReport,
    PipelineStage,
    StageProcessingError,
    classify_failure,
    processing_stage,
)


class _SinMemoria(Exception):
    """Stub con el nombre de la excepción de CUDA, que no se importa aquí."""

    __name__ = "OutOfMemoryError"


class TooManyRequests(Exception):
    """Stub con el nombre de la excepción de deep_translator."""


class TaxonomiaDeEtapasTests(unittest.TestCase):
    def test_los_valores_no_cambian_los_informes_ya_escritos(self):
        """El enum hereda de `str` para que los `fallidas/*.error.json` sigan siendo válidos."""
        self.assertEqual(PipelineStage.LIMPIEZA, "limpieza")
        self.assertEqual(PipelineStage.GUARDAR_LIMPIEZA, "guardar_limpieza")
        self.assertEqual(PipelineStage.OCR_TRADUCCION_RENDER, "ocr_traduccion_render")
        self.assertEqual(PipelineStage.GUARDAR_TRADUCCION, "guardar_traduccion")

    def test_la_etapa_sobrevive_al_envoltorio(self):
        logger = logging.getLogger("test")
        with self.assertRaises(StageProcessingError) as ctx:
            with processing_stage(PipelineStage.LIMPIEZA, logger=logger, page_index=3, filename="0003.jpg"):
                raise ValueError("mascara vacia")

        self.assertEqual(ctx.exception.stage, "limpieza")
        self.assertEqual(ctx.exception.original_error_type, "ValueError")


class ClasificacionDeFallosTests(unittest.TestCase):
    def test_memoria(self):
        self.assertEqual(classify_failure(MemoryError("sin memoria")), FailureKind.MEMORIA)
        self.assertEqual(
            classify_failure(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")),
            FailureKind.MEMORIA,
        )

    def test_proveedor_externo(self):
        self.assertEqual(classify_failure(TooManyRequests("429")), FailureKind.PROVEEDOR_EXTERNO)

    def test_modelo_no_disponible(self):
        self.assertEqual(
            classify_failure(FileNotFoundError("models/inpainting/lama.ckpt")),
            FailureKind.MODELO_NO_DISPONIBLE,
        )

    def test_entrada_invalida(self):
        self.assertEqual(classify_failure(ValueError("bbox degenerada")), FailureKind.ENTRADA_INVALIDA)

    def test_lo_que_no_encaja_no_se_fuerza(self):
        """Inventar una familia para todo sería peor que admitir que no se sabe."""
        self.assertEqual(classify_failure(ZeroDivisionError("x/0")), FailureKind.DESCONOCIDO)


class InformeDeFalloTests(unittest.TestCase):
    def test_clasifica_la_excepcion_original_no_el_envoltorio(self):
        """Un `StageProcessingError` siempre sería 'desconocido' y no diría nada útil."""
        original = MemoryError("CUDA out of memory")
        envuelto = StageProcessingError(PipelineStage.LIMPIEZA, original, page_index=0, filename="0001.jpg")

        informe = PageFailureReport.from_exception(envuelto, page_index=0, filename="0001.jpg")

        self.assertEqual(informe.stage, "limpieza")
        self.assertEqual(informe.error_type, "MemoryError")
        self.assertEqual(informe.failure_kind, FailureKind.MEMORIA.value)

    def test_un_fallo_sin_etapa_queda_marcado_como_desconocido(self):
        informe = PageFailureReport.from_exception(
            ZeroDivisionError("x/0"), page_index=1, filename="0002.jpg"
        )

        self.assertEqual(informe.stage, "unknown")
        self.assertEqual(informe.failure_kind, FailureKind.DESCONOCIDO.value)

    def test_el_informe_sigue_siendo_serializable(self):
        informe = PageFailureReport.from_exception(
            ValueError("mala caja"), page_index=0, filename="0001.jpg"
        )
        datos = informe.to_dict()

        self.assertEqual(datos["failure_kind"], FailureKind.ENTRADA_INVALIDA.value)
        self.assertIn("traceback", datos)


if __name__ == "__main__":
    unittest.main()
