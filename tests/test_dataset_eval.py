"""Guardia de regresión sobre ``dataset_eval``.

Estas pruebas no necesitan GPU ni modelos: puntúan la predicción cruda ya guardada en cada
caso (``prediccion_base/``) contra la verdad de referencia humana y la comparan con
``baseline.json``. Lo que vigilan es que la lógica de medición y los datos sigan alineados.

Para medir una ejecución **nueva** del pipeline (eso sí requiere GPU y modelos):

    python -m parallel_manga_translator.quality.eval_dataset score \\
        --predictions <carpeta outputs de la ejecución> --fail-on-regression

o, desde pytest, exportando la ruta y dejando que se active la prueba opcional:

    PMT_EVAL_PREDICTIONS=<carpeta outputs> python -m pytest tests/test_dataset_eval.py -q
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from parallel_manga_translator.quality.eval_dataset import (
    DEFAULT_TOLERANCE,
    TRACKED_METRICS,
    VERDICT_IMPROVED,
    VERDICT_NO_DATA,
    VERDICT_REGRESSED,
    VERDICT_UNCHANGED,
    _default_dataset_dir,
    build_ground_truth_page,
    compare_reports,
    compare_summaries,
    detection_bbox,
    discover_cases,
    resolve_prediction_jsons,
    score_case,
)
from parallel_manga_translator.quality.evaluation_manager import load_ground_truth_file

DATASET_DIR = Path(os.environ.get("PMT_EVAL_DATASET") or _default_dataset_dir())
CASES = discover_cases(DATASET_DIR)


def _require_cases() -> None:
    if not CASES:
        raise unittest.SkipTest(f"No hay casos de evaluación en {DATASET_DIR}")


class DatasetIntegrityTests(unittest.TestCase):
    """El dataset describe lo que realmente contiene."""

    def setUp(self) -> None:
        _require_cases()

    def test_cada_caso_declara_sus_paginas(self) -> None:
        for case in CASES:
            with self.subTest(case=case.name):
                gt_files = case.ground_truth_files()
                declared = case.meta.get("pages") or []
                self.assertEqual(len(gt_files), len(declared))
                self.assertEqual(
                    [p.stem for p in gt_files],
                    [str(row["page"]) for row in declared],
                )

    def test_totales_del_caso_coinciden_con_la_verdad_de_referencia(self) -> None:
        for case in CASES:
            with self.subTest(case=case.name):
                total = 0
                manual = 0
                for path in case.ground_truth_files():
                    page = json.loads(path.read_text(encoding="utf-8"))
                    total += len(page["regions"])
                    manual += sum(1 for r in page["regions"] if r.get("manual"))
                totals = case.meta.get("totals") or {}
                self.assertEqual(total, totals.get("gt_regions"))
                self.assertEqual(manual, totals.get("anadidas_por_humano"))

    def test_las_paginas_existen_si_estan_copiadas(self) -> None:
        for case in CASES:
            if not case.pages_dir.is_dir():
                self.skipTest(f"{case.name}: paginas/ no está materializado (se ignora en git)")
            with self.subTest(case=case.name):
                for row in case.meta.get("pages") or []:
                    self.assertTrue(
                        (case.pages_dir / str(row["image"])).is_file(),
                        f"Falta la página {row['image']} en {case.pages_dir}",
                    )

    def test_la_prediccion_base_esta_completa(self) -> None:
        for case in CASES:
            with self.subTest(case=case.name):
                transcription, translation = resolve_prediction_jsons(case.base_prediction_dir)
                self.assertIsNotNone(transcription, "Falta Transcripción.json")
                self.assertIsNotNone(translation, "Falta Traducción.json")


class GroundTruthShapeTests(unittest.TestCase):
    """La verdad de referencia es cargable por el evaluador y tiene la caja correcta."""

    def setUp(self) -> None:
        _require_cases()

    def test_cada_pagina_carga_con_el_evaluador(self) -> None:
        for case in CASES:
            for path in case.ground_truth_files():
                with self.subTest(case=case.name, page=path.stem):
                    page = load_ground_truth_file(path)
                    self.assertEqual(page["page"], path.stem)
                    for region in page["regions"]:
                        x, y, w, h = region["bbox"]
                        self.assertGreater(w, 0)
                        self.assertGreater(h, 0)
                        self.assertGreaterEqual(x, 0)
                        self.assertGreaterEqual(y, 0)

    def test_el_origen_de_cada_region_es_conocido(self) -> None:
        for case in CASES:
            for path in case.ground_truth_files():
                page = json.loads(path.read_text(encoding="utf-8"))
                with self.subTest(case=case.name, page=path.stem):
                    for region in page["regions"]:
                        self.assertIn(region["origen"], ("pipeline", "manual"))
                        self.assertEqual(region["manual"], region["origen"] == "manual")

    def test_la_caja_de_deteccion_no_es_la_de_maquetacion(self) -> None:
        """Guarda contra la trampa del formato de bbox.

        ``manifest["regions"][i]["bbox"]`` es la caja de texto renderizado, encogida dentro
        del globo. Si alguien vuelve a usarla como caja de detección, los emparejamientos
        caen en picado. La regla: cuando hay ``ui_layout.original_region_bbox``, esa manda.
        """
        layout_box = [100, 100, 50, 40]
        balloon_box = [90, 90, 80, 70]
        region = {"bbox": layout_box, "ui_layout": {"original_region_bbox": balloon_box}}
        self.assertEqual(detection_bbox(region), balloon_box)
        self.assertEqual(detection_bbox({"bbox": layout_box}), layout_box)

    def test_las_regiones_borradas_no_entran_en_la_verdad_de_referencia(self) -> None:
        page = {
            "index": 0,
            "output_filename": "0007.jpg",
            "source_filename": "07.jpg",
            "brush_strokes": [{"x": 1}],
            "regions": [
                {"bbox": [0, 0, 10, 10], "deleted": True, "style": "dialogo", "type": "dialogue"},
                {"bbox": [20, 20, 10, 10], "style": "dialogo", "type": "dialogue"},
                {"bbox": [40, 40, 10, 10], "manual": True, "style": "onomatopeya", "type": "manual"},
            ],
        }
        built = build_ground_truth_page(page)
        self.assertEqual(built["page"], "0007")
        self.assertEqual(built["page_number"], 7)
        self.assertEqual(len(built["regions"]), 2)
        self.assertEqual(built["descartadas_por_humano"], 1)
        self.assertEqual(built["anadidas_por_humano"], 1)
        self.assertEqual(built["brush_strokes"], 1)
        # Una región añadida a mano conserva un tipo comparable con lo que predice el pipeline.
        self.assertEqual(built["regions"][1]["tipo"], "sfx")
        self.assertEqual(built["regions"][1]["tipo_original"], "manual")


class BaselineReproducibilityTests(unittest.TestCase):
    """Puntuar ``prediccion_base/`` tiene que devolver exactamente ``baseline.json``."""

    def setUp(self) -> None:
        _require_cases()

    def test_baseline_presente_y_reproducible(self) -> None:
        for case in CASES:
            with self.subTest(case=case.name):
                baseline = case.load_baseline()
                self.assertIsNotNone(baseline, f"Falta {case.baseline_path}")
                report = score_case(case)
                self.assertEqual(report["summary"], baseline["summary"])
                self.assertEqual(report["by_type"], baseline["by_type"])

    def test_el_baseline_empareja_cajas_de_verdad(self) -> None:
        """Si el IoU medio se desploma, la verdad de referencia dejó de ser comparable."""
        for case in CASES:
            with self.subTest(case=case.name):
                summary = (case.load_baseline() or {}).get("summary") or {}
                self.assertGreater(
                    summary.get("mean_iou", 0.0),
                    0.8,
                    "Las cajas emparejadas deberían ser casi idénticas; revisa detection_bbox().",
                )
                self.assertGreater(summary.get("matched_regions", 0), 0)

    def test_la_prediccion_base_no_regresa_contra_su_propio_baseline(self) -> None:
        for case in CASES:
            with self.subTest(case=case.name):
                comparison = compare_reports(case.load_baseline(), score_case(case))
                self.assertEqual(comparison["verdict"], VERDICT_UNCHANGED, comparison["regressed"])


class ComparisonLogicTests(unittest.TestCase):
    """La comparación distingue mejora, regresión y ruido en la dirección correcta."""

    @staticmethod
    def _summary(**overrides):
        base = {
            "detection_precision": 0.50,
            "detection_recall": 0.50,
            "detection_f1": 0.50,
            "mean_iou": 0.90,
            "mean_ocr_cer": 0.10,
            "mean_translation_cer": 0.40,
        }
        base.update(overrides)
        return base

    def test_subir_una_metrica_de_deteccion_es_mejora(self) -> None:
        result = compare_summaries(self._summary(), self._summary(detection_f1=0.60))
        self.assertEqual(result["metrics"]["detection_f1"]["verdict"], VERDICT_IMPROVED)
        self.assertEqual(result["verdict"], VERDICT_IMPROVED)

    def test_subir_una_tasa_de_error_es_regresion(self) -> None:
        result = compare_summaries(self._summary(), self._summary(mean_translation_cer=0.55))
        self.assertEqual(result["metrics"]["mean_translation_cer"]["verdict"], VERDICT_REGRESSED)
        self.assertEqual(result["verdict"], VERDICT_REGRESSED)

    def test_bajar_una_tasa_de_error_es_mejora(self) -> None:
        result = compare_summaries(self._summary(), self._summary(mean_ocr_cer=0.02))
        self.assertEqual(result["metrics"]["mean_ocr_cer"]["verdict"], VERDICT_IMPROVED)

    def test_un_cambio_menor_que_la_tolerancia_es_ruido(self) -> None:
        result = compare_summaries(
            self._summary(), self._summary(detection_f1=0.50 + DEFAULT_TOLERANCE / 2)
        )
        self.assertEqual(result["metrics"]["detection_f1"]["verdict"], VERDICT_UNCHANGED)
        self.assertEqual(result["verdict"], VERDICT_UNCHANGED)

    def test_una_regresion_manda_sobre_las_mejoras(self) -> None:
        result = compare_summaries(
            self._summary(), self._summary(detection_f1=0.80, mean_ocr_cer=0.30)
        )
        self.assertEqual(result["improved"], ["detection_f1"])
        self.assertEqual(result["regressed"], ["mean_ocr_cer"])
        self.assertEqual(result["verdict"], VERDICT_REGRESSED)

    def test_una_metrica_sin_datos_no_dictamina(self) -> None:
        result = compare_summaries(self._summary(mean_ocr_cer=None), self._summary(mean_ocr_cer=None))
        self.assertEqual(result["metrics"]["mean_ocr_cer"]["verdict"], VERDICT_NO_DATA)
        self.assertEqual(result["verdict"], VERDICT_UNCHANGED)

    def test_se_vigilan_todas_las_metricas_declaradas(self) -> None:
        result = compare_summaries(self._summary(), self._summary())
        self.assertEqual(tuple(result["metrics"]), TRACKED_METRICS)

    def test_la_comparacion_tambien_desglosa_por_tipo(self) -> None:
        baseline = {"summary": self._summary(), "by_type": {"dialogue": self._summary()}}
        current = {
            "summary": self._summary(detection_f1=0.70),
            "by_type": {"dialogue": self._summary(mean_translation_cer=0.60)},
        }
        comparison = compare_reports(baseline, current)
        self.assertEqual(comparison["verdict"], VERDICT_IMPROVED)
        self.assertEqual(comparison["by_type"]["dialogue"]["verdict"], VERDICT_REGRESSED)


class FreshRunRegressionTests(unittest.TestCase):
    """Puerta de calidad para una ejecución real del pipeline (requiere GPU y modelos)."""

    def test_una_ejecucion_nueva_no_regresa(self) -> None:
        predictions = os.environ.get("PMT_EVAL_PREDICTIONS")
        if not predictions:
            self.skipTest(
                "Define PMT_EVAL_PREDICTIONS con la carpeta outputs/ de una ejecución "
                "sobre dataset_eval/<caso>/paginas para activar esta comprobación."
            )
        _require_cases()
        predictions_dir = Path(predictions)
        self.assertTrue(predictions_dir.is_dir(), f"No existe {predictions_dir}")

        # Una ejecución cubre un solo caso: sus páginas, sus idiomas y sus motores. Puntuar
        # esa salida contra el resto de casos compararía cosas distintas y daría una
        # regresión falsa, así que hay que decir cuál se midió.
        wanted = os.environ.get("PMT_EVAL_CASE", "").strip()
        if wanted:
            cases = [c for c in CASES if c.name == wanted]
            self.assertTrue(cases, f"PMT_EVAL_CASE={wanted!r} no existe. Casos: {[c.name for c in CASES]}")
        elif len(CASES) == 1:
            cases = list(CASES)
        else:
            self.fail(
                "Define PMT_EVAL_CASE con el caso que ejecutaste; hay varios: "
                f"{[c.name for c in CASES]}"
            )

        regressions = []
        for case in cases:
            baseline = case.load_baseline()
            if baseline is None:
                continue
            report = score_case(case, predictions_dir=predictions_dir)
            comparison = compare_reports(baseline, report)
            if comparison["verdict"] == VERDICT_REGRESSED:
                regressions.append((case.name, comparison["regressed"]))
        self.assertEqual(regressions, [], f"Regresión frente a la validación humana: {regressions}")


if __name__ == "__main__":
    unittest.main()
