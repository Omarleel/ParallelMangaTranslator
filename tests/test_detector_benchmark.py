"""Banco de detectores: lo que fija el contrato de una comparación honesta.

Tres cosas que se prueban aquí porque son justo las que hacían incomparables a dos
detectores: que se puntúe contra las dos convenciones de caja, que el subconjunto sin
sesgo se reporte aparte, y que fusiones y divisiones se cuenten (no se ven en P/R).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parallel_manga_translator.quality.detector_benchmark import (
    CRITERIA,
    BenchmarkCase,
    GtRegion,
    coverage,
    format_report,
    iou,
    match_greedy,
    score_detector,
    structural_diagnostics,
    _aviso_para,
)


def _region(page="0001", bubble=(0, 0, 100, 100), text=(20, 20, 40, 40), manual=False) -> GtRegion:
    return GtRegion(page=page, bubble_box=bubble, text_box=text, kind="dialogue", manual=manual)


def test_coverage_does_not_depend_on_the_shape_of_the_candidate() -> None:
    """Es la métrica libre de convención: '¿encontró este texto?', no '¿con qué forma?'."""
    texto = (20, 20, 40, 40)
    globo = (0, 0, 100, 100)      # mucho mayor que el texto
    bloque = (20, 20, 40, 40)     # exactamente el texto

    assert coverage(texto, globo) == pytest.approx(1.0)
    assert coverage(texto, bloque) == pytest.approx(1.0)
    # El IoU sí depende, y por eso una sola convención decide el ganador.
    assert iou(texto, globo) < 0.2
    assert iou(texto, bloque) == pytest.approx(1.0)


def test_the_same_prediction_wins_or_loses_depending_on_the_box_convention() -> None:
    """El hallazgo que motiva el banco, fijado como test."""
    region = _region(bubble=(0, 0, 100, 100), text=(30, 30, 40, 40))
    como_globo = [(0, 0, 100, 100)]
    como_texto = [(30, 30, 40, 40)]

    assert len(match_greedy(como_globo, [region], "globo", 0.5)) == 1
    assert len(match_greedy(como_globo, [region], "texto", 0.5)) == 0
    assert len(match_greedy(como_texto, [region], "texto", 0.5)) == 1
    assert len(match_greedy(como_texto, [region], "globo", 0.5)) == 0
    # Bajo cobertura las dos convenciones encuentran el texto.
    assert len(match_greedy(como_globo, [region], "cobertura", 0.8)) == 1
    assert len(match_greedy(como_texto, [region], "cobertura", 0.8)) == 1


def test_matching_is_one_to_one_and_takes_the_best_first() -> None:
    region = _region(text=(0, 0, 100, 100))
    regular = (0, 0, 60, 100)
    perfecta = (0, 0, 100, 100)

    emparejados = match_greedy([regular, perfecta], [region], "texto", 0.5)

    assert len(emparejados) == 1, "Una region no puede consumir dos predicciones."
    pred_index, _gt_index, score = emparejados[0]
    assert pred_index == 1 and score == pytest.approx(1.0)


def test_merges_and_splits_are_counted_because_precision_hides_them() -> None:
    regiones = {"0001": [_region(text=(0, 0, 40, 40)), _region(text=(60, 0, 40, 40))]}

    fusion = structural_diagnostics({"0001": [(0, 0, 100, 40)]}, regiones)
    assert fusion["fusiones"] == 1, "Una caja que se come dos bloques es un error."

    division = structural_diagnostics({"0001": [(0, 0, 20, 40), (20, 0, 20, 40)]}, regiones)
    assert division["divisiones"] == 1


def test_the_unbiased_subset_is_scored_apart() -> None:
    """Las regiones a mano son la única comparación limpia: no salen de ningún detector."""
    regiones = {"0001": [
        _region(text=(0, 0, 40, 40), manual=False),
        _region(text=(60, 0, 40, 40), manual=True),
    ]}
    # Solo acierta la region manual.
    predicciones = {"0001": [(60, 0, 40, 40)]}

    informe = score_detector(predicciones, regiones)

    assert informe["criterios"]["texto/manual"]["gt"] == 1
    assert informe["criterios"]["texto/manual"]["recall"] == pytest.approx(1.0)
    assert informe["criterios"]["texto/todas"]["recall"] == pytest.approx(0.5)
    # En el subconjunto manual la precisión no es interpretable y se deja a cero.
    assert informe["criterios"]["texto/manual"]["precision"] == 0.0


def test_every_criterion_is_reported_for_both_subsets() -> None:
    informe = score_detector({"0001": [(0, 0, 10, 10)]}, {"0001": [_region()]})
    for criterion in CRITERIA:
        for subset in ("todas", "manual"):
            assert f"{criterion}/{subset}" in informe["criterios"]


def test_a_case_loads_its_two_boxes_and_the_manual_flag(tmp_path: Path) -> None:
    case_dir = tmp_path / "xx_01"
    (case_dir / "ground_truth").mkdir(parents=True)
    (case_dir / "case.json").write_text(json.dumps({
        "name": "xx_01",
        "options": {"source_language": "Japonés"},
        "pages": [{"page": "0001", "image": "0001.png"}],
    }), encoding="utf-8")
    (case_dir / "ground_truth" / "0001.json").write_text(json.dumps({
        "page": "0001",
        "regions": [
            {"bbox": [0, 0, 100, 100], "bbox_texto": [20, 20, 40, 40], "tipo": "dialogue", "manual": True},
            {"bbox": [200, 0, 50, 50], "tipo": "sfx", "manual": False},
        ],
    }), encoding="utf-8")

    case = BenchmarkCase(case_dir)
    regiones = case.ground_truth()["0001"]

    assert case.source_language == "Japonés"
    assert regiones[0].bubble_box == (0, 0, 100, 100)
    assert regiones[0].text_box == (20, 20, 40, 40)
    assert regiones[0].manual is True
    # Sin bbox_texto se cae a la caja de globo en vez de descartar la region.
    assert regiones[1].text_box == (200, 0, 50, 50)


def test_the_report_always_carries_the_bias_warning() -> None:
    """Un numero de este banco sin su aviso se malinterpreta: el GT favorece a un lado."""
    informe = {
        "case": "xx_01", "pages": 1, "gt_regions": 1, "gt_manual_regions": 1,
        "aviso": "El subconjunto 'manual' es el unico no derivado del detector actual.",
        "detectores": {"yolo": {
            "predicted_regions": 1, "seconds": 0.1,
            "criterios": score_detector({"0001": [(0, 0, 10, 10)]}, {"0001": [_region()]})["criterios"],
            "estructura": {"fusiones": 0, "divisiones": 0},
        }},
    }
    texto = format_report(informe)
    assert "AVISO" in texto and "manual" in texto
    assert "globo/todas" in texto and "texto/todas" in texto and "cobertura/todas" in texto


def test_the_warning_depends_on_what_the_ground_truth_boxes_are() -> None:
    """Un mismo aviso para los dos tipos de caso es peor que ninguno.

    En un caso `region` el subconjunto 'todas' se puede leer; en uno `globo`/`mixta` esas
    cajas son la salida del detector que genero el caso y favorecen a ese lado.
    """
    limpio = _aviso_para("region")
    sesgado = _aviso_para("mixta")

    assert "todas" in limpio and "OJO" not in limpio
    assert "OJO" in sesgado and "manual" in sesgado
    assert sesgado != limpio
    # Un caso sin declararla no debe pasar por bueno en silencio.
    assert "desconocida" in _aviso_para("").lower() or "reconstruye" in _aviso_para("")


def test_the_case_exposes_the_box_convention(tmp_path) -> None:
    case_dir = tmp_path / "xx_02"
    (case_dir / "ground_truth").mkdir(parents=True)
    (case_dir / "case.json").write_text(json.dumps({
        "name": "xx_02",
        "options": {"source_language": "Japonés"},
        "convencion_cajas": "region",
        "pages": [],
    }), encoding="utf-8")

    assert BenchmarkCase(case_dir).box_convention == "region"

    (case_dir / "case.json").write_text(json.dumps({"name": "xx_02", "pages": []}), encoding="utf-8")
    assert BenchmarkCase(case_dir).box_convention == "desconocida"
