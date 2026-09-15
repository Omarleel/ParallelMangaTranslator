"""La fuente de regiones se elige por trabajo desde la UI.

El defecto es `yolo` y conviene saber por qué, porque estuvo en `rtdetr` un rato: **yolo es
el único que da el polígono del globo**, y la limpieza está construida sobre él. RT-DETR
localiza el texto bastante mejor pero sólo da cajas, y como fuente de regiones limpia peor
sobre material real. Su sitio es `ocr.detection_engine`, no éste.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parallel_manga_translator.detection.region_source_factory import REGION_SOURCES
from parallel_manga_translator.ui.job_manager import JobManager, JobOptions, JobState, normalize_region_source

STATIC = Path(__file__).resolve().parents[1] / "parallel_manga_translator" / "ui" / "static"


def test_the_default_is_the_one_that_gives_the_bubble_polygon() -> None:
    """La limpieza depende del polígono del globo, y sólo yolo lo da."""
    assert JobOptions().region_source == "yolo"


def test_the_value_is_validated_against_the_registry_not_a_copy_of_the_list() -> None:
    for nombre in REGION_SOURCES:
        assert normalize_region_source(nombre) == nombre
    assert normalize_region_source("") == "yolo", "Un formulario vacío no debe romper el trabajo."
    with pytest.raises(ValueError) as error:
        normalize_region_source("detector_inventado")
    assert "detector_inventado" in str(error.value)


def test_the_choice_reaches_the_pipeline_config(tmp_path: Path) -> None:
    """`quality` no se replicaba por trabajo: la opción se perdía por el camino."""
    manager = JobManager(jobs_root=tmp_path / "jobs", start_worker=False)
    job = JobState(
        job_id="job-rs",
        title="x",
        root_dir=str(tmp_path / "jobs" / "job-rs"),
        input_dir=str(tmp_path / "entrada"),
        output_dir=str(tmp_path / "salida"),
        options=JobOptions(region_source="comic_text_detector"),
    )

    config = manager._build_config_for_job(job)

    assert config.quality.region_source == "comic_text_detector"


def test_a_job_without_the_option_keeps_the_current_detector(tmp_path: Path) -> None:
    manager = JobManager(jobs_root=tmp_path / "jobs", start_worker=False)
    job = JobState(
        job_id="job-rs2",
        title="x",
        root_dir=str(tmp_path / "jobs" / "job-rs2"),
        input_dir=str(tmp_path / "entrada"),
        output_dir=str(tmp_path / "salida"),
        options=JobOptions(),
    )

    assert manager._build_config_for_job(job).quality.region_source == "yolo"


def test_the_form_offers_the_choice_and_sends_it() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'id="regionSource"' in html and 'name="region_source"' in html
    assert 'value="comic_text_detector"' in html
    # El aviso importa tanto como el desplegable: sin el, se elige a ciegas. Y lo que hay
    # que explicar es que esta opcion NO es donde se mejora la localizacion del texto.
    assert "polígono del globo" in html
    assert "OCR de" in html and "detección" in html, "Debe decir dónde SÍ se cambia el localizador."
    assert "data.append('region_source', regionSource?.value || 'yolo');" in javascript


def test_the_work_view_names_the_source_only_when_it_is_not_the_default() -> None:
    """Si un trabajo sale raro, lo primero que hay que poder ver es que se cambió esto."""
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "function regionSourceLabel(value)" in javascript
    assert "opts.region_source && opts.region_source !== 'yolo'" in javascript


def test_the_endpoint_accepts_the_field() -> None:
    app_source = (
        Path(__file__).resolve().parents[1] / "parallel_manga_translator" / "ui" / "app.py"
    ).read_text(encoding="utf-8")

    assert 'region_source: str = Form(default="yolo")' in app_source
    assert "region_source=normalize_region_source(region_source)," in app_source
