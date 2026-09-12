"""Promoción de un trabajo corregido de la UI a caso de `dataset_eval`.

El editor no reimplementa el formato del banco: llama al mismo constructor que el CLI
(`eval_dataset build`). Lo que se prueba aquí es el contrato del botón — qué cuenta como
"corregido", qué nombre se propone, y las dos negativas que protegen el banco: un caso
sin validación humana y un sobrescrito accidental.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from parallel_manga_translator.quality import eval_dataset
from parallel_manga_translator.ui.job_manager import JobManager, JobOptions, JobState, PageState


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((32, 32, 3), value, dtype=np.uint8))


def _region(index: int, *, manual: bool = False, deleted: bool = False, modified: bool = False) -> dict:
    return {
        "index": index,
        "bbox": [4 + index, 4, 10, 10],
        "source_bbox": [4 + index, 4, 10, 10],
        "original_text": "げんぶん",
        "translated_text": "texto",
        "style": "dialogo",
        "type": "manual" if manual else "dialogue",
        "manual": manual,
        "deleted": deleted,
        "modified": modified,
        "visible": True,
        "restore_original": False,
        "rotation_angle": 0,
    }


def _build_manager(tmp_path: Path, *, corrected: bool = True) -> tuple[JobManager, JobState]:
    jobs_root = tmp_path / "jobs"
    root_dir = jobs_root / "job-ds"
    input_dir = root_dir / "entrada"
    output_dir = root_dir / "outputs"

    pages = []
    for index in range(2):
        source = f"p{index + 1}.png"
        output_name = f"{index + 1:04d}.png"
        _write_image(input_dir / source, 40)
        _write_image(output_dir / "corregida" / output_name, 90)
        regions = [_region(0)]
        if corrected and index == 0:
            regions.append(_region(1, manual=True))
            regions.append(_region(2, deleted=True))
        pages.append(
            PageState(
                index=index,
                source_filename=source,
                output_filename=output_name,
                status="ready",
                original_path=str(input_dir / source),
                clean_path=str(output_dir / "limpieza" / output_name),
                translated_path=str(output_dir / "traduccion" / output_name),
                corrected_path=str(output_dir / "corregida" / output_name),
                corrections_path=str(output_dir / "correcciones" / f"{index + 1:04d}.json"),
                regions=regions,
            )
        )

    # Predicción cruda de la ejecución original: es lo que puntúa la línea base.
    for folder, filename in (("limpieza", "Transcripción.json"), ("traduccion", "Traducción.json")):
        path = output_dir / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"paginas": []}, ensure_ascii=False), encoding="utf-8")

    job = JobState(
        job_id="job-ds",
        title="Caso de prueba",
        root_dir=str(root_dir),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        status="ready",
        pages=pages,
        options=JobOptions(source_language="Japonés", target_language="Español"),
    )
    manager = JobManager(jobs_root=jobs_root, start_worker=False)
    manager._jobs[job.job_id] = job
    manager.manifests.save(job)
    return manager, job


def _point_dataset_at(monkeypatch, dataset_dir: Path) -> None:
    monkeypatch.setattr(eval_dataset, "resolve_dataset_dir", lambda *a, **k: dataset_dir)


def test_preview_counts_only_the_work_a_human_did(monkeypatch, tmp_path: Path) -> None:
    manager, _job = _build_manager(tmp_path)
    _point_dataset_at(monkeypatch, tmp_path / "banco")

    preview = manager.dataset_case_preview("job-ds")

    assert preview["pages"] == 2
    assert preview["corrected_pages"] == 1, "Solo la primera página tiene intervención humana."
    assert preview["anadidas_por_humano"] == 1
    assert preview["descartadas_por_humano"] == 1
    # La región borrada no cuenta como referencia: 2 vivas en la página 1 y 1 en la 2.
    assert preview["gt_regions"] == 3
    assert preview["suggested_name"] == "ja_01", "La convención del banco es idioma + número."
    assert preview["ready"] is True


def test_the_suggested_name_skips_cases_that_already_exist(monkeypatch, tmp_path: Path) -> None:
    manager, _job = _build_manager(tmp_path)
    dataset_dir = tmp_path / "banco"
    (dataset_dir / "ja_01").mkdir(parents=True)
    (dataset_dir / "ja_01" / "case.json").write_text("{}", encoding="utf-8")
    _point_dataset_at(monkeypatch, dataset_dir)

    assert manager.dataset_case_preview("job-ds")["suggested_name"] == "ja_02"


def test_building_a_case_writes_the_ground_truth_and_the_metadata(monkeypatch, tmp_path: Path) -> None:
    manager, _job = _build_manager(tmp_path)
    dataset_dir = tmp_path / "banco"
    _point_dataset_at(monkeypatch, dataset_dir)

    result = manager.export_job_to_dataset("job-ds", name="ja_07")

    case_dir = dataset_dir / "ja_07"
    assert result["case"] == "ja_07"
    assert Path(result["case_dir"]) == case_dir
    assert (case_dir / "case.json").is_file()
    assert sorted(p.name for p in (case_dir / "ground_truth").glob("*.json")) == ["0001.json", "0002.json"]
    # Las páginas de entrada se copian para poder reejecutar el pipeline sobre el caso.
    assert (case_dir / "paginas" / "0001.png").is_file()
    assert (case_dir / "referencia" / "0001.png").is_file()

    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    assert meta["origin_job"] == "job-ds"
    assert meta["totals"]["gt_regions"] == 3
    assert meta["totals"]["anadidas_por_humano"] == 1
    # El editor prometió estas cifras en la vista previa: tienen que ser las mismas.
    assert meta["totals"]["gt_regions"] == result["gt_regions"]


def test_a_job_nobody_corrected_is_refused(monkeypatch, tmp_path: Path) -> None:
    """Un caso sin validación humana mediría el pipeline contra su propia salida."""
    manager, _job = _build_manager(tmp_path, corrected=False)
    _point_dataset_at(monkeypatch, tmp_path / "banco")

    try:
        manager.export_job_to_dataset("job-ds", name="ja_01")
    except ValueError as error:
        assert "corrección manual" in str(error)
    else:
        raise AssertionError("Sin correcciones manuales no debe poder construirse un caso.")

    assert not (tmp_path / "banco" / "ja_01").exists()


def test_an_existing_case_is_never_overwritten_by_accident(monkeypatch, tmp_path: Path) -> None:
    manager, _job = _build_manager(tmp_path)
    dataset_dir = tmp_path / "banco"
    _point_dataset_at(monkeypatch, dataset_dir)
    manager.export_job_to_dataset("job-ds", name="ja_01")

    try:
        manager.export_job_to_dataset("job-ds", name="ja_01")
    except FileExistsError as error:
        assert "ja_01" in str(error)
    else:
        raise AssertionError("Regenerar un caso existente exige pedirlo explícitamente.")

    again = manager.export_job_to_dataset("job-ds", name="ja_01", overwrite=True)
    assert again["case"] == "ja_01"


def test_the_case_name_cannot_escape_the_dataset_folder(monkeypatch, tmp_path: Path) -> None:
    """El nombre acaba siendo una carpeta y llega de un formulario del navegador."""
    manager, _job = _build_manager(tmp_path)
    dataset_dir = tmp_path / "banco"
    _point_dataset_at(monkeypatch, dataset_dir)

    result = manager.export_job_to_dataset("job-ds", name="../../fuera/ja_99")

    case_dir = Path(result["case_dir"]).resolve()
    assert dataset_dir.resolve() in case_dir.parents
    assert not (tmp_path / "fuera").exists()


def test_a_running_job_is_not_frozen_mid_flight(monkeypatch, tmp_path: Path) -> None:
    manager, job = _build_manager(tmp_path)
    _point_dataset_at(monkeypatch, tmp_path / "banco")
    job.status = "processing"

    try:
        manager.export_job_to_dataset("job-ds", name="ja_01")
    except ValueError as error:
        assert "termine" in str(error)
    else:
        raise AssertionError("Un trabajo en marcha aún puede cambiar de regiones.")
