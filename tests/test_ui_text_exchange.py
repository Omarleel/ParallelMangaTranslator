"""Exportar el guión del trabajo entero, editarlo fuera y volver a importarlo.

Lo que se vigila aquí es lo que hace útil la función: que la exportación lleve las cajas
—sin ellas se traduce sin saber cuánto espacio hay— y que la importación empareje por
identidad. Emparejar por posición pasaría estos tests si las regiones estuvieran en orden;
por eso casi todos los casos reordenan, borran o mezclan regiones a propósito.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from parallel_manga_translator.ui import manual_renderer
from parallel_manga_translator.ui.job_manager import JobManager, JobState, PageState
from parallel_manga_translator.ui.page_regions import merge_page_regions
from parallel_manga_translator.ui.text_exchange import (
    FORMAT_NAME,
    build_job_export,
    export_filename,
    plan_job_import,
)


class _SolidTextRenderer:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def render_with_layouts(self, image, boxes, texts, **kwargs):
        result = image.copy()
        for x, y, w, h in boxes:
            result[y : y + h, x : x + w] = 220
        return result


def _write_uniform(path: Path, value: int, size: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((size, size, 3), value, dtype=np.uint8))


def _transcription(page_no: int, items) -> dict:
    return {
        "Transcripción": [
            {
                "Página": page_no,
                "Globos de texto": [
                    {
                        "Índice": indice,
                        "Coordenadas": [[x, y], [x + w, y + h]],
                        "Texto": texto,
                        "Estilo": "dialogo",
                    }
                    for indice, (x, y, w, h), texto in items
                ],
            }
        ]
    }


def _build_job(tmp_path: Path, pages: int = 2) -> tuple[JobManager, JobState]:
    jobs_root = tmp_path / "jobs"
    root_dir = jobs_root / "job-id"
    input_dir = root_dir / "entrada"
    output_dir = root_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_states = []
    for index in range(pages):
        nombre = f"{index + 1:04d}.png"
        original = input_dir / nombre
        clean = output_dir / "limpieza" / nombre
        translated = output_dir / "traduccion" / nombre
        _write_uniform(original, 40)
        _write_uniform(clean, 100)
        _write_uniform(translated, 160)
        page_states.append(
            PageState(
                index=index,
                source_filename=nombre,
                output_filename=nombre,
                status="ready",
                original_path=str(original),
                clean_path=str(clean),
                translated_path=str(translated),
                corrected_path=str(output_dir / "corregida" / nombre),
                corrections_path=str(output_dir / "correcciones" / f"{index + 1:04d}.json"),
            )
        )

    job = JobState(
        job_id="job-id",
        title="Capítulo 01",
        root_dir=str(root_dir),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        status="ready",
        pages=page_states,
    )
    for page in job.pages:
        page.regions = merge_page_regions(
            job,
            page,
            _transcription(
                page.index + 1,
                [(0, (5, 5, 20, 12), f"one p{page.index + 1}"), (1, (30, 5, 20, 12), f"two p{page.index + 1}")],
            ),
            {"Traducción": []},
        )
    manager = JobManager(jobs_root=jobs_root, start_worker=False)
    manager._jobs[job.job_id] = job
    manager.manifests.save(job)
    return manager, job


def test_export_carries_every_form_of_the_box(tmp_path: Path) -> None:
    _manager, job = _build_job(tmp_path, pages=1)
    export = build_job_export(job, {0: (800, 1200)})

    assert export["formato"] == FORMAT_NAME
    assert export["titulo"] == "Capítulo 01"
    pagina = export["paginas"][0]
    assert (pagina["ancho"], pagina["alto"]) == (800, 1200)
    region = pagina["regiones"][0]
    # Traducir sin saber cuánto sitio hay es traducir a ciegas: van las dos formas de la
    # caja, la del pipeline y la del editor.
    assert region["bbox"] == [5, 5, 20, 12]
    assert region["coordenadas"] == [[5, 5], [25, 17]]
    assert region["bbox_deteccion"] == [5, 5, 20, 12]
    assert region["region_uid"] == "p0001r0000"
    assert region["texto_original"] == "one p1"


def test_import_matches_by_identity_not_by_position(tmp_path: Path) -> None:
    _manager, job = _build_job(tmp_path, pages=1)
    # El archivo llega con las regiones al revés, como si alguien las hubiera ordenado por
    # texto en una hoja de cálculo. Por posición esto asignaría cada traducción al globo
    # equivocado y nadie se enteraría.
    payload = {
        "formato": FORMAT_NAME,
        "paginas": [
            {
                "indice": 0,
                "regiones": [
                    {"region_uid": "p0001r0001", "indice": 0, "texto_traducido": "DOS"},
                    {"region_uid": "p0001r0000", "indice": 1, "texto_traducido": "UNO"},
                ],
            }
        ],
    }
    plan = plan_job_import(job, payload)

    assert plan.changed_regions == 2
    por_uid = {region["region_uid"]: region for region in plan.pages[0].regions}
    assert por_uid["p0001r0000"]["translated_text"] == "UNO"
    assert por_uid["p0001r0001"]["translated_text"] == "DOS"
    assert all(region["modified"] for region in plan.pages[0].regions)


def test_import_only_touches_the_fields_present_in_the_file(tmp_path: Path) -> None:
    _manager, job = _build_job(tmp_path, pages=1)
    payload = {
        "formato": FORMAT_NAME,
        "paginas": [{"indice": 0, "regiones": [{"region_uid": "p0001r0000", "texto_traducido": "hola"}]}],
    }
    plan = plan_job_import(job, payload)
    region = next(r for r in plan.pages[0].regions if r["region_uid"] == "p0001r0000")

    assert region["translated_text"] == "hola"
    # Un archivo que solo trae traducción no puede borrar la transcripción.
    assert region["original_text"] == "one p1"
    assert plan.pages[0].changed == 1


def test_import_reports_what_it_cannot_match(tmp_path: Path) -> None:
    """Una identidad desconocida se informa; jamás se reasigna al globo de esa posición.

    Es la trampa que este módulo existe para evitar: la primera versión caía al índice
    cuando el `region_uid` no aparecía, y colocaba la traducción de otro trabajo en el
    primer globo de la página sin decir nada.
    """
    _manager, job = _build_job(tmp_path, pages=1)
    payload = {
        "formato": FORMAT_NAME,
        "paginas": [
            {"indice": 0, "regiones": [{"region_uid": "p0001r0099", "texto_traducido": "de otro trabajo"}]},
            {"indice": 7, "regiones": [{"region_uid": "p0008r0000", "texto_traducido": "página que no existe"}]},
        ],
    }
    plan = plan_job_import(job, payload)

    assert plan.changed_regions == 0
    assert plan.unmatched == ["p0001r0099"]
    assert plan.unknown_pages == [7]


def test_import_rejects_a_file_that_is_not_ours(tmp_path: Path) -> None:
    _manager, job = _build_job(tmp_path, pages=1)
    with pytest.raises(ValueError, match="Formato desconocido"):
        plan_job_import(job, {"formato": "otra-cosa", "paginas": [{"indice": 0, "regiones": []}]})
    with pytest.raises(ValueError, match="no trae páginas"):
        plan_job_import(job, {"formato": FORMAT_NAME, "paginas": []})


def test_round_trip_through_the_manager_rerenders_the_pages(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, job = _build_job(tmp_path, pages=2)

    export = manager.export_job_texts("job-id")
    assert len(export["paginas"]) == 2

    # Traducimos fuera: solo la página 2, y solo uno de sus globos.
    export["paginas"][1]["regiones"][1]["texto_traducido"] = "traducido fuera"

    simulacion = manager.import_job_texts("job-id", export, dry_run=True)
    assert simulacion["simulacion"] is True
    assert simulacion["regiones_actualizadas"] == 1
    assert simulacion["paginas_actualizadas"] == [2]
    # Simular no escribe: el trabajo sigue como estaba.
    assert not Path(job.pages[1].corrections_path).exists()

    resultado = manager.import_job_texts("job-id", export)
    assert resultado["regiones_actualizadas"] == 1
    assert resultado["paginas_actualizadas"] == [2]
    assert resultado["sin_emparejar_total"] == 0

    guardadas = manager._jobs["job-id"].pages[1].regions
    por_uid = {r["region_uid"]: r for r in guardadas}
    assert por_uid["p0002r0001"]["translated_text"] == "traducido fuera"
    # La página 1 no se toca si su texto no cambió.
    assert not Path(job.pages[0].corrections_path).exists()
    # Y lo importado pasa por el camino de corrección de siempre: queda en disco y la
    # página compuesta se regenera, no solo el manifiesto.
    escritas = json.loads(Path(job.pages[1].corrections_path).read_text(encoding="utf-8"))["regions"]
    assert any(r.get("region_uid") == "p0002r0001" for r in escritas)
    assert Path(job.pages[1].corrected_path).exists()


def test_import_works_on_a_job_that_never_rendered(monkeypatch, tmp_path: Path) -> None:
    """Un trabajo en modo «limpiar» o «limpiar y transcribir» no tiene página traducida.

    La composición no la necesita —parte de la capa de fondo, que es la limpieza—, pero el
    guardado la exigía por un resto del render incremental antiguo. Importar un guion en uno
    de esos trabajos fallaba con «faltan imágenes base» teniendo la limpieza delante.
    """
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, job = _build_job(tmp_path, pages=1)
    Path(job.pages[0].translated_path).unlink()
    assert not Path(job.pages[0].translated_path).exists()

    export = manager.export_job_texts("job-id")
    export["paginas"][0]["regiones"][0]["texto_traducido"] = "traducido fuera"
    resultado = manager.import_job_texts("job-id", export)

    assert resultado["regiones_actualizadas"] == 1
    assert resultado["paginas_actualizadas"] == [1]
    guardadas = {r["region_uid"]: r for r in manager._jobs["job-id"].pages[0].regions}
    assert guardadas["p0001r0000"]["translated_text"] == "traducido fuera"
    # Y la página compuesta se escribe, sobre la limpieza.
    assert Path(job.pages[0].corrected_path).exists()


def test_a_clean_only_job_can_still_be_exported(tmp_path: Path) -> None:
    """El resultado de «solo limpiar» son las páginas limpias, y hay que poder bajárselas.

    El ZIP solo aceptaba corregida o traducida, así que un trabajo que terminó bien se
    saltaba todas las páginas y moría con «no hay páginas listas para exportar».
    """
    import zipfile

    manager, job = _build_job(tmp_path, pages=1)
    Path(job.pages[0].translated_path).unlink()

    destino = manager.create_export_zip("job-id")

    with zipfile.ZipFile(destino) as archivo:
        nombres = archivo.namelist()
        manifiesto = json.loads(archivo.read("manifest_export.json").decode("utf-8"))
    assert any(n.startswith("imagenes_finales/") for n in nombres), nombres
    assert manifiesto["pages"][0]["variant"] == "limpieza"


def test_import_skips_pages_that_are_not_ready(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, job = _build_job(tmp_path, pages=2)
    job.pages[1].status = "pending"

    export = manager.export_job_texts("job-id")
    export["paginas"][1]["regiones"][0]["texto_traducido"] = "sobre una página sin limpiar"
    resultado = manager.import_job_texts("job-id", export)

    assert resultado["paginas_omitidas"] == [2]
    assert resultado["regiones_actualizadas"] == 0
    assert not Path(job.pages[1].corrections_path).exists()


def test_export_filename_survives_a_non_ascii_title() -> None:
    assert export_filename("Capítulo 01") == "Cap_tulo 01_textos.json"
    assert export_filename("") == "trabajo_textos.json"


def test_the_ui_exposes_both_actions() -> None:
    static = Path(__file__).resolve().parents[1] / "parallel_manga_translator" / "ui" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    javascript = (static / "app.js").read_text(encoding="utf-8")

    for elemento in ("exportTextsBtn", "importTextsBtn", "importTextsInput", "textsModal", "confirmTextsBtn"):
        assert elemento in html, f"Falta {elemento} en la UI."
    # La confirmación no es decorativa: se pide al servidor el informe sin escribir nada.
    assert "dry_run=true" in javascript
    assert "function submitTextImport" in javascript
