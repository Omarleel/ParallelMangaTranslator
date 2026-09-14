"""La corrección humana debe saber qué región de la ejecución corrige.

Sin esto, emparejar una corrección con lo que vio el pipeline hay que reconstruirlo a
posteriori, y las tres formas de hacerlo fallan en silencio: el editor reordena la lista,
reasigna `index` al borrar y encoge la caja al área de texto. Estos tests fijan justo esos
tres casos, que son los que rompieron cuatro mediciones de transcripción.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from parallel_manga_translator.models.region_identity import is_manual_uid, run_region_uid
from parallel_manga_translator.quality.eval_dataset import build_ground_truth_page
from parallel_manga_translator.quality.ocr_crop_pairing import (
    index_ocr_crops,
    pair_ground_truth_with_crops,
)
from parallel_manga_translator.ui import manual_renderer
from parallel_manga_translator.ui.job_manager import JobManager, JobState, PageState
from parallel_manga_translator.ui.page_regions import apply_saved_corrections, merge_page_regions


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
    image = np.full((size, size, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _build_ready_manager(tmp_path: Path) -> tuple[JobManager, JobState, PageState]:
    jobs_root = tmp_path / "jobs"
    root_dir = jobs_root / "job-id"
    input_dir = root_dir / "entrada"
    output_dir = root_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = input_dir / "page.png"
    clean = output_dir / "limpieza" / "page.png"
    translated = output_dir / "traduccion" / "page.png"
    _write_uniform(original, 40)
    _write_uniform(clean, 100)
    _write_uniform(translated, 160)

    page = PageState(
        index=0,
        source_filename="page.png",
        output_filename="page.png",
        status="ready",
        original_path=str(original),
        clean_path=str(clean),
        translated_path=str(translated),
        corrected_path=str(output_dir / "corregida" / "page.png"),
        corrections_path=str(output_dir / "correcciones" / "page.json"),
    )
    job = JobState(
        job_id="job-id",
        title="identidad de región",
        root_dir=str(root_dir),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        status="ready",
        pages=[page],
    )
    manager = JobManager(jobs_root=jobs_root, start_worker=False)
    manager._jobs[job.job_id] = job
    manager.manifests.save(job)
    return manager, job, page


def _payload(index: int, bbox, *, uid: str = "", text: str = "texto", **extra) -> dict:
    region = {
        "index": index,
        "bbox": list(bbox),
        "source_bbox": list(bbox),
        "text": text,
        "original_text": "original",
        "style": "dialogo",
        "restore_original": False,
        "visible": True,
        "modified": True,
        "manual": False,
        "deleted": False,
        "auto_font_size": True,
        "font_size": None,
        "rotation_angle": 0,
        "text_align": "center",
        "vertical_align": "middle",
        "line_spacing_factor": 1,
        "text_offset_x": 0,
        "text_offset_y": 0,
        "ui_layout": None,
    }
    if uid:
        region["region_uid"] = uid
    region.update(extra)
    return region


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


def _regions_by_uid(regions):
    return {str(region.get("region_uid") or ""): region for region in regions}


def test_merge_stamps_run_identity_and_run_bbox(tmp_path: Path) -> None:
    job = JobState(job_id="j", title="t", root_dir=str(tmp_path), input_dir=str(tmp_path), output_dir=str(tmp_path))
    page = PageState(index=2, source_filename="p.png", output_filename="p.png")
    trans = _transcription(3, [(0, (10, 10, 30, 20), "hola"), (5, (60, 40, 25, 15), "adios")])

    regions = merge_page_regions(job, page, trans, {"Traducción": []})

    assert [r["region_uid"] for r in regions] == [run_region_uid(3, 0), run_region_uid(3, 5)]
    # La identidad sale del `Índice` del pipeline, no de la posición en la lista: el
    # segundo globo es el índice 5 porque el filtro de idioma descartó los de en medio.
    assert regions[1]["region_uid"] == "p0003r0005"
    assert regions[0]["run_bbox"] == [10, 10, 30, 20]


def test_corrections_preserve_identity_even_reordered(tmp_path: Path) -> None:
    job = JobState(job_id="j", title="t", root_dir=str(tmp_path), input_dir=str(tmp_path), output_dir=str(tmp_path))
    page = PageState(index=0, source_filename="p.png", output_filename="p.png")
    trans = _transcription(1, [(0, (10, 10, 30, 20), "uno"), (1, (60, 40, 25, 15), "dos")])
    regions = merge_page_regions(job, page, trans, {"Traducción": []})

    corrections = [
        {"index": 1, "bbox": [61, 41, 20, 10], "text": "DOS", "modified": True, "region_uid": "p0001r0001"},
        {"index": 0, "bbox": [11, 11, 25, 15], "text": "UNO", "modified": True, "region_uid": "p0001r0000"},
        {"index": 9, "bbox": [5, 5, 8, 8], "text": "añadida", "manual": True, "region_uid": "manual-abc12345"},
    ]
    merged = apply_saved_corrections(regions, corrections)
    by_uid = _regions_by_uid(merged)

    assert by_uid["p0001r0000"]["translated_text"] == "UNO"
    assert by_uid["p0001r0001"]["translated_text"] == "DOS"
    # La caja de la ejecución no la toca el editor aunque encoja `bbox` al área de texto.
    assert by_uid["p0001r0000"]["run_bbox"] == [10, 10, 30, 20]
    assert is_manual_uid(by_uid["manual-abc12345"]["region_uid"])


def test_identity_survives_reorder_delete_and_rerender(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, job, page = _build_ready_manager(tmp_path)
    page.regions = merge_page_regions(
        job,
        page,
        _transcription(1, [(0, (5, 5, 20, 12), "uno"), (1, (30, 5, 20, 12), "dos"), (2, (5, 30, 20, 12), "tres")]),
        {"Traducción": []},
    )
    manager.manifests.save(job)
    uids = [r["region_uid"] for r in page.regions]
    assert uids == ["p0001r0000", "p0001r0001", "p0001r0002"]

    # El humano borra la primera y reordena las otras dos: los `index` se reasignan.
    manager.save_manual_render(
        "job-id",
        0,
        [
            _payload(0, (30, 6, 18, 10), uid=uids[2], text="TRES"),
            _payload(1, (6, 31, 18, 10), uid=uids[1], text="DOS"),
            _payload(2, (5, 5, 20, 12), uid=uids[0], deleted=True),
        ],
        [],
        operation="render",
    )

    guardadas = _regions_by_uid(manager._jobs["job-id"].pages[0].regions)
    assert set(guardadas) == set(uids)
    assert guardadas[uids[2]]["translated_text"] == "TRES"
    assert guardadas[uids[1]]["translated_text"] == "DOS"
    assert guardadas[uids[0]]["deleted"] is True
    # Y lo que de verdad importa: cada corrección sigue apuntando a la caja que detectó
    # la ejecución, no a la que el editor dejó después de mover y encoger.
    assert guardadas[uids[1]]["run_bbox"] == [30, 5, 20, 12]
    assert guardadas[uids[2]]["run_bbox"] == [5, 30, 20, 12]

    escritas = json.loads(Path(page.corrections_path).read_text(encoding="utf-8"))["regions"]
    assert {str(r.get("region_uid")) for r in escritas} == set(uids)


def test_manual_region_receives_its_own_stable_identity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, job, page = _build_ready_manager(tmp_path)
    page.regions = merge_page_regions(job, page, _transcription(1, [(0, (5, 5, 20, 12), "uno")]), {"Traducción": []})
    manager.manifests.save(job)

    manager.save_manual_render(
        "job-id",
        0,
        [_payload(0, (5, 5, 20, 12), uid="p0001r0000"), _payload(1, (30, 30, 15, 10), manual=True, text="mía")],
        [],
        operation="render",
    )
    guardadas = manager._jobs["job-id"].pages[0].regions
    asignado = str(guardadas[1]["region_uid"])
    assert is_manual_uid(asignado), "Una región dibujada a mano no corrige ninguna del run, pero debe ser seguible."

    # El navegador la devuelve con su identidad; un segundo guardado no debe renombrarla.
    manager.save_manual_render(
        "job-id",
        0,
        [
            _payload(0, (31, 31, 15, 10), uid=asignado, manual=True, text="mía, movida"),
            _payload(1, (5, 5, 20, 12), uid="p0001r0000"),
        ],
        [],
        operation="render",
    )
    guardadas = _regions_by_uid(manager._jobs["job-id"].pages[0].regions)
    assert guardadas[asignado]["translated_text"] == "mía, movida"
    assert set(guardadas) == {asignado, "p0001r0000"}


def test_corrections_written_before_identity_existed_recover_it_by_index(monkeypatch, tmp_path: Path) -> None:
    """Un cliente antiguo no manda `region_uid`; se recupera del manifiesto, no se pierde."""
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, job, page = _build_ready_manager(tmp_path)
    page.regions = merge_page_regions(
        job, page, _transcription(1, [(0, (5, 5, 20, 12), "uno"), (1, (30, 5, 20, 12), "dos")]), {"Traducción": []}
    )
    manager.manifests.save(job)

    manager.save_manual_render(
        "job-id",
        0,
        [_payload(0, (5, 5, 20, 12), text="UNO"), _payload(1, (30, 5, 20, 12), text="DOS")],
        [],
        operation="render",
    )
    guardadas = manager._jobs["job-id"].pages[0].regions
    assert [r["region_uid"] for r in guardadas] == ["p0001r0000", "p0001r0001"]


def test_ground_truth_page_carries_identity() -> None:
    page = {
        "index": 0,
        "output_filename": "0001.png",
        "regions": [
            {
                "region_uid": "p0001r0000",
                "run_bbox": [10, 10, 40, 30],
                "bbox": [14, 14, 30, 20],
                "original_text": "hello",
                "translated_text": "hola",
                "type": "dialogue",
            },
            {"region_uid": "p0001r0001", "bbox": [80, 10, 20, 20], "deleted": True},
        ],
    }
    built = build_ground_truth_page(page)
    assert [r["region_uid"] for r in built["regions"]] == ["p0001r0000"]
    assert built["regions"][0]["bbox_run"] == [10, 10, 40, 30]
    assert built["descartadas_por_humano"] == 1


def _dump_crop(root: Path, page_index: int, indice: int, uid: str | None, bbox) -> None:
    carpeta = root / f"pagina_{page_index:04d}"
    carpeta.mkdir(parents=True, exist_ok=True)
    _write_uniform(carpeta / f"region_{indice:02d}_enmascarado.png", 200, size=12)
    _write_uniform(carpeta / f"region_{indice:02d}_preparado.png", 255, size=12)
    meta = {"indice": indice, "bbox": list(bbox), "text_bbox": list(bbox), "kind": "dialogue"}
    if uid:
        meta["region_uid"] = uid
    (carpeta / f"region_{indice:02d}.json").write_text(json.dumps(meta), encoding="utf-8")


def test_pairing_uses_identity_and_reports_what_it_cannot_match(tmp_path: Path) -> None:
    dump = tmp_path / "crops"
    _dump_crop(dump, 0, 0, "p0001r0000", (10, 10, 40, 30))
    _dump_crop(dump, 0, 1, "p0001r0001", (60, 10, 30, 30))
    _dump_crop(dump, 0, 2, None, (90, 10, 20, 20))  # volcado de una versión anterior

    crops = index_ocr_crops(dump)
    assert set(crops) == {"p0001r0000", "p0001r0001"}, "Un recorte sin identidad no se empareja a la fuerza."

    pages = [
        {
            "page": "0001",
            "regions": [
                {"region_uid": "p0001r0000", "texto_original": "HELLO", "bbox_texto": [14, 14, 30, 20]},
                {"region_uid": "p0001r0009", "texto_original": "otra ejecución"},
                {"texto_original": "caso antiguo, sin identidad"},
            ],
        }
    ]
    report = pair_ground_truth_with_crops(pages, crops)

    assert report.emparejadas == 1
    assert report.pairs[0].reference_text == "HELLO"
    assert report.pairs[0].prepared_path is not None
    assert report.pairs[0].crop_bbox == [10, 10, 40, 30]
    assert report.sin_recorte == ["p0001r0009"]
    assert report.sin_correccion == ["p0001r0001"]
    assert report.sin_identidad == 1
