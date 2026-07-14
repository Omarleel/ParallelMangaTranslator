from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from parallel_manga_translator.ui import manual_renderer
from parallel_manga_translator.ui.job_manager import JobManager, JobState, PageState
from parallel_manga_translator.ui.manual_renderer import BrushStroke, ManualRegion, render_manual_page


class _SolidTextRenderer:
    """Renderer determinista para probar el orden de composición de la UI."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def render_with_layouts(self, image, boxes, texts, **kwargs):
        result = image.copy()
        for x, y, w, h in boxes:
            result[y : y + h, x : x + w] = 220
        return result


def _write_uniform(path: Path, value: int, size: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size, size, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _build_ready_manager(tmp_path: Path) -> tuple[JobManager, JobState, PageState]:
    jobs_root = tmp_path / "jobs"
    root_dir = jobs_root / "job-test"
    input_dir = root_dir / "entrada"
    output_dir = root_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = input_dir / "page.png"
    clean = output_dir / "limpieza" / "page.png"
    translated = output_dir / "traduccion" / "page.png"
    corrected = output_dir / "corregida" / "page.png"
    corrections = output_dir / "correcciones" / "page.json"

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
        corrected_path=str(corrected),
        corrections_path=str(corrections),
    )
    job = JobState(
        job_id="job-test",
        title="UI persistence test",
        root_dir=str(root_dir),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        status="ready",
        pages=[page],
    )
    manager = JobManager(jobs_root=jobs_root, start_worker=False)
    manager._jobs[job.job_id] = job
    manager._save_manifest(job)
    return manager, job, page


def _region_payload(*, bbox, source_bbox, text="texto") -> dict:
    return {
        "index": 0,
        "bbox": list(bbox),
        "source_bbox": list(source_bbox),
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


def test_original_restore_brush_is_composited_after_text(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)

    original = tmp_path / "original.png"
    clean = tmp_path / "clean.png"
    translated = tmp_path / "translated.png"
    output = tmp_path / "output.png"
    _write_uniform(original, 40)
    _write_uniform(clean, 100)
    _write_uniform(translated, 160)

    region = ManualRegion(index=0, bbox=(8, 8, 20, 20), source_bbox=(8, 8, 20, 20), text="texto", modified=True)
    stroke = BrushStroke(points=[(18, 18)], radius=3, mode="mask_eraser")

    render_manual_page(
        clean_path=clean,
        original_path=original,
        translated_path=translated,
        output_path=output,
        regions=[region],
        brush_strokes=[stroke],
    )

    rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert rendered is not None
    assert np.all(rendered[18, 18] == 40), "La restauración original debe prevalecer sobre el texto rasterizado."
    assert np.all(rendered[10, 10] == 220), "Fuera de la pincelada, la región sigue renderizando su texto."


def test_saved_region_advances_source_bbox_so_second_move_clears_previous_position(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, _job, page = _build_ready_manager(tmp_path)

    translated = cv2.imread(page.translated_path, cv2.IMREAD_COLOR)
    assert translated is not None
    translated[4:12, 4:12] = 230  # texto automático en la posición A
    assert cv2.imwrite(page.translated_path, translated)

    box_a = [4, 4, 8, 8]
    box_b = [18, 4, 8, 8]
    box_c = [30, 4, 8, 8]

    first = manager.save_manual_render(
        "job-test",
        0,
        [_region_payload(bbox=box_b, source_bbox=box_a)],
        [],
        operation="render",
    )
    assert first["regions"][0]["source_bbox"] == box_b

    second_payload = _region_payload(
        bbox=box_c,
        source_bbox=first["regions"][0]["source_bbox"],
        text=first["regions"][0]["translated_text"],
    )
    second = manager.save_manual_render("job-test", 0, [second_payload], [], operation="render")

    rendered = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert rendered is not None
    assert np.all(rendered[8, 8] == 100), "La posición automática original debe quedar limpia."
    assert np.all(rendered[8, 22] == 100), "La posición del primer movimiento debe limpiarse al mover otra vez."
    assert np.all(rendered[8, 34] == 220), "El texto debe existir solo en la posición más reciente."
    assert second["regions"][0]["source_bbox"] == box_c


def test_restore_original_stroke_survives_later_regular_save(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, _job, page = _build_ready_manager(tmp_path)

    bbox = [8, 8, 20, 20]
    stroke = {
        "points": [[18, 18]],
        "radius": 3,
        "mode": "mask_eraser",
        "applied": False,
    }
    restored = manager.save_manual_render(
        "job-test",
        0,
        [_region_payload(bbox=bbox, source_bbox=bbox)],
        [stroke],
        operation="mask_eraser",
    )

    assert restored["brush_strokes"], "La restauración debe conservarse como parte del estado persistente."
    assert restored["brush_strokes"][0]["mode"] == "mask_eraser"

    # Un guardado posterior de texto/región no debe volver a pintar encima de la zona restaurada.
    manager.save_manual_render(
        "job-test",
        0,
        [_region_payload(bbox=bbox, source_bbox=bbox, text="texto actualizado")],
        restored["brush_strokes"],
        operation="render",
    )

    rendered = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert rendered is not None
    assert np.all(rendered[18, 18] == 40)
    assert np.all(rendered[10, 10] == 220)


def test_corrected_transcription_can_be_retranslated_without_running_ocr(monkeypatch, tmp_path: Path) -> None:
    manager, _job, _page = _build_ready_manager(tmp_path)

    fake_config = SimpleNamespace(
        translation=SimpleNamespace(idioma_entrada="Japonés", idioma_salida="Español"),
        character_memory=None,
    )
    monkeypatch.setattr(manager, "_build_config_for_job", lambda job: fake_config)

    import parallel_manga_translator.ui.job_manager as job_manager_module
    from parallel_manga_translator.translation.translator_manager import TranslatorManager

    monkeypatch.setattr(job_manager_module, "set_active_config", lambda config: None)

    class FakeTranslator:
        def traducir_textos(self, texts):
            return [f"TRADUCIDO: {texts[0]}"]

    monkeypatch.setattr(TranslatorManager, "from_config", classmethod(lambda cls, *args, **kwargs: FakeTranslator()))

    result = manager.translate_manual_text("job-test", 0, "texto fuente corregido")

    assert result["original_text"] == "texto fuente corregido"
    assert result["translated_text"] == "TRADUCIDO: texto fuente corregido"
    assert result["source_language"] == "Japonés"
    assert result["target_language"] == "Español"
