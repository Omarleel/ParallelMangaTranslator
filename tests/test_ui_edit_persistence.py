from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from parallel_manga_translator.ui import manual_renderer
from parallel_manga_translator.inpainting import LamaLarge
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
    manager.manifests.save(job)
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


def test_text_region_is_composited_above_original_restore_brush(monkeypatch, tmp_path: Path) -> None:
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
    assert np.all(rendered[18, 18] == 220), "El texto de región debe quedar siempre por encima de cualquier trazo de fondo."
    assert np.all(rendered[10, 10] == 220), "Toda la región modificada conserva su texto rasterizado."


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


def test_baked_brush_stroke_does_not_replay_on_later_regular_save(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, _job, page = _build_ready_manager(tmp_path)

    bbox = [8, 8, 20, 20]
    # Primera acción: restaurar original fuera de la caja de texto.
    stroke = {
        "points": [[32, 32]],
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

    assert restored["brush_strokes"] == [], "Los trazos ya horneados no deben quedar como reglas persistentes."
    first = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert first is not None
    assert np.all(first[32, 32] == 40)

    # Segunda acción en la misma zona: limpiar. Debe ganar la acción nueva y no
    # reaparecer el restaurador histórico en un guardado posterior.
    clean_stroke = {
        "points": [[32, 32]],
        "radius": 3,
        "mode": "restore_clean",
        "applied": False,
    }
    cleaned = manager.save_manual_render(
        "job-test",
        0,
        [_region_payload(bbox=bbox, source_bbox=bbox, text="texto actualizado")],
        [clean_stroke],
        operation="render",
    )
    assert cleaned["brush_strokes"] == []

    rendered = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert rendered is not None
    assert np.all(rendered[32, 32] == 100), "La acción más reciente debe prevalecer en una zona reutilizada."
    assert np.all(rendered[18, 18] == 220), "El texto de región permanece como capa superior."


def test_corrected_transcription_can_be_retranslated_without_running_ocr(monkeypatch, tmp_path: Path) -> None:
    manager, _job, _page = _build_ready_manager(tmp_path)

    fake_config = SimpleNamespace(
        translation=SimpleNamespace(idioma_entrada="Japonés", idioma_salida="Español"),
        character_memory=None,
    )
    monkeypatch.setattr(manager, "_build_config_for_job", lambda job: fake_config)

    from parallel_manga_translator.translation.translator_manager import TranslatorManager


    class FakeTranslator:
        def traducir_textos(self, texts):
            return [f"TRADUCIDO: {texts[0]}"]

    monkeypatch.setattr(TranslatorManager, "from_config", classmethod(lambda cls, *args, **kwargs: FakeTranslator()))

    result = manager.translate_manual_text("job-test", 0, "texto fuente corregido")

    assert result["original_text"] == "texto fuente corregido"
    assert result["translated_text"] == "TRADUCIDO: texto fuente corregido"
    assert result["source_language"] == "Japonés"
    assert result["target_language"] == "Español"


def test_history_can_restore_translation_position_without_raster_ghost(monkeypatch, tmp_path: Path) -> None:
    """Una instantánea del historial debe reconstruirse desde fondo, no desde texto horneado."""
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, _job, page = _build_ready_manager(tmp_path)

    box_a = [4, 4, 8, 8]
    box_b = [20, 4, 8, 8]

    moved = manager.save_manual_render(
        "job-test",
        0,
        [_region_payload(bbox=box_b, source_bbox=box_a)],
        [],
        operation="render",
        background_revision="base",
    )
    assert moved["background_revision"] == "base"

    # Simula Ctrl+Z: la instantánea previa tenía la región en A. Incluso si la
    # región histórica no estaba marcada como modificada, debe dibujarse desde la
    # capa de fondo y desaparecer por completo el texto rasterizado de B.
    undo_payload = _region_payload(bbox=box_a, source_bbox=box_a)
    undo_payload["modified"] = False
    undone = manager.save_manual_render(
        "job-test",
        0,
        [undo_payload],
        [],
        operation="render",
        background_revision="base",
    )
    assert undone["background_revision"] == "base"

    rendered = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert rendered is not None
    assert np.all(rendered[8, 8] == 220), "Deshacer debe devolver el texto a la caja histórica."
    assert np.all(rendered[8, 24] == 100), "La posición movida no puede quedar rasterizada como fantasma."

    # Simula Ctrl+Y sobre la misma revisión de fondo.
    redone = manager.save_manual_render(
        "job-test",
        0,
        [_region_payload(bbox=box_b, source_bbox=box_a)],
        [],
        operation="render",
        background_revision="base",
    )
    assert redone["background_revision"] == "base"
    rendered = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert rendered is not None
    assert np.all(rendered[8, 8] == 100)
    assert np.all(rendered[8, 24] == 220), "Rehacer debe volver a mover la región sin duplicar texto."


def test_background_revision_makes_brush_cleanup_undoable_and_redoable(monkeypatch, tmp_path: Path) -> None:
    """El historial de UI puede cambiar entre revisiones inmutables de limpieza."""
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, _job, page = _build_ready_manager(tmp_path)

    bbox = [8, 8, 20, 20]
    region = _region_payload(bbox=bbox, source_bbox=bbox)
    stroke = {
        "points": [[32, 32]],
        "radius": 3,
        "mode": "mask_eraser",
        "applied": False,
    }
    brushed = manager.save_manual_render(
        "job-test",
        0,
        [region],
        [stroke],
        operation="mask_eraser",
        background_revision="base",
    )
    brushed_revision = brushed["background_revision"]
    assert brushed_revision != "base"
    assert Path(page.manual_background_path).exists()
    rendered = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert rendered is not None and np.all(rendered[32, 32] == 40)

    # Ctrl+Z: seleccionar la revisión de fondo que estaba en la instantánea previa.
    undone = manager.save_manual_render(
        "job-test",
        0,
        [region],
        [],
        operation="render",
        background_revision="base",
    )
    assert undone["background_revision"] == "base"
    rendered = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert rendered is not None and np.all(rendered[32, 32] == 100)

    # Ctrl+Y: la instantánea de rehacer conserva la revisión creada por el pincel.
    redone = manager.save_manual_render(
        "job-test",
        0,
        [region],
        [],
        operation="render",
        background_revision=brushed_revision,
    )
    assert redone["background_revision"] == brushed_revision
    rendered = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert rendered is not None and np.all(rendered[32, 32] == 40)


def test_inpaint_revision_is_preserved_when_restoring_original_elsewhere(monkeypatch, tmp_path: Path) -> None:
    """Restaurar B debe partir de la revisión que ya contiene el inpaint de A."""
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, _job, page = _build_ready_manager(tmp_path)

    # El fake hace muy visible la zona reconstruida: 205 en cualquier píxel enmascarado.
    monkeypatch.setattr(
        manager.manual_edits,
        "_manual_inpaint_callable",
        lambda _model: lambda image, mask: np.where((mask > 0)[..., None], 205, image).astype(np.uint8),
    )

    bbox = [4, 4, 8, 8]
    region = _region_payload(bbox=bbox, source_bbox=bbox)
    inpaint_stroke = {
        "points": [[28, 12]],
        "radius": 2,
        "mode": "inpaint",
        "applied": False,
    }
    inpainted = manager.save_manual_render(
        "job-test",
        0,
        [region],
        [inpaint_stroke],
        operation="inpaint",
        background_revision="base",
        inpaint_model="opencv-tela",
    )
    inpaint_revision = inpainted["background_revision"]
    assert inpaint_revision != "base"

    after_inpaint = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert after_inpaint is not None
    assert np.all(after_inpaint[12, 28] == 205)

    restore_stroke = {
        "points": [[28, 30]],
        "radius": 2,
        "mode": "mask_eraser",
        "applied": False,
    }
    restored = manager.save_manual_render(
        "job-test",
        0,
        [region],
        [restore_stroke],
        operation="mask_eraser",
        background_revision=inpaint_revision,
    )

    assert restored["background_revision"] not in {"base", inpaint_revision}
    final = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert final is not None
    assert np.all(final[12, 28] == 205), "Restaurar otra zona no puede descartar un inpaint ya horneado."
    assert np.all(final[30, 28] == 40), "La nueva pincelada sí debe recuperar el manga original en su propia zona."

def test_two_consecutive_restore_original_brush_commits_accumulate(monkeypatch, tmp_path: Path) -> None:
    """Dos aplicaciones separadas del mismo pincel deben encadenarse sobre la revisión previa."""
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, _job, page = _build_ready_manager(tmp_path)

    bbox = [4, 4, 8, 8]
    region = _region_payload(bbox=bbox, source_bbox=bbox)
    stroke_a = {
        "points": [[28, 12]],
        "radius": 2,
        "mode": "mask_eraser",
        "applied": False,
    }
    first = manager.save_manual_render(
        "job-test",
        0,
        [region],
        [stroke_a],
        operation="mask_eraser",
        background_revision="base",
    )
    revision_a = first["background_revision"]
    assert revision_a != "base"

    stroke_b = {
        "points": [[28, 30]],
        "radius": 2,
        "mode": "mask_eraser",
        "applied": False,
    }
    second = manager.save_manual_render(
        "job-test",
        0,
        [region],
        [stroke_b],
        operation="mask_eraser",
        background_revision=revision_a,
    )
    assert second["background_revision"] not in {"base", revision_a}

    rendered = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert rendered is not None
    assert np.all(rendered[12, 28] == 40), "La primera restauración debe sobrevivir a la segunda aplicación."
    assert np.all(rendered[30, 28] == 40), "La segunda restauración también debe quedar aplicada."


def test_two_consecutive_clean_brush_commits_accumulate_after_inpaint(monkeypatch, tmp_path: Path) -> None:
    """Limpiar A y luego B no puede devolver A al fondo inpaintado anterior."""
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, _job, page = _build_ready_manager(tmp_path)
    monkeypatch.setattr(
        manager.manual_edits,
        "_manual_inpaint_callable",
        lambda _model: lambda image, mask: np.where((mask > 0)[..., None], 205, image).astype(np.uint8),
    )

    bbox = [4, 4, 8, 8]
    region = _region_payload(bbox=bbox, source_bbox=bbox)
    seed_inpaint = {
        "points": [[28, 12], [28, 30]],
        "radius": 4,
        "mode": "inpaint",
        "applied": False,
    }
    inpainted = manager.save_manual_render(
        "job-test",
        0,
        [region],
        [seed_inpaint],
        operation="inpaint",
        background_revision="base",
        inpaint_model="opencv-tela",
    )
    inpaint_revision = inpainted["background_revision"]

    clean_a = {
        "points": [[28, 12]],
        "radius": 2,
        "mode": "restore_clean",
        "applied": False,
    }
    first = manager.save_manual_render(
        "job-test",
        0,
        [region],
        [clean_a],
        operation="render",
        background_revision=inpaint_revision,
    )
    revision_a = first["background_revision"]

    clean_b = {
        "points": [[28, 30]],
        "radius": 2,
        "mode": "restore_clean",
        "applied": False,
    }
    manager.save_manual_render(
        "job-test",
        0,
        [region],
        [clean_b],
        operation="render",
        background_revision=revision_a,
    )

    rendered = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert rendered is not None
    assert np.all(rendered[12, 28] == 100), "La primera limpieza debe mantenerse después de limpiar una segunda zona."
    assert np.all(rendered[30, 28] == 100), "La segunda limpieza debe quedar acumulada sobre la primera."



def test_two_consecutive_inpaint_commits_accumulate(monkeypatch, tmp_path: Path) -> None:
    """Aplicar inpaint en A y después en B debe conservar la reconstrucción de A."""
    monkeypatch.setattr(manual_renderer, "TextRenderer", _SolidTextRenderer)
    manager, _job, page = _build_ready_manager(tmp_path)
    monkeypatch.setattr(
        manager.manual_edits,
        "_manual_inpaint_callable",
        lambda _model: lambda image, mask: np.where((mask > 0)[..., None], 205, image).astype(np.uint8),
    )

    bbox = [4, 4, 8, 8]
    region = _region_payload(bbox=bbox, source_bbox=bbox)
    first = manager.save_manual_render(
        "job-test",
        0,
        [region],
        [{"points": [[28, 12]], "radius": 2, "mode": "inpaint", "applied": False}],
        operation="inpaint",
        background_revision="base",
        inpaint_model="opencv-tela",
    )
    revision_a = first["background_revision"]

    second = manager.save_manual_render(
        "job-test",
        0,
        [region],
        [{"points": [[28, 30]], "radius": 2, "mode": "inpaint", "applied": False}],
        operation="inpaint",
        background_revision=revision_a,
        inpaint_model="opencv-tela",
    )
    assert second["background_revision"] not in {"base", revision_a}

    rendered = cv2.imread(page.corrected_path, cv2.IMREAD_COLOR)
    assert rendered is not None
    assert np.all(rendered[12, 28] == 205), "El primer inpaint debe sobrevivir al segundo commit."
    assert np.all(rendered[30, 28] == 205), "El segundo inpaint debe aplicarse sobre la revisión ya modificada."


def test_manual_neural_inpaint_uses_large_context_crop_and_preserves_pixels_outside_mask(tmp_path: Path) -> None:
    """El pincel neural procesa una ventana local, pero solo puede modificar su máscara."""
    base_path = tmp_path / "large_base.png"
    output_path = tmp_path / "large_out.png"
    image = np.full((1800, 2200, 3), 100, dtype=np.uint8)
    # Marcadores fuera de la máscara para detectar cualquier contaminación del crop.
    image[500:520, 700:720] = 77
    image[1250:1270, 1450:1470] = 133
    assert cv2.imwrite(str(base_path), image)

    seen_shapes: list[tuple[int, int]] = []

    def fake_heavy_inpaint(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        seen_shapes.append(crop.shape[:2])
        # Deliberadamente cambia TODO el crop. El wrapper de producción debe aceptar
        # únicamente los píxeles cubiertos por la máscara para impedir costuras.
        return np.full_like(crop, 205)

    fake_heavy_inpaint._pmt_crop_profile = {
        "enabled": True,
        "min_side": 1024,
        "context_px": 384,
        "alignment": 64,
        "max_model_side": 1536,
    }
    changed = manual_renderer.apply_pending_inpaint_only(
        base_path=base_path,
        output_path=output_path,
        brush_strokes=[BrushStroke(points=[(1100, 900)], radius=5, mode="inpaint")],
        inpaint_fn=fake_heavy_inpaint,
    )

    assert changed is True
    assert seen_shapes == [(1024, 1024)], "Una pincelada pequeña no debe ejecutar LaMa sobre la página 1800x2200 completa."
    rendered = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert rendered is not None
    assert np.all(rendered[900, 1100] == 205)
    assert np.all(rendered[500, 700] == 77), "Fuera de la máscara el fondo debe conservarse bit a bit."
    assert np.all(rendered[1260, 1460] == 133), "El recorte no puede introducir costuras ni cambios colaterales."
    assert np.all(rendered[900, 900] == 100), "Incluso dentro del crop, fuera de la máscara no se copia la salida del modelo."


def test_manual_inpaint_profiles_keep_lama_large_fp32_and_generous_context(tmp_path: Path) -> None:
    """La optimización no reduce precisión ni cambia el modelo seleccionado."""
    manager = JobManager(jobs_root=tmp_path / "jobs", start_worker=False)
    large_callable = manager.manual_edits._manual_inpaint_callable("lama_large_512px")
    mpe_callable = manager.manual_edits._manual_inpaint_callable("lama_mpe")
    aot_callable = manager.manual_edits._manual_inpaint_callable("aot")

    assert large_callable._pmt_crop_profile == {
        "enabled": True,
        "min_side": 1024,
        "context_px": 384,
        "alignment": 64,
        "max_model_side": 1536,
    }
    assert mpe_callable._pmt_crop_profile["min_side"] == 896
    assert aot_callable._pmt_crop_profile["min_side"] == 768
    assert LamaLarge().precision == "fp32", "La ruta optimizada no debe bajar LaMa Large a fp16/bf16."


def test_large_inpaint_area_falls_back_to_full_page_context(tmp_path: Path) -> None:
    """Si el crop ya costaría lo mismo que el modelo completo, se conserva la ruta histórica."""
    base_path = tmp_path / "fallback_base.png"
    output_path = tmp_path / "fallback_out.png"
    image = np.full((1800, 2200, 3), 100, dtype=np.uint8)
    assert cv2.imwrite(str(base_path), image)
    seen_shapes: list[tuple[int, int]] = []

    def fake_heavy_inpaint(input_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        seen_shapes.append(input_image.shape[:2])
        return input_image.copy()

    fake_heavy_inpaint._pmt_crop_profile = {
        "enabled": True,
        "min_side": 1024,
        "context_px": 384,
        "alignment": 64,
        "max_model_side": 1536,
    }
    manual_renderer.apply_pending_inpaint_only(
        base_path=base_path,
        output_path=output_path,
        brush_strokes=[BrushStroke(points=[(700, 900), (1500, 900)], radius=8, mode="inpaint")],
        inpaint_fn=fake_heavy_inpaint,
    )

    assert seen_shapes == [(1800, 2200)], "Una máscara extensa debe conservar todo el contexto global de la ruta anterior."
