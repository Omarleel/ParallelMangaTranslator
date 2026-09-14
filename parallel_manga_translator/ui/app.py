from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from parallel_manga_translator.config.constants import normalizar_modelo_inpaint
from parallel_manga_translator.ui.job_manager import (
    JobManager,
    JobOptions,
    job_to_public,
    normalize_choice,
    normalize_pipeline_mode,
    normalize_region_source,
)

STATIC_DIR = Path(__file__).parent / "static"
manager = JobManager(config_path=os.getenv("PMT_CONFIG", "config.yaml"))

app = FastAPI(
    title="Parallel Manga Translator UI",
    description="Asistente local de revisión para traducción, limpieza y corrección manual de páginas de manga.",
    version="0.6.0",
)
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "PMT_UI_ALLOWED_ORIGINS",
        "http://127.0.0.1:7860,http://localhost:7860",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class RegionPatch(BaseModel):
    index: int
    bbox: List[float] = Field(min_length=4, max_length=4)
    text: str = ""
    style: str = "dialogo"
    restore_original: bool = False
    visible: bool = True
    modified: bool = False
    manual: bool = False
    deleted: bool = False
    original_text: str = ""
    source_bbox: Optional[List[float]] = Field(default=None, min_length=4, max_length=4)
    auto_font_size: bool = True
    font_size: Optional[int] = None
    rotation_angle: float = 0.0
    text_align: str = "center"
    vertical_align: str = "middle"
    line_spacing_factor: float = 1.0
    text_offset_x: float = 0.0
    text_offset_y: float = 0.0
    ui_layout: Optional[Dict[str, Any]] = None
    # Qué región de la ejecución corrige esta caja. El editor reasigna `index`, así que
    # sin esto una corrección no se puede volver a emparejar con lo que vio el pipeline.
    region_uid: str = ""
    run_bbox: Optional[List[float]] = Field(default=None, min_length=4, max_length=4)


class BrushStrokePatch(BaseModel):
    points: List[List[float]] = Field(default_factory=list)
    radius: int = 18
    mode: str = "restore_original"
    applied: bool = False
    #: RGB del pincel de pintar. Lo elige el cuentagotas de la UI.
    color: Optional[List[int]] = Field(default=None, min_length=3, max_length=3)


class RenderRequest(BaseModel):
    regions: List[RegionPatch]
    brush_strokes: List[BrushStrokePatch] = Field(default_factory=list)
    operation: str = "render"
    inpaint_model: str = "job"
    background_revision: str = "base"


class RegionPreviewRequest(BaseModel):
    region: RegionPatch


class OcrRegionRequest(BaseModel):
    bbox: List[float] = Field(min_length=4, max_length=4)
    translate: bool = True


class TranslateRegionRequest(BaseModel):
    original_text: str = ""


class DatasetCaseRequest(BaseModel):
    name: str = ""
    copy_images: bool = True
    copy_reference: bool = True
    overwrite: bool = False
    refresh_baseline: bool = True


class RetranslateJobRequest(BaseModel):
    translator: str = "llm"
    target_language: Optional[str] = None
    overwrite_manual: bool = False
    llm_model: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/jobs")
def list_jobs(include_pages: bool = True):
    return {"jobs": manager.list_jobs(include_pages=include_pages)}


@app.post("/api/jobs")
def create_job(
    title: str = Form(default=""),
    images: List[UploadFile] = File(default=[]),
    zip_file: Optional[UploadFile] = File(default=None),
    source_language: str = Form(default="Japonés"),
    target_language: str = Form(default="Español"),
    detection_engine: str = Form(default="auto"),
    transcription_engine: str = Form(default="auto"),
    translator: str = Form(default="llm"),
    region_source: str = Form(default="yolo"),
    modo: str = Form(default="traducir"),
    inpaint_model: str = Form(default="auto"),
    page_max_retries: int = Form(default=2),
    retry_backoff_seconds: float = Form(default=2.0),
):
    try:
        options = JobOptions(
            source_language=source_language or "Japonés",
            target_language=target_language or "Español",
            detection_engine=normalize_choice(detection_engine, "auto"),
            transcription_engine=normalize_choice(transcription_engine, "auto"),
            translator=normalize_choice(translator, "llm"),
            region_source=normalize_region_source(region_source),
            modo=normalize_pipeline_mode(modo),
            inpaint_model=normalizar_modelo_inpaint(inpaint_model, "auto"),
            page_max_retries=max(0, min(20, int(page_max_retries))),
            retry_backoff_seconds=max(0.0, min(300.0, float(retry_backoff_seconds))),
        )
        job = manager.create_job_from_uploads(files=images, zip_file=zip_file, title=title, options=options)
        manager.start_job(job.job_id)
        return job_to_public(job)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """Borra un trabajo y su carpeta. La confirmación la pide el editor."""
    try:
        return manager.delete_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="El trabajo no existe.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return job_to_public(manager.get_job(job_id))
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str):
    try:
        return job_to_public(manager.pause_job(job_id))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    try:
        return job_to_public(manager.resume_job(job_id))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/translation-runs")
def translation_runs(job_id: str):
    try:
        return manager.translation_runs_public(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/retranslate")
def retranslate_job(job_id: str, request: RetranslateJobRequest):
    try:
        job = manager.retranslate_job(
            job_id,
            translator=normalize_choice(request.translator, "llm"),
            target_language=request.target_language,
            overwrite_manual=bool(request.overwrite_manual),
            llm_model=request.llm_model,
        )
        return job_to_public(job)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return job_to_public(manager.cancel_job(job_id))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/pages/{page_index}")
def get_page(job_id: str, page_index: int):
    try:
        job = manager.get_job(job_id)
        page = job.pages[page_index]
        from parallel_manga_translator.ui.job_manager import page_to_public

        return page_to_public(page, job_id)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail="La página no existe.") from exc
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/pages/{page_index}/image/{variant}")
def get_page_image(job_id: str, page_index: int, variant: str):
    try:
        path = manager.image_path(job_id, page_index, variant)
        if not path.exists():
            raise HTTPException(status_code=404, detail="La imagen todavía no está disponible.")
        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return FileResponse(str(path), media_type=media_type)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/dataset-case")
def dataset_case_preview(job_id: str):
    """Cifras y nombre sugerido para convertir este trabajo en caso de dataset_eval."""
    try:
        return manager.dataset_case_preview(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="El trabajo no existe.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/dataset-case")
def build_dataset_case(job_id: str, request: DatasetCaseRequest):
    """Construye el caso con el mismo constructor que el CLI `eval_dataset build`."""
    try:
        return manager.export_job_to_dataset(
            job_id,
            name=request.name,
            copy_images=bool(request.copy_images),
            copy_reference=bool(request.copy_reference),
            overwrite=bool(request.overwrite),
            refresh_baseline=bool(request.refresh_baseline),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/textos")
def export_job_texts(job_id: str):
    """Descarga el texto de todo el trabajo con las coordenadas de cada globo."""
    from parallel_manga_translator.ui.text_exchange import export_filename

    try:
        payload = manager.export_job_texts(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="El trabajo no existe.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    nombre = export_filename(str(payload.get("titulo") or job_id))
    return JSONResponse(payload, headers={"Content-Disposition": f'attachment; filename="{nombre}"'})


@app.post("/api/jobs/{job_id}/textos")
def import_job_texts(job_id: str, payload: Dict[str, Any] = Body(...), dry_run: bool = False):
    """Aplica un archivo de textos editado fuera y vuelve a renderizar lo que cambie.

    `dry_run=true` devuelve el mismo informe sin tocar nada: es lo que la UI enseña para
    confirmar, porque una importación cambia el trabajo entero de una vez.
    """
    try:
        return manager.import_job_texts(job_id, payload, dry_run=bool(dry_run))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="El trabajo no existe.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/export")
def export_job(job_id: str):
    try:
        path = manager.create_export_zip(job_id)
        return FileResponse(str(path), media_type="application/zip", filename=path.name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/pages/{page_index}/render")
def render_page(job_id: str, page_index: int, request: RenderRequest):
    try:
        return manager.save_manual_render(
            job_id,
            page_index,
            [region.model_dump() if hasattr(region, "model_dump") else region.dict() for region in request.regions],
            [stroke.model_dump() if hasattr(stroke, "model_dump") else stroke.dict() for stroke in request.brush_strokes],
            operation=request.operation,
            inpaint_model=request.inpaint_model,
            background_revision=request.background_revision,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/pages/{page_index}/ocr-region")
def ocr_manual_region(job_id: str, page_index: int, request: OcrRegionRequest):
    try:
        return manager.transcribe_manual_region(job_id, page_index, request.bbox, translate=request.translate)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/pages/{page_index}/translate-region")
def translate_manual_region(job_id: str, page_index: int, request: TranslateRegionRequest):
    try:
        return manager.translate_manual_text(job_id, page_index, request.original_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/pages/{page_index}/region-preview")
def render_region_preview(job_id: str, page_index: int, request: RegionPreviewRequest):
    try:
        content = manager.render_region_preview(
            job_id,
            page_index,
            request.region.model_dump() if hasattr(request.region, "model_dump") else request.region.dict(),
        )
        return Response(content=content, media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/pages/{page_index}/region-metrics")
def region_metrics(job_id: str, page_index: int, request: RegionPreviewRequest):
    try:
        return manager.resolve_region_metrics(
            job_id,
            page_index,
            request.region.model_dump() if hasattr(request.region, "model_dump") else request.region.dict(),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/pages/{page_index}/reset")
def reset_page(job_id: str, page_index: int):
    try:
        return manager.reset_manual_render(job_id, page_index)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.on_event("shutdown")
def shutdown_job_manager() -> None:
    manager.shutdown(timeout=2.0)


def main() -> None:
    import uvicorn

    host = os.getenv("PMT_UI_HOST", "127.0.0.1")
    port = int(os.getenv("PMT_UI_PORT", "7860"))
    uvicorn.run("parallel_manga_translator.ui.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
