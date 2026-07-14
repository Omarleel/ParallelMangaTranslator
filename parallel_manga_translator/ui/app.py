from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from parallel_manga_translator.config.constants import normalizar_modelo_inpaint
from parallel_manga_translator.ui.job_manager import JobManager, JobOptions, normalize_choice, job_to_public

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
    allow_methods=["GET", "POST"],
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


class BrushStrokePatch(BaseModel):
    points: List[List[float]] = Field(default_factory=list)
    radius: int = 18
    mode: str = "restore_original"
    applied: bool = False


class RenderRequest(BaseModel):
    regions: List[RegionPatch]
    brush_strokes: List[BrushStrokePatch] = Field(default_factory=list)
    operation: str = "render"


class RegionPreviewRequest(BaseModel):
    region: RegionPatch


class OcrRegionRequest(BaseModel):
    bbox: List[float] = Field(min_length=4, max_length=4)
    translate: bool = True


class TranslateRegionRequest(BaseModel):
    original_text: str = ""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": manager.list_jobs()}


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
            inpaint_model=normalizar_modelo_inpaint(inpaint_model, "auto"),
            page_max_retries=max(0, min(20, int(page_max_retries))),
            retry_backoff_seconds=max(0.0, min(300.0, float(retry_backoff_seconds))),
        )
        job = manager.create_job_from_uploads(files=images, zip_file=zip_file, title=title, options=options)
        manager.start_job(job.job_id)
        return job_to_public(job)
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
