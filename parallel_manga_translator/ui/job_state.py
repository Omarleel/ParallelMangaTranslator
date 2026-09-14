"""Modelo de datos de un trabajo de la UI.

Vivía dentro de `job_manager.py`, junto al orquestador que lo manipula. Sacarlo no es
cosmético: mientras el estado estuvo ahí, cualquier colaborador que necesitara leer o
escribir un `JobState` tenía que importar el módulo del god object entero, lo que cerraba
un ciclo y hacía imposible extraer nada. Este módulo no importa nada de la UI.

`manifest.json` es la fuente de verdad de un trabajo; SQLite sólo guarda el orden de la
cola. Estas dataclases son exactamente lo que se serializa ahí.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from parallel_manga_translator.io.image_naming import normalized_page_output_name

#: Campos de `PageState` que guardan rutas absolutas y hay que reanclar si el trabajo
#: cambia de carpeta. Si añades un campo de ruta a `PageState`, va aquí también, o se
#: quedará apuntando al sitio viejo sin que nada falle hasta que alguien abra la página.
PAGE_PATH_FIELDS = (
    "original_path",
    "clean_path",
    "translated_path",
    "corrected_path",
    "corrections_path",
    "manual_background_path",
)


def normalize_choice(value: Any, default: str = "auto") -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"", "none", "null", "nil", "default"}:
        return default
    return normalized


def rebase_stored_path(value: str, old_root: str, new_root: Path) -> str:
    """Reancla una ruta absoluta guardada bajo `old_root` para que apunte a `new_root`."""
    if not value or not old_root:
        return value
    stored = value.replace("\\", "/")
    old = old_root.replace("\\", "/").rstrip("/")
    if not old:
        return value
    candidate = stored if os.name != "nt" else stored.lower()
    prefix = old if os.name != "nt" else old.lower()
    if not (candidate + "/").startswith(prefix + "/"):
        return value
    suffix = stored[len(old):].lstrip("/")
    return str(new_root / suffix) if suffix else str(new_root)


def page_of(job: "JobState", page_index: int) -> "PageState":
    """La página `page_index` de un trabajo, o `IndexError` si no existe.

    Es una comprobación de rango, no lógica de orquestación: vivía en `JobManager` y seis
    métodos del editor la llamaban por `self`, lo que ataba el editor al manager sin razón.
    """
    if page_index < 0 or page_index >= len(job.pages):
        raise IndexError("La página solicitada no existe.")
    return job.pages[page_index]


def normalized_output_name(filename: str, page_index: int) -> str:
    return normalized_page_output_name(filename, page_index)




@dataclass
class TranslationEvent:
    """Evento persistente de una solicitud de traducción/retraducción."""

    timestamp: float = field(default_factory=time.time)
    kind: str = "info"
    level: str = "info"
    message: str = ""
    page_index: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationRunState:
    """Una solicitud concreta de traducción, separada del trabajo que la contiene.

    Se guarda en ``manifest.json`` para poder cerrar PMT y continuar otro día con el
    mismo proveedor/modelo y exactamente las páginas que faltaban.
    """

    run_id: str
    operation: str = "retranslate"
    translator: str = "llm"
    provider: str = "groq"
    model: str = ""
    target_language: str = "Español"
    overwrite_manual: bool = False
    status: str = "queued"  # queued | processing | paused | completed | failed | cancelled
    page_indices: List[int] = field(default_factory=list)
    pending_pages: List[int] = field(default_factory=list)
    completed_pages: List[int] = field(default_factory=list)
    current_page: Optional[int] = None
    message: str = ""
    last_error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    updated_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    events: List[TranslationEvent] = field(default_factory=list)

@dataclass
class JobOptions:
    source_language: str = "Japonés"
    target_language: str = "Español"
    detection_engine: str = "auto"
    transcription_engine: str = "auto"
    translator: str = "llm"  # google | llm
    # Fuente de regiones por trabajo: yolo | comic_text_detector. Se elige por trabajo
    # y no globalmente porque su ventaja depende del material, medido en dataset_eval.
    region_source: str = "yolo"
    # Que se le pide al pipeline: traducir (todo), limpiar (solo borrar el texto) o
    # limpiar_transcribir (deja Transcripción.json, sin traducir ni rotular).
    modo: str = "traducir"
    inpaint_model: str = "auto"
    page_max_retries: int = 2
    retry_backoff_seconds: float = 2.0


@dataclass
class PageState:
    index: int
    source_filename: str
    output_filename: str
    status: str = "pending"  # pending | processing | ready | failed
    message: str = ""
    original_path: str = ""
    clean_path: str = ""
    translated_path: str = ""
    corrected_path: str = ""
    corrections_path: str = ""
    manual_background_path: str = ""
    background_revision: str = "base"
    regions: List[Dict[str, Any]] = field(default_factory=list)
    brush_strokes: List[Dict[str, Any]] = field(default_factory=list)
    attempt_count: int = 0
    last_error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    updated_at: float = field(default_factory=time.time)

    @property
    def display_status(self) -> str:
        if self.corrected_path and Path(self.corrected_path).exists():
            return "corrected"
        return self.status


@dataclass
class JobState:
    job_id: str
    title: str
    root_dir: str
    input_dir: str
    output_dir: str
    status: str = "queued"  # queued | processing | ready | failed
    message: str = "Esperando inicio del procesamiento."
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    pages: List[PageState] = field(default_factory=list)
    processed_count: int = 0
    failed_count: int = 0
    active_page: int = 0
    options: JobOptions = field(default_factory=JobOptions)
    pause_requested: bool = False
    cancel_requested: bool = False
    resume_requested: bool = False
    recovery_count: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    # Qué debe hacer el worker con este trabajo cuando lo tome de la cola. La
    # retraducción reutiliza la misma cola persistente y el mismo worker único, así
    # que necesita decir con qué operación vuelve a entrar.
    pending_operation: str = "process"  # process | retranslate
    retranslate_pages: List[int] = field(default_factory=list)
    # Historial persistente de solicitudes de traducción. La solicitud activa conserva
    # proveedor/modelo y páginas pendientes para poder reanudar incluso otro día.
    translation_runs: List[TranslationRunState] = field(default_factory=list)
    active_translation_run_id: str = ""

    @property
    def total_count(self) -> int:
        return len(self.pages)

    @property
    def progress(self) -> float:
        total = self.total_count or 1
        return round(((self.processed_count + self.failed_count) / total) * 100, 2)


def page_to_public(page: PageState, job_id: str) -> Dict[str, Any]:
    corrected_exists = bool(page.corrected_path and Path(page.corrected_path).exists())
    return {
        "index": page.index,
        "source_filename": page.source_filename,
        "output_filename": page.output_filename,
        "status": page.status,
        "display_status": "corrected" if corrected_exists else page.status,
        "message": page.message,
        "attempt_count": page.attempt_count,
        "last_error": page.last_error,
        "started_at": page.started_at,
        "completed_at": page.completed_at,
        "regions": page.regions,
        "brush_strokes": page.brush_strokes,
        "background_revision": page.background_revision or "base",
        "has_corrected": corrected_exists,
        "images": {
            "original": f"/api/jobs/{job_id}/pages/{page.index}/image/original",
            "clean": f"/api/jobs/{job_id}/pages/{page.index}/image/clean",
            "background": f"/api/jobs/{job_id}/pages/{page.index}/image/background",
            "translated": f"/api/jobs/{job_id}/pages/{page.index}/image/translated",
            "corrected": f"/api/jobs/{job_id}/pages/{page.index}/image/corrected",
            "current": f"/api/jobs/{job_id}/pages/{page.index}/image/current",
        },
        "updated_at": page.updated_at,
    }


def job_to_public(job: JobState) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "title": job.title,
        "status": job.status,
        "message": job.message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "processed_count": job.processed_count,
        "failed_count": job.failed_count,
        "total_count": job.total_count,
        "progress": job.progress,
        "active_page": job.active_page,
        "pause_requested": job.pause_requested,
        "cancel_requested": job.cancel_requested,
        "resume_requested": job.resume_requested,
        "recovery_count": job.recovery_count,
        "pending_operation": job.pending_operation or "process",
        "retranslate_pending": len(job.retranslate_pages or []),
        "active_translation_run_id": job.active_translation_run_id or "",
        "translation_run_count": len(job.translation_runs or []),
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "options": asdict(job.options) if hasattr(job.options, "__dataclass_fields__") else job.options,
        "pages": [page_to_public(page, job.job_id) for page in job.pages],
    }



def active_translation_run(job: "JobState") -> Optional[TranslationRunState]:
    run_id = str(getattr(job, "active_translation_run_id", "") or "")
    if not run_id:
        return None
    for run in reversed(getattr(job, "translation_runs", []) or []):
        if str(getattr(run, "run_id", "")) == run_id:
            return run
    return None


def append_translation_event(
    job: "JobState",
    kind: str,
    message: str,
    *,
    level: str = "info",
    page_index: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
    max_events: int = 600,
) -> Optional[TranslationEvent]:
    run = active_translation_run(job)
    if run is None:
        return None
    event = TranslationEvent(
        kind=str(kind or "info"),
        level=str(level or "info"),
        message=str(message or ""),
        page_index=page_index,
        details=dict(details or {}),
    )
    run.events.append(event)
    if max_events > 0 and len(run.events) > max_events:
        del run.events[:-max_events]
    run.updated_at = event.timestamp
    return event


def translation_run_to_public(run: TranslationRunState) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "operation": run.operation,
        "translator": run.translator,
        "provider": run.provider,
        "model": run.model,
        "target_language": run.target_language,
        "overwrite_manual": run.overwrite_manual,
        "status": run.status,
        "page_indices": list(run.page_indices),
        "pending_pages": list(run.pending_pages),
        "completed_pages": list(run.completed_pages),
        "current_page": run.current_page,
        "message": run.message,
        "last_error": run.last_error,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "updated_at": run.updated_at,
        "finished_at": run.finished_at,
        "events": [asdict(event) for event in run.events],
    }


def is_retranslating(job: "JobState") -> bool:
    """¿Este trabajo vuelve a la cola para retraducir, no para procesar?"""
    return str(job.pending_operation or "process") == "retranslate"


def recount(job: "JobState") -> None:
    """Recalcula los contadores derivados del estado de las páginas."""
    job.processed_count = sum(1 for item in job.pages if item.status == "ready")
    job.failed_count = sum(1 for item in job.pages if item.status == "failed")


def finalize_cancelled(job: "JobState") -> None:
    """Deja el trabajo en su estado terminal cancelado, conservando lo ya terminado."""
    now = time.time()
    for page in job.pages:
        if page.status in {"pending", "processing"}:
            page.status = "cancelled"
            page.message = "Cancelada por el usuario."
            page.completed_at = now
            page.updated_at = now
    job.status = "cancelled"
    job.message = "Trabajo cancelado. Las páginas ya terminadas se conservaron."
    job.cancel_requested = True
    job.pause_requested = False
    job.resume_requested = False
    job.finished_at = now
    job.updated_at = now
    recount(job)


__all__ = [
    "PAGE_PATH_FIELDS",
    "JobOptions",
    "JobState",
    "PageState",
    "TranslationEvent",
    "TranslationRunState",
    "active_translation_run",
    "append_translation_event",
    "finalize_cancelled",
    "is_retranslating",
    "job_to_public",
    "normalize_choice",
    "page_to_public",
    "recount",
    "normalized_output_name",
    "page_of",
    "rebase_stored_path",
    "translation_run_to_public",
]
