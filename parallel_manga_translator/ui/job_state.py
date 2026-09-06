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
from typing import Any, Dict, List

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
class JobOptions:
    source_language: str = "Japonés"
    target_language: str = "Español"
    detection_engine: str = "auto"
    transcription_engine: str = "auto"
    translator: str = "llm"  # google | llm
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
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "options": asdict(job.options) if hasattr(job.options, "__dataclass_fields__") else job.options,
        "pages": [page_to_public(page, job.job_id) for page in job.pages],
    }


__all__ = [
    "PAGE_PATH_FIELDS",
    "JobOptions",
    "JobState",
    "PageState",
    "job_to_public",
    "normalize_choice",
    "page_to_public",
    "normalized_output_name",
    "page_of",
    "rebase_stored_path",
]
