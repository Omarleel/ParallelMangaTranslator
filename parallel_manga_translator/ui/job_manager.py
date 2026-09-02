from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2

from parallel_manga_translator.cli import build_default_config, build_image_processor, prepare_assets, prepare_runtime
from parallel_manga_translator.config.constants import normalizar_modelo_inpaint
from parallel_manga_translator.inpainting import AOTInpainter, LamaInpainterMPE, LamaLarge, OpenCVInpainter
from parallel_manga_translator.config.runtime_config import set_active_config
from parallel_manga_translator.infrastructure.execution_control import (
    ExecutionControl,
    JobCancelledError,
    JobPausedError,
    execution_control_scope,
)
from parallel_manga_translator.infrastructure.logging_config import configure_logging, get_logger
from parallel_manga_translator.infrastructure.gpu_scheduler import gpu_slot
from parallel_manga_translator.io.image_naming import normalized_page_output_name
from parallel_manga_translator.ui.manual_renderer import apply_background_brush_strokes, apply_pending_inpaint_only, parse_brush_strokes, parse_manual_regions, read_corrections, read_corrections_payload, render_manual_composite, render_manual_page, render_manual_region_preview, resolve_manual_region_metrics, write_corrections
from parallel_manga_translator.ui.queue_adapter import CapturingJsonQueue
from parallel_manga_translator.ui.persistent_queue import PersistentJobQueue

logger = get_logger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PROJECT_JOBS_DIR = Path(os.getenv("PMT_UI_JOBS_DIR", ".pmt_ui_jobs")).resolve()


def normalize_choice(value: Any, default: str = "auto") -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"", "none", "null", "nil", "default"}:
        return default
    return normalized


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

    @property
    def total_count(self) -> int:
        return len(self.pages)

    @property
    def progress(self) -> float:
        total = self.total_count or 1
        return round(((self.processed_count + self.failed_count) / total) * 100, 2)


def normalized_output_name(filename: str, page_index: int) -> str:
    return normalized_page_output_name(filename, page_index)


def natural_sort_key(filename: str) -> List[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", Path(filename).name)]


def safe_flat_name(original_name: str, used: set[str]) -> str:
    candidate = Path(original_name).name.replace("\x00", "")
    candidate = candidate or f"page_{len(used) + 1}.jpg"
    stem, ext = os.path.splitext(candidate)
    ext = ext.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise ValueError("Formato de imagen no soportado.")
    safe_stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(". ") or f"page_{len(used) + 1}"
    candidate = f"{safe_stem}{ext}"
    base = candidate
    counter = 2
    while candidate.lower() in used:
        candidate = f"{safe_stem}_{counter}{ext}"
        counter += 1
    used.add(candidate.lower())
    return candidate


def safe_export_slug(value: str, fallback: str = "manga") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip()).strip("._-")
    return slug[:80] or fallback


def unique_arcname(folder: str, filename: str, used: set[str]) -> str:
    candidate = f"{folder}/{Path(filename).name}"
    if candidate.lower() not in used:
        used.add(candidate.lower())
        return candidate
    stem, ext = os.path.splitext(Path(filename).name)
    counter = 2
    while True:
        candidate = f"{folder}/{stem}_{counter}{ext}"
        if candidate.lower() not in used:
            used.add(candidate.lower())
            return candidate
        counter += 1


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
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "options": asdict(job.options) if hasattr(job.options, "__dataclass_fields__") else job.options,
        "pages": [page_to_public(page, job.job_id) for page in job.pages],
    }


class JobManager:
    def __init__(
        self,
        jobs_root: Path = PROJECT_JOBS_DIR,
        config_path: str = "config.yaml",
        *,
        start_worker: bool = True,
    ) -> None:
        self.jobs_root = Path(jobs_root).resolve()
        self.config_path = config_path
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, JobState] = {}
        self._lock = threading.RLock()
        # El pipeline usa configuración global y modelos CUDA grandes. Un único worker
        # garantiza aislamiento entre trabajos y evita duplicar memoria de GPU.
        self._processing_lock = threading.Lock()
        self._assets_prepared = False
        self._manual_inpainters: Dict[str, Any] = {}
        self._manual_inpaint_lock = threading.Lock()
        self._queue = PersistentJobQueue(self.jobs_root / "queue.sqlite3")
        self._queue_condition = threading.Condition(threading.RLock())
        self._controls: Dict[str, ExecutionControl] = {}
        self._shutdown = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._queue.recover_processing()
        self._recover_interrupted_jobs()
        if start_worker:
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="pmt-persistent-worker",
                daemon=True,
            )
            self._worker_thread.start()

    def create_job_from_paths(self, *, title: str, input_dir: Path, output_dir: Path, root_dir: Path, options: JobOptions | None = None) -> JobState:
        image_files = [p.name for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        image_files = sorted(image_files, key=natural_sort_key)
        if not image_files:
            raise ValueError("No se encontraron imágenes válidas (.jpg, .jpeg, .png, .bmp o .webp).")

        pages = []
        for idx, filename in enumerate(image_files):
            output_name = normalized_output_name(filename, idx)
            pages.append(
                PageState(
                    index=idx,
                    source_filename=filename,
                    output_filename=output_name,
                    original_path=str(input_dir / filename),
                    clean_path=str(output_dir / "limpieza" / output_name),
                    translated_path=str(output_dir / "traduccion" / output_name),
                    corrected_path=str(output_dir / "corregida" / output_name),
                    corrections_path=str(output_dir / "correcciones" / f"{Path(output_name).stem}.json"),
                )
            )
        job = JobState(
            job_id=root_dir.name,
            title=title,
            root_dir=str(root_dir),
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            pages=pages,
            options=options or JobOptions(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._save_manifest(job)
        return job

    def create_job_from_uploads(self, files: Sequence[Any], zip_file: Optional[Any] = None, title: str = "", options: JobOptions | None = None) -> JobState:
        job_id = uuid.uuid4().hex[:12]
        root_dir = self.jobs_root / job_id
        input_dir = root_dir / "entrada"
        output_dir = root_dir / "outputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()

        if zip_file is not None and getattr(zip_file, "filename", ""):
            archive_path = root_dir / "upload.zip"
            self._copy_upload_file(zip_file, archive_path)
            self._extract_zip_images(archive_path, input_dir, used)
            title = title or Path(zip_file.filename).stem or "manga"

        for file in files or []:
            filename = getattr(file, "filename", "") or ""
            if not filename or Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            target_name = safe_flat_name(filename, used)
            self._copy_upload_file(file, input_dir / target_name)
            title = title or Path(filename).parent.name or "manga"

        return self.create_job_from_paths(
            title=title.strip() or "Manga sin título",
            input_dir=input_dir,
            output_dir=output_dir,
            root_dir=root_dir,
            options=options or JobOptions(),
        )

    @staticmethod
    def _copy_upload_file(upload: Any, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as out:
            upload.file.seek(0)
            shutil.copyfileobj(upload.file, out)

    def _extract_zip_images(self, archive_path: Path, input_dir: Path, used: set[str]) -> None:
        max_uncompressed = 1024 * 1024 * 1024  # 1 GB límite razonable para evitar zips accidentales enormes.
        total = 0
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                ext = Path(info.filename).suffix.lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                total += info.file_size
                if total > max_uncompressed:
                    raise ValueError("El zip supera el límite de 1 GB de imágenes descomprimidas.")
                target_name = safe_flat_name(info.filename, used)
                with archive.open(info) as src, (input_dir / target_name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    def start_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        with self._lock:
            if job.status in {"ready", "cancelled"}:
                raise ValueError(f"El trabajo ya está en estado {job.status}.")
            if job.status == "failed":
                for page in job.pages:
                    if page.status == "failed":
                        page.status = "pending"
                        page.message = "Pendiente de un nuevo intento solicitado por el usuario."
                        page.attempt_count = 0
                        page.last_error = ""
                        page.completed_at = 0.0
                self._recount(job)
            job.pause_requested = False
            job.cancel_requested = False
            job.resume_requested = False
            job.status = "queued"
            job.message = "Trabajo añadido a la cola persistente."
            job.updated_at = time.time()
            self._save_manifest(job)
        self._queue.enqueue(job.job_id)
        self._wake_worker()

    def pause_job(self, job_id: str) -> JobState:
        job = self.get_job(job_id)
        with self._lock:
            if job.status in {"ready", "failed", "cancelled"}:
                raise ValueError(f"No se puede pausar un trabajo en estado {job.status}.")
            job.pause_requested = True
            job.updated_at = time.time()
            control = self._controls.get(job_id)
            if control is not None:
                job.status = "pausing"
                job.message = "Pausa solicitada; se detendrá en el próximo punto seguro."
                control.request_pause()
            else:
                self._queue.remove(job_id)
                job.status = "paused"
                job.message = "Trabajo pausado en la cola."
            self._save_manifest(job)
        return job

    def resume_job(self, job_id: str) -> JobState:
        job = self.get_job(job_id)
        with self._lock:
            if job.status not in {"paused", "pausing", "resuming"}:
                raise ValueError("El trabajo no está pausado.")
            job.pause_requested = False
            job.updated_at = time.time()
            control = self._controls.get(job_id)
            if control is not None and not control.pause_triggered:
                # La pausa aún no llegó a un checkpoint: se puede retirar sin perder
                # el trabajo actual ni repetir la página.
                job.status = "processing"
                job.message = "Solicitud de pausa retirada; el procesamiento continúa."
                job.resume_requested = False
                control.resume()
            elif control is not None:
                # El worker ya está desenrollando la página. Se reencolará al salir.
                job.status = "resuming"
                job.message = "Reanudación solicitada; el trabajo volverá a la cola al liberar el worker."
                job.resume_requested = True
            else:
                job.status = "queued"
                job.message = "Trabajo reanudado y devuelto a la cola."
                job.resume_requested = False
                self._queue.enqueue(job_id, preserve_time=False)
            self._save_manifest(job)
        self._wake_worker()
        return job

    def cancel_job(self, job_id: str) -> JobState:
        job = self.get_job(job_id)
        with self._lock:
            if job.status in {"ready", "failed", "cancelled"}:
                return job
            job.cancel_requested = True
            job.pause_requested = False
            job.resume_requested = False
            job.updated_at = time.time()
            self._queue.remove(job_id)
            control = self._controls.get(job_id)
            if control is not None:
                job.status = "cancelling"
                job.message = "Cancelación solicitada; se detendrá en el próximo punto seguro."
                control.request_cancel()
            else:
                self._finalize_cancelled_job(job)
            self._save_manifest(job)
        self._wake_worker()
        return job

    def get_job(self, job_id: str) -> JobState:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = self._load_manifest(job_id)
                self._jobs[job_id] = job
            return job

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            loaded = list(self._jobs.values())
        known = {job.job_id for job in loaded}
        for manifest in self.jobs_root.glob("*/manifest.json"):
            job_id = manifest.parent.name
            if job_id not in known:
                try:
                    job = self._load_manifest(job_id)
                    loaded.append(job)
                    with self._lock:
                        self._jobs[job_id] = job
                except Exception:
                    continue
        result = []
        for job in sorted(loaded, key=lambda item: item.created_at, reverse=True):
            payload = job_to_public(job)
            payload["queue_position"] = self._queue.position(job.job_id)
            result.append(payload)
        return result

    def shutdown(self, timeout: float = 2.0) -> None:
        self._shutdown.set()
        self._wake_worker()
        thread = self._worker_thread
        if thread and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))

    def _wake_worker(self) -> None:
        with self._queue_condition:
            self._queue_condition.notify_all()

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            job_id = self._queue.pop_next()
            if not job_id:
                with self._queue_condition:
                    self._queue_condition.wait(timeout=0.75)
                continue
            try:
                job = self.get_job(job_id)
                if job.status in {"paused", "cancelled", "ready"}:
                    continue
                self._run_job(job_id)
            except Exception as exc:  # el worker no debe morir por un manifiesto defectuoso
                logger.exception("El worker persistente falló para %s: %s", job_id, exc)
                try:
                    job = self.get_job(job_id)
                    self._mark_job(job, status="failed", message=f"Error del worker: {exc}")
                except Exception:
                    pass
            finally:
                self._queue.complete(job_id)

    def _recover_interrupted_jobs(self) -> None:
        manifest_ids = set()
        for manifest in self.jobs_root.glob("*/manifest.json"):
            job_id = manifest.parent.name
            manifest_ids.add(job_id)
            try:
                job = self._load_manifest(job_id)
            except Exception as exc:
                logger.warning("No se pudo recuperar el manifiesto %s: %s", manifest, exc)
                self._queue.remove(job_id)
                continue

            changed = False
            for page in job.pages:
                if page.status == "processing":
                    if Path(page.clean_path).exists() and Path(page.translated_path).exists():
                        page.status = "ready"
                        page.message = "Página recuperada desde archivos ya generados."
                        page.completed_at = page.completed_at or time.time()
                    else:
                        page.status = "pending"
                        page.message = "Página interrumpida recuperada; pendiente de reintento."
                    page.updated_at = time.time()
                    changed = True

            self._recount(job)
            if job.cancel_requested or job.status == "cancelling":
                self._finalize_cancelled_job(job)
                self._queue.remove(job_id)
                changed = True
            elif job.pause_requested or job.status in {"paused", "pausing"}:
                job.status = "paused"
                job.pause_requested = True
                job.message = "Trabajo pausado recuperado después del reinicio."
                self._queue.remove(job_id)
                changed = True
            elif job.status in {"processing", "queued", "resuming"}:
                if all(page.status == "ready" for page in job.pages):
                    job.status = "ready"
                    job.message = "Trabajo recuperado: todas las páginas estaban completas."
                    job.finished_at = job.finished_at or time.time()
                    self._queue.remove(job_id)
                else:
                    job.status = "queued"
                    job.message = "Trabajo recuperado y devuelto a la cola persistente."
                    job.recovery_count += 1
                    self._queue.enqueue(job_id)
                changed = True
            elif job.status in {"ready", "failed", "cancelled"}:
                self._queue.remove(job_id)

            if changed:
                job.updated_at = time.time()
                self._save_manifest(job)
            self._jobs[job_id] = job

        for stale_id in set(self._queue.all_ids()) - manifest_ids:
            self._queue.remove(stale_id)

    def _on_control_paused(self, job_id: str) -> None:
        try:
            job = self.get_job(job_id)
            with self._lock:
                job.status = "paused"
                job.pause_requested = True
                job.message = "Trabajo pausado en un punto seguro."
                job.updated_at = time.time()
                self._save_manifest(job)
        except Exception:
            logger.exception("No se pudo guardar la pausa del trabajo %s", job_id)

    def _on_control_resumed(self, job_id: str) -> None:
        try:
            job = self.get_job(job_id)
            with self._lock:
                job.status = "processing"
                job.pause_requested = False
                job.resume_requested = False
                job.message = "Procesamiento reanudado."
                job.updated_at = time.time()
                self._save_manifest(job)
        except Exception:
            logger.exception("No se pudo guardar la reanudación del trabajo %s", job_id)

    @staticmethod
    def _recount(job: JobState) -> None:
        job.processed_count = sum(1 for item in job.pages if item.status == "ready")
        job.failed_count = sum(1 for item in job.pages if item.status == "failed")

    def _finalize_cancelled_job(self, job: JobState) -> None:
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
        self._recount(job)

    @staticmethod
    def _remove_partial_outputs(page: PageState) -> None:
        for path_value in (page.clean_path, page.translated_path):
            path = Path(path_value)
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _cooperative_backoff(control: ExecutionControl, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            control.checkpoint()
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

    def image_path(self, job_id: str, page_index: int, variant: str) -> Path:
        job = self.get_job(job_id)
        page = self._page(job, page_index)
        if variant == "original":
            return Path(page.original_path)
        if variant == "clean":
            return Path(page.clean_path)
        if variant == "translated":
            return Path(page.translated_path)
        if variant == "corrected":
            return Path(page.corrected_path)
        if variant == "current":
            corrected = Path(page.corrected_path)
            return corrected if corrected.exists() else Path(page.translated_path)
        raise ValueError("Variante de imagen no soportada.")

    def create_export_zip(self, job_id: str) -> Path:
        job = self.get_job(job_id)
        export_dir = Path(job.root_dir) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_name = f"{safe_export_slug(job.title, job.job_id)}_resultado.zip"
        export_path = export_dir / export_name
        if export_path.exists():
            export_path.unlink()

        used_names: set[str] = set()
        exported_pages: List[Dict[str, Any]] = []
        with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for page in sorted(job.pages, key=lambda item: item.index):
                corrected = Path(page.corrected_path)
                translated = Path(page.translated_path)
                if corrected.exists():
                    image_path = corrected
                    variant = "corregida"
                elif page.status == "ready" and translated.exists():
                    image_path = translated
                    variant = "traduccion"
                else:
                    continue

                image_arcname = unique_arcname("imagenes_finales", page.output_filename, used_names)
                archive.write(image_path, image_arcname)

                correction_path = Path(page.corrections_path)
                correction_arcname = ""
                if correction_path.exists():
                    correction_arcname = unique_arcname("correcciones", correction_path.name, used_names)
                    archive.write(correction_path, correction_arcname)

                exported_pages.append(
                    {
                        "page_index": page.index,
                        "source_filename": page.source_filename,
                        "output_filename": page.output_filename,
                        "variant": variant,
                        "image": image_arcname,
                        "corrections": correction_arcname,
                    }
                )

            if not exported_pages:
                raise ValueError("Todavía no hay páginas listas para exportar.")

            manifest = {
                "title": job.title,
                "job_id": job.job_id,
                "exported_at": time.time(),
                "included_pages": len(exported_pages),
                "total_pages": len(job.pages),
                "note": "Cada imagen final usa la corrección manual si existe; si no, usa la traducción automática lista.",
                "pages": exported_pages,
            }
            archive.writestr("manifest_export.json", json.dumps(manifest, ensure_ascii=False, indent=2))

        return export_path

    def _resolve_manual_inpaint_model(self, job: JobState, requested: str | None) -> str:
        raw = str(requested or "job").strip().lower()
        if raw in {"job", "configured", "config", "default"}:
            raw = str(getattr(job.options, "inpaint_model", "auto") or "auto")
        model = normalizar_modelo_inpaint(raw, "auto")
        # El pincel manual siempre trabaja con una máscara raster. B/N necesita cajas
        # de detección, así que para este flujo se usa LaMa como opción automática
        # de calidad y OpenCV solo cuando el usuario lo selecciona explícitamente.
        if model in {"auto", "B/N"}:
            return "lama_mpe"
        return model

    def _manual_inpaint_callable(self, model_name: str):
        factories = {
            "opencv-tela": OpenCVInpainter,
            "lama_mpe": LamaInpainterMPE,
            "lama_large_512px": LamaLarge,
            "aot": AOTInpainter,
        }
        factory = factories.get(model_name)
        if factory is None:
            raise ValueError(f"Modelo de inpainting manual no soportado: {model_name}")

        def run(image, mask):
            if model_name == "opencv-tela":
                return OpenCVInpainter().inpaint(image, mask)
            with self._manual_inpaint_lock:
                inpainter = self._manual_inpainters.get(model_name)
                if inpainter is None:
                    inpainter = factory()
                    self._manual_inpainters[model_name] = inpainter

                async def infer():
                    if getattr(inpainter, "model", None) is None and hasattr(inpainter, "_load"):
                        await inpainter._load()
                    with gpu_slot("ui.manual_inpaint", enabled=True):
                        if hasattr(inpainter, "_inpaint"):
                            result = inpainter._inpaint(image, mask)
                        else:
                            result = inpainter.inpaint(image, mask)
                        if asyncio.iscoroutine(result):
                            result = await result
                        return result

                return asyncio.run(infer())

        # Perfil de recorte conservador. No cambia modelo, precisión ni pesos: solo
        # evita ejecutar la red sobre partes lejanas de la página que el pincel no
        # puede modificar. LaMa Large conserva una ventana mínima de 1024x1024 y
        # 384 px de contexto alrededor de trazos mayores; sigue en fp32.
        crop_profiles = {
            "lama_large_512px": {"enabled": True, "min_side": 1024, "context_px": 384, "alignment": 64, "max_model_side": 1536},
            "lama_mpe": {"enabled": True, "min_side": 896, "context_px": 320, "alignment": 64, "max_model_side": 1024},
            "aot": {"enabled": True, "min_side": 768, "context_px": 256, "alignment": 64, "max_model_side": 1024},
        }
        run._pmt_crop_profile = crop_profiles.get(model_name, {"enabled": False})
        return run

    def _inpaint_backup_path(self, job: JobState, page: PageState) -> Path:
        backup_dir = Path(job.output_dir) / ".ui_backups" / "before_inpaint"
        return backup_dir / page.output_filename

    @staticmethod
    def _without_mask_erasers(brush_strokes):
        return [
            stroke
            for stroke in brush_strokes
            if (stroke.mode or "").strip().lower() not in {"mask_eraser", "erase_mask", "eraser"}
        ]

    def _manual_background_dir(self, job: JobState, page: PageState) -> Path:
        return Path(job.output_dir) / ".ui_backgrounds" / f"page_{page.index:04d}"

    def _background_revision_path(self, job: JobState, page: PageState, revision: str) -> Path:
        safe_revision = re.sub(r"[^A-Za-z0-9_-]+", "", str(revision or ""))
        if not safe_revision:
            raise ValueError("La revisión de fondo no es válida.")
        return self._manual_background_dir(job, page) / f"{safe_revision}.png"

    def _resolve_background_revision(
        self,
        job: JobState,
        page: PageState,
        requested_revision: str | None,
    ) -> tuple[str, Path]:
        revision = str(requested_revision or page.background_revision or "base").strip() or "base"
        if revision == "base":
            clean_path = Path(page.clean_path)
            if not clean_path.exists():
                raise ValueError("Falta la imagen limpia para reconstruir el fondo.")
            return "base", clean_path

        revision_path = self._background_revision_path(job, page, revision)
        if revision_path.exists():
            return revision, revision_path

        # Compatibilidad con manifiestos creados por versiones intermedias: si la
        # revisión actual apunta a un archivo explícito, se acepta y se archiva para
        # que desde este momento también pueda participar en deshacer/rehacer.
        legacy_path = Path(page.manual_background_path) if page.manual_background_path else None
        if revision == page.background_revision and legacy_path and legacy_path.exists():
            revision_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, revision_path)
            return revision, revision_path
        raise ValueError("La revisión de fondo solicitada ya no está disponible.")

    def _new_background_revision_path(self, job: JobState, page: PageState) -> tuple[str, Path]:
        revision = uuid.uuid4().hex
        path = self._background_revision_path(job, page, revision)
        path.parent.mkdir(parents=True, exist_ok=True)
        return revision, path

    @staticmethod
    def _has_background_brush_strokes(brush_strokes) -> bool:
        return any(
            (stroke.mode or "restore_clean").strip().lower() != "inpaint"
            for stroke in brush_strokes
        )

    def save_manual_render(
        self,
        job_id: str,
        page_index: int,
        regions_payload: Sequence[Dict[str, Any]],
        brush_strokes_payload: Sequence[Dict[str, Any]] | None = None,
        operation: str = "render",
        inpaint_model: str | None = None,
        background_revision: str | None = None,
    ) -> Dict[str, Any]:
        job = self.get_job(job_id)
        page = self._page(job, page_index)
        if page.status != "ready":
            raise ValueError("La página todavía no está lista para edición.")
        clean_path = Path(page.clean_path)
        original_path = Path(page.original_path)
        translated_path = Path(page.translated_path)
        if not clean_path.exists() or not original_path.exists() or not translated_path.exists():
            raise ValueError("Faltan imágenes base para renderizar la corrección.")
        image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudo leer la imagen limpia.")
        h, w = image.shape[:2]
        regions = parse_manual_regions(regions_payload, w, h)
        brush_strokes = parse_brush_strokes(brush_strokes_payload or [], w, h)
        normalized_operation = (operation or "render").strip().lower()

        # El historial de la UI guarda esta revisión junto con bbox/texto. Al
        # deshacer/rehacer, el servidor vuelve exactamente a la capa de limpieza
        # correspondiente y después recompone las regiones, sin reutilizar texto
        # rasterizado de una versión posterior.
        active_revision, active_background = self._resolve_background_revision(
            job, page, background_revision
        )
        next_revision = active_revision
        next_background = active_background

        if normalized_operation == "inpaint":
            target_revision, target_path = self._new_background_revision_path(job, page)
            selected_manual_model = self._resolve_manual_inpaint_model(job, inpaint_model)
            changed = apply_pending_inpaint_only(
                base_path=active_background,
                output_path=target_path,
                brush_strokes=brush_strokes,
                inpaint_fn=self._manual_inpaint_callable(selected_manual_model),
            )
            if changed:
                next_revision, next_background = target_revision, target_path
            else:
                target_path.unlink(missing_ok=True)
            for stroke in brush_strokes:
                if (stroke.mode or "").strip().lower() == "inpaint":
                    stroke.applied = True
            brush_strokes = self._without_mask_erasers(brush_strokes)
        elif self._has_background_brush_strokes(brush_strokes):
            target_revision, target_path = self._new_background_revision_path(job, page)
            changed = apply_background_brush_strokes(
                base_path=active_background,
                clean_path=clean_path,
                original_path=original_path,
                output_path=target_path,
                brush_strokes=brush_strokes,
            )
            if changed:
                next_revision, next_background = target_revision, target_path
            else:
                target_path.unlink(missing_ok=True)

        # La salida final se reconstruye siempre como dos capas: fondo/limpieza y
        # regiones de texto. Nunca parte de corrected_path, que puede contener una
        # posición de texto ya rasterizada y causar fantasmas al deshacer un movimiento.
        render_manual_composite(
            background_path=next_background,
            original_path=original_path,
            output_path=page.corrected_path,
            regions=regions,
        )

        # Las pinceladas ya quedaron incorporadas en una revisión inmutable del fondo.
        # El identificador de esa revisión sí permanece en el historial del navegador.
        brush_strokes = []

        # Conservamos source_bbox como metadato de compatibilidad. Ya no se necesita
        # para borrar texto previo porque la composición nunca usa una imagen que tenga
        # texto editable rasterizado, pero avanzar el origen sigue siendo útil para
        # proyectos y pruebas creados con versiones anteriores.
        for region in regions:
            region.source_bbox = region.bbox

        write_corrections(page.corrections_path, regions, brush_strokes)
        with self._lock:
            previous_by_index = {
                int(region.get("index", idx)): region
                for idx, region in enumerate(page.regions)
                if isinstance(region, dict)
            }
            payload_by_index = {
                int(region.get("index", idx)): region
                for idx, region in enumerate(regions_payload)
                if isinstance(region, dict)
            }
            page.regions = [
                {
                    **previous_by_index.get(region.index, {}),
                    "index": region.index,
                    "bbox": list(region.bbox),
                    "source_bbox": list(region.source_bbox or region.bbox),
                    "original_text": payload_by_index.get(region.index, {}).get("original_text", previous_by_index.get(region.index, {}).get("original_text", "")),
                    "translated_text": region.text,
                    "style": region.style,
                    "type": previous_by_index.get(region.index, {}).get("type", "manual" if region.manual else "dialogue"),
                    "restore_original": region.restore_original,
                    "visible": region.visible,
                    "modified": region.modified,
                    "manual": region.manual or bool(previous_by_index.get(region.index, {}).get("manual", False)),
                    "deleted": region.deleted,
                    "auto_font_size": region.auto_font_size,
                    "font_size": region.font_size,
                    "rotation_angle": region.rotation_angle,
                    "ui_layout": region.ui_layout or previous_by_index.get(region.index, {}).get("ui_layout"),
                }
                for region in regions
            ]
            page.brush_strokes = []
            page.background_revision = next_revision
            page.manual_background_path = "" if next_revision == "base" else str(next_background)
            page.updated_at = time.time()
            job.updated_at = page.updated_at
            self._save_manifest(job)
        return page_to_public(page, job_id)


    def render_region_preview(self, job_id: str, page_index: int, region_payload: Dict[str, Any]) -> bytes:
        """Rasteriza una región con la misma ruta usada por el guardado final."""
        job = self.get_job(job_id)
        page = self._page(job, page_index)
        if page.status != "ready":
            raise ValueError("La página todavía no está lista para edición.")
        clean_path = Path(page.clean_path)
        original_path = Path(page.original_path)
        if not clean_path.exists() or not original_path.exists():
            raise ValueError("Faltan imágenes base para previsualizar la región.")
        image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudo leer la imagen limpia.")
        h, w = image.shape[:2]
        regions = parse_manual_regions([region_payload], w, h)
        if not regions:
            raise ValueError("La región de previsualización no es válida.")
        return render_manual_region_preview(
            clean_path=clean_path,
            original_path=original_path,
            region=regions[0],
        )

    def resolve_region_metrics(self, job_id: str, page_index: int, region_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Resuelve la tipografía y geometría exactas de una región sin guardar cambios."""
        job = self.get_job(job_id)
        page = self._page(job, page_index)
        if page.status != "ready":
            raise ValueError("La página todavía no está lista para edición.")
        clean_path = Path(page.clean_path)
        if not clean_path.exists():
            raise ValueError("Falta la imagen limpia para medir la región.")
        image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudo leer la imagen limpia.")
        h, w = image.shape[:2]
        regions = parse_manual_regions([region_payload], w, h)
        if not regions:
            raise ValueError("La región de medición no es válida.")
        return resolve_manual_region_metrics(clean_path=clean_path, region=regions[0])

    def reset_manual_render(self, job_id: str, page_index: int) -> Dict[str, Any]:
        job = self.get_job(job_id)
        page = self._page(job, page_index)
        for raw in [page.corrected_path, page.corrections_path]:
            path = Path(raw)
            if path.exists():
                path.unlink()
        background_dir = self._manual_background_dir(job, page)
        if background_dir.exists():
            shutil.rmtree(background_dir, ignore_errors=True)
        # Restaurar regiones desde JSON del pipeline si existe.
        page.regions = self._merge_page_regions(job, page)
        page.brush_strokes = []
        page.background_revision = "base"
        page.manual_background_path = ""
        page.updated_at = time.time()
        job.updated_at = page.updated_at
        self._save_manifest(job)
        return page_to_public(page, job_id)

    def _build_config_for_job(self, job: JobState):
        config = build_default_config(self.config_path)
        options = job.options if isinstance(job.options, JobOptions) else JobOptions(**dict(job.options or {}))
        translator = normalize_choice(options.translator, "llm")
        method = "LLM" if translator == "llm" else "Tradicional"
        traditional_provider = "google" if translator == "google" else config.translation.traditional_provider
        llm = replace(config.translation.llm, provider=config.translation.llm.provider or "groq")
        translation = replace(
            config.translation,
            idioma_entrada=options.source_language or config.translation.idioma_entrada,
            idioma_salida=options.target_language or config.translation.idioma_salida,
            metodo_traduccion=method,
            traditional_provider=traditional_provider,
            llm=llm,
            project_dir=job.input_dir,
            modelo_inpaint=normalizar_modelo_inpaint(options.inpaint_model, config.translation.modelo_inpaint),
        )
        ocr = replace(
            config.ocr,
            detection_engine=normalize_choice(options.detection_engine, "auto"),
            transcription_engine=normalize_choice(options.transcription_engine, "auto"),
        )
        processing = replace(
            config.processing,
            ruta_carpeta_entrada=job.input_dir,
            cache_dir=str(Path(job.root_dir) / ".cache"),
        )
        logging = replace(config.logging, file=str(Path(job.root_dir) / "job.log"))
        return replace(config, translation=translation, ocr=ocr, processing=processing, logging=logging)

    def translate_manual_text(self, job_id: str, page_index: int, original_text: str) -> Dict[str, Any]:
        """Traduce una transcripción corregida manualmente sin volver a ejecutar OCR."""
        job = self.get_job(job_id)
        page = self._page(job, page_index)
        if page.status != "ready":
            raise ValueError("La página todavía no está lista para traducir.")

        source_text = str(original_text or "")
        if not source_text.strip():
            return {
                "original_text": source_text,
                "translated_text": "",
            }

        config = self._build_config_for_job(job)
        set_active_config(config)
        from parallel_manga_translator.translation.translator_manager import TranslatorManager

        control = ExecutionControl()
        with execution_control_scope(control):
            translator = TranslatorManager.from_config(config.translation, config.character_memory)
            translated_text = translator.traducir_textos([source_text])[0]

        with self._lock:
            job.updated_at = time.time()
            self._save_manifest(job)

        return {
            "original_text": source_text,
            "translated_text": str(translated_text or ""),
            "source_language": config.translation.idioma_entrada,
            "target_language": config.translation.idioma_salida,
            "translator": normalize_choice(job.options.translator, "llm"),
        }

    def transcribe_manual_region(self, job_id: str, page_index: int, bbox: Sequence[float], translate: bool = True) -> Dict[str, Any]:
        job = self.get_job(job_id)
        page = self._page(job, page_index)
        if page.status != "ready":
            raise ValueError("La página todavía no está lista para OCR manual.")
        image = cv2.imread(str(page.original_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudo leer la imagen original para OCR.")
        h, w = image.shape[:2]
        from parallel_manga_translator.ui.manual_renderer import _safe_box  # reutiliza validación central

        x, y, bw, bh = _safe_box(bbox, w, h)
        crop = image[y:y + bh, x:x + bw]
        if crop.size == 0:
            raise ValueError("La región seleccionada no contiene píxeles válidos.")

        config = self._build_config_for_job(job)
        set_active_config(config)
        from parallel_manga_translator.ocr.ocr_manager import OcrManager

        ocr = OcrManager(config.translation.idioma_entrada, config.ocr)
        original_text = ocr.extract_texts([crop])[0] if crop.size else ""
        rotation_angle = 0.0
        rotation_confidence = 0.0
        try:
            from parallel_manga_translator.geometry.text_orientation import estimate_text_rotation
            from parallel_manga_translator.ocr.text_detection import TextDetectionFactory

            detector = TextDetectionFactory.create(config.translation.idioma_entrada, config.ocr)
            rotation = estimate_text_rotation(detector.detect_text_boxes(crop))
            rotation_angle = float(rotation.get("angle", 0.0) or 0.0)
            rotation_confidence = float(rotation.get("confidence", 0.0) or 0.0)
        except Exception:
            pass
        translated_text = ""
        if translate and original_text.strip():
            translated_text = self.translate_manual_text(job_id, page_index, original_text)["translated_text"]
        return {
            "bbox": [x, y, bw, bh],
            "original_text": original_text,
            "translated_text": translated_text,
            "rotation_angle": rotation_angle,
            "rotation_confidence": rotation_confidence,
            "source_language": config.translation.idioma_entrada,
            "target_language": config.translation.idioma_salida,
            "ocr_engine": normalize_choice(job.options.transcription_engine, "auto"),
            "translator": normalize_choice(job.options.translator, "llm"),
        }

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        with self._processing_lock:
            if job.status in {"paused", "cancelled", "ready"}:
                return

            config = self._build_config_for_job(job)
            control = ExecutionControl(
                on_paused=lambda: self._on_control_paused(job_id),
                on_resumed=lambda: self._on_control_resumed(job_id),
            )
            with self._lock:
                self._controls[job_id] = control
                job.status = "processing"
                job.message = "Preparando modelos y recursos…"
                job.started_at = job.started_at or time.time()
                job.finished_at = 0.0
                job.updated_at = time.time()
                self._save_manifest(job)

            try:
                with execution_control_scope(control):
                    control.checkpoint()
                    prepare_runtime()
                    if not self._assets_prepared:
                        prepare_assets()
                        self._assets_prepared = True

                    processing = replace(
                        config.processing,
                        ruta_carpeta_entrada=job.input_dir,
                        batch_size=1,
                        usar_paralelismo=False,
                        max_workers=1,
                        cache_dir=str(Path(job.root_dir) / ".cache"),
                    )
                    translation = replace(config.translation, project_dir=job.input_dir)
                    config = replace(config, processing=processing, translation=translation)
                    set_active_config(config)
                    configure_logging(log_file=config.logging.file, level=config.logging.level)

                    output_dir = Path(job.output_dir)
                    clean_dir = output_dir / "limpieza"
                    translation_dir = output_dir / "traduccion"
                    corrected_dir = output_dir / "corregida"
                    corrections_dir = output_dir / "correcciones"
                    for directory in [clean_dir, translation_dir, corrected_dir, corrections_dir]:
                        directory.mkdir(parents=True, exist_ok=True)

                    trans_queue = CapturingJsonQueue(clean_dir / "Transcripción.json")
                    trad_queue = CapturingJsonQueue(translation_dir / "Traducción.json")
                    trans_queue.put({"agregar_entrada": {"Título": job.title, "Páginas": len(job.pages)}})
                    trad_queue.put({"agregar_entrada": {"Título": job.title, "Páginas": len(job.pages)}})

                    processor = build_image_processor(config)
                    max_attempts = max(1, int(job.options.page_max_retries) + 1)
                    retry_backoff = max(0.0, float(job.options.retry_backoff_seconds))

                    for page in job.pages:
                        control.checkpoint()
                        if page.status == "ready" and Path(page.clean_path).exists() and Path(page.translated_path).exists():
                            continue
                        if page.status == "cancelled":
                            continue
                        if page.attempt_count >= max_attempts and page.status == "failed":
                            continue

                        page_succeeded = False
                        while page.attempt_count < max_attempts and not page_succeeded:
                            control.checkpoint()
                            with self._lock:
                                page.attempt_count += 1
                                page.status = "processing"
                                page.message = (
                                    f"Procesando OCR, limpieza, traducción y renderizado "
                                    f"(intento {page.attempt_count}/{max_attempts})…"
                                )
                                page.last_error = ""
                                page.started_at = page.started_at or time.time()
                                page.updated_at = time.time()
                                job.active_page = page.index
                                job.status = "processing"
                                job.message = f"Procesando página {page.index + 1}/{len(job.pages)}."
                                job.updated_at = page.updated_at
                                self._save_manifest(job)

                            self._remove_partial_outputs(page)
                            try:
                                processor.procesar(
                                    job.input_dir,
                                    str(clean_dir),
                                    str(translation_dir),
                                    {page.index: page.source_filename},
                                    trans_queue,
                                    trad_queue,
                                )
                                control.checkpoint()
                                trans_queue.put({"ordenar_por_paginas": {"tipo": "Transcripción"}})
                                trad_queue.put({"ordenar_por_paginas": {"tipo": "Traducción"}})
                                if not Path(page.translated_path).exists() or not Path(page.clean_path).exists():
                                    raise RuntimeError("El pipeline terminó sin generar las imágenes esperadas.")

                                with self._lock:
                                    page.regions = self._merge_page_regions(job, page, trans_queue.data, trad_queue.data)
                                    saved_payload = read_corrections_payload(page.corrections_path)
                                    corrections = saved_payload.get("regions", [])
                                    if corrections:
                                        page.regions = self._apply_saved_corrections(page.regions, corrections)
                                    page.brush_strokes = saved_payload.get("brush_strokes", [])
                                    page.status = "ready"
                                    page.message = "Lista para revisión."
                                    page.last_error = ""
                                    page.completed_at = time.time()
                                    page.updated_at = page.completed_at
                                    self._recount(job)
                                    job.updated_at = page.updated_at
                                    self._save_manifest(job)
                                page_succeeded = True

                            except JobPausedError:
                                self._remove_partial_outputs(page)
                                with self._lock:
                                    # Una pausa no consume un intento de procesamiento de página.
                                    page.attempt_count = max(0, page.attempt_count - 1)
                                    page.status = "pending"
                                    page.message = "Página pausada; se retomará desde el inicio al reanudar."
                                    page.updated_at = time.time()
                                    job.updated_at = page.updated_at
                                    self._save_manifest(job)
                                raise
                            except JobCancelledError:
                                with self._lock:
                                    page.status = "cancelled"
                                    page.message = "Cancelada por el usuario."
                                    page.completed_at = time.time()
                                    page.updated_at = page.completed_at
                                    self._save_manifest(job)
                                raise
                            except Exception as exc:
                                logger.exception(
                                    "Error procesando página %s del job %s, intento %s/%s: %s",
                                    page.index + 1,
                                    job_id,
                                    page.attempt_count,
                                    max_attempts,
                                    exc,
                                )
                                with self._lock:
                                    page.last_error = str(exc)
                                    page.updated_at = time.time()
                                    if page.attempt_count < max_attempts:
                                        page.status = "pending"
                                        page.message = (
                                            f"Intento {page.attempt_count}/{max_attempts} falló; "
                                            "se reintentará automáticamente."
                                        )
                                    else:
                                        page.status = "failed"
                                        page.message = f"Falló después de {max_attempts} intentos: {exc}"
                                        page.completed_at = page.updated_at
                                    self._recount(job)
                                    job.updated_at = page.updated_at
                                    self._save_manifest(job)
                                if page.attempt_count < max_attempts:
                                    self._cooperative_backoff(
                                        control,
                                        retry_backoff * (2 ** max(0, page.attempt_count - 1)),
                                    )

                    with self._lock:
                        self._recount(job)
                        job.finished_at = time.time()
                        job.updated_at = job.finished_at
                        job.pause_requested = False
                        if job.failed_count == 0:
                            job.status = "ready"
                            job.message = "Procesamiento finalizado."
                        else:
                            job.status = "failed"
                            job.message = "Finalizado con páginas fallidas."
                        self._save_manifest(job)

            except JobPausedError:
                requeue = False
                with self._lock:
                    if job.cancel_requested or control.cancel_requested:
                        self._finalize_cancelled_job(job)
                    elif job.resume_requested:
                        job.status = "queued"
                        job.message = "Trabajo reanudado y devuelto a la cola."
                        job.pause_requested = False
                        job.resume_requested = False
                        job.updated_at = time.time()
                        self._queue.enqueue(job_id, preserve_time=False)
                        requeue = True
                    else:
                        job.status = "paused"
                        job.message = "Trabajo pausado. Las páginas terminadas se conservaron."
                        job.pause_requested = True
                        job.updated_at = time.time()
                    self._save_manifest(job)
                if requeue:
                    self._wake_worker()
            except JobCancelledError:
                with self._lock:
                    self._finalize_cancelled_job(job)
                    self._save_manifest(job)
            except Exception as exc:
                logger.exception("Error preparando o ejecutando job %s: %s", job_id, exc)
                with self._lock:
                    job.status = "failed"
                    job.message = f"Error general: {exc}"
                    job.finished_at = time.time()
                    job.updated_at = job.finished_at
                    self._save_manifest(job)
                failure_path = Path(job.root_dir) / "error.log"
                failure_path.write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                with self._lock:
                    self._controls.pop(job_id, None)

    def _merge_page_regions(
        self,
        job: JobState,
        page: PageState,
        trans_data: Optional[Dict[str, Any]] = None,
        trad_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        trans_data = trans_data or self._read_json(Path(job.output_dir) / "limpieza" / "Transcripción.json")
        trad_data = trad_data or self._read_json(Path(job.output_dir) / "traduccion" / "Traducción.json")
        page_no = page.index + 1
        originals = self._page_items(trans_data, "Transcripción", page_no)
        translations = self._page_items(trad_data, "Traducción", page_no)
        by_index: Dict[int, Dict[str, Any]] = {}

        for item in originals:
            idx = int(item.get("Índice", len(by_index)))
            coords = item.get("Coordenadas") or [[0, 0], [0, 0]]
            bbox = self._coords_to_bbox(coords)
            by_index.setdefault(idx, {"index": idx})
            by_index[idx].update(
                {
                    "bbox": bbox,
                    "source_bbox": bbox,
                    "original_text": item.get("Texto", ""),
                    "style": item.get("Estilo", "dialogo"),
                    "type": item.get("Tipo", "dialogue"),
                    "confidence": item.get("Confianza", 0),
                    "restore_original": False,
                    "visible": True,
                    "modified": False,
                    "deleted": False,
                    "auto_font_size": True,
                    "font_size": None,
                    "rotation_angle": item.get("Ángulo de texto", item.get("rotation_angle", 0.0)),
                    "rotation_confidence": item.get("Confianza de inclinación", item.get("rotation_confidence", 0.0)),
                    "ui_layout": item.get("Layout UI") or item.get("ui_layout"),
                }
            )
        for item in translations:
            idx = int(item.get("Índice", len(by_index)))
            coords = item.get("Coordenadas") or [[0, 0], [0, 0]]
            bbox = self._coords_to_bbox(coords)
            by_index.setdefault(idx, {"index": idx})
            by_index[idx].update(
                {
                    "bbox": by_index[idx].get("bbox") or bbox,
                    "source_bbox": by_index[idx].get("source_bbox") or bbox,
                    "translated_text": item.get("Texto", ""),
                    "style": item.get("Estilo", by_index[idx].get("style", "dialogo")),
                    "type": item.get("Tipo", by_index[idx].get("type", "dialogue")),
                    "confidence": item.get("Confianza", by_index[idx].get("confidence", 0)),
                    "restore_original": by_index[idx].get("restore_original", False),
                    "visible": by_index[idx].get("visible", True),
                    "modified": by_index[idx].get("modified", False),
                    "deleted": by_index[idx].get("deleted", False),
                    "auto_font_size": by_index[idx].get("auto_font_size", True),
                    "font_size": by_index[idx].get("font_size"),
                    "rotation_angle": item.get("Ángulo de texto", item.get("rotation_angle", by_index[idx].get("rotation_angle", 0.0))),
                    "rotation_confidence": item.get("Confianza de inclinación", item.get("rotation_confidence", by_index[idx].get("rotation_confidence", 0.0))),
                    "ui_layout": item.get("Layout UI") or item.get("ui_layout") or by_index[idx].get("ui_layout"),
                }
            )
        return [by_index[idx] for idx in sorted(by_index)]

    @staticmethod
    def _apply_saved_corrections(regions: List[Dict[str, Any]], corrections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        corrected_by_index = {int(item.get("index", idx)): item for idx, item in enumerate(corrections) if isinstance(item, dict)}
        merged: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for idx, region in enumerate(regions):
            region_index = int(region.get("index", idx))
            correction = corrected_by_index.get(region_index)
            if correction:
                source_bbox = correction.get("source_bbox") or region.get("source_bbox") or region.get("bbox")
                region = {
                    **region,
                    "bbox": correction.get("bbox", region.get("bbox")),
                    "source_bbox": source_bbox,
                    "translated_text": correction.get("text", region.get("translated_text", "")),
                    "style": correction.get("style", region.get("style", "dialogo")),
                    "restore_original": bool(correction.get("restore_original", False)),
                    "visible": bool(correction.get("visible", True)),
                    "modified": bool(correction.get("modified", True)),
                    "manual": bool(correction.get("manual", region.get("manual", False))),
                    "deleted": bool(correction.get("deleted", False)),
                    "auto_font_size": bool(correction.get("auto_font_size", True)),
                    "font_size": correction.get("font_size"),
                    "rotation_angle": correction.get("rotation_angle", region.get("rotation_angle", 0.0)),
                    "ui_layout": correction.get("ui_layout") or region.get("ui_layout"),
                }
            else:
                region = {
                    **region,
                    "source_bbox": region.get("source_bbox") or region.get("bbox"),
                    "modified": bool(region.get("modified", False)),
                    "auto_font_size": region.get("auto_font_size", True),
                    "font_size": region.get("font_size"),
                    "rotation_angle": region.get("rotation_angle", 0.0),
                    "ui_layout": region.get("ui_layout"),
                }
            seen.add(region_index)
            merged.append(region)

        for correction_index, correction in sorted(corrected_by_index.items()):
            if correction_index in seen:
                continue
            bbox = correction.get("bbox") or [0, 0, 1, 1]
            merged.append({
                "index": correction_index,
                "bbox": bbox,
                "source_bbox": correction.get("source_bbox") or bbox,
                "original_text": correction.get("original_text", ""),
                "translated_text": correction.get("text", correction.get("translated_text", "")),
                "style": correction.get("style", "dialogo"),
                "type": correction.get("type", "manual"),
                "confidence": correction.get("confidence", 0),
                "restore_original": bool(correction.get("restore_original", False)),
                "visible": bool(correction.get("visible", True)),
                "modified": bool(correction.get("modified", True)),
                "manual": True,
                "deleted": bool(correction.get("deleted", False)),
                "auto_font_size": bool(correction.get("auto_font_size", True)),
                "font_size": correction.get("font_size"),
                "rotation_angle": correction.get("rotation_angle", 0.0),
                "ui_layout": correction.get("ui_layout"),
            })
        return merged

    @staticmethod
    def _page_items(data: Dict[str, Any], key: str, page_no: int) -> List[Dict[str, Any]]:
        pages = data.get(key, []) if isinstance(data, dict) else []
        if not isinstance(pages, list):
            return []
        page = next((item for item in pages if isinstance(item, dict) and item.get("Página") == page_no), None)
        items = page.get("Globos de texto", []) if isinstance(page, dict) else []
        return items if isinstance(items, list) else []

    @staticmethod
    def _coords_to_bbox(coords: Any) -> List[int]:
        try:
            (x1, y1), (x2, y2) = coords
            return [int(x1), int(y1), max(1, int(x2) - int(x1)), max(1, int(y2) - int(y1))]
        except Exception:
            return [0, 0, 1, 1]

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _page(self, job: JobState, page_index: int) -> PageState:
        if page_index < 0 or page_index >= len(job.pages):
            raise IndexError("La página solicitada no existe.")
        return job.pages[page_index]

    def _mark_job(self, job: JobState, *, status: str, message: str) -> None:
        with self._lock:
            job.status = status
            job.message = message
            job.updated_at = time.time()
            self._save_manifest(job)

    def _save_manifest(self, job: JobState) -> None:
        manifest = Path(job.root_dir) / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(job)
        tmp_path = manifest.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(manifest)

    def _load_manifest(self, job_id: str) -> JobState:
        manifest = self.jobs_root / job_id / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError("No existe ese trabajo de UI.")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("El manifiesto del trabajo no es válido.")
        page_fields = PageState.__dataclass_fields__
        pages = [
            PageState(**{k: v for k, v in page.items() if k in page_fields})
            for page in data.get("pages", [])
            if isinstance(page, dict)
        ]
        data["pages"] = pages
        options = data.get("options", {})
        if isinstance(options, dict):
            data["options"] = JobOptions(**{k: v for k, v in options.items() if k in JobOptions.__dataclass_fields__})
        elif not isinstance(options, JobOptions):
            data["options"] = JobOptions()
        job_fields = JobState.__dataclass_fields__
        return JobState(**{k: v for k, v in data.items() if k in job_fields})
