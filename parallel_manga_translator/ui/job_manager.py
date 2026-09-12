from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


from parallel_manga_translator.bootstrap import build_default_config
from parallel_manga_translator.config.constants import normalizar_modelo_inpaint
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ui.job_manifest_store import JobManifestStore
from parallel_manga_translator.ui.job_execution import JobExecutionSupport
from parallel_manga_translator.ui.job_runner import JobRunner
from parallel_manga_translator.ui.retranslation_runner import RetranslationRunner
from parallel_manga_translator.ui.manual_edit_service import ManualEditService
# Reexportados a proposito: `ui/app.py` los importa desde aqui y son parte del API
# publica del modulo, no imports incidentales.
from parallel_manga_translator.ui.job_state import (  # noqa: F401
    JobOptions,
    JobState,
    PageState,
    TranslationRunState,
    active_translation_run,
    append_translation_event,
    job_to_public,
    finalize_cancelled,
    is_retranslating,
    normalize_choice,
    page_of,
    recount,
    page_to_public,
    normalized_output_name,
    rebase_stored_path,
    translation_run_to_public,
)
from parallel_manga_translator.ui.retranslator import region_is_retranslatable
from parallel_manga_translator.ui.persistent_queue import PersistentJobQueue

logger = get_logger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Prefijos de los casos de `dataset_eval` (ja_01, en_01…), por idioma de origen.
DATASET_LANGUAGE_CODES = {
    "japonés": "ja", "japones": "ja",
    "inglés": "en", "ingles": "en",
    "coreano": "ko",
    "chino": "zh",
    "español": "es", "espanol": "es",
    "portugués": "pt", "portugues": "pt",
    "francés": "fr", "frances": "fr",
    "italiano": "it",
}
PROJECT_JOBS_DIR = Path(os.getenv("PMT_UI_JOBS_DIR", ".pmt_ui_jobs")).resolve()



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
        # Persistencia del manifiesto: se construye antes de `_recover_interrupted_jobs`,
        # que ya lee trabajos de disco.
        self.manifests = JobManifestStore(self.jobs_root)
        self.manual_edits = ManualEditService(
            get_job=self.get_job,
            manifests=self.manifests,
            lock=self._lock,
            config_for_job=self._build_config_for_job,
        )
        self._queue = PersistentJobQueue(self.jobs_root / "queue.sqlite3")
        self._queue_condition = threading.Condition(threading.RLock())
        self._shutdown = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        # Andamiaje de ejecución compartido por los dos runners. `config_for_job` se pasa
        # como lambda y no como método enlazado para que se resuelva en cada llamada: así,
        # sustituirlo sobre el manager (cosa que hacen los tests) sigue surtiendo efecto.
        self._execution = JobExecutionSupport(
            get_job=self.get_job,
            manifests=self.manifests,
            state_lock=self._lock,
            queue=self._queue,
            wake_worker=self._wake_worker,
            config_for_job=lambda job: self._build_config_for_job(job),
        )
        self.job_runner = JobRunner(self._execution)
        self.retranslation_runner = RetranslationRunner(self._execution)
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
            self.manifests.save(job)
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
                recount(job)
            job.pause_requested = False
            job.cancel_requested = False
            job.resume_requested = False
            job.status = "queued"
            job.message = "Trabajo añadido a la cola persistente."
            job.updated_at = time.time()
            self.manifests.save(job)
        self._queue.enqueue(job.job_id)
        self._wake_worker()

    def retranslate_job(
        self,
        job_id: str,
        *,
        translator: str | None = None,
        target_language: str | None = None,
        overwrite_manual: bool = False,
        llm_model: str | None = None,
    ) -> JobState:
        """Reencola un trabajo terminado para volver a traducirlo con otro traductor.

        No repite detección, OCR ni inpainting: reutiliza la imagen limpia y la
        transcripción ya guardadas. Por defecto respeta las páginas con correcciones
        manuales; `overwrite_manual` las descarta para retraducirlas también.
        """
        job = self.get_job(job_id)
        with self._lock:
            if job.status not in {"ready", "failed", "cancelled"}:
                raise ValueError(f"No se puede retraducir un trabajo en estado {job.status}.")

            options = job.options if isinstance(job.options, JobOptions) else JobOptions(**dict(job.options or {}))
            selected = normalize_choice(translator, options.translator or "llm")
            if selected not in {"google", "llm"}:
                raise ValueError("El traductor debe ser 'google' (tradicional) o 'llm'.")

            base_config = build_default_config(self.config_path)
            provider = "google"
            model = ""
            if selected == "llm":
                provider = str(base_config.translation.llm.provider or "groq").strip().lower() or "groq"
                model = str(llm_model or base_config.translation.llm.model or "").strip()
                if not model:
                    raise ValueError("La retraducción LLM necesita un modelo configurado.")

            targets, skipped = self._retranslation_targets(job, overwrite_manual=overwrite_manual)
            if not targets:
                if skipped:
                    raise ValueError(
                        "Todas las páginas retraducibles tienen correcciones manuales. "
                        "Activa la opción de sobrescribirlas si quieres retraducirlas igualmente."
                    )
                raise ValueError(
                    "No hay páginas retraducibles: hacen falta páginas listas con su imagen limpia y su transcripción."
                )

            options.translator = selected
            if target_language and str(target_language).strip():
                options.target_language = str(target_language).strip()
            job.options = options

            if overwrite_manual:
                for index in targets:
                    self.manual_edits.discard_edits(job, job.pages[index])

            run_id = uuid.uuid4().hex[:10]
            run = TranslationRunState(
                run_id=run_id,
                translator=selected,
                provider=provider,
                model=model,
                target_language=options.target_language,
                overwrite_manual=bool(overwrite_manual),
                status="queued",
                page_indices=list(targets),
                pending_pages=list(targets),
                message=self._retranslation_queue_message(selected, len(targets), len(skipped)),
            )
            job.translation_runs.append(run)
            # Evita que años de uso hagan crecer indefinidamente el manifiesto; se
            # conservan las solicitudes más recientes, que son las útiles en la UI.
            if len(job.translation_runs) > 50:
                job.translation_runs = job.translation_runs[-50:]
            job.active_translation_run_id = run_id
            append_translation_event(
                job,
                "queued",
                run.message,
                details={"provider": provider, "model": model, "pages": list(targets)},
            )

            job.pending_operation = "retranslate"
            job.retranslate_pages = list(targets)
            job.pause_requested = False
            job.cancel_requested = False
            job.resume_requested = False
            job.finished_at = 0.0
            job.status = "queued"
            job.message = self._retranslation_queue_message(selected, len(targets), len(skipped))
            job.updated_at = time.time()
            self.manifests.save(job)
        self._queue.enqueue(job.job_id)
        self._wake_worker()
        return job

    @staticmethod
    def _retranslation_queue_message(translator: str, targets: int, skipped: int) -> str:
        motor = "Google" if translator == "google" else "LLM"
        mensaje = f"Retraducción con {motor} en cola: {targets} página(s)."
        if skipped:
            mensaje += f" {skipped} conservan su corrección manual."
        return mensaje


    def _retranslation_targets(self, job: JobState, *, overwrite_manual: bool) -> tuple[List[int], List[int]]:
        """Separa las páginas que se pueden retraducir de las que se conservan."""
        targets: List[int] = []
        skipped: List[int] = []
        for page in job.pages:
            if page.status != "ready" or not Path(page.clean_path).exists():
                continue
            if not any(region_is_retranslatable(region) for region in page.regions):
                continue
            if not overwrite_manual and page.corrected_path and Path(page.corrected_path).exists():
                skipped.append(page.index)
                continue
            targets.append(page.index)
        return targets, skipped

    def pause_job(self, job_id: str) -> JobState:
        job = self.get_job(job_id)
        with self._lock:
            if job.status in {"ready", "failed", "cancelled"}:
                raise ValueError(f"No se puede pausar un trabajo en estado {job.status}.")
            job.pause_requested = True
            job.updated_at = time.time()
            run = active_translation_run(job) if is_retranslating(job) else None
            if run is not None:
                run.status = "paused"
                run.message = "Pausa solicitada por el usuario."
                append_translation_event(job, "pause_requested", run.message)
            control = self._execution.controls.get(job_id)
            if control is not None:
                job.status = "pausing"
                job.message = "Pausa solicitada; se detendrá en el próximo punto seguro."
                control.request_pause()
            else:
                self._queue.remove(job_id)
                job.status = "paused"
                job.message = "Trabajo pausado en la cola."
            self.manifests.save(job)
        return job

    def resume_job(self, job_id: str) -> JobState:
        job = self.get_job(job_id)
        with self._lock:
            if job.status not in {"paused", "pausing", "resuming"}:
                raise ValueError("El trabajo no está pausado.")
            job.pause_requested = False
            job.updated_at = time.time()
            run = active_translation_run(job) if is_retranslating(job) else None
            if run is not None:
                run.status = "queued"
                run.last_error = ""
                run.message = "Reanudación solicitada; continuará por las páginas pendientes."
                append_translation_event(job, "resume_requested", run.message)
            control = self._execution.controls.get(job_id)
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
            self.manifests.save(job)
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
            control = self._execution.controls.get(job_id)
            if control is not None:
                job.status = "cancelling"
                job.message = "Cancelación solicitada; se detendrá en el próximo punto seguro."
                control.request_cancel()
            elif is_retranslating(job):
                self.retranslation_runner.finish(job, message="Retraducción cancelada antes de empezar.", outcome="cancelled")
            else:
                finalize_cancelled(job)
            self.manifests.save(job)
        self._wake_worker()
        return job

    # ------------------------------------------------------------------
    # Promoción a dataset_eval
    # ------------------------------------------------------------------
    # Un trabajo corregido a mano es exactamente la materia prima del banco de
    # regresión: alguien ya decidió, página a página, qué globo sobraba y qué texto
    # era el bueno. Esto lo empaqueta sin salir del editor, llamando al MISMO
    # constructor que usa `python -m parallel_manga_translator.quality.eval_dataset
    # build`, para que no haya dos formatos de caso conviviendo.

    def dataset_case_preview(self, job_id: str) -> Dict[str, Any]:
        """Lo que el editor necesita para plantear el envío: nombre, cifras y avisos."""
        from parallel_manga_translator.quality.eval_dataset import resolve_dataset_dir

        job = self.get_job(job_id)
        dataset_dir = resolve_dataset_dir()
        existing = sorted(
            path.parent.name
            for path in dataset_dir.glob("*/case.json")
        ) if dataset_dir.is_dir() else []
        summary = self._human_correction_summary(job)
        suggested = self._suggested_case_name(job, existing)
        return {
            "job_id": job.job_id,
            "title": job.title,
            "status": job.status,
            "dataset_dir": str(dataset_dir),
            "suggested_name": suggested,
            "existing_cases": existing,
            "ready": job.status in {"ready", "failed", "cancelled"},
            **summary,
        }

    def export_job_to_dataset(
        self,
        job_id: str,
        *,
        name: str = "",
        copy_images: bool = True,
        copy_reference: bool = True,
        overwrite: bool = False,
        refresh_baseline: bool = True,
    ) -> Dict[str, Any]:
        """Construye (o actualiza) un caso de `dataset_eval` a partir de este trabajo."""
        from parallel_manga_translator.quality.eval_dataset import (
            build_case_from_ui_job,
            load_case,
            resolve_dataset_dir,
            score_case,
            write_baseline,
        )

        job = self.get_job(job_id)
        with self._lock:
            if self._execution.controls.get(job_id) is not None or job.status not in {"ready", "failed", "cancelled"}:
                raise ValueError("Espera a que el trabajo termine (o cancélalo) antes de enviarlo a dataset_eval.")
            # El caso se construye leyendo `manifest.json` del disco: hay que asegurarse
            # de que refleja las últimas correcciones antes de copiar nada.
            self.manifests.save(job)
            root_dir = self._job_root_dir(job)
        if root_dir is None:
            raise ValueError("No se encuentra la carpeta de este trabajo.")

        summary = self._human_correction_summary(job)
        if summary["corrected_pages"] == 0:
            # Una "verdad de referencia" que nadie ha validado no es verdad de nada: el
            # caso mediría el pipeline contra su propia salida y siempre daría 1.0.
            raise ValueError(
                "Este trabajo no tiene ninguna corrección manual. Corrige al menos una página "
                "antes de convertirlo en caso de dataset_eval."
            )

        dataset_dir = resolve_dataset_dir()
        case_name = self._safe_case_name(name) or self._suggested_case_name(job, [])
        case_dir = dataset_dir / case_name
        if case_dir.exists() and not overwrite:
            raise FileExistsError(f"Ya existe el caso «{case_name}». Marca sobrescribir para regenerarlo.")

        dataset_dir.mkdir(parents=True, exist_ok=True)
        built_dir = build_case_from_ui_job(
            root_dir,
            dataset_dir,
            name=case_name,
            copy_images=bool(copy_images),
            copy_reference=bool(copy_reference),
        )

        case = load_case(built_dir)
        totals = dict(case.meta.get("totals") or {})
        baseline: Optional[Dict[str, Any]] = None
        baseline_error = ""
        if refresh_baseline:
            try:
                report = score_case(case)
                write_baseline(case, report)
                baseline = dict(report.get("summary") or {})
            except Exception as exc:  # noqa: BLE001 - el caso ya es válido sin baseline
                # Sin `prediccion_base/` no hay nada que puntuar. El caso sirve igual:
                # se avisa y se deja que el usuario genere la línea base cuando quiera.
                baseline_error = str(exc)
                logger.warning("Caso %s construido sin baseline: %s", case_name, exc)

        logger.info("Caso de dataset_eval construido desde la UI: %s (%s)", case_name, job_id)
        return {
            "case": case_name,
            "case_dir": str(built_dir),
            "dataset_dir": str(dataset_dir),
            "overwritten": bool(overwrite and case_dir.exists()),
            "totals": totals,
            "baseline": baseline,
            "baseline_error": baseline_error,
            "score_command": (
                "python -m parallel_manga_translator.quality.eval_dataset score "
                f"--case {case_name} --predictions <carpeta outputs> --fail-on-regression"
            ),
            **summary,
        }

    @staticmethod
    def _safe_case_name(name: str) -> str:
        """El nombre acaba siendo una carpeta: se limita a lo que no puede escaparse."""
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "").strip()).strip("._-")
        return slug[:60]

    def _suggested_case_name(self, job: JobState, existing: Sequence[str]) -> str:
        """`ja_01`, `en_02`… la convención que ya usa el banco."""
        language = str(getattr(job.options, "source_language", "") or "").strip().lower()
        code = DATASET_LANGUAGE_CODES.get(language, "")
        if not code:
            code = re.sub(r"[^a-z]+", "", language)[:2] or "case"
        taken = set(existing)
        for number in range(1, 100):
            candidate = f"{code}_{number:02d}"
            if candidate not in taken:
                return candidate
        return f"{code}_{job.job_id[:6]}"

    @staticmethod
    def _human_correction_summary(job: JobState) -> Dict[str, Any]:
        """Cuánto trabajo humano lleva encima, con las mismas cuentas que el caso.

        Se reutiliza `build_ground_truth_page` a propósito: si el editor promete N
        regiones de referencia, tienen que ser exactamente las que acabe escribiendo
        `build_case_from_ui_job`, no una estimación parecida.
        """
        from parallel_manga_translator.quality.eval_dataset import build_ground_truth_page

        pages = 0
        corrected_pages = 0
        gt_regions = 0
        added = 0
        discarded = 0
        strokes = 0
        for page in job.pages:
            pages += 1
            gt_page = build_ground_truth_page(
                {
                    "index": page.index,
                    "output_filename": page.output_filename,
                    "source_filename": page.source_filename,
                    "regions": page.regions,
                    "brush_strokes": page.brush_strokes,
                }
            )
            gt_regions += len(gt_page["regions"])
            added += gt_page["anadidas_por_humano"]
            discarded += gt_page["descartadas_por_humano"]
            strokes += gt_page["brush_strokes"]
            touched = (
                gt_page["anadidas_por_humano"]
                or gt_page["descartadas_por_humano"]
                or gt_page["brush_strokes"]
                or any(region.get("modificado") for region in gt_page["regions"])
            )
            if touched:
                corrected_pages += 1
        return {
            "pages": pages,
            "corrected_pages": corrected_pages,
            "gt_regions": gt_regions,
            "anadidas_por_humano": added,
            "descartadas_por_humano": discarded,
            "brush_strokes": strokes,
        }

    def delete_job(self, job_id: str) -> Dict[str, Any]:
        """Borra un trabajo y todo lo que escribió en disco. No tiene vuelta atrás.

        Un trabajo vivo no se borra a traición: se exige cancelarlo antes. Matar el
        directorio mientras el worker escribe en él dejaría el manifiesto a medias y el
        hilo lanzando errores de ruta en cada página.
        """
        job = self.get_job(job_id)
        with self._lock:
            active = self._execution.controls.get(job_id) is not None
            if active or job.status not in {"ready", "failed", "cancelled"}:
                raise ValueError("Cancela el trabajo antes de borrarlo.")
            self._queue.remove(job_id)
            self._jobs.pop(job_id, None)
            root_dir = self._job_root_dir(job)
            title = job.title
        folder_removed = True
        if root_dir is not None and root_dir.exists():
            folder_removed = self._remove_job_folder(root_dir)
        if folder_removed:
            logger.info("Trabajo borrado por el usuario: %s (%s)", job_id, title)
        else:
            logger.warning(
                "Trabajo %s (%s) quitado del historial, pero su carpeta no se pudo borrar del todo: %s",
                job_id, title, root_dir,
            )
        return {
            "job_id": job_id,
            "title": title,
            "deleted": True,
            # El trabajo desaparece del historial aunque el sistema de archivos se
            # resista. Decirlo es la diferencia entre un aviso y 200 MB fantasma.
            "folder_removed": folder_removed,
        }

    @staticmethod
    def _remove_job_folder(root_dir: Path) -> bool:
        """Borra la carpeta del trabajo y dice la verdad sobre el resultado.

        `ignore_errors=True` era mentira piadosa: dejaba restos y seguía informando de
        un borrado limpio. En Windows los dos motivos habituales son un archivo marcado
        de solo lectura (se reintenta quitando el atributo) y una ruta que pasa de los
        260 caracteres de MAX_PATH, donde ni siquiera se puede abrir para borrar.
        """
        def retry_readonly(func, path, _exc_info):
            try:
                os.chmod(path, 0o600)
                func(path)
            except OSError:
                pass

        # El manifiesto primero, y aparte: `list_jobs` reconstruye el historial
        # escaneando `*/manifest.json`, así que mientras ese archivo exista el trabajo
        # reaparece en la lista. Es el más pequeño y el que casi siempre se deja borrar;
        # si el resto se resiste, quedan archivos huérfanos pero no un trabajo zombi.
        with contextlib.suppress(OSError):
            (root_dir / "manifest.json").unlink(missing_ok=True)

        try:
            shutil.rmtree(root_dir, onerror=retry_readonly)
        except OSError:
            return False
        return not root_dir.exists()

    def _job_root_dir(self, job: JobState) -> Optional[Path]:
        """Directorio del trabajo, solo si de verdad cuelga de `jobs_root`.

        El manifiesto guarda rutas absolutas y puede venir de otra máquina o de una
        copia movida a mano: se resuelve y se comprueba la pertenencia antes de borrar
        nada recursivamente.
        """
        candidates = [self.jobs_root / job.job_id]
        if job.root_dir:
            candidates.append(Path(job.root_dir))
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved == self.jobs_root or self.jobs_root not in resolved.parents:
                continue
            if resolved.is_dir():
                return resolved
        return None

    def get_job(self, job_id: str) -> JobState:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = self.manifests.load(job_id)
                self._jobs[job_id] = job
            return job

    def list_jobs(self, include_pages: bool = True) -> List[Dict[str, Any]]:
        with self._lock:
            loaded = list(self._jobs.values())
        known = {job.job_id for job in loaded}
        for manifest in self.jobs_root.glob("*/manifest.json"):
            job_id = manifest.parent.name
            if job_id not in known:
                try:
                    job = self.manifests.load(job_id)
                    loaded.append(job)
                    with self._lock:
                        self._jobs[job_id] = job
                except Exception:
                    continue
        result = []
        for job in sorted(loaded, key=lambda item: item.created_at, reverse=True):
            payload = job_to_public(job)
            payload["queue_position"] = self._queue.position(job.job_id)
            if not include_pages:
                # El historial solo necesita la ficha del trabajo. Con proyectos de
                # cientos de páginas, arrastrar todas sus regiones convertiría un
                # listado en una respuesta de varios megabytes.
                pages = payload.pop("pages", [])
                payload["page_count"] = len(pages)
                payload["corrected_count"] = sum(1 for page in pages if page.get("has_corrected"))
            result.append(payload)
        return result

    def translation_runs_public(self, job_id: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        config = build_default_config(self.config_path)
        return {
            "active_run_id": job.active_translation_run_id or "",
            "default_llm_provider": str(config.translation.llm.provider or "groq"),
            "default_llm_model": str(config.translation.llm.model or ""),
            "runs": [translation_run_to_public(run) for run in reversed(job.translation_runs or [])],
        }

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
                job = self.manifests.load(job_id)
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

            recount(job)
            if job.cancel_requested or job.status == "cancelling":
                finalize_cancelled(job)
                self._queue.remove(job_id)
                changed = True
            elif job.pause_requested or job.status in {"paused", "pausing"}:
                job.status = "paused"
                job.pause_requested = True
                job.message = "Trabajo pausado recuperado después del reinicio."
                run = active_translation_run(job) if is_retranslating(job) else None
                if run is not None:
                    run.status = "paused"
                    run.pending_pages = list(job.retranslate_pages)
                    run.message = job.message
                self._queue.remove(job_id)
                changed = True
            elif job.status in {"processing", "queued", "resuming"}:
                if not is_retranslating(job) and all(page.status == "ready" for page in job.pages):
                    job.status = "ready"
                    job.message = "Trabajo recuperado: todas las páginas estaban completas."
                    job.finished_at = job.finished_at or time.time()
                    self._queue.remove(job_id)
                else:
                    job.status = "queued"
                    job.message = "Trabajo recuperado y devuelto a la cola persistente."
                    job.recovery_count += 1
                    run = active_translation_run(job) if is_retranslating(job) else None
                    if run is not None:
                        run.status = "queued"
                        run.pending_pages = list(job.retranslate_pages)
                        run.message = job.message
                        append_translation_event(
                            job,
                            "recovered",
                            "PMT se reinició durante la retraducción; se conservaron las páginas pendientes.",
                            details={"pending_pages": list(job.retranslate_pages)},
                        )
                    self._queue.enqueue(job_id)
                changed = True
            elif job.status in {"ready", "failed", "cancelled"}:
                self._queue.remove(job_id)

            with self._lock:
                if changed:
                    job.updated_at = time.time()
                    self.manifests.save(job)
                self._jobs[job_id] = job

        for stale_id in set(self._queue.all_ids()) - manifest_ids:
            self._queue.remove(stale_id)







    def image_path(self, job_id: str, page_index: int, variant: str) -> Path:
        job = self.get_job(job_id)
        page = page_of(job, page_index)
        if variant == "original":
            return Path(page.original_path)
        if variant == "clean":
            return Path(page.clean_path)
        if variant == "translated":
            return Path(page.translated_path)
        if variant == "corrected":
            return Path(page.corrected_path)
        if variant == "background":
            # Capa de fondo vigente: la limpieza salvo que el pincel manual ya haya
            # creado una revisión. El editor la usa para tapar en vivo el texto que
            # todavía está rasterizado en la posición anterior de una región.
            manual_background = Path(page.manual_background_path) if page.manual_background_path else None
            if manual_background is not None and manual_background.exists():
                return manual_background
            return Path(page.clean_path)
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

    # --- Edicion manual: la implementacion vive en ManualEditService ---

    def save_manual_render(self, job_id: str, page_index: int, regions_payload: Sequence[Dict[str, Any]], brush_strokes_payload: Sequence[Dict[str, Any]] | None=None, operation: str='render', inpaint_model: str | None=None, background_revision: str | None=None):
        return self.manual_edits.save_manual_render(job_id=job_id, page_index=page_index, regions_payload=regions_payload, brush_strokes_payload=brush_strokes_payload, operation=operation, inpaint_model=inpaint_model, background_revision=background_revision)

    def render_region_preview(self, job_id: str, page_index: int, region_payload: Dict[str, Any]):
        return self.manual_edits.render_region_preview(job_id=job_id, page_index=page_index, region_payload=region_payload)

    def reset_manual_render(self, job_id: str, page_index: int):
        return self.manual_edits.reset_manual_render(job_id=job_id, page_index=page_index)

    def resolve_region_metrics(self, job_id: str, page_index: int, region_payload: Dict[str, Any]):
        return self.manual_edits.resolve_region_metrics(job_id=job_id, page_index=page_index, region_payload=region_payload)

    def transcribe_manual_region(self, job_id: str, page_index: int, bbox: Sequence[float], translate: bool=True):
        return self.manual_edits.transcribe_manual_region(job_id=job_id, page_index=page_index, bbox=bbox, translate=translate)

    def translate_manual_text(self, job_id: str, page_index: int, original_text: str):
        return self.manual_edits.translate_manual_text(job_id=job_id, page_index=page_index, original_text=original_text)
















    def _build_config_for_job(self, job: JobState):
        config = build_default_config(self.config_path)
        options = job.options if isinstance(job.options, JobOptions) else JobOptions(**dict(job.options or {}))
        active_run = active_translation_run(job) if is_retranslating(job) else None
        translator = normalize_choice(active_run.translator if active_run is not None else options.translator, "llm")
        target_language = active_run.target_language if active_run is not None else options.target_language
        method = "LLM" if translator == "llm" else "Tradicional"
        traditional_provider = "google" if translator == "google" else config.translation.traditional_provider
        llm_provider = config.translation.llm.provider or "groq"
        llm_model = config.translation.llm.model
        if active_run is not None and active_run.translator == "llm":
            llm_provider = active_run.provider or llm_provider
            llm_model = active_run.model or llm_model
        llm = replace(config.translation.llm, provider=llm_provider, model=llm_model)
        translation = replace(
            config.translation,
            idioma_entrada=options.source_language or config.translation.idioma_entrada,
            idioma_salida=target_language or config.translation.idioma_salida,
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



    def _run_job(self, job_id: str) -> None:
        """Despacha al runner que toca; la ejecución vive en ellos, no aquí.

        Se conserva el nombre porque es el punto de entrada del worker y el que usan los
        tests para ejecutar un trabajo de forma síncrona.
        """
        job = self.get_job(job_id)
        if is_retranslating(job):
            self.retranslation_runner.run(job_id)
        else:
            self.job_runner.run(job_id)











    def _mark_job(self, job: JobState, *, status: str, message: str) -> None:
        with self._lock:
            job.status = status
            job.message = message
            job.updated_at = time.time()
            self.manifests.save(job)
