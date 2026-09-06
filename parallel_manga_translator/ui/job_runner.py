"""Ejecución de un trabajo: sus páginas por el pipeline, con reintentos.

Era `JobManager._run_job`, 226 líneas dentro de una clase que además coordinaba la cola,
el ciclo de vida y la edición manual. Aquí queda sólo la ejecución: qué páginas se saltan,
cuántos intentos tiene cada una, qué se borra antes de reintentar y cómo queda el trabajo
al terminar.

Lo que rodea a la ejecución —tomar el worker único, registrar el control de pausa,
preparar recursos, reaccionar a una pausa— lo aporta `JobExecutionSupport`, compartido con
`RetranslationRunner`.

Dos decisiones de comportamiento que conviene no perder de vista al leer el bucle:

- **Una pausa no consume un intento.** Se devuelve el contador y la página vuelve a
  pendiente; al reanudar se rehace desde el principio.
- **Antes de cada intento se borran las salidas parciales** de esa página. Sin eso, un
  fallo a mitad deja una imagen limpia sin traducir que el reintento daría por buena.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import replace
from pathlib import Path

from parallel_manga_translator.bootstrap import build_image_processor
from parallel_manga_translator.infrastructure.execution_control import (
    ExecutionControl,
    JobCancelledError,
    JobPausedError,
    execution_control_scope,
)
from parallel_manga_translator.infrastructure.logging_config import configure_logging, get_logger
from parallel_manga_translator.ui.job_execution import JobExecutionSupport
from parallel_manga_translator.ui.job_state import PageState, finalize_cancelled, recount
from parallel_manga_translator.ui.manual_renderer import read_corrections_payload
from parallel_manga_translator.ui.page_regions import apply_saved_corrections, merge_page_regions
from parallel_manga_translator.ui.queue_adapter import CapturingJsonQueue

logger = get_logger(__name__)


class JobRunner:
    """Procesa las páginas de un trabajo. No sabe de colas, de HTTP ni de edición manual."""

    def __init__(self, support: JobExecutionSupport) -> None:
        self._support = support

    def run(self, job_id: str) -> None:
        """Procesa las páginas pendientes de un trabajo, con reintentos por página."""
        job = self._support.get_job(job_id)
        with self._support.processing_lock:
            if job.status in {"paused", "cancelled", "ready"}:
                return

            config = self._support.config_for_job(job)
            with self._support.executing(job, message="Preparando modelos y recursos…", mark_started=True) as control:
                try:
                    with execution_control_scope(control):
                        control.checkpoint()
                        self._support.prepare_environment()

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
                                with self._support.state_lock:
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
                                    self._support.manifests.save(job)

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

                                    with self._support.state_lock:
                                        page.regions = merge_page_regions(job, page, trans_queue.data, trad_queue.data)
                                        saved_payload = read_corrections_payload(page.corrections_path)
                                        corrections = saved_payload.get("regions", [])
                                        if corrections:
                                            page.regions = apply_saved_corrections(page.regions, corrections)
                                        page.brush_strokes = saved_payload.get("brush_strokes", [])
                                        page.status = "ready"
                                        page.message = "Lista para revisión."
                                        page.last_error = ""
                                        page.completed_at = time.time()
                                        page.updated_at = page.completed_at
                                        recount(job)
                                        job.updated_at = page.updated_at
                                        self._support.manifests.save(job)
                                    page_succeeded = True

                                except JobPausedError:
                                    self._remove_partial_outputs(page)
                                    with self._support.state_lock:
                                        # Una pausa no consume un intento de procesamiento de página.
                                        page.attempt_count = max(0, page.attempt_count - 1)
                                        page.status = "pending"
                                        page.message = "Página pausada; se retomará desde el inicio al reanudar."
                                        page.updated_at = time.time()
                                        job.updated_at = page.updated_at
                                        self._support.manifests.save(job)
                                    raise
                                except JobCancelledError:
                                    with self._support.state_lock:
                                        page.status = "cancelled"
                                        page.message = "Cancelada por el usuario."
                                        page.completed_at = time.time()
                                        page.updated_at = page.completed_at
                                        self._support.manifests.save(job)
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
                                    with self._support.state_lock:
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
                                        recount(job)
                                        job.updated_at = page.updated_at
                                        self._support.manifests.save(job)
                                    if page.attempt_count < max_attempts:
                                        self._cooperative_backoff(
                                            control,
                                            retry_backoff * (2 ** max(0, page.attempt_count - 1)),
                                        )

                        with self._support.state_lock:
                            recount(job)
                            job.finished_at = time.time()
                            job.updated_at = job.finished_at
                            job.pause_requested = False
                            if job.failed_count == 0:
                                job.status = "ready"
                                job.message = "Procesamiento finalizado."
                            else:
                                job.status = "failed"
                                job.message = "Finalizado con páginas fallidas."
                            self._support.manifests.save(job)

                except JobPausedError:
                    self._support.handle_pause(
                        job,
                        control,
                        on_cancelled=finalize_cancelled,
                        resume_message="Trabajo reanudado y devuelto a la cola.",
                        paused_message="Trabajo pausado. Las páginas terminadas se conservaron.",
                    )
                except JobCancelledError:
                    with self._support.state_lock:
                        finalize_cancelled(job)
                        self._support.manifests.save(job)
                except Exception as exc:
                    logger.exception("Error preparando o ejecutando job %s: %s", job_id, exc)
                    with self._support.state_lock:
                        job.status = "failed"
                        job.message = f"Error general: {exc}"
                        job.finished_at = time.time()
                        job.updated_at = job.finished_at
                        self._support.manifests.save(job)
                    failure_path = Path(job.root_dir) / "error.log"
                    failure_path.write_text(traceback.format_exc(), encoding="utf-8")

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


__all__ = ["JobRunner"]
