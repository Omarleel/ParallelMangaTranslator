"""Ejecución persistente de retraducciones.

Una retraducción no es un segundo pipeline completo: reutiliza limpieza, OCR y regiones ya
persistidas. Desde esta versión cada solicitud tiene además un ``TranslationRunState``
propio. Eso permite cerrar PMT, volver otro día y continuar con el mismo modelo y las
mismas páginas pendientes; los errores recuperables pausan la solicitud en vez de
convertirla en un estado terminal que olvida el progreso.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence

from parallel_manga_translator.infrastructure.execution_control import (
    JobCancelledError,
    JobPausedError,
    execution_control_scope,
)
from parallel_manga_translator.infrastructure.logging_config import configure_logging, get_logger
from parallel_manga_translator.ui.job_execution import JobExecutionSupport
from parallel_manga_translator.ui.job_state import (
    JobState,
    TranslationRunState,
    active_translation_run,
    append_translation_event,
    is_retranslating,
    normalize_choice,
    recount,
)
from parallel_manga_translator.ui.page_regions import push_retranslated_page
from parallel_manga_translator.ui.queue_adapter import CapturingJsonQueue
from parallel_manga_translator.ui.retranslator import JobRetranslator

logger = get_logger(__name__)


class RetranslationRunner:
    """Ejecuta la retraducción. Encolarla y decidir qué páginas entran es del manager."""

    def __init__(self, support: JobExecutionSupport) -> None:
        self._support = support

    def _ensure_run(self, job: JobState, config) -> TranslationRunState:
        """Crea un registro para manifiestos antiguos que tenían retraducción pendiente."""
        run = active_translation_run(job)
        if run is not None:
            return run

        elegido = normalize_choice(job.options.translator, "llm")
        provider = "google"
        model = ""
        if elegido == "llm":
            provider = str(config.translation.llm.provider or "groq")
            model = str(config.translation.llm.model or "")
        run = TranslationRunState(
            run_id=uuid.uuid4().hex[:10],
            translator=elegido,
            provider=provider,
            model=model,
            target_language=str(job.options.target_language or config.translation.idioma_salida),
            status="queued",
            page_indices=list(job.retranslate_pages),
            pending_pages=list(job.retranslate_pages),
            message="Solicitud antigua recuperada y convertida al historial persistente.",
        )
        job.translation_runs.append(run)
        job.active_translation_run_id = run.run_id
        append_translation_event(job, "recovered", run.message)
        return run

    def _provider_event_callback(self, job: JobState):
        """Devuelve un sink ligero que el proveedor LLM puede usar durante sus retries."""
        def emit(kind: str, message: str, *, level: str = "info", details=None) -> None:
            with self._support.state_lock:
                run = active_translation_run(job)
                page_index = run.current_page if run is not None else None
                append_translation_event(
                    job,
                    kind,
                    message,
                    level=level,
                    page_index=page_index,
                    details=dict(details or {}),
                )
                job.updated_at = time.time()
                self._support.manifests.save(job)
        return emit

    def _attach_provider_event_sink(self, retranslator: JobRetranslator, job: JobState) -> None:
        translator = getattr(retranslator, "translator", None)
        manager = getattr(translator, "translator_manager", None)
        provider = getattr(manager, "provider", None)
        if provider is not None:
            setattr(provider, "translation_event_callback", self._provider_event_callback(job))

    def run(self, job_id: str) -> None:
        """Vuelve a traducir las páginas pendientes sin repetir limpieza ni OCR."""
        job = self._support.get_job(job_id)
        with self._support.processing_lock:
            if job.status in {"paused", "cancelled"} or not is_retranslating(job):
                return

            config = self._support.config_for_job(job)
            with self._support.executing(job, message="Preparando la retraducción…") as control:
                try:
                    with execution_control_scope(control):
                        control.checkpoint()
                        self._support.prepare_environment()
                        configure_logging(log_file=config.logging.file, level=config.logging.level)

                        with self._support.state_lock:
                            run = self._ensure_run(job, config)
                            resumed = bool(run.started_at)
                            run.status = "processing"
                            run.started_at = run.started_at or time.time()
                            run.finished_at = 0.0
                            run.last_error = ""
                            run.message = "Retraducción reanudada." if resumed else "Retraducción iniciada."
                            run.pending_pages = list(job.retranslate_pages)
                            append_translation_event(
                                job,
                                "resumed" if resumed else "started",
                                run.message,
                                details={"provider": run.provider, "model": run.model},
                            )
                            self._support.manifests.save(job)

                        translation_dir = Path(job.output_dir) / "traduccion"
                        translation_dir.mkdir(parents=True, exist_ok=True)
                        trad_queue = CapturingJsonQueue(translation_dir / "Traducción.json")
                        retranslator = JobRetranslator(config)
                        self._attach_provider_event_sink(retranslator, job)

                        elegido = normalize_choice(run.translator, "llm")
                        motor = "Google" if elegido == "google" else f"{run.provider}/{run.model}".strip("/")
                        pendientes = [index for index in job.retranslate_pages if 0 <= index < len(job.pages)]
                        total_original = len(run.page_indices) or len(pendientes)

                        for page_index in pendientes:
                            control.checkpoint()
                            page = job.pages[page_index]
                            with self._support.state_lock:
                                run.current_page = page.index
                                run.status = "processing"
                                run.message = f"Retraduciendo página {page.index + 1} con {motor}."
                                job.active_page = page.index
                                job.message = run.message
                                page.message = f"Retraduciendo con {motor}…"
                                page.updated_at = time.time()
                                job.updated_at = page.updated_at
                                append_translation_event(
                                    job,
                                    "page_started",
                                    run.message,
                                    page_index=page.index,
                                )
                                self._support.manifests.save(job)

                            resultados = retranslator.retranslate_page(
                                page_index=page.index,
                                clean_path=page.clean_path,
                                output_path=page.translated_path,
                                regions=page.regions,
                            )
                            control.checkpoint()

                            with self._support.state_lock:
                                page.regions = self._apply_retranslation(page.regions, resultados)
                                push_retranslated_page(trad_queue, page, resultados)
                                page.message = "Retraducida y lista para revisión."
                                page.last_error = ""
                                page.completed_at = time.time()
                                page.updated_at = page.completed_at
                                job.retranslate_pages = [idx for idx in job.retranslate_pages if idx != page_index]
                                run.pending_pages = list(job.retranslate_pages)
                                if page_index not in run.completed_pages:
                                    run.completed_pages.append(page_index)
                                run.current_page = None
                                run.message = (
                                    f"Página {page.index + 1} completada. "
                                    f"Quedan {len(run.pending_pages)} de {total_original}."
                                )
                                job.updated_at = page.updated_at
                                append_translation_event(
                                    job,
                                    "page_completed",
                                    run.message,
                                    page_index=page.index,
                                )
                                self._support.manifests.save(job)

                            # Sólo puede ocurrir si el usuario habilitó explícitamente el
                            # fallback. Se conserva la página ya escrita, pero la solicitud
                            # queda pausada para no mezclar silenciosamente motores.
                            if elegido == "llm" and getattr(retranslator, "llm_fallback_reason", ""):
                                self.pause_for_error(
                                    job,
                                    RuntimeError(retranslator.llm_fallback_reason),
                                    message_prefix=(
                                        f"Retraducción pausada tras la página {page.index + 1} de {total_original}: "
                                        "el LLM cambió a traductor tradicional"
                                    ),
                                )
                                return

                        with self._support.state_lock:
                            self.finish(job, message=f"Retraducción con {motor} finalizada.", outcome="completed")
                            self._support.manifests.save(job)

                except JobPausedError:
                    self._support.handle_pause(
                        job,
                        control,
                        on_cancelled=lambda trabajo: self.finish(
                            trabajo,
                            message="Retraducción cancelada. Las páginas ya retraducidas se conservaron.",
                            outcome="cancelled",
                        ),
                        resume_message="Retraducción reanudada y devuelta a la cola.",
                        paused_message="Retraducción pausada. Continuará en la página pendiente al reanudar.",
                    )
                    with self._support.state_lock:
                        run = active_translation_run(job)
                        if run is not None:
                            run.status = "queued" if job.status == "queued" else "paused"
                            run.message = job.message
                            append_translation_event(
                                job,
                                "resumed" if job.status == "queued" else "paused",
                                job.message,
                            )
                            self._support.manifests.save(job)
                except JobCancelledError:
                    with self._support.state_lock:
                        self.finish(
                            job,
                            message="Retraducción cancelada. Las páginas ya retraducidas se conservaron.",
                            outcome="cancelled",
                        )
                        self._support.manifests.save(job)
                except Exception as exc:
                    logger.exception("Error retraduciendo el job %s: %s", job_id, exc)
                    with self._support.state_lock:
                        self.pause_for_error(job, exc)
                        self._support.manifests.save(job)

    def pause_for_error(self, job: JobState, exc: BaseException, *, message_prefix: str = "Retraducción pausada por error") -> None:
        """Conserva progreso y pendientes para que el usuario pueda reanudar después."""
        error = (str(exc).strip() or exc.__class__.__name__).rstrip(" .")
        run = active_translation_run(job)
        if run is not None:
            run.status = "paused"
            run.last_error = error
            run.message = f"{message_prefix}: {error}"
            run.pending_pages = list(job.retranslate_pages)
            append_translation_event(
                job,
                "error",
                run.message,
                level="error",
                page_index=run.current_page,
                details={"pending_pages": list(run.pending_pages)},
            )
        job.status = "paused"
        job.pause_requested = True
        job.resume_requested = False
        job.cancel_requested = False
        job.message = (
            f"{message_prefix}: {error}. El progreso quedó guardado; "
            "puedes cerrar PMT y usar Reanudar más tarde."
        )
        job.finished_at = 0.0
        job.updated_at = time.time()
        self._support.queue.remove(job.job_id)

    def finish(self, job: JobState, *, message: str, outcome: str = "completed") -> None:
        """Cierra la solicitud activa y devuelve el trabajo a su estado terminal."""
        now = time.time()
        run = active_translation_run(job)
        if run is not None:
            run.status = outcome
            run.current_page = None
            run.pending_pages = [] if outcome == "completed" else list(job.retranslate_pages)
            run.message = message
            run.finished_at = now
            run.updated_at = now
            append_translation_event(
                job,
                "completed" if outcome == "completed" else outcome,
                message,
                level="info" if outcome == "completed" else "warning",
            )

        job.pending_operation = "process"
        job.retranslate_pages = []
        job.active_translation_run_id = ""
        job.pause_requested = False
        job.resume_requested = False
        recount(job)
        if job.failed_count:
            job.status = "failed"
        elif any(page.status == "cancelled" for page in job.pages):
            job.status = "cancelled"
        else:
            job.status = "ready"
        job.cancel_requested = job.status == "cancelled"
        job.message = message
        job.finished_at = now
        job.updated_at = now
        self._support.queue.remove(job.job_id)

    @staticmethod
    def _apply_retranslation(regions: Sequence[Dict[str, Any]], resultados: Sequence[Any]) -> List[Dict[str, Any]]:
        """Vuelca la nueva traducción sobre las regiones guardadas del manifiesto."""
        por_indice = {resultado.index: resultado for resultado in resultados}
        actualizadas: List[Dict[str, Any]] = []
        for posicion, region in enumerate(regions):
            if not isinstance(region, dict):
                continue
            resultado = por_indice.get(int(region.get("index", posicion)))
            if resultado is None:
                actualizadas.append(region)
                continue
            actualizadas.append({
                **region,
                "translated_text": resultado.translated_text,
                "style": resultado.style,
                "modified": False,
            })
        return actualizadas


__all__ = ["RetranslationRunner"]
