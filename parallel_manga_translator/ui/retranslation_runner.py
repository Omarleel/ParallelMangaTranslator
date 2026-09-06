"""Retraducción de un trabajo ya terminado, sin repetir limpieza ni OCR.

Era `JobManager._run_retranslation` y sus dos ayudantes. Reutiliza `JobRetranslator`, que
es quien sabe traducir y rotular una página; aquí está el recorrido: qué páginas quedan
pendientes, cómo se refleja cada una en el manifiesto y en `Traducción.json`, y en qué
estado terminal queda el trabajo.

La retraducción **reentra por la misma cola persistente y el mismo worker único** que el
procesamiento normal. Por eso comparte `JobExecutionSupport` con `JobRunner` en vez de
tener su propio entorno: son dos cuerpos distintos sobre el mismo andamiaje.

Una regla de comportamiento que no es evidente y conviene no tocar: si se pidió LLM y el
proveedor degrada en silencio al traductor tradicional (típicamente al quedarse sin
tokens), la retraducción **se detiene** en esa página en vez de seguir. Quien pidió LLM
sobre un tomo terminado no debe acabar con media obra traducida por otro motor sin
enterarse.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from parallel_manga_translator.infrastructure.execution_control import (
    JobCancelledError,
    JobPausedError,
    execution_control_scope,
)
from parallel_manga_translator.infrastructure.logging_config import configure_logging, get_logger
from parallel_manga_translator.ui.job_execution import JobExecutionSupport
from parallel_manga_translator.ui.job_state import JobState, is_retranslating, normalize_choice, recount
from parallel_manga_translator.ui.page_regions import push_retranslated_page
from parallel_manga_translator.ui.queue_adapter import CapturingJsonQueue
from parallel_manga_translator.ui.retranslator import JobRetranslator

logger = get_logger(__name__)


class RetranslationRunner:
    """Ejecuta la retraducción. Encolarla y decidir qué páginas entran es del manager."""

    def __init__(self, support: JobExecutionSupport) -> None:
        self._support = support

    def run(self, job_id: str) -> None:
        """Vuelve a traducir las páginas marcadas sin repetir limpieza ni OCR."""
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

                        translation_dir = Path(job.output_dir) / "traduccion"
                        translation_dir.mkdir(parents=True, exist_ok=True)
                        trad_queue = CapturingJsonQueue(translation_dir / "Traducción.json")
                        retranslator = JobRetranslator(config)

                        elegido = normalize_choice(job.options.translator, "llm")
                        motor = "Google" if elegido == "google" else "LLM"
                        pendientes = [index for index in job.retranslate_pages if 0 <= index < len(job.pages)]
                        total = len(pendientes)
                        degradacion = ""
                        detenida_en = 0
                        for posicion, page_index in enumerate(pendientes, start=1):
                            control.checkpoint()
                            page = job.pages[page_index]
                            with self._support.state_lock:
                                job.active_page = page.index
                                job.message = f"Retraduciendo página {posicion}/{total} con {motor}."
                                page.message = f"Retraduciendo con {motor}…"
                                page.updated_at = time.time()
                                job.updated_at = page.updated_at
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
                                job.retranslate_pages = [index for index in job.retranslate_pages if index != page_index]
                                job.updated_at = page.updated_at
                                self._support.manifests.save(job)

                            # El proveedor LLM cae al traductor tradicional en silencio cuando
                            # se queda sin tokens. Quien pidió LLM sobre un trabajo terminado
                            # no debe acabar con medio tomo traducido por otro motor sin
                            # enterarse: se para aquí y se deja el resto intacto.
                            if elegido == "llm" and retranslator.llm_fallback_reason:
                                degradacion = retranslator.llm_fallback_reason
                                detenida_en = posicion
                                break

                        with self._support.state_lock:
                            if degradacion:
                                self.finish(
                                    job,
                                    message=(
                                        f"Retraducción detenida en la página {detenida_en} de {total}: {degradacion} "
                                        "Esas páginas quedaron con traducción tradicional; el resto no se tocó. "
                                        "Vuelve a lanzarla cuando el LLM esté disponible, o elige Google a propósito."
                                    ),
                                )
                            else:
                                self.finish(job, message=f"Retraducción con {motor} finalizada.")
                            self._support.manifests.save(job)

                except JobPausedError:
                    self._support.handle_pause(
                        job,
                        control,
                        on_cancelled=lambda trabajo: self.finish(trabajo, message="Retraducción cancelada. Las páginas ya retraducidas se conservaron."),
                        resume_message="Retraducción reanudada y devuelta a la cola.",
                        paused_message="Retraducción pausada. Continuará en la página pendiente al reanudar.",
                    )
                except JobCancelledError:
                    with self._support.state_lock:
                        self.finish(
                            job,
                            message="Retraducción cancelada. Las páginas ya retraducidas se conservaron.",
                        )
                        self._support.manifests.save(job)
                except Exception as exc:
                    logger.exception("Error retraduciendo el job %s: %s", job_id, exc)
                    with self._support.state_lock:
                        # Las páginas siguen siendo válidas: solo falló el intento de
                        # retraducción, así que el trabajo vuelve a su estado terminal.
                        self.finish(job, message=f"La retraducción falló: {exc}")
                        self._support.manifests.save(job)

    def finish(self, job: JobState, *, message: str) -> None:
        """Devuelve el trabajo a su estado terminal después de una retraducción."""
        job.pending_operation = "process"
        job.retranslate_pages = []
        job.pause_requested = False
        job.resume_requested = False
        recount(job)
        # Retraducir no cambia qué páginas existen: el trabajo vuelve al estado
        # terminal que tenía, incluido el de un trabajo cancelado a medias.
        if job.failed_count:
            job.status = "failed"
        elif any(page.status == "cancelled" for page in job.pages):
            job.status = "cancelled"
        else:
            job.status = "ready"
        job.cancel_requested = job.status == "cancelled"
        job.message = message
        job.finished_at = time.time()
        job.updated_at = job.finished_at
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
                # `ui_layout` se conserva a propósito: describe el hueco del globo que
                # salió de la máscara de limpieza, y esa geometría no cambia al
                # retraducir. El texto nuevo se reajusta dentro de ella al renderizar.
                # La región vuelve a ser salida automática: cualquier marca de edición
                # manual previa ya se descartó antes de encolar la retraducción.
                "modified": False,
            })
        return actualizadas


__all__ = ["RetranslationRunner"]
