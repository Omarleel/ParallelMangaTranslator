"""Lo que comparten los dos runners: el andamiaje para ejecutar un trabajo.

`_run_job` y `_run_retranslation` eran 226 y 136 líneas dentro de `JobManager`, y medido
por AST necesitaban **exactamente las mismas trece cosas** de él. No era casualidad: las
dos empiezan igual (tomar el worker único, construir la configuración del trabajo,
registrar un `ExecutionControl`, preparar runtime y fuentes) y terminan igual (reaccionar
a la pausa, cerrar el log del trabajo, desregistrar el control). Sólo cambia el cuerpo.

Eso es lo único que se extrae aquí. Cada runner conserva su propio cuerpo y su propio
manejo de errores, porque ahí sí divergen: uno reintenta páginas y borra salidas
parciales, el otro se detiene si el LLM degrada a traductor tradicional.

Este objeto NO decide qué se ejecuta. Es el entorno: el lock que garantiza un solo
trabajo a la vez, el registro de controles vivo que consultan pausar/reanudar/cancelar, y
la preparación de recursos que sólo debe ocurrir una vez por proceso.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator

from parallel_manga_translator.bootstrap import prepare_assets, prepare_runtime
from parallel_manga_translator.infrastructure.execution_control import ExecutionControl
from parallel_manga_translator.infrastructure.logging_config import close_log_file, get_logger
from parallel_manga_translator.ui.job_manifest_store import JobManifestStore
from parallel_manga_translator.ui.job_state import JobState

logger = get_logger(__name__)


class JobExecutionSupport:
    """Entorno de ejecución compartido por `JobRunner` y `RetranslationRunner`.

    Recibe del `JobManager` lo que es suyo —buscar un trabajo, persistirlo, el lock de
    estado, la cola y cómo despertar al worker— y aporta lo que es del entorno de
    ejecución: el lock de procesamiento, los controles vivos y la preparación de recursos.

    `config_for_job` se recibe como invocable, no como valor, a propósito: se resuelve en
    cada llamada, de modo que sustituirlo en el manager (cosa que hacen los tests) sigue
    surtiendo efecto aquí.
    """

    def __init__(
        self,
        *,
        get_job: Callable[[str], JobState],
        manifests: JobManifestStore,
        state_lock: threading.RLock,
        queue: Any,
        wake_worker: Callable[[], None],
        config_for_job: Callable[[JobState], Any],
    ) -> None:
        self.get_job = get_job
        self.manifests = manifests
        self.state_lock = state_lock
        self.queue = queue
        self.wake_worker = wake_worker
        self.config_for_job = config_for_job

        # El pipeline carga modelos CUDA grandes. Un único trabajo en vuelo garantiza
        # aislamiento entre trabajos y evita duplicar memoria de GPU.
        self.processing_lock = threading.Lock()
        # Controles vivos por trabajo: los consultan pausar, reanudar y cancelar.
        self.controls: Dict[str, ExecutionControl] = {}
        self._assets_prepared = False

    # -- callbacks del ExecutionControl ---------------------------------------------

    def on_paused(self, job_id: str) -> None:
        try:
            job = self.get_job(job_id)
            with self.state_lock:
                job.status = "paused"
                job.pause_requested = True
                job.message = "Trabajo pausado en un punto seguro."
                job.updated_at = time.time()
                self.manifests.save(job)
        except Exception:
            logger.exception("No se pudo guardar la pausa del trabajo %s", job_id)

    def on_resumed(self, job_id: str) -> None:
        try:
            job = self.get_job(job_id)
            with self.state_lock:
                job.status = "processing"
                job.pause_requested = False
                job.resume_requested = False
                job.message = "Procesamiento reanudado."
                job.updated_at = time.time()
                self.manifests.save(job)
        except Exception:
            logger.exception("No se pudo guardar la reanudación del trabajo %s", job_id)

    # -- andamiaje -------------------------------------------------------------------

    @contextmanager
    def executing(self, job: JobState, *, message: str, mark_started: bool = False) -> Iterator[ExecutionControl]:
        """Registra el control, marca el trabajo en curso y limpia al salir.

        El cierre del log del trabajo no es opcional ni cosmético: el `FileHandler` de
        `job.log` queda enganchado al logger del paquete y, sin cerrarlo, Windows no deja
        borrar la carpeta del trabajo y los logs de los trabajos siguientes se duplican
        en ese fichero.
        """
        control = ExecutionControl(
            on_paused=lambda: self.on_paused(job.job_id),
            on_resumed=lambda: self.on_resumed(job.job_id),
        )
        with self.state_lock:
            self.controls[job.job_id] = control
            job.status = "processing"
            job.message = message
            if mark_started:
                job.started_at = job.started_at or time.time()
            job.finished_at = 0.0
            job.updated_at = time.time()
            self.manifests.save(job)
        try:
            yield control
        finally:
            close_log_file(str(Path(job.root_dir) / "job.log"))
            with self.state_lock:
                self.controls.pop(job.job_id, None)

    def prepare_environment(self) -> None:
        """Secretos, memoria de GPU, y fuentes y modelos una sola vez por proceso.

        No configura el logging a propósito: cada runner lo hace en su sitio, y ese sitio
        no es el mismo. `JobRunner` reconstruye la configuración del trabajo antes de
        llamar, así que moverlo aquí cambiaría el orden de una operación con efectos.
        """
        prepare_runtime()
        if not self._assets_prepared:
            # El rotulado de la retraducción necesita las mismas fuentes que el pipeline.
            prepare_assets()
            self._assets_prepared = True

    def handle_pause(
        self,
        job: JobState,
        control: ExecutionControl,
        *,
        on_cancelled: Callable[[JobState], None],
        resume_message: str,
        paused_message: str,
    ) -> None:
        """Qué hacer cuando la ejecución se detiene en un punto seguro.

        Tres desenlaces posibles, y los tres los comparten ambos runners: el usuario
        canceló mientras tanto, pidió reanudar (vuelve a la cola), o simplemente pausó.
        Lo único que cambia entre runners es cómo se deja un trabajo cancelado y qué se
        le cuenta al usuario, y eso llega por parámetro.
        """
        requeue = False
        with self.state_lock:
            if job.cancel_requested or control.cancel_requested:
                on_cancelled(job)
            elif job.resume_requested:
                job.status = "queued"
                job.message = resume_message
                job.pause_requested = False
                job.resume_requested = False
                job.updated_at = time.time()
                self.queue.enqueue(job.job_id, preserve_time=False)
                requeue = True
            else:
                job.status = "paused"
                job.message = paused_message
                job.pause_requested = True
                job.updated_at = time.time()
            self.manifests.save(job)
        if requeue:
            self.wake_worker()


__all__ = ["JobExecutionSupport"]
