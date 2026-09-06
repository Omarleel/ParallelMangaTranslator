"""Lectura y escritura de `manifest.json`, la fuente de verdad de un trabajo de la UI.

Estaba repartido en `JobManager` como cuatro métodos privados con 34 sitios de llamada.
Que el guardado sea atómico y sobreviva al antivirus era, por tanto, una convención que
había que respetar en cada uno de esos sitios; aquí es una propiedad de un objeto que se
puede probar solo.

En Windows el escáner on-access del antivirus abre los archivos recién escritos y, mientras
sostiene el handle, renombrar sobre el destino devuelve `ERROR_ACCESS_DENIED` (WinError 5):
renombrar sobre un archivo existente necesita acceso de borrado sobre él. Es transitorio, y
perder el guardado aborta el trabajo entero, así que se reintenta con espera creciente.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ui.job_state import (
    PAGE_PATH_FIELDS,
    JobOptions,
    JobState,
    PageState,
    rebase_stored_path,
)

logger = get_logger(__name__)

MANIFEST_FILENAME = "manifest.json"

#: Reintentos del `replace` final. Seis con espera exponencial desde 50 ms cubren de sobra
#: lo que tarda un escáner en soltar un JSON pequeño.
MANIFEST_REPLACE_RETRIES = 6
MANIFEST_REPLACE_BACKOFF = 0.05


class JobManifestStore:
    """Persiste un `JobState` en disco. No sabe nada de colas, hilos de trabajo ni HTTP."""

    def __init__(self, jobs_root: Path) -> None:
        self.jobs_root = Path(jobs_root)
        # El temporal tiene nombre fijo por trabajo, así que dos hilos guardando a la vez
        # escribirían sobre el mismo archivo y uno podría tenerlo abierto mientras el otro
        # lo renombra: el mismo WinError 5, esta vez sin antivirus de por medio. Antes esto
        # dependía de que los 34 sitios de llamada tomaran el lock del manager; ahora la
        # garantía es local y no se puede saltar por olvido.
        self._write_lock = threading.Lock()

    def manifest_path(self, job: JobState) -> Path:
        return Path(job.root_dir) / MANIFEST_FILENAME

    def save(self, job: JobState) -> None:
        """Escribe el manifiesto de forma atómica. Propaga si no lo consigue.

        Callar el fallo sería peor que abortar: el manifiesto es lo único que sabe qué
        páginas van hechas, así que perderlo en silencio deja el trabajo mintiendo.
        """
        with self._write_lock:
            manifest = self.manifest_path(job)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = manifest.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8"
            )

            for intento in range(1, MANIFEST_REPLACE_RETRIES + 1):
                try:
                    tmp_path.replace(manifest)
                    return
                except PermissionError:
                    if intento == MANIFEST_REPLACE_RETRIES:
                        logger.warning(
                            "No se pudo reemplazar %s tras %s intentos. Suele ser un antivirus "
                            "sosteniendo el archivo recién escrito; excluir la carpeta de trabajos "
                            "del escaneo en tiempo real lo evita.",
                            manifest,
                            intento,
                        )
                        raise
                    time.sleep(MANIFEST_REPLACE_BACKOFF * (2 ** (intento - 1)))

    def load(self, job_id: str) -> JobState:
        """Reconstruye un `JobState` desde disco, reanclando rutas si la carpeta se movió."""
        manifest = self.jobs_root / job_id / MANIFEST_FILENAME
        if not manifest.exists():
            raise FileNotFoundError("No existe ese trabajo de UI.")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("El manifiesto del trabajo no es válido.")

        # Se filtra por los campos declarados: un manifiesto de una versión anterior puede
        # traer claves que ya no existen, y eso no debe impedir abrir el trabajo.
        page_fields = PageState.__dataclass_fields__
        data["pages"] = [
            PageState(**{k: v for k, v in page.items() if k in page_fields})
            for page in data.get("pages", [])
            if isinstance(page, dict)
        ]
        options = data.get("options", {})
        if isinstance(options, dict):
            data["options"] = JobOptions(
                **{k: v for k, v in options.items() if k in JobOptions.__dataclass_fields__}
            )
        elif not isinstance(options, JobOptions):
            data["options"] = JobOptions()

        job = JobState(**{k: v for k, v in data.items() if k in JobState.__dataclass_fields__})
        if self.rebase_paths(job):
            logger.info("Rutas del trabajo %s reancladas a %s.", job_id, job.root_dir)
            self.save(job)
        return job

    def rebase_paths(self, job: JobState) -> bool:
        """Corrige las rutas absolutas de un trabajo cuya carpeta se movió de sitio."""
        new_root = self.jobs_root / job.job_id
        old_root = job.root_dir
        if not old_root or Path(old_root) == new_root:
            return False
        job.root_dir = str(new_root)
        job.input_dir = rebase_stored_path(job.input_dir, old_root, new_root)
        job.output_dir = rebase_stored_path(job.output_dir, old_root, new_root)
        for page in job.pages:
            for attribute in PAGE_PATH_FIELDS:
                setattr(page, attribute, rebase_stored_path(getattr(page, attribute), old_root, new_root))
        return True


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_REPLACE_BACKOFF",
    "MANIFEST_REPLACE_RETRIES",
    "JobManifestStore",
]
