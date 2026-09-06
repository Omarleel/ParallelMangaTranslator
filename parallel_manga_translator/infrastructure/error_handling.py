from __future__ import annotations

import json
import logging
import re
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from parallel_manga_translator.infrastructure.execution_control import JobControlError


class PipelineStage(str, Enum):
    """Etapas del pipeline, con los nombres que ya se escriben en los informes de fallo.

    Existían como cadenas sueltas en cada `processing_stage(...)`. Un typo creaba una
    etapa fantasma que nadie detectaba hasta leer un `fallidas/*.error.json`. Hereda de
    `str` a propósito: serializa igual que antes, así que los informes ya escritos y
    cualquier herramienta que los lea siguen siendo válidos.
    """

    LIMPIEZA = "limpieza"
    GUARDAR_LIMPIEZA = "guardar_limpieza"
    OCR_TRADUCCION_RENDER = "ocr_traduccion_render"
    GUARDAR_TRADUCCION = "guardar_traduccion"
    #: Valor por defecto de `PageFailureReport.from_exception` cuando el fallo llega sin
    #: envoltorio de etapa.
    DESCONOCIDA = "unknown"

    def __str__(self) -> str:
        """Serializa como su valor, no como `PipelineStage.LIMPIEZA`.

        Heredar de `str` hace que la comparación `== "limpieza"` funcione, pero en
        Python < 3.11 el `__str__` que gana es el de `Enum`, así que `str(etapa)` daría
        el nombre cualificado. `PageFailureReport` hace exactamente ese `str(...)`, de
        modo que sin esto los informes de fallo cambiarían de formato en silencio.
        """
        return self.value


class FailureKind(str, Enum):
    """Qué *clase* de fallo fue, además de en qué etapa ocurrió.

    La etapa dice dónde; esto dice si merece la pena reintentar. En un pipeline de IA la
    mayoría de los fallos son de recursos o de proveedor externo —transitorios— y se
    confunden con errores de código porque todos llegan como `Exception`.
    """

    MEMORIA = "memoria"
    PROVEEDOR_EXTERNO = "proveedor_externo"
    MODELO_NO_DISPONIBLE = "modelo_no_disponible"
    ENTRADA_INVALIDA = "entrada_invalida"
    DESCONOCIDO = "desconocido"

    def __str__(self) -> str:
        return self.value


#: Clasificación por nombre de clase, para no importar torch/requests aquí sólo para esto.
_TIPOS_MEMORIA = {"OutOfMemoryError", "MemoryError", "CudaError"}
_TIPOS_PROVEEDOR = {
    "TooManyRequests", "TranslationNotFound", "RequestError", "ServerException",
    "ConnectionError", "Timeout", "ReadTimeout", "HTTPError", "APIError", "APIStatusError",
}
_TIPOS_ENTRADA = {"ValueError", "TypeError", "KeyError", "IndexError"}


def classify_failure(exc: BaseException) -> FailureKind:
    """Clasifica un fallo en una familia accionable."""
    nombre = type(exc).__name__
    mensaje = str(exc).lower()

    if nombre in _TIPOS_MEMORIA or "out of memory" in mensaje or "cuda error" in mensaje:
        return FailureKind.MEMORIA
    if nombre in _TIPOS_PROVEEDOR:
        return FailureKind.PROVEEDOR_EXTERNO
    if isinstance(exc, FileNotFoundError) or "no se pudo cargar" in mensaje or "no existe el modelo" in mensaje:
        return FailureKind.MODELO_NO_DISPONIBLE
    if nombre in _TIPOS_ENTRADA:
        return FailureKind.ENTRADA_INVALIDA
    return FailureKind.DESCONOCIDO


@dataclass
class PageFailureReport:
    """Reporte persistente y fácil de inspeccionar para una página fallida."""

    page_index: int
    filename: str
    stage: str
    error_type: str
    error_message: str
    traceback: str
    #: Familia del fallo. La etapa dice dónde ocurrió; esto dice de qué tipo es.
    failure_kind: str = FailureKind.DESCONOCIDO.value
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        *,
        page_index: int,
        filename: str,
        default_stage: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PageFailureReport":
        if isinstance(exc, StageProcessingError):
            original = exc.original_exception
            stage = exc.stage
            tb = exc.formatted_traceback
            error_type = exc.original_error_type
            error_message = exc.original_error_message
            merged_metadata = dict(exc.metadata)
            if metadata:
                merged_metadata.update(metadata)
        else:
            original = exc
            stage = default_stage
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            error_type = type(original).__name__
            error_message = str(original)
            merged_metadata = dict(metadata or {})
        return cls(
            page_index=int(page_index),
            filename=str(filename),
            stage=str(stage or default_stage),
            error_type=str(error_type),
            error_message=str(error_message),
            traceback=tb,
            # Se clasifica la excepción original, no el envoltorio de etapa: un
            # StageProcessingError siempre sería "desconocido" y no diría nada.
            failure_kind=classify_failure(original).value,
            metadata=merged_metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StageProcessingError(RuntimeError):
    """Error con contexto de etapa para no perder dónde falló la página."""

    def __init__(
        self,
        stage: str,
        original_exception: Exception,
        *,
        page_index: int | None = None,
        filename: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.stage = stage
        self.original_exception = original_exception
        self.page_index = page_index
        self.filename = filename
        self.metadata = dict(metadata or {})
        self.original_error_type = type(original_exception).__name__
        self.original_error_message = str(original_exception)
        self.formatted_traceback = "".join(
            traceback.format_exception(type(original_exception), original_exception, original_exception.__traceback__)
        )
        super().__init__(
            f"Fallo en etapa '{stage}'"
            f"{f' para {filename}' if filename else ''}: "
            f"{self.original_error_type}: {self.original_error_message}"
        )


def unwrap_original_exception(exc: Exception) -> Exception:
    return exc.original_exception if isinstance(exc, StageProcessingError) else exc


@contextmanager
def processing_stage(
    stage: str,
    *,
    logger: logging.Logger,
    page_index: int | None = None,
    filename: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    """Context manager para loggear una etapa y relanzar con contexto estructurado."""
    try:
        logger.debug("Iniciando etapa | stage=%s | page=%s | file=%s", stage, page_index, filename)
        yield
        logger.debug("Etapa completada | stage=%s | page=%s | file=%s", stage, page_index, filename)
    except (StageProcessingError, JobControlError):
        raise
    except Exception as exc:
        logger.exception("Fallo en etapa | stage=%s | page=%s | file=%s | error=%s", stage, page_index, filename, exc)
        raise StageProcessingError(stage, exc, page_index=page_index, filename=filename, metadata=metadata) from exc


def _safe_report_name(page_index: int, filename: str) -> str:
    stem = Path(filename).stem or "pagina"
    suffix = Path(filename).suffix or ""
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "pagina"
    return f"{page_index + 1:04d}_{safe_stem}{suffix}.error.json"


def write_failure_report(output_root: str | Path, report: PageFailureReport) -> Path:
    failure_dir = Path(output_root) / "fallidas"
    failure_dir.mkdir(parents=True, exist_ok=True)
    path = failure_dir / _safe_report_name(report.page_index, report.filename)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
