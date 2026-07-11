from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PageMetrics:
    page_index: int
    filename: str
    status: str = "ok"
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    duration_seconds: float = 0.0
    detected_regions: int = 0
    detected_bubbles: int = 0
    detected_sfx: int = 0
    ocr_empty: int = 0
    translations_empty: int = 0
    image_width: int = 0
    image_height: int = 0
    retries: int = 0
    timings: Dict[str, float] = field(default_factory=dict)

    def finish(self) -> "PageMetrics":
        self.finished_at = time.time()
        self.duration_seconds = round(self.finished_at - self.started_at, 4)
        return self


class MetricsWriter:
    def __init__(self, output_root: str) -> None:
        self.output_root = Path(output_root)
        self.pages_dir = self.output_root / "metricas" / "paginas"
        self.pages_dir.mkdir(parents=True, exist_ok=True)

    def write_page(self, metrics: PageMetrics) -> None:
        metrics.finish()
        file_path = self.pages_dir / f"{metrics.page_index + 1:04d}.json"
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(metrics), fh, ensure_ascii=False, indent=2)

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _iso_timestamp(value: Optional[float]) -> Optional[str]:
        if value is None or value <= 0:
            return None
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()

    @classmethod
    def _rows_for_execution(
        cls,
        rows: List[Dict[str, Any]],
        execution_started_at: Optional[float],
        execution_finished_at: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Devuelve solo métricas creadas durante la ejecución actual.

        Esto evita que una reanudación mezcle páginas antiguas con las procesadas
        ahora y mida como trabajo el tiempo transcurrido entre ambas ejecuciones.
        """
        if execution_started_at is None or execution_finished_at is None:
            return rows

        tolerance = 1.0
        lower = execution_started_at - tolerance
        upper = execution_finished_at + tolerance
        selected: List[Dict[str, Any]] = []
        for row in rows:
            started = cls._as_float(row.get("started_at"))
            finished = cls._as_float(row.get("finished_at"))
            if started <= 0 and finished <= 0:
                continue
            anchor = started if started > 0 else finished
            if lower <= anchor <= upper:
                selected.append(row)
        return selected

    @classmethod
    def aggregate(
        cls,
        output_root: str,
        *,
        execution_started_at: Optional[float] = None,
        execution_finished_at: Optional[float] = None,
        processing_started_at: Optional[float] = None,
        processing_finished_at: Optional[float] = None,
        execution_duration_seconds: Optional[float] = None,
        processing_duration_seconds: Optional[float] = None,
        total_input_pages: Optional[int] = None,
        execution_mode: Optional[str] = None,
    ) -> Optional[str]:
        root = Path(output_root)
        pages_dir = root / "metricas" / "paginas"
        if not pages_dir.exists():
            return None

        rows: List[Dict[str, Any]] = []
        for file_path in sorted(pages_dir.glob("*.json")):
            try:
                rows.append(json.loads(file_path.read_text(encoding="utf-8")))
            except Exception:
                continue
        if not rows:
            return None

        execution_rows = cls._rows_for_execution(
            rows,
            execution_started_at,
            execution_finished_at,
        )

        page_durations = [
            max(0.0, cls._as_float(row.get("duration_seconds")))
            for row in execution_rows
        ]
        accumulated_page_seconds = sum(page_durations)
        accumulated_stage_seconds = sum(
            max(0.0, cls._as_float(stage_seconds))
            for row in execution_rows
            for stage_seconds in (row.get("timings") or {}).values()
        )
        accumulated_page_wait_seconds = max(
            0.0,
            accumulated_page_seconds - accumulated_stage_seconds,
        )

        if execution_duration_seconds is not None:
            real_total_seconds = max(0.0, execution_duration_seconds)
        elif execution_started_at is not None and execution_finished_at is not None:
            real_total_seconds = max(0.0, execution_finished_at - execution_started_at)
        else:
            starts = [
                cls._as_float(row.get("started_at"))
                for row in execution_rows
                if cls._as_float(row.get("started_at")) > 0
            ]
            finishes = [
                cls._as_float(row.get("finished_at"))
                for row in execution_rows
                if cls._as_float(row.get("finished_at")) > 0
            ]
            real_total_seconds = (
                max(0.0, max(finishes) - min(starts))
                if starts and finishes
                else accumulated_page_seconds
            )

        if processing_duration_seconds is not None:
            real_processing_seconds = max(0.0, processing_duration_seconds)
        elif processing_started_at is not None and processing_finished_at is not None:
            real_processing_seconds = max(0.0, processing_finished_at - processing_started_at)
        else:
            real_processing_seconds = real_total_seconds

        processed_this_run = len(execution_rows)
        average_real_seconds = (
            real_processing_seconds / processed_this_run
            if processed_this_run
            else 0.0
        )
        average_accumulated_seconds = (
            accumulated_page_seconds / processed_this_run
            if processed_this_run
            else 0.0
        )
        overlap_factor = (
            accumulated_page_seconds / real_processing_seconds
            if real_processing_seconds > 0
            else 0.0
        )
        initialization_seconds = (
            max(0.0, processing_started_at - execution_started_at)
            if execution_started_at is not None and processing_started_at is not None
            else 0.0
        )
        finalization_seconds = (
            max(0.0, execution_finished_at - processing_finished_at)
            if execution_finished_at is not None and processing_finished_at is not None
            else max(0.0, real_total_seconds - real_processing_seconds - initialization_seconds)
        )
        skipped_this_run = (
            max(0, int(total_input_pages) - processed_this_run)
            if total_input_pages is not None
            else 0
        )

        summary = {
            "paginas": len(rows),
            "ok": sum(1 for row in rows if row.get("status") == "ok"),
            "fallidas": sum(1 for row in rows if row.get("status") != "ok"),
            # Tiempo de pared real de esta ejecución, no la suma de páginas solapadas.
            "duracion_total_segundos": round(real_total_seconds, 3),
            "duracion_procesamiento_segundos": round(real_processing_seconds, 3),
            "duracion_inicializacion_segundos": round(initialization_seconds, 3),
            "duracion_finalizacion_exportacion_segundos": round(finalization_seconds, 3),
            "promedio_segundos_por_pagina": round(average_real_seconds, 3),
            "paginas_procesadas_en_esta_ejecucion": processed_this_run,
            "paginas_omitidas_en_esta_ejecucion": skipped_this_run,
            # Métricas acumuladas útiles para evaluar paralelismo/pipeline.
            "trabajo_acumulado_paginas_segundos": round(accumulated_page_seconds, 3),
            "trabajo_acumulado_etapas_segundos": round(accumulated_stage_seconds, 3),
            "espera_acumulada_paginas_segundos": round(accumulated_page_wait_seconds, 3),
            "promedio_trabajo_acumulado_segundos_por_pagina": round(average_accumulated_seconds, 3),
            "factor_solapamiento": round(overlap_factor, 3),
            "modo_ejecucion": execution_mode or "desconocido",
            "inicio_ejecucion_utc": cls._iso_timestamp(execution_started_at),
            "fin_ejecucion_utc": cls._iso_timestamp(execution_finished_at),
            "regiones_detectadas": sum(int(row.get("detected_regions") or 0) for row in rows),
            "globos_detectados": sum(int(row.get("detected_bubbles") or 0) for row in rows),
            "onomatopeyas_detectadas": sum(int(row.get("detected_sfx") or 0) for row in rows),
            "ocr_vacios": sum(int(row.get("ocr_empty") or 0) for row in rows),
            "traducciones_vacias": sum(int(row.get("translations_empty") or 0) for row in rows),
            "paginas_con_error": [row.get("filename") for row in rows if row.get("status") != "ok"],
        }
        report = {"resumen": summary, "paginas": rows}
        report_path = root / "metricas" / "reporte.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(report_path)