from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
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
    def aggregate(output_root: str) -> Optional[str]:
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
        summary = {
            "paginas": len(rows),
            "ok": sum(1 for row in rows if row.get("status") == "ok"),
            "fallidas": sum(1 for row in rows if row.get("status") != "ok"),
            "duracion_total_segundos": round(sum(float(row.get("duration_seconds") or 0) for row in rows), 3),
            "promedio_segundos_por_pagina": round(sum(float(row.get("duration_seconds") or 0) for row in rows) / max(1, len(rows)), 3),
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
