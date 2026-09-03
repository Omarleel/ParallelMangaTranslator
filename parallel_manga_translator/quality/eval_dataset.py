"""Dataset de regresión construido a partir de trabajos validados por humanos en la UI.

Un *caso* de ``dataset_eval`` empaqueta tres cosas:

* ``paginas/``          -> imágenes de entrada, renombradas a ``0001.jpg``… para fijar el
                           número de página con el que el pipeline las numera.
* ``ground_truth/``     -> una verdad de referencia por página, extraída del ``manifest.json``
                           del trabajo: las regiones que el humano dejó vivas tras corregir,
                           con su bbox final y sus textos finales.
* ``prediccion_base/``  -> ``Transcripción.json`` / ``Traducción.json`` **crudos** de la
                           ejecución original, es decir lo que el pipeline predijo antes de
                           que nadie lo tocara. De ahí sale la línea base.

Con eso, medir si un cambio mejora o empeora es: volver a pasar el pipeline sobre
``paginas/``, puntuar contra ``ground_truth/`` y comparar el resumen con ``baseline.json``.

Semántica de la corrección humana (definida por la UI, ver ``docs/08-UI-EDITOR.md``):

* ``deleted``  -> el humano borró la región: falso positivo del detector.
* ``manual``   -> el humano la añadió a mano: falso negativo del detector (sin texto OCR).
* ``modified`` -> el humano editó el texto renderizado.

Advertencia honesta sobre esta verdad de referencia: una región borrada puede serlo porque
la detección era mala *o* porque el humano decidió no traducir ese texto. Ambas cosas se
miden igual aquí, porque ambas son "lo que el humano quería ver publicado".
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from parallel_manga_translator.quality.evaluation_manager import (
    EvaluationConfig,
    EvaluationManager,
    _page_number_from_name,
    load_ground_truth_file,
    load_pipeline_predictions,
)

PAGES_DIRNAME = "paginas"
REFERENCE_DIRNAME = "referencia"
GROUND_TRUTH_DIRNAME = "ground_truth"
BASE_PREDICTION_DIRNAME = "prediccion_base"
CASE_FILENAME = "case.json"
BASELINE_FILENAME = "baseline.json"
TRANSCRIPTION_FILENAME = "Transcripción.json"
TRANSLATION_FILENAME = "Traducción.json"

#: Métricas donde subir es mejorar.
HIGHER_IS_BETTER: Tuple[str, ...] = (
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "mean_iou",
)
#: Métricas donde bajar es mejorar (tasas de error).
LOWER_IS_BETTER: Tuple[str, ...] = ("mean_ocr_cer", "mean_translation_cer")

TRACKED_METRICS: Tuple[str, ...] = HIGHER_IS_BETTER + LOWER_IS_BETTER

#: Cambios por debajo de esto se consideran ruido, no señal.
DEFAULT_TOLERANCE = 0.005

VERDICT_IMPROVED = "mejora"
VERDICT_REGRESSED = "regresión"
VERDICT_UNCHANGED = "sin cambio"
VERDICT_NO_DATA = "sin datos"


# --------------------------------------------------------------------------------------
# Construcción del dataset desde un trabajo de la UI
# --------------------------------------------------------------------------------------


def _effective_type(region: Mapping[str, Any]) -> str:
    """Tipo comparable entre verdad de referencia y predicción.

    Las regiones añadidas a mano quedan marcadas con ``type == "manual"`` en el manifest,
    que no es un tipo que el pipeline pueda predecir nunca. Se reconstruye a partir del
    estilo para que el desglose por tipo no las deje siempre fuera.
    """
    tipo = str(region.get("type") or "").strip()
    if tipo and tipo != "manual":
        return tipo
    estilo = str(region.get("style") or "").strip().lower()
    return "sfx" if estilo == "onomatopeya" else "dialogue"


def _int_box(value: Any) -> Optional[List[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None


def detection_bbox(region: Mapping[str, Any]) -> Optional[List[int]]:
    """Caja comparable con la que emite el detector.

    Cuidado, aquí está la trampa del formato: ``region["bbox"]`` del manifest es la caja de
    **maquetación del texto** (la que el editor encoge dentro del globo para renderizar), no
    la del globo detectado. La caja del globo vive en ``ui_layout.original_region_bbox`` y es
    la única que se puede comparar contra ``Coordenadas`` de ``Transcripción.json``.
    Las regiones añadidas a mano no tienen esa clave: ahí la caja dibujada por el humano es
    la única referencia disponible.
    """
    layout = region.get("ui_layout")
    if isinstance(layout, Mapping):
        box = _int_box(layout.get("original_region_bbox"))
        if box is not None and box[2] > 0 and box[3] > 0:
            return box
    return _int_box(region.get("bbox"))


def ground_truth_region(region: Mapping[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """Convierte una región corregida del manifest en una región de verdad de referencia."""
    bbox = detection_bbox(region)
    if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
        return None
    return {
        "index": index,
        "bbox": bbox,
        "bbox_texto": _int_box(region.get("bbox")),
        "origen": "manual" if region.get("manual") else "pipeline",
        "tipo": _effective_type(region),
        "tipo_original": region.get("type") or "",
        "estilo": region.get("style") or "",
        "texto_original": str(region.get("original_text") or ""),
        "texto_traducido": str(region.get("translated_text") or ""),
        "manual": bool(region.get("manual")),
        "modificado": bool(region.get("modified")),
        "angulo": float(region.get("rotation_angle") or 0.0),
    }


def build_ground_truth_page(page: Mapping[str, Any]) -> Dict[str, Any]:
    """Verdad de referencia de una página: lo que el humano dejó vivo tras corregir."""
    stem = Path(str(page.get("output_filename") or "")).stem or f"{int(page.get('index', 0)) + 1:04d}"
    kept: List[Dict[str, Any]] = []
    deleted = 0
    manual = 0
    for region in page.get("regions") or []:
        if not isinstance(region, Mapping):
            continue
        if region.get("deleted"):
            deleted += 1
            continue
        converted = ground_truth_region(region, len(kept))
        if converted is None:
            continue
        if converted["manual"]:
            manual += 1
        kept.append(converted)
    return {
        "page": stem,
        "page_number": _page_number_from_name(stem),
        "source_image": page.get("output_filename") or "",
        "original_filename": page.get("source_filename") or "",
        "regions": kept,
        "brush_strokes": len(page.get("brush_strokes") or []),
        "descartadas_por_humano": deleted,
        "anadidas_por_humano": manual,
    }


def build_case_from_ui_job(
    job_dir: Path,
    dest_dir: Path,
    *,
    name: Optional[str] = None,
    copy_images: bool = True,
    copy_reference: bool = True,
) -> Path:
    """Materializa un caso de ``dataset_eval`` a partir de un trabajo corregido de la UI.

    Devuelve la ruta del caso creado. Es idempotente: reescribe la verdad de referencia y
    los metadatos, y solo copia imágenes que falten o hayan cambiado de tamaño.
    """
    job_dir = Path(job_dir)
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"El trabajo no tiene manifest.json: {job_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    case_name = name or str(manifest.get("job_id") or job_dir.name)
    case_dir = Path(dest_dir) / case_name
    gt_dir = case_dir / GROUND_TRUTH_DIRNAME
    pages_dir = case_dir / PAGES_DIRNAME
    reference_dir = case_dir / REFERENCE_DIRNAME
    base_pred_dir = case_dir / BASE_PREDICTION_DIRNAME
    for directory in (gt_dir, base_pred_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if copy_images:
        pages_dir.mkdir(parents=True, exist_ok=True)
    if copy_reference:
        reference_dir.mkdir(parents=True, exist_ok=True)

    input_dir = job_dir / "entrada"
    outputs_dir = job_dir / "outputs"
    page_rows: List[Dict[str, Any]] = []

    for page in manifest.get("pages") or []:
        if not isinstance(page, Mapping):
            continue
        gt_page = build_ground_truth_page(page)
        stem = gt_page["page"]
        (gt_dir / f"{stem}.json").write_text(
            json.dumps(gt_page, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        image_name = str(page.get("output_filename") or f"{stem}.jpg")
        if copy_images:
            source_image = input_dir / str(page.get("source_filename") or "")
            _copy_if_needed(source_image, pages_dir / image_name)
        if copy_reference:
            _copy_if_needed(outputs_dir / "corregida" / image_name, reference_dir / image_name)

        page_rows.append(
            {
                "page": stem,
                "page_number": gt_page["page_number"],
                "image": image_name,
                "original_filename": gt_page["original_filename"],
                "gt_regions": len(gt_page["regions"]),
                "descartadas_por_humano": gt_page["descartadas_por_humano"],
                "anadidas_por_humano": gt_page["anadidas_por_humano"],
                "brush_strokes": gt_page["brush_strokes"],
            }
        )

    _copy_if_needed(outputs_dir / "limpieza" / TRANSCRIPTION_FILENAME, base_pred_dir / TRANSCRIPTION_FILENAME)
    _copy_if_needed(outputs_dir / "traduccion" / TRANSLATION_FILENAME, base_pred_dir / TRANSLATION_FILENAME)

    case = {
        "name": case_name,
        "title": manifest.get("title") or "",
        "origin_job": manifest.get("job_id") or job_dir.name,
        "options": dict(manifest.get("options") or {}),
        "pages": page_rows,
        "totals": {
            "pages": len(page_rows),
            "gt_regions": sum(r["gt_regions"] for r in page_rows),
            "descartadas_por_humano": sum(r["descartadas_por_humano"] for r in page_rows),
            "anadidas_por_humano": sum(r["anadidas_por_humano"] for r in page_rows),
            "brush_strokes": sum(r["brush_strokes"] for r in page_rows),
        },
        "built_at": time.time(),
    }
    (case_dir / CASE_FILENAME).write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_dir


def _copy_if_needed(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


# --------------------------------------------------------------------------------------
# Carga y puntuación de casos
# --------------------------------------------------------------------------------------


@dataclass
class EvalCase:
    name: str
    root: Path
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ground_truth_dir(self) -> Path:
        return self.root / GROUND_TRUTH_DIRNAME

    @property
    def pages_dir(self) -> Path:
        return self.root / PAGES_DIRNAME

    @property
    def reference_dir(self) -> Path:
        return self.root / REFERENCE_DIRNAME

    @property
    def base_prediction_dir(self) -> Path:
        return self.root / BASE_PREDICTION_DIRNAME

    @property
    def baseline_path(self) -> Path:
        return self.root / BASELINE_FILENAME

    def ground_truth_files(self) -> List[Path]:
        return sorted(p for p in self.ground_truth_dir.glob("*.json") if p.is_file())

    def load_baseline(self) -> Optional[Dict[str, Any]]:
        if not self.baseline_path.is_file():
            return None
        return json.loads(self.baseline_path.read_text(encoding="utf-8"))


def load_case(case_dir: Path) -> EvalCase:
    case_dir = Path(case_dir)
    case_file = case_dir / CASE_FILENAME
    if not case_file.is_file():
        raise FileNotFoundError(f"No es un caso de dataset_eval (falta {CASE_FILENAME}): {case_dir}")
    meta = json.loads(case_file.read_text(encoding="utf-8"))
    return EvalCase(name=str(meta.get("name") or case_dir.name), root=case_dir, meta=meta)


def discover_cases(dataset_dir: Path) -> List[EvalCase]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        return []
    return [load_case(child) for child in sorted(dataset_dir.iterdir()) if (child / CASE_FILENAME).is_file()]


def resolve_prediction_jsons(location: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Encuentra ``Transcripción.json`` / ``Traducción.json`` bajo una carpeta de salida.

    Acepta tanto la carpeta ``outputs`` de una ejecución (con ``limpieza/`` y ``traduccion/``)
    como una carpeta plana que contenga ya los dos ficheros.
    """
    location = Path(location)
    candidates = [
        (location / "limpieza" / TRANSCRIPTION_FILENAME, location / "traduccion" / TRANSLATION_FILENAME),
        (location / TRANSCRIPTION_FILENAME, location / TRANSLATION_FILENAME),
    ]
    for transcription, translation in candidates:
        if transcription.is_file() or translation.is_file():
            return (transcription if transcription.is_file() else None, translation if translation.is_file() else None)
    return None, None


def _filter_regions(regions: Iterable[Mapping[str, Any]], tipo: str) -> List[Dict[str, Any]]:
    return [dict(r) for r in regions if str(r.get("type") or "dialogue") == tipo]


def score_case(
    case: EvalCase,
    *,
    predictions_dir: Optional[Path] = None,
    transcription_json: Optional[Path] = None,
    translation_json: Optional[Path] = None,
    config: Optional[EvaluationConfig] = None,
) -> Dict[str, Any]:
    """Puntúa una predicción contra la verdad de referencia del caso.

    Sin argumentos de predicción usa ``prediccion_base/``, es decir reproduce la línea base.
    """
    if predictions_dir is not None:
        transcription_json, translation_json = resolve_prediction_jsons(predictions_dir)
    if transcription_json is None and translation_json is None:
        transcription_json, translation_json = resolve_prediction_jsons(case.base_prediction_dir)
    if transcription_json is None and translation_json is None:
        raise FileNotFoundError(
            f"No se encontraron {TRANSCRIPTION_FILENAME} ni {TRANSLATION_FILENAME} para el caso {case.name}"
        )

    gt_files = case.ground_truth_files()
    if not gt_files:
        raise FileNotFoundError(f"El caso {case.name} no tiene verdad de referencia en {case.ground_truth_dir}")

    manager = EvaluationManager(config or EvaluationConfig())
    predictions = load_pipeline_predictions(transcription_json, translation_json)

    gt_pages = [load_ground_truth_file(path) for path in gt_files]
    evaluations = []
    for gt_page in gt_pages:
        page_number = _page_number_from_name(str(gt_page.get("page") or ""))
        evaluations.append(manager.evaluate_page(gt_page, predictions.get(page_number or -1, [])))
    report = manager.aggregate_report(evaluations)

    tipos = sorted({str(r.get("type") or "dialogue") for page in gt_pages for r in page["regions"]})
    by_type: Dict[str, Any] = {}
    for tipo in tipos:
        subset = []
        for gt_page in gt_pages:
            page_number = _page_number_from_name(str(gt_page.get("page") or ""))
            filtered_gt = {"page": gt_page["page"], "regions": _filter_regions(gt_page["regions"], tipo)}
            filtered_pred = _filter_regions(predictions.get(page_number or -1, []), tipo)
            subset.append(manager.evaluate_page(filtered_gt, filtered_pred))
        by_type[tipo] = manager.aggregate_report(subset)["summary"]

    report["case"] = case.name
    report["by_type"] = by_type
    report["prediction_source"] = {
        "transcription": str(transcription_json) if transcription_json else None,
        "translation": str(translation_json) if translation_json else None,
    }
    report["generated_at"] = time.time()
    return report


def write_baseline(case: EvalCase, report: Mapping[str, Any]) -> Path:
    payload = {
        "case": case.name,
        "summary": dict(report.get("summary") or {}),
        "by_type": {k: dict(v) for k, v in (report.get("by_type") or {}).items()},
        "generated_at": report.get("generated_at", time.time()),
        "note": "Puntuación de prediccion_base/ contra ground_truth/. Regenerar solo cuando se acepte un nuevo nivel de calidad.",
    }
    case.baseline_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return case.baseline_path


# --------------------------------------------------------------------------------------
# Comparación contra la línea base
# --------------------------------------------------------------------------------------


def compare_metric(name: str, baseline: Any, current: Any, tolerance: float = DEFAULT_TOLERANCE) -> Dict[str, Any]:
    direction = "lower" if name in LOWER_IS_BETTER else "higher"
    if baseline is None or current is None:
        return {
            "metric": name,
            "direction": direction,
            "baseline": baseline,
            "current": current,
            "delta": None,
            "verdict": VERDICT_NO_DATA,
        }
    delta = round(float(current) - float(baseline), 6)
    if abs(delta) <= tolerance:
        verdict = VERDICT_UNCHANGED
    elif (delta > 0) == (direction == "higher"):
        verdict = VERDICT_IMPROVED
    else:
        verdict = VERDICT_REGRESSED
    return {
        "metric": name,
        "direction": direction,
        "baseline": round(float(baseline), 6),
        "current": round(float(current), 6),
        "delta": delta,
        "verdict": verdict,
    }


def compare_summaries(
    baseline_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
    tolerance: float = DEFAULT_TOLERANCE,
    metrics: Sequence[str] = TRACKED_METRICS,
) -> Dict[str, Any]:
    """Compara dos resúmenes y dictamina mejora / regresión / sin cambio."""
    rows = [
        compare_metric(name, baseline_summary.get(name), current_summary.get(name), tolerance)
        for name in metrics
    ]
    regressed = [r["metric"] for r in rows if r["verdict"] == VERDICT_REGRESSED]
    improved = [r["metric"] for r in rows if r["verdict"] == VERDICT_IMPROVED]
    if regressed:
        verdict = VERDICT_REGRESSED
    elif improved:
        verdict = VERDICT_IMPROVED
    else:
        verdict = VERDICT_UNCHANGED
    return {
        "tolerance": tolerance,
        "metrics": {r["metric"]: r for r in rows},
        "improved": improved,
        "regressed": regressed,
        "verdict": verdict,
    }


def compare_reports(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, Any]:
    """Compara un reporte fresco contra un ``baseline.json``, global y por tipo de región."""
    result = compare_summaries(
        baseline.get("summary") or {}, current.get("summary") or {}, tolerance
    )
    baseline_types = baseline.get("by_type") or {}
    current_types = current.get("by_type") or {}
    result["by_type"] = {
        tipo: compare_summaries(baseline_types.get(tipo) or {}, current_types.get(tipo) or {}, tolerance)
        for tipo in sorted(set(baseline_types) | set(current_types))
    }
    result["case"] = current.get("case") or baseline.get("case")
    return result


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _format_number(value: Any) -> str:
    if value is None:
        return "    -"
    return f"{float(value):.4f}"


def format_comparison(comparison: Mapping[str, Any]) -> str:
    lines = [f"Caso: {comparison.get('case') or '?'}    tolerancia={comparison.get('tolerance')}"]
    lines.append(f"{'métrica':<24}{'base':>10}{'actual':>10}{'delta':>10}  veredicto")
    for name, row in (comparison.get("metrics") or {}).items():
        delta = row.get("delta")
        delta_text = "     -" if delta is None else f"{delta:+.4f}"
        lines.append(
            f"{name:<24}{_format_number(row.get('baseline')):>10}{_format_number(row.get('current')):>10}"
            f"{delta_text:>10}  {row.get('verdict')}"
        )
    by_type = comparison.get("by_type") or {}
    for tipo, sub in by_type.items():
        lines.append(f"  [{tipo}] {sub.get('verdict')}"
                     + (f" (regresión en: {', '.join(sub.get('regressed') or [])})" if sub.get("regressed") else ""))
    lines.append(f"VEREDICTO GLOBAL: {comparison.get('verdict')}")
    return "\n".join(lines)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def evaluation_settings():
    """Sección `evaluation` de `config.yaml`, la única fuente de estas rutas y umbrales.

    Degrada a los valores por defecto del dataclass si el YAML no se puede cargar: el arnés
    debe poder ejecutarse sobre una copia recién clonada sin un `config.yaml` válido.
    """
    from parallel_manga_translator.config.app_config import EvaluationSettings

    try:
        from parallel_manga_translator.config.config_manager import ConfigManager

        return ConfigManager(str(_repo_root() / "config.yaml")).build_application_config().evaluation
    except Exception:
        return EvaluationSettings()


def _default_dataset_dir() -> Path:
    """Carpeta del dataset de evaluación según `evaluation.dataset_dir`.

    No se confunde con `processing.input_dir`: ése es el manga que traduce el usuario.
    """
    configured = Path(str(evaluation_settings().dataset_dir or "dataset_eval"))
    return configured if configured.is_absolute() else _repo_root() / configured


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Construye y puntúa el dataset de regresión validado por humanos (dataset_eval)."
    )
    parser.add_argument("--dataset", default="", help="Carpeta dataset_eval (por defecto la del repositorio).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Crea o actualiza un caso desde un trabajo corregido de la UI.")
    p_build.add_argument("--job", required=True, help="Carpeta del trabajo (.pmt_ui_jobs/<id>).")
    p_build.add_argument("--name", default="", help="Nombre del caso (por defecto, el id del trabajo).")
    p_build.add_argument("--no-images", action="store_true", help="No copiar las páginas de entrada.")
    p_build.add_argument("--no-reference", action="store_true", help="No copiar los renders corregidos.")
    p_build.add_argument("--baseline", action="store_true", help="Recalcular baseline.json tras construir.")

    p_list = sub.add_parser("list", help="Lista los casos disponibles.")

    p_score = sub.add_parser("score", help="Puntúa una ejecución contra la verdad de referencia.")
    p_score.add_argument("--case", default="", help="Nombre del caso; si se omite, todos.")
    p_score.add_argument("--predictions", default="", help="Carpeta outputs/ de la ejecución a medir.")
    p_score.add_argument("--output", default="", help="Carpeta donde escribir los reportes JSON.")
    p_score.add_argument("--iou", type=float, default=None, help="Umbral IoU; por defecto, evaluation.iou_threshold.")
    p_score.add_argument("--tolerance", type=float, default=None, help="Ruido tolerado; por defecto, evaluation.tolerance.")
    p_score.add_argument("--fail-on-regression", action="store_true", help="Devuelve 1 si alguna métrica empeora.")

    p_baseline = sub.add_parser("refresh-baseline", help="Regenera baseline.json desde prediccion_base/.")
    p_baseline.add_argument("--case", default="", help="Nombre del caso; si se omite, todos.")
    p_baseline.add_argument("--iou", type=float, default=None)

    args = parser.parse_args(argv)
    settings = evaluation_settings()
    dataset_dir = Path(args.dataset) if args.dataset else _default_dataset_dir()
    iou = settings.iou_threshold if getattr(args, "iou", None) is None else args.iou
    tolerance = settings.tolerance if getattr(args, "tolerance", None) is None else args.tolerance

    if args.command == "build":
        case_dir = build_case_from_ui_job(
            Path(args.job),
            dataset_dir,
            name=args.name or None,
            copy_images=not args.no_images,
            copy_reference=not args.no_reference,
        )
        case = load_case(case_dir)
        print(f"Caso construido en {case_dir}")
        print(json.dumps(case.meta.get("totals", {}), ensure_ascii=False, indent=2))
        if args.baseline:
            report = score_case(case)
            write_baseline(case, report)
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "list":
        cases = discover_cases(dataset_dir)
        if not cases:
            print(f"No hay casos en {dataset_dir}")
            return 0
        for case in cases:
            totals = case.meta.get("totals", {})
            options = case.meta.get("options", {})
            print(
                f"{case.name:<18} {totals.get('pages', 0):>3} páginas  "
                f"{totals.get('gt_regions', 0):>4} regiones  "
                f"{options.get('source_language', '?')} -> {options.get('target_language', '?')}"
            )
        return 0

    cases = discover_cases(dataset_dir)
    if args.case:
        cases = [c for c in cases if c.name == args.case]
        if not cases:
            print(f"Caso no encontrado: {args.case}")
            return 2

    if args.command == "refresh-baseline":
        for case in cases:
            report = score_case(case, config=EvaluationConfig(iou_threshold=iou))
            path = write_baseline(case, report)
            print(f"{case.name}: baseline escrito en {path}")
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0

    # score
    exit_code = 0
    output_dir = Path(args.output) if args.output else None
    for case in cases:
        predictions_dir = Path(args.predictions) if args.predictions else None
        report = score_case(
            case,
            predictions_dir=predictions_dir,
            config=EvaluationConfig(iou_threshold=iou),
        )
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{case.name}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        baseline = case.load_baseline()
        if baseline is None:
            print(f"{case.name}: sin baseline.json, solo resumen")
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
            continue
        comparison = compare_reports(baseline, report, tolerance=tolerance)
        print(format_comparison(comparison))
        print()
        if comparison["verdict"] == VERDICT_REGRESSED and args.fail_on_regression:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
