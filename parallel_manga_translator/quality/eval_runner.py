"""Ejecuta un caso de ``dataset_eval`` midiendo detección, OCR e inpainting.

Por qué existe además de ``eval_dataset``:

``eval_dataset score`` puntúa una salida ya generada. Para iterar sobre la detección y
la limpieza hace falta *generar* esa salida, y hacerlo con el pipeline completo cuesta
una traducción por región (red, cuota y no determinismo) que no aporta nada cuando lo
que se está tocando es la detección o el inpainting.

Este runner ejecuta **las mismas funciones que usa el pipeline real** hasta justo antes
de traducir:

``pipeline_limpieza_y_ocr`` (limpieza -> extraer_regiones -> transcribir) -> filtro de
idioma de origen

y escribe un ``Transcripción.json`` con el mismo formato que emite
``rendering_pipeline_mixin._push_original_texts_to_queue``, de modo que se puntúa con el
mismo evaluador. Lo que no se ejecuta es traducción y renderizado; por eso
``mean_translation_cer`` queda fuera de la comparación (ver ``RUNNER_METRICS``).

Además mide la calidad de la limpieza, que ``eval_dataset`` no cubre:

* ``residual_ink``   fracción de la tinta original que sigue visible tras limpiar (↓).
* ``outside_damage`` diferencia media fuera de las máscaras a borrar, 0-255 (↓).
* ``verifier_pass_rate`` regiones que aprueban el ``VisualInpaintVerifier`` (↑).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.config.app_config import ApplicationConfig
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.processing.pipeline import PageContext, pipeline_limpieza_y_ocr
from parallel_manga_translator.quality.eval_dataset import (
    DEFAULT_TOLERANCE,
    EvalCase,
    compare_summaries,
    format_comparison,
    load_case,
    score_case,
)
from parallel_manga_translator.quality.evaluation_manager import EvaluationConfig
from parallel_manga_translator.quality.visual_inpaint_verifier import VisualInpaintVerifier

logger = get_logger(__name__)

TRANSCRIPTION_FILENAME = "Transcripción.json"
CLEAN_DIRNAME = "limpieza"
RUN_REPORT_FILENAME = "reporte.json"
REGIONS_FILENAME = "regiones.json"
DEFAULT_RUNS_DIRNAME = "ejecuciones"

#: Métricas de ``eval_dataset`` que este runner sí puede medir. La traducción queda
#: fuera a propósito: no se ejecuta, y compararla daría una regresión falsa.
RUNNER_METRICS: Tuple[str, ...] = (
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "mean_iou",
    "mean_ocr_cer",
)

#: Métricas de limpieza, con su dirección de mejora.
INPAINT_HIGHER_IS_BETTER: Tuple[str, ...] = ("verifier_pass_rate",)
INPAINT_LOWER_IS_BETTER: Tuple[str, ...] = ("residual_ink", "outside_damage")
INPAINT_METRICS: Tuple[str, ...] = INPAINT_LOWER_IS_BETTER + INPAINT_HIGHER_IS_BETTER

#: Umbral de tinta: un píxel cuenta como tinta si está esta cantidad de luma por
#: debajo de la mediana del fondo de su región.
INK_LUMA_MARGIN = 45.0


# --------------------------------------------------------------------------------------
# Configuración de la ejecución
# --------------------------------------------------------------------------------------


@dataclass
class RunSettings:
    """Ajustes de una ejecución de medición."""

    case_name: str
    run_dir: Path
    pages: Optional[int] = None
    debug_artifacts: bool = False
    overrides: Dict[str, Any] = field(default_factory=dict)


def _case_options(case: EvalCase) -> Dict[str, Any]:
    options = case.meta.get("options")
    return dict(options) if isinstance(options, Mapping) else {}


def build_case_config(
    case: EvalCase,
    run_dir: Path,
    *,
    base_config_path: str = "config.yaml",
    debug_artifacts: bool = False,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ApplicationConfig:
    """Construye la configuración de la ejecución a partir de ``case.json``.

    Reproducir los motores del caso es lo que hace comparable el resultado con
    ``baseline.json``; puntuar con otros motores mediría la diferencia entre
    configuraciones, no el efecto de un cambio de código (ver ``dataset_eval/README.md``).
    """
    from parallel_manga_translator.cli import build_default_config

    config = build_default_config(base_config_path)
    options = _case_options(case)

    translation = config.translation
    if options.get("source_language"):
        translation = replace(translation, idioma_entrada=str(options["source_language"]))
    if options.get("target_language"):
        translation = replace(translation, idioma_salida=str(options["target_language"]))
    if options.get("inpaint_model"):
        translation = replace(translation, modelo_inpaint=str(options["inpaint_model"]))

    ocr = config.ocr
    if options.get("detection_engine"):
        ocr = replace(ocr, detection_engine=str(options["detection_engine"]))
    if options.get("transcription_engine"):
        ocr = replace(ocr, transcription_engine=str(options["transcription_engine"]))

    # Aísla salidas y caché de la ejecución: los artefactos de debug del detector
    # cuelgan de processing.ruta_carpeta_entrada.
    processing = replace(
        config.processing,
        ruta_carpeta_entrada=str(run_dir),
        cache_dir=str(run_dir / ".cache"),
        usar_paralelismo=False,
    )
    quality = replace(
        config.quality,
        bubble_merge_debug=bool(debug_artifacts),
        visual_inpaint_debug=bool(debug_artifacts),
    )
    secciones = {"quality": quality, "translation": translation, "ocr": ocr}
    for key, value in dict(overrides or {}).items():
        seccion, _, campo = key.rpartition(".")
        seccion = seccion or "quality"
        if seccion not in secciones:
            raise ValueError(f"Sección de override desconocida: {seccion} (usa {sorted(secciones)})")
        destino = secciones[seccion]
        if not hasattr(destino, campo):
            raise ValueError(f"Override desconocido para {type(destino).__name__}: {campo}")
        secciones[seccion] = replace(destino, **{campo: value})
    quality, translation, ocr = secciones["quality"], secciones["translation"], secciones["ocr"]

    config = replace(config, translation=translation, ocr=ocr, processing=processing, quality=quality)

    return config


# --------------------------------------------------------------------------------------
# Métricas de limpieza
# --------------------------------------------------------------------------------------


def _luma(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _region_clean_mask(region: TextRegion, shape: Tuple[int, int]) -> np.ndarray:
    mask = region.clean_mask if region.clean_mask is not None else None
    if mask is None or mask.size == 0:
        return np.zeros(shape, dtype=np.uint8)
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def measure_inpaint(
    original: np.ndarray,
    cleaned: np.ndarray,
    regions: Sequence[TextRegion],
    verifier: VisualInpaintVerifier,
) -> Dict[str, Any]:
    """Mide la limpieza de una página sin necesitar una plancha limpia de referencia.

    No hay ground truth de limpieza en ``dataset_eval``, así que se miden tres cosas
    comprobables sobre la propia página: cuánta tinta original sobrevive dentro de lo
    que se quería borrar, cuánto cambió la página fuera de esas máscaras, y qué dice el
    verificador visual que ya usa el pipeline.

    ``regions`` deben ser sólo las regiones que el pipeline decidió limpiar
    (``CleanManga.regiones_a_limpiar``). Medir sobre todas contaría como tinta residual
    el arte que se conserva a propósito, por ejemplo las onomatopeyas en modo ``keep``.
    """
    shape = original.shape[:2]
    before_luma = _luma(original)
    after_luma = _luma(cleaned)

    union = np.zeros(shape, dtype=np.uint8)
    ink_total = 0
    ink_residual = 0
    verifier_passed = 0
    verifier_scores: List[float] = []
    failed_checks: Dict[str, int] = {}
    without_mask: Dict[str, int] = {}

    for region in regions:
        mask = _region_clean_mask(region, shape)
        if cv2.countNonZero(mask) == 0:
            # Una región que el pipeline iba a limpiar y acabó sin máscara de tinta es
            # texto que nunca se borra. El origen de la máscara dice por qué.
            source = str((getattr(region, "metadata", None) or {}).get("clean_mask_source") or "sin_origen")
            key = f"{region.kind}:{source}"
            without_mask[key] = without_mask.get(key, 0) + 1
            continue
        union = cv2.bitwise_or(union, mask)

        selection = mask > 0
        background = float(np.median(after_luma[selection]))
        threshold = background - INK_LUMA_MARGIN
        ink_before = selection & (before_luma < threshold)
        count_before = int(np.count_nonzero(ink_before))
        if count_before:
            ink_total += count_before
            ink_residual += int(np.count_nonzero(ink_before & (after_luma < threshold)))

        report = verifier.evaluate(original, cleaned, mask, context_mask=region.mask)
        verifier_scores.append(float(report.score))
        if report.passed:
            verifier_passed += 1
        for name in report.failed_checks:
            failed_checks[name] = failed_checks.get(name, 0) + 1

    evaluated = len(verifier_scores)
    outside = cv2.bitwise_not(cv2.dilate(union, np.ones((9, 9), np.uint8), iterations=1))
    outside_pixels = int(cv2.countNonZero(outside))
    outside_damage = (
        float(np.mean(np.abs(before_luma[outside > 0] - after_luma[outside > 0]))) if outside_pixels else 0.0
    )

    return {
        "regions_evaluated": evaluated,
        "regions_without_clean_mask": sum(without_mask.values()),
        "regions_without_clean_mask_detail": without_mask,
        "clean_mask_pixels": int(cv2.countNonZero(union)),
        "residual_ink": round(ink_residual / ink_total, 6) if ink_total else None,
        "ink_pixels": ink_total,
        "outside_damage": round(outside_damage, 4),
        "verifier_pass_rate": round(verifier_passed / evaluated, 4) if evaluated else None,
        "verifier_score": round(float(np.mean(verifier_scores)), 4) if verifier_scores else None,
        "verifier_failed_checks": failed_checks,
    }


def _aggregate_inpaint(pages: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def weighted(metric: str, weight_key: str) -> Optional[float]:
        total_weight = 0
        total = 0.0
        for page in pages:
            value = page.get(metric)
            weight = page.get(weight_key) or 0
            if value is None or not weight:
                continue
            total += float(value) * float(weight)
            total_weight += float(weight)
        return round(total / total_weight, 6) if total_weight else None

    failed: Dict[str, int] = {}
    without_mask: Dict[str, int] = {}
    for page in pages:
        for name, count in (page.get("verifier_failed_checks") or {}).items():
            failed[name] = failed.get(name, 0) + int(count)
        for name, count in (page.get("regions_without_clean_mask_detail") or {}).items():
            without_mask[name] = without_mask.get(name, 0) + int(count)

    damages = [float(p["outside_damage"]) for p in pages if p.get("outside_damage") is not None]
    return {
        "pages": len(pages),
        "regions_evaluated": sum(int(p.get("regions_evaluated") or 0) for p in pages),
        "residual_ink": weighted("residual_ink", "ink_pixels"),
        "outside_damage": round(float(np.mean(damages)), 4) if damages else None,
        "verifier_pass_rate": weighted("verifier_pass_rate", "regions_evaluated"),
        "verifier_score": weighted("verifier_score", "regions_evaluated"),
        "verifier_failed_checks": dict(sorted(failed.items(), key=lambda kv: -kv[1])),
        "regions_without_clean_mask": sum(without_mask.values()),
        "regions_without_clean_mask_detail": dict(sorted(without_mask.items(), key=lambda kv: -kv[1])),
    }


# --------------------------------------------------------------------------------------
# Ejecución de un caso
# --------------------------------------------------------------------------------------


def _page_rows(translate_manga, cuadros: Sequence, textos: Sequence[str], regiones: Sequence[TextRegion]) -> List[Dict[str, Any]]:
    """Reproduce las filas que el pipeline empuja a ``Transcripción.json``.

    Mismo criterio que ``_push_original_texts_to_queue``: una región cuyo texto no pasa
    el filtro de idioma de origen no llega al JSON y, por tanto, no cuenta como
    predicción. Ese filtro es parte de la detección efectiva, no un detalle de formato.
    """
    textos_limpios = [translate_manga.normalizar_texto_ocr(texto) for texto in textos]
    flags = translate_manga._source_language_flags_for_texts(textos_limpios)
    estilos = translate_manga._clasificar_estilos_texto(textos_limpios)

    rows: List[Dict[str, Any]] = []
    for idx, ((x, y, w, h), texto) in enumerate(zip(cuadros, textos_limpios)):
        if idx < len(flags) and not flags[idx]:
            continue
        region = regiones[idx] if idx < len(regiones) else None
        row: Dict[str, Any] = {
            "Índice": idx,
            "Coordenadas": [[int(x), int(y)], [int(x + w), int(y + h)]],
            "Texto": texto,
            "Estilo": estilos[idx] if idx < len(estilos) else "dialogo",
        }
        if region is not None:
            tx, ty, tw, th = region.text_bbox
            row.update({
                "Tipo": region.kind,
                "Confianza": round(float(region.confidence), 4),
                "Coordenadas texto original": [[int(tx), int(ty)], [int(tx + tw), int(ty + th)]],
                "Fuente máscara": region.metadata.get("mask_source", ""),
            })
        rows.append(row)
    return rows


def _region_dump(
    regiones: Sequence[TextRegion],
    cuadros: Sequence,
    textos: Sequence[str],
) -> List[Dict[str, Any]]:
    """Metadata por región para analizar señales de detección fuera del pipeline.

    Interesa sobre todo poder cruzar lo que leyó el localizador (EasyOCR, en
    ``source_text_hint``) con lo que leyó el transcriptor (MangaOCR/Paddle): en texto real
    ambos coinciden, sobre arte de onomatopeyas producen basura distinta.
    """
    filas: List[Dict[str, Any]] = []
    for idx, region in enumerate(regiones):
        metadata = getattr(region, "metadata", {}) or {}
        filas.append({
            "index": idx,
            "kind": region.kind,
            "bbox": [int(v) for v in region.bbox],
            "text_bbox": [int(v) for v in region.text_bbox],
            "render_bbox": [int(v) for v in (cuadros[idx] if idx < len(cuadros) else region.bbox)],
            "confidence": round(float(region.confidence), 4),
            "detections_count": int(region.detections_count or 0),
            "hint_localizador": str(region.source_text_hint or ""),
            "texto_transcrito": str(textos[idx] if idx < len(textos) else ""),
            "mask_source": metadata.get("mask_source", ""),
            "free_text_confidence": metadata.get("free_text_confidence"),
            "free_text_filter_reason": metadata.get("free_text_filter_reason", ""),
            "free_text_onomatopoeia": bool(metadata.get("free_text_onomatopoeia")),
            "specialized_ocr_guard_text": metadata.get("specialized_ocr_guard_text", ""),
            "clean_mask_source": metadata.get("clean_mask_source", ""),
        })
    return filas


def run_case(case: EvalCase, settings: RunSettings, *, base_config_path: str = "config.yaml") -> Dict[str, Any]:
    """Ejecuta limpieza + OCR sobre las páginas del caso y escribe las predicciones."""
    pages_dir = case.pages_dir
    images = sorted(p for p in pages_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    if not images:
        raise FileNotFoundError(
            f"El caso {case.name} no tiene imágenes en {pages_dir}. Rematerialízalas con "
            f"'python -m parallel_manga_translator.quality.eval_dataset build'."
        )
    if settings.pages:
        images = images[: settings.pages]

    run_dir = settings.run_dir
    clean_dir = run_dir / CLEAN_DIRNAME
    clean_dir.mkdir(parents=True, exist_ok=True)

    config = build_case_config(
        case,
        run_dir,
        base_config_path=base_config_path,
        debug_artifacts=settings.debug_artifacts,
        overrides=settings.overrides,
    )
    from parallel_manga_translator.cli import build_image_processor, prepare_assets, prepare_runtime

    prepare_runtime()
    prepare_assets()
    processor = build_image_processor(config)
    clean_manga = processor.clean_manga
    translate_manga = processor.translate_manga
    # La misma composicion que ejecuta produccion, recortada donde acaba la medicion.
    # Antes esta secuencia estaba reescrita a mano aqui, y por tanto podia divergir del
    # pipeline real sin que ningun test lo notara: justo el banco de pruebas.
    pipeline = pipeline_limpieza_y_ocr(clean_manga, translate_manga)
    verifier = VisualInpaintVerifier(accept_score=float(config.quality.visual_inpaint_accept_score))

    transcription_pages: List[Dict[str, Any]] = []
    inpaint_pages: List[Dict[str, Any]] = []
    region_dumps: List[Dict[str, Any]] = []
    started = time.time()

    for index, image_path in enumerate(images):
        page_started = time.time()
        imagen = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if imagen is None:
            logger.warning("No se pudo leer %s", image_path)
            continue

        clean_manga.set_debug_page_context(index, source_filename=image_path.name, output_filename=image_path.name)
        # Mismo contrato que ImageProcessor: sin este contexto, visual_inpaint_debug no
        # escribe nada y `--debug-artifacts` se queda sin los informes por región.
        if settings.debug_artifacts:
            clean_manga.set_visual_inpaint_debug_context(
                output_root=config.processing.ruta_carpeta_salida,
                page_index=index,
                filename=image_path.name,
            )
        else:
            clean_manga.clear_visual_inpaint_debug_context()
        translate_manga.insertar_json_queue(index, None, None)
        try:
            contexto = pipeline.run(PageContext(imagen=imagen))
        finally:
            clean_manga.clear_debug_page_context()

        imagen_limpia = contexto.imagen_limpia
        regiones = contexto.regiones
        ordenadas = contexto.regiones_ordenadas
        cuadros, textos = contexto.cuadros, contexto.textos

        cv2.imencode(".jpg", imagen_limpia)[1].tofile(str(clean_dir / image_path.name))

        rows = _page_rows(translate_manga, cuadros, textos, ordenadas)
        transcription_pages.append({"Página": index + 1, "Globos de texto": rows})
        region_dumps.append({"page": index + 1, "regions": _region_dump(ordenadas, cuadros, textos)})

        globos_limpiados, libres_limpiados = clean_manga.regiones_a_limpiar(imagen, regiones)
        page_inpaint = measure_inpaint(imagen, imagen_limpia, globos_limpiados + libres_limpiados, verifier)
        page_inpaint["page"] = index + 1
        page_inpaint["seconds"] = round(time.time() - page_started, 2)
        inpaint_pages.append(page_inpaint)

        logger.info(
            "Página %s/%s: %s regiones detectadas, %s en JSON, %.1fs",
            index + 1,
            len(images),
            len(ordenadas),
            len(rows),
            time.time() - page_started,
        )

    (run_dir / TRANSCRIPTION_FILENAME).write_text(
        json.dumps({"Transcripción": transcription_pages}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / REGIONS_FILENAME).write_text(
        json.dumps({"paginas": region_dumps}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "case": case.name,
        "run_dir": str(run_dir),
        "pages": len(transcription_pages),
        "seconds": round(time.time() - started, 2),
        "options": _case_options(case),
        "overrides": dict(settings.overrides),
        "inpaint": {"summary": _aggregate_inpaint(inpaint_pages), "pages": inpaint_pages},
    }


def compare_inpaint(
    baseline: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, Any]:
    """Compara las métricas de limpieza contra una ejecución previa, si la hay."""
    rows: Dict[str, Any] = {}
    improved: List[str] = []
    regressed: List[str] = []
    for name in INPAINT_METRICS:
        base_value = (baseline or {}).get(name)
        current_value = current.get(name)
        direction = "higher" if name in INPAINT_HIGHER_IS_BETTER else "lower"
        if base_value is None or current_value is None:
            rows[name] = {"metric": name, "direction": direction, "baseline": base_value,
                          "current": current_value, "delta": None, "verdict": "sin datos"}
            continue
        # ``outside_damage`` vive en escala 0-255; la tolerancia de las tasas no aplica.
        metric_tolerance = tolerance * 255 if name == "outside_damage" else tolerance
        delta = round(float(current_value) - float(base_value), 6)
        if abs(delta) <= metric_tolerance:
            verdict = "sin cambio"
        elif (delta > 0) == (direction == "higher"):
            verdict = "mejora"
            improved.append(name)
        else:
            verdict = "regresión"
            regressed.append(name)
        rows[name] = {"metric": name, "direction": direction, "baseline": round(float(base_value), 6),
                      "current": round(float(current_value), 6), "delta": delta, "verdict": verdict}
    return {"tolerance": tolerance, "metrics": rows, "improved": improved, "regressed": regressed}


def _format_inpaint(comparison: Mapping[str, Any]) -> str:
    lines = [f"{'limpieza':<24}{'previo':>10}{'actual':>10}{'delta':>10}  veredicto"]
    for name, row in (comparison.get("metrics") or {}).items():
        base = row.get("baseline")
        current = row.get("current")
        delta = row.get("delta")
        lines.append(
            f"{name:<24}"
            f"{('-' if base is None else f'{float(base):.4f}'):>10}"
            f"{('-' if current is None else f'{float(current):.4f}'):>10}"
            f"{('-' if delta is None else f'{delta:+.4f}'):>10}  {row.get('verdict')}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_overrides(values: Sequence[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Override mal formado (usa clave=valor): {item}")
        key, raw = item.split("=", 1)
        raw = raw.strip()
        if raw.lower() in {"true", "false"}:
            value: Any = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        overrides[key.strip()] = value
    return overrides


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Ejecuta limpieza + OCR sobre un caso de dataset_eval y mide detección e inpainting."
    )
    parser.add_argument("--case", required=True, help="Nombre del caso, por ejemplo ja_01.")
    parser.add_argument("--dataset-dir", default="dataset_eval", help="Carpeta del banco de pruebas.")
    parser.add_argument("--label", default="actual", help="Nombre de la ejecución dentro de dataset_eval/ejecuciones.")
    parser.add_argument("--pages", type=int, default=0, help="Limita el número de páginas (0 = todas).")
    parser.add_argument("--config", default="config.yaml", help="Configuración base.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="Override de configuración: --set bubble_confidence=0.30 (quality por "
                             "defecto) o --set translation.modelo_inpaint=lama_mpe.")
    parser.add_argument("--debug-artifacts", action="store_true", help="Guarda debug de globos e inpaint.")
    parser.add_argument("--compare-with", default="", help="Etiqueta de otra ejecución para comparar la limpieza.")
    parser.add_argument("--fail-on-regression", action="store_true", help="Código de salida 1 si algo empeora.")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    case = load_case(dataset_dir / args.case)
    run_dir = dataset_dir / DEFAULT_RUNS_DIRNAME / args.label / case.name
    run_dir.mkdir(parents=True, exist_ok=True)

    settings = RunSettings(
        case_name=case.name,
        run_dir=run_dir,
        pages=args.pages or None,
        debug_artifacts=bool(args.debug_artifacts),
        overrides=_parse_overrides(args.overrides),
    )
    run_report = run_case(case, settings, base_config_path=args.config)

    evaluation_config = EvaluationConfig(iou_threshold=0.5)
    page_numbers = list(range(1, run_report["pages"] + 1)) if args.pages else None
    detection = score_case(case, predictions_dir=run_dir, config=evaluation_config, page_numbers=page_numbers)
    baseline = case.load_baseline() or {}
    comparison = compare_summaries(
        baseline.get("summary") or {}, detection.get("summary") or {}, DEFAULT_TOLERANCE, metrics=RUNNER_METRICS
    )
    comparison["case"] = case.name
    comparison["by_type"] = {
        tipo: compare_summaries(
            (baseline.get("by_type") or {}).get(tipo) or {},
            (detection.get("by_type") or {}).get(tipo) or {},
            DEFAULT_TOLERANCE,
            metrics=RUNNER_METRICS,
        )
        for tipo in sorted(set(baseline.get("by_type") or {}) | set(detection.get("by_type") or {}))
    }

    previous_inpaint: Optional[Dict[str, Any]] = None
    if args.compare_with:
        previous_path = dataset_dir / DEFAULT_RUNS_DIRNAME / args.compare_with / case.name / RUN_REPORT_FILENAME
        if previous_path.is_file():
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            previous_inpaint = ((previous.get("run") or {}).get("inpaint") or {}).get("summary")
        else:
            logger.warning("No existe la ejecución previa %s", previous_path)
    inpaint_comparison = compare_inpaint(previous_inpaint, run_report["inpaint"]["summary"])

    payload = {
        "run": run_report,
        "detection": detection,
        "comparison": comparison,
        "inpaint_comparison": inpaint_comparison,
    }
    (run_dir / RUN_REPORT_FILENAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(format_comparison(comparison))
    print()
    print(_format_inpaint(inpaint_comparison))
    summary = run_report["inpaint"]["summary"]
    print(
        f"\nregiones limpiadas={summary['regions_evaluated']} "
        f"sin_mascara={summary['regions_without_clean_mask']} "
        f"verificador_score={summary['verifier_score']} "
        f"fallos={summary['verifier_failed_checks']}"
    )
    if summary["regions_without_clean_mask_detail"]:
        print(f"sin máscara de tinta: {summary['regions_without_clean_mask_detail']}")
    print(f"ejecución: {run_dir}  ({run_report['seconds']}s)")

    regressed = bool(comparison.get("regressed")) or bool(inpaint_comparison.get("regressed"))
    return 1 if (args.fail_on_regression and regressed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
