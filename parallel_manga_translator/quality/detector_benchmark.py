"""Banco para comparar detectores entre sí, no contra el pipeline actual.

`eval_dataset` responde "¿mejora este cambio el pipeline?" y para eso está bien. No sirve
para "¿qué detector es mejor?", y el motivo está medido: su caja de referencia es
`ui_layout.original_region_bbox`, **la salida del detector actual corregida a mano**. En
las tres corridas de control, el detector actual empareja con IoU 1.00 exacto contra su
propio ground truth. Cualquier otro detector pierde por construcción.

Este módulo arregla las tres cosas que lo impedían:

1. **Objetivo neutral.** Se puntúa contra `bbox_texto` —dónde está el texto— además de
   contra la caja de globo. Medido sobre `en_02`: contra globo el detector actual gana
   70-39; contra texto pierde 23-59. La conclusión dependía por completo de la
   convención, así que se reportan las dos y nunca una sola.

2. **Subconjunto sin sesgo.** Las regiones con `origen: manual` las dibujó una persona
   sin ver la salida de ningún detector. Son 47 en `ja_01`, 8 en `en_01` y 13 en `en_02`,
   y son la única comparación limpia. El banco separa siempre "todas" de "manual".

3. **Solo detección.** No corre OCR, limpieza ni inpaint, así que el número no llega
   mezclado con el resto del pipeline y una comparación cuesta segundos, no 7 minutos.

Advertencia que va en el propio informe: esto mide **localización**, no el resultado
final de la página. Está documentado que un F1 de detector no predice el del pipeline
cuando hay otra fuente de regiones. Sirve para descartar rápido y para entender *por qué*
un detector falla; promocionar uno exige medirlo después con `eval_runner`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2

from parallel_manga_translator.infrastructure.logging_config import get_logger

logger = get_logger(__name__)

Box = Tuple[int, int, int, int]

#: Criterios de emparejamiento. Cada uno responde una pregunta distinta y por eso se
#: reportan los tres: quedarse con uno es lo que hacía incomparables a dos detectores.
CRITERIA = ("globo", "texto", "cobertura")

DEFAULT_IOU = 0.50
DEFAULT_COVERAGE = 0.80


# ----------------------------------------------------------------------------------
# Geometría
# ----------------------------------------------------------------------------------

def area(box: Sequence[int]) -> int:
    return max(0, int(box[2])) * max(0, int(box[3]))


def intersection(a: Sequence[int], b: Sequence[int]) -> int:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: Sequence[int], b: Sequence[int]) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def coverage(target: Sequence[int], candidate: Sequence[int]) -> float:
    """Qué fracción de `target` cubre `candidate`. No depende de la forma del candidato."""
    objetivo = area(target)
    return intersection(target, candidate) / objetivo if objetivo > 0 else 0.0


# ----------------------------------------------------------------------------------
# Modelo del banco
# ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class GtRegion:
    page: str
    bubble_box: Box
    text_box: Box
    kind: str
    manual: bool


@dataclass(frozen=True)
class DetectorPrediction:
    page: str
    boxes: List[Box] = field(default_factory=list)


@dataclass(frozen=True)
class CriterionScore:
    criterion: str
    subset: str
    gt: int
    predicted: int
    matched: int
    precision: float
    recall: float
    f1: float
    mean_score: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "subset": self.subset,
            "gt": self.gt,
            "predicted": self.predicted,
            "matched": self.matched,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_score": round(self.mean_score, 4),
        }


def _score_for(criterion: str, prediction: Box, region: GtRegion) -> float:
    if criterion == "globo":
        return iou(prediction, region.bubble_box)
    if criterion == "texto":
        return iou(prediction, region.text_box)
    if criterion == "cobertura":
        return coverage(region.text_box, prediction)
    raise ValueError(f"Criterio desconocido: {criterion}")


def _threshold_for(criterion: str, iou_threshold: float, coverage_threshold: float) -> float:
    return coverage_threshold if criterion == "cobertura" else iou_threshold


def match_greedy(
    predictions: Sequence[Box],
    regions: Sequence[GtRegion],
    criterion: str,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Empareja uno a uno, mejor puntuación primero.

    Greedy y no óptimo a propósito: es el mismo criterio que usa el evaluador del banco,
    así que los números de los dos sitios se pueden leer juntos.
    """
    candidatos: List[Tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, region in enumerate(regions):
            score = _score_for(criterion, prediction, region)
            if score >= threshold:
                candidatos.append((score, pred_index, gt_index))
    candidatos.sort(reverse=True)

    usados_pred: set[int] = set()
    usados_gt: set[int] = set()
    emparejados: List[Tuple[int, int, float]] = []
    for score, pred_index, gt_index in candidatos:
        if pred_index in usados_pred or gt_index in usados_gt:
            continue
        usados_pred.add(pred_index)
        usados_gt.add(gt_index)
        emparejados.append((pred_index, gt_index, score))
    return emparejados


def structural_diagnostics(
    predictions_by_page: Mapping[str, List[Box]],
    regions_by_page: Mapping[str, List[GtRegion]],
    *,
    merge_threshold: float = 0.5,
    split_threshold: float = 0.3,
) -> Dict[str, int]:
    """Fusiones y divisiones, que no dependen de la convención de caja.

    Una predicción que se come dos bloques de texto es un error aunque su IoU sea alto,
    y un bloque partido en dos predicciones también. Ninguna de las dos cosas se ve en
    precision/recall.
    """
    fusiones = 0
    divisiones = 0
    for page, regions in regions_by_page.items():
        boxes = predictions_by_page.get(page, [])
        for box in boxes:
            cubiertos = sum(1 for region in regions if coverage(region.text_box, box) >= merge_threshold)
            if cubiertos >= 2:
                fusiones += 1
        for region in regions:
            tocando = sum(1 for box in boxes if coverage(region.text_box, box) >= split_threshold)
            if tocando >= 2:
                divisiones += 1
    return {"fusiones": fusiones, "divisiones": divisiones}


# ----------------------------------------------------------------------------------
# Caso y carga
# ----------------------------------------------------------------------------------

class BenchmarkCase:
    """Un caso de `dataset_eval` visto como banco de detección."""

    def __init__(self, case_dir: Path) -> None:
        self.dir = Path(case_dir)
        meta_path = self.dir / "case.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"El caso no tiene case.json: {self.dir}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.name = str(self.meta.get("name") or self.dir.name)

    @property
    def source_language(self) -> str:
        return str((self.meta.get("options") or {}).get("source_language") or "Japonés")

    def pages(self, limit: int = 0) -> List[Tuple[str, Path]]:
        """(stem de la página, ruta de la imagen). Las imágenes no se versionan."""
        filas = self.meta.get("pages") or []
        salida: List[Tuple[str, Path]] = []
        for fila in filas:
            stem = str(fila.get("page") or "")
            imagen = self.dir / "paginas" / str(fila.get("image") or "")
            if stem and imagen.is_file():
                salida.append((stem, imagen))
        if limit > 0:
            salida = salida[:limit]
        return salida

    def ground_truth(self) -> Dict[str, List[GtRegion]]:
        regiones: Dict[str, List[GtRegion]] = {}
        for archivo in sorted((self.dir / "ground_truth").glob("*.json")):
            data = json.loads(archivo.read_text(encoding="utf-8"))
            page = str(data.get("page") or archivo.stem)
            filas: List[GtRegion] = []
            for region in data.get("regions") or []:
                bubble = region.get("bbox")
                text = region.get("bbox_texto") or bubble
                if not bubble or not text:
                    continue
                filas.append(
                    GtRegion(
                        page=page,
                        bubble_box=tuple(int(v) for v in bubble[:4]),  # type: ignore[arg-type]
                        text_box=tuple(int(v) for v in text[:4]),  # type: ignore[arg-type]
                        kind=str(region.get("tipo") or "unknown"),
                        manual=bool(region.get("manual")),
                    )
                )
            regiones[page] = filas
        return regiones


# ----------------------------------------------------------------------------------
# Ejecución de detectores
# ----------------------------------------------------------------------------------

def run_detector(case: BenchmarkCase, detector: str, *, limit: int = 0) -> Dict[str, List[Box]]:
    """Corre SOLO la detección de un detector registrado sobre las páginas del caso."""
    from dataclasses import replace

    from parallel_manga_translator.config.app_config import ProcessingConfig, QualityConfig
    from parallel_manga_translator.detection.region_source_factory import create_region_source

    quality = replace(QualityConfig(), region_source=detector)
    source = create_region_source(
        case.source_language,
        quality_config=quality,
        processing_config=ProcessingConfig(),
    )

    predicciones: Dict[str, List[Box]] = {}
    for stem, ruta in case.pages(limit=limit):
        imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        if imagen is None:
            logger.warning("No se pudo leer %s", ruta)
            predicciones[stem] = []
            continue
        regiones = source.detect_primary_bubble_regions(imagen)
        predicciones[stem] = [tuple(int(v) for v in region.bbox[:4]) for region in regiones]  # type: ignore[misc]
    return predicciones


# ----------------------------------------------------------------------------------
# Puntuación
# ----------------------------------------------------------------------------------

def score_detector(
    predictions_by_page: Mapping[str, List[Box]],
    regions_by_page: Mapping[str, List[GtRegion]],
    *,
    iou_threshold: float = DEFAULT_IOU,
    coverage_threshold: float = DEFAULT_COVERAGE,
) -> Dict[str, Any]:
    subsets = {
        "todas": lambda region: True,
        # Las dibujadas a mano son la unica comparacion limpia entre detectores.
        "manual": lambda region: region.manual,
    }
    resultados: Dict[str, Any] = {"criterios": {}, "estructura": structural_diagnostics(predictions_by_page, regions_by_page)}

    total_predichas = sum(len(v) for v in predictions_by_page.values())
    for criterion in CRITERIA:
        threshold = _threshold_for(criterion, iou_threshold, coverage_threshold)
        for subset, keep in subsets.items():
            gt_total = 0
            matched = 0
            scores: List[float] = []
            for page, regions in regions_by_page.items():
                filtradas = [r for r in regions if keep(r)]
                gt_total += len(filtradas)
                if not filtradas:
                    continue
                boxes = predictions_by_page.get(page, [])
                for _pred, _gt, score in match_greedy(boxes, filtradas, criterion, threshold):
                    matched += 1
                    scores.append(score)
            # La precisión del subconjunto manual no es interpretable —las predicciones
            # sobre regiones no manuales no son falsos positivos— así que solo se
            # reporta recall ahí. En "todas" sí se reportan las tres.
            precision = matched / total_predichas if total_predichas and subset == "todas" else 0.0
            recall = matched / gt_total if gt_total else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            resultados["criterios"][f"{criterion}/{subset}"] = CriterionScore(
                criterion=criterion,
                subset=subset,
                gt=gt_total,
                predicted=total_predichas,
                matched=matched,
                precision=precision,
                recall=recall,
                f1=f1,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
            ).as_dict()
    return resultados


def benchmark_case(
    case: BenchmarkCase,
    detectors: Sequence[str],
    *,
    limit: int = 0,
    iou_threshold: float = DEFAULT_IOU,
    coverage_threshold: float = DEFAULT_COVERAGE,
) -> Dict[str, Any]:
    regiones = case.ground_truth()
    if limit > 0:
        paginas = {stem for stem, _ in case.pages(limit=limit)}
        regiones = {k: v for k, v in regiones.items() if k in paginas}

    informe: Dict[str, Any] = {
        "case": case.name,
        "source_language": case.source_language,
        "pages": len(regiones),
        "gt_regions": sum(len(v) for v in regiones.values()),
        "gt_manual_regions": sum(1 for v in regiones.values() for r in v if r.manual),
        "iou_threshold": iou_threshold,
        "coverage_threshold": coverage_threshold,
        "generated_at": time.time(),
        "aviso": (
            "Mide localizacion, no resultado final de pagina. El subconjunto 'manual' es el "
            "unico no derivado del detector actual; 'todas' favorece al detector que genero "
            "el ground truth."
        ),
        "detectores": {},
    }
    for detector in detectors:
        inicio = time.time()
        predicciones = run_detector(case, detector, limit=limit)
        informe["detectores"][detector] = {
            "predicted_regions": sum(len(v) for v in predicciones.values()),
            "seconds": round(time.time() - inicio, 2),
            **score_detector(
                predicciones,
                regiones,
                iou_threshold=iou_threshold,
                coverage_threshold=coverage_threshold,
            ),
        }
    return informe


# ----------------------------------------------------------------------------------
# Presentación
# ----------------------------------------------------------------------------------

def format_report(informe: Mapping[str, Any]) -> str:
    lineas: List[str] = []
    detectores = list(informe.get("detectores") or {})
    lineas.append(
        f"Caso {informe['case']} ({informe['pages']} paginas, {informe['gt_regions']} regiones GT, "
        f"{informe['gt_manual_regions']} dibujadas a mano)"
    )
    if not detectores:
        return "\n".join(lineas)

    ancho = max(12, max(len(d) for d in detectores) + 2)
    cabecera = "criterio / subconjunto".ljust(26) + "".join(d.ljust(ancho) for d in detectores)
    lineas.append(cabecera)
    lineas.append("-" * len(cabecera))

    for criterion in CRITERIA:
        for subset in ("todas", "manual"):
            clave = f"{criterion}/{subset}"
            fila = f"{criterion}/{subset}".ljust(26)
            for detector in detectores:
                datos = informe["detectores"][detector]["criterios"].get(clave, {})
                if subset == "manual":
                    fila += f"R={datos.get('recall', 0):.3f}".ljust(ancho)
                else:
                    fila += f"F1={datos.get('f1', 0):.3f}".ljust(ancho)
            lineas.append(fila)

    lineas.append("-" * len(cabecera))
    fila = "predichas".ljust(26)
    for detector in detectores:
        fila += str(informe["detectores"][detector]["predicted_regions"]).ljust(ancho)
    lineas.append(fila)
    for clave, etiqueta in (("fusiones", "fusiones (2+ bloques)"), ("divisiones", "divisiones (bloque en 2+)")):
        fila = etiqueta.ljust(26)
        for detector in detectores:
            fila += str(informe["detectores"][detector]["estructura"].get(clave, 0)).ljust(ancho)
        lineas.append(fila)
    fila = "segundos".ljust(26)
    for detector in detectores:
        fila += str(informe["detectores"][detector]["seconds"]).ljust(ancho)
    lineas.append(fila)
    lineas.append("")
    lineas.append(f"AVISO: {informe['aviso']}")
    return "\n".join(lineas)


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    from parallel_manga_translator.quality.eval_dataset import resolve_dataset_dir

    parser = argparse.ArgumentParser(description="Compara detectores de regiones entre si sobre dataset_eval.")
    parser.add_argument("--case", default="", help="Caso; si se omite, todos los del banco.")
    parser.add_argument("--dataset-dir", default="", help="Carpeta del banco (por defecto, la de config.yaml).")
    parser.add_argument("--detectors", default="yolo,comic_text_detector", help="Lista separada por comas.")
    parser.add_argument("--pages", type=int, default=0, help="Limita el numero de paginas (0 = todas).")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument("--output", default="", help="Carpeta donde escribir el informe JSON.")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else resolve_dataset_dir()
    detectores = [d.strip() for d in str(args.detectors).split(",") if d.strip()]
    if args.case:
        casos = [BenchmarkCase(dataset_dir / args.case)]
    else:
        casos = [BenchmarkCase(p.parent) for p in sorted(dataset_dir.glob("*/case.json"))]
    if not casos:
        print(f"No hay casos en {dataset_dir}")
        return 2

    for caso in casos:
        informe = benchmark_case(
            caso,
            detectores,
            limit=args.pages,
            iou_threshold=args.iou,
            coverage_threshold=args.coverage,
        )
        print(format_report(informe))
        print()
        if args.output:
            destino = Path(args.output)
            destino.mkdir(parents=True, exist_ok=True)
            ruta = destino / f"detectores_{caso.name}.json"
            ruta.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"informe: {ruta}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
