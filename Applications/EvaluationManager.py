from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Box = Tuple[int, int, int, int]


def normalize_text_for_eval(text: Any) -> str:
    text = str(text or "")
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def char_error_rate(reference: Any, hypothesis: Any) -> float:
    ref = normalize_text_for_eval(reference)
    hyp = normalize_text_for_eval(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    return levenshtein_distance(ref, hyp) / max(1, len(ref))


def _sequence_levenshtein(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if list(reference) == list(hypothesis):
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, 1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ref_item == hyp_item else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def word_error_rate(reference: Any, hypothesis: Any) -> float:
    ref_words = normalize_text_for_eval(reference).split()
    hyp_words = normalize_text_for_eval(hypothesis).split()
    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0
    return _sequence_levenshtein(ref_words, hyp_words) / max(1, len(ref_words))


def box_area(box: Box) -> int:
    return max(0, int(box[2])) * max(0, int(box[3]))


def box_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _box_from_value(value: Any) -> Optional[Box]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        x, y, w, h = value
        return int(round(x)), int(round(y)), int(round(w)), int(round(h))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            (x1, y1), (x2, y2) = value
            return int(round(x1)), int(round(y1)), max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))
        except Exception:
            return None
    return None


def region_from_json(row: Mapping[str, Any], page: Optional[str] = None, index: int = 0) -> Dict[str, Any]:
    box = (
        _box_from_value(row.get("bbox"))
        or _box_from_value(row.get("box"))
        or _box_from_value(row.get("coordenadas"))
        or _box_from_value(row.get("Coordenadas"))
    )
    if box is None:
        raise ValueError(f"Región sin bbox/coordenadas válidas en página {page or '?'} índice {index}")
    source_text = (
        row.get("text_ja")
        or row.get("texto_original")
        or row.get("Texto original")
        or row.get("source_text")
        or row.get("Texto")
        or ""
    )
    translated_text = (
        row.get("translation_es")
        or row.get("texto_traducido")
        or row.get("Texto traducido")
        or row.get("translation")
        or row.get("Texto")
        or ""
    )
    return {
        "page": page,
        "index": index,
        "bbox": box,
        "type": row.get("type") or row.get("tipo") or row.get("Tipo") or "dialogue",
        "source_text": str(source_text or ""),
        "translated_text": str(translated_text or ""),
        "raw": dict(row),
    }


def load_ground_truth_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        page_name = path.stem
        rows = data
    elif isinstance(data, Mapping):
        page_name = str(data.get("page") or data.get("filename") or path.stem)
        rows = data.get("regions") or data.get("Globos de texto") or data.get("items") or []
    else:
        raise TypeError(f"Ground truth inválido: {path}")
    if not isinstance(rows, list):
        raise TypeError(f"La lista de regiones no es válida: {path}")
    return {
        "page": page_name,
        "regions": [region_from_json(row, page_name, idx) for idx, row in enumerate(rows) if isinstance(row, Mapping)],
    }


def _page_number_from_name(name: str) -> Optional[int]:
    match = re.search(r"(\d+)", str(name or ""))
    return int(match.group(1)) if match else None


def _load_pipeline_pages(path: Path, root_key: str) -> Dict[int, List[Dict[str, Any]]]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get(root_key, []) if isinstance(data, Mapping) else []
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for page_row in rows if isinstance(rows, list) else []:
        if not isinstance(page_row, Mapping):
            continue
        page_number = page_row.get("Página") or page_row.get("page") or page_row.get("pagina")
        try:
            page_number = int(page_number)
        except Exception:
            continue
        regions = page_row.get("Globos de texto") or page_row.get("regions") or []
        pages[page_number] = [region_from_json(row, str(page_number), idx) for idx, row in enumerate(regions) if isinstance(row, Mapping)]
    return pages


def load_pipeline_predictions(transcription_json: Optional[Path], translation_json: Optional[Path]) -> Dict[int, List[Dict[str, Any]]]:
    trans_pages = _load_pipeline_pages(transcription_json, "Transcripción") if transcription_json else {}
    trad_pages = _load_pipeline_pages(translation_json, "Traducción") if translation_json else {}
    all_pages = sorted(set(trans_pages) | set(trad_pages))
    result: Dict[int, List[Dict[str, Any]]] = {}
    for page in all_pages:
        source_regions = trans_pages.get(page, [])
        translated_regions = trad_pages.get(page, [])
        max_len = max(len(source_regions), len(translated_regions))
        merged: List[Dict[str, Any]] = []
        for idx in range(max_len):
            src = source_regions[idx] if idx < len(source_regions) else {}
            trg = translated_regions[idx] if idx < len(translated_regions) else {}
            box = src.get("bbox") or trg.get("bbox")
            merged.append({
                "page": str(page),
                "index": idx,
                "bbox": box,
                "type": src.get("type") or trg.get("type") or "dialogue",
                "source_text": src.get("source_text", ""),
                "translated_text": trg.get("source_text") or trg.get("translated_text") or "",
                "raw": {"transcription": src.get("raw", {}), "translation": trg.get("raw", {})},
            })
        result[page] = merged
    return result


def greedy_match_regions(gt_regions: Sequence[Mapping[str, Any]], pred_regions: Sequence[Mapping[str, Any]], iou_threshold: float = 0.5):
    candidates = []
    for gi, gt in enumerate(gt_regions):
        for pi, pred in enumerate(pred_regions):
            iou = box_iou(gt["bbox"], pred["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    matches = []
    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append({"gt_index": gi, "pred_index": pi, "iou": round(float(iou), 4)})
    missing = [i for i in range(len(gt_regions)) if i not in used_gt]
    extra = [i for i in range(len(pred_regions)) if i not in used_pred]
    return matches, missing, extra


@dataclass
class EvaluationConfig:
    iou_threshold: float = 0.5
    pass_cer_threshold: float = 0.08
    pass_translation_cer_threshold: float = 0.35


@dataclass
class PageEvaluation:
    page: str
    page_number: Optional[int]
    gt_regions: int
    predicted_regions: int
    matched_regions: int
    missing_regions: int
    extra_regions: int
    detection_precision: float
    detection_recall: float
    detection_f1: float
    mean_iou: float
    mean_ocr_cer: Optional[float]
    mean_translation_cer: Optional[float]
    matches: List[Dict[str, Any]] = field(default_factory=list)
    missing_gt_indices: List[int] = field(default_factory=list)
    extra_pred_indices: List[int] = field(default_factory=list)


class EvaluationManager:
    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()

    @staticmethod
    def _mean(values: Iterable[float]) -> Optional[float]:
        values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        return round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0

    def evaluate_page(self, gt_page: Mapping[str, Any], pred_regions: Sequence[Mapping[str, Any]]) -> PageEvaluation:
        gt_regions = list(gt_page.get("regions") or [])
        matches, missing, extra = greedy_match_regions(gt_regions, pred_regions, self.config.iou_threshold)
        detailed_matches: List[Dict[str, Any]] = []
        ocr_cers: List[float] = []
        translation_cers: List[float] = []
        for match in matches:
            gt = gt_regions[match["gt_index"]]
            pred = pred_regions[match["pred_index"]]
            ocr_cer = char_error_rate(gt.get("source_text", ""), pred.get("source_text", "")) if gt.get("source_text") else None
            translation_cer = char_error_rate(gt.get("translated_text", ""), pred.get("translated_text", "")) if gt.get("translated_text") else None
            if ocr_cer is not None:
                ocr_cers.append(ocr_cer)
            if translation_cer is not None:
                translation_cers.append(translation_cer)
            detailed = dict(match)
            detailed.update({
                "gt_bbox": gt.get("bbox"),
                "pred_bbox": pred.get("bbox"),
                "gt_source_text": gt.get("source_text", ""),
                "pred_source_text": pred.get("source_text", ""),
                "gt_translation": gt.get("translated_text", ""),
                "pred_translation": pred.get("translated_text", ""),
                "ocr_cer": round(float(ocr_cer), 4) if ocr_cer is not None else None,
                "translation_cer": round(float(translation_cer), 4) if translation_cer is not None else None,
            })
            detailed_matches.append(detailed)

        precision = round(len(matches) / max(1, len(pred_regions)), 4)
        recall = round(len(matches) / max(1, len(gt_regions)), 4)
        return PageEvaluation(
            page=str(gt_page.get("page") or ""),
            page_number=_page_number_from_name(str(gt_page.get("page") or "")),
            gt_regions=len(gt_regions),
            predicted_regions=len(pred_regions),
            matched_regions=len(matches),
            missing_regions=len(missing),
            extra_regions=len(extra),
            detection_precision=precision,
            detection_recall=recall,
            detection_f1=self._f1(precision, recall),
            mean_iou=self._mean([m["iou"] for m in matches]) or 0.0,
            mean_ocr_cer=self._mean(ocr_cers),
            mean_translation_cer=self._mean(translation_cers),
            matches=detailed_matches,
            missing_gt_indices=missing,
            extra_pred_indices=extra,
        )

    def evaluate_dataset(
        self,
        ground_truth_dir: Path,
        transcription_json: Optional[Path] = None,
        translation_json: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        gt_files = sorted([p for p in ground_truth_dir.glob("*.json") if p.is_file()])
        if not gt_files:
            raise FileNotFoundError(f"No hay JSON de ground truth en {ground_truth_dir}")
        predictions = load_pipeline_predictions(transcription_json, translation_json)
        pages: List[PageEvaluation] = []
        for gt_file in gt_files:
            gt_page = load_ground_truth_file(gt_file)
            page_number = _page_number_from_name(str(gt_page.get("page") or gt_file.stem))
            pred_regions = predictions.get(page_number or -1, [])
            pages.append(self.evaluate_page(gt_page, pred_regions))
        report = self.aggregate_report(pages)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def aggregate_report(self, pages: Sequence[PageEvaluation]) -> Dict[str, Any]:
        total_gt = sum(p.gt_regions for p in pages)
        total_pred = sum(p.predicted_regions for p in pages)
        total_matches = sum(p.matched_regions for p in pages)
        precision = round(total_matches / max(1, total_pred), 4)
        recall = round(total_matches / max(1, total_gt), 4)
        ocr_cers = [p.mean_ocr_cer for p in pages if p.mean_ocr_cer is not None]
        translation_cers = [p.mean_translation_cer for p in pages if p.mean_translation_cer is not None]
        summary = {
            "pages": len(pages),
            "gt_regions": total_gt,
            "predicted_regions": total_pred,
            "matched_regions": total_matches,
            "missing_regions": sum(p.missing_regions for p in pages),
            "extra_regions": sum(p.extra_regions for p in pages),
            "detection_precision": precision,
            "detection_recall": recall,
            "detection_f1": self._f1(precision, recall),
            "mean_iou": self._mean([p.mean_iou for p in pages]) or 0.0,
            "mean_ocr_cer": self._mean([v for v in ocr_cers if v is not None]),
            "mean_translation_cer": self._mean([v for v in translation_cers if v is not None]),
            "iou_threshold": self.config.iou_threshold,
            "pass_ocr_cer_threshold": self.config.pass_cer_threshold,
            "pass_translation_cer_threshold": self.config.pass_translation_cer_threshold,
        }
        summary["passes_ocr_threshold"] = (
            summary["mean_ocr_cer"] is not None and summary["mean_ocr_cer"] <= self.config.pass_cer_threshold
        )
        summary["passes_translation_threshold"] = (
            summary["mean_translation_cer"] is not None and summary["mean_translation_cer"] <= self.config.pass_translation_cer_threshold
        )
        return {"summary": summary, "pages": [asdict(p) for p in pages]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evalúa precisión de detección, OCR y traducción contra ground truth manual.")
    parser.add_argument("--ground-truth", required=True, help="Carpeta con JSON anotados manualmente.")
    parser.add_argument("--transcription-json", default="", help="Ruta a Limpieza/Transcripción.json")
    parser.add_argument("--translation-json", default="", help="Ruta a Traduccion/Traducción.json")
    parser.add_argument("--output", default="", help="Ruta del reporte de evaluación JSON.")
    parser.add_argument("--iou", type=float, default=0.5, help="Umbral IoU para considerar una región detectada correcta.")
    args = parser.parse_args(argv)

    manager = EvaluationManager(EvaluationConfig(iou_threshold=args.iou))
    report = manager.evaluate_dataset(
        ground_truth_dir=Path(args.ground_truth),
        transcription_json=Path(args.transcription_json) if args.transcription_json else None,
        translation_json=Path(args.translation_json) if args.translation_json else None,
        output_path=Path(args.output) if args.output else None,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
