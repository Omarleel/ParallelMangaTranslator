"""Empareja lo que vio el OCR con lo que el humano escribió, por identidad.

Hasta ahora medir la transcripción exigía reconstruir la correspondencia entre el recorte
que el pipeline pasó al OCR y la región que el humano corrigió. Se intentó por posición en
la lista (el editor la reordena), por índice filtrando borradas (el editor también borra) y
por solapamiento de cajas (el editor las encoge al área de texto). Las tres fallan, y fallan
en silencio: producen una media de CER perfectamente creíble sobre parejas equivocadas.

Desde que la corrección guarda `region_uid`, no hay nada que reconstruir. Este módulo solo
junta las dos mitades por ese token y **dice en voz alta** lo que no ha podido emparejar,
que es la parte que las medidas anteriores se callaban.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

MASKED_SUFFIX = "_enmascarado.png"
PREPARED_SUFFIX = "_preparado.png"


@dataclass(frozen=True)
class CropRecord:
    """Un recorte volcado por `quality.ocr_crop_debug_dir`."""

    region_uid: str
    indice: int
    bbox: List[int]
    text_bbox: List[int]
    kind: str
    masked_path: Optional[Path]
    prepared_path: Optional[Path]


@dataclass(frozen=True)
class CropPair:
    """Un recorte y la transcripción que el humano dejó para esa misma región."""

    region_uid: str
    reference_text: str
    masked_path: Optional[Path]
    prepared_path: Optional[Path]
    crop_bbox: List[int]
    corrected_bbox: Optional[List[int]]
    page: str


@dataclass
class PairingReport:
    pairs: List[CropPair] = field(default_factory=list)
    #: Regiones corregidas cuya identidad no existe en el volcado: el volcado es de otra
    #: ejecución, o la región la dibujó el humano y no la vio ningún OCR.
    sin_recorte: List[str] = field(default_factory=list)
    #: Recortes que el humano no dejó corregidos: los borró, o los dejó vacíos.
    sin_correccion: List[str] = field(default_factory=list)
    #: Regiones de casos creados antes de que la corrección guardara su identidad.
    sin_identidad: int = 0

    @property
    def emparejadas(self) -> int:
        return len(self.pairs)

    def to_dict(self) -> Dict[str, Any]:
        parejas = []
        for pair in self.pairs:
            fila = {k: v for k, v in asdict(pair).items() if k not in ("masked_path", "prepared_path")}
            fila["masked_path"] = str(pair.masked_path) if pair.masked_path else ""
            fila["prepared_path"] = str(pair.prepared_path) if pair.prepared_path else ""
            parejas.append(fila)
        return {
            "emparejadas": self.emparejadas,
            "sin_recorte": self.sin_recorte,
            "sin_correccion": self.sin_correccion,
            "sin_identidad": self.sin_identidad,
            "parejas": parejas,
        }


def _int_box(value: Any) -> List[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    try:
        return [int(round(float(v))) for v in list(value)[:4]]
    except (TypeError, ValueError):
        return []


def index_ocr_crops(dump_dir: Path) -> Dict[str, CropRecord]:
    """Indexa por `region_uid` los recortes volcados bajo `dump_dir`.

    Un volcado sin `region_uid` es de una versión anterior y se ignora a propósito: sin
    identidad no hay emparejamiento fiable, y fingir uno es justo el error que este módulo
    existe para evitar.
    """
    dump_dir = Path(dump_dir)
    crops: Dict[str, CropRecord] = {}
    for meta_path in sorted(dump_dir.rglob("region_*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, Mapping):
            continue
        region_uid = str(meta.get("region_uid") or "")
        if not region_uid:
            continue
        stem = meta_path.stem
        masked = meta_path.with_name(stem + MASKED_SUFFIX)
        prepared = meta_path.with_name(stem + PREPARED_SUFFIX)
        crops[region_uid] = CropRecord(
            region_uid=region_uid,
            indice=int(meta.get("indice", -1) or -1),
            bbox=_int_box(meta.get("bbox")),
            text_bbox=_int_box(meta.get("text_bbox")),
            kind=str(meta.get("kind") or ""),
            masked_path=masked if masked.is_file() else None,
            prepared_path=prepared if prepared.is_file() else None,
        )
    return crops


def pair_ground_truth_with_crops(
    pages: Iterable[Mapping[str, Any]],
    crops: Mapping[str, CropRecord],
) -> PairingReport:
    """Cruza las páginas de verdad de referencia con los recortes, solo por identidad."""
    report = PairingReport()
    vistos: Set[str] = set()
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        nombre = str(page.get("page") or "")
        for region in page.get("regions") or []:
            if not isinstance(region, Mapping):
                continue
            region_uid = str(region.get("region_uid") or "")
            if not region_uid:
                report.sin_identidad += 1
                continue
            crop = crops.get(region_uid)
            if crop is None:
                report.sin_recorte.append(region_uid)
                continue
            vistos.add(region_uid)
            report.pairs.append(
                CropPair(
                    region_uid=region_uid,
                    reference_text=str(region.get("texto_original") or ""),
                    masked_path=crop.masked_path,
                    prepared_path=crop.prepared_path,
                    crop_bbox=list(crop.bbox),
                    corrected_bbox=_int_box(region.get("bbox_texto")) or None,
                    page=nombre,
                )
            )
    report.sin_correccion = sorted(set(crops) - vistos)
    return report


def load_case_pages(case_dir: Path) -> List[Dict[str, Any]]:
    """Lee las páginas de verdad de referencia de un caso de `dataset_eval`."""
    from parallel_manga_translator.quality.eval_dataset import load_case

    case = load_case(Path(case_dir))
    pages: List[Dict[str, Any]] = []
    for path in case.ground_truth_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, Mapping):
            pages.append(dict(data))
    return pages


def format_report(report: PairingReport) -> str:
    con_texto = sum(1 for pair in report.pairs if pair.reference_text.strip())
    lineas = [
        f"parejas por identidad          : {report.emparejadas}",
        f"  de ellas con texto corregido : {con_texto}",
        f"corregidas sin recorte         : {len(report.sin_recorte)}",
        f"recortes sin corregir          : {len(report.sin_correccion)}",
        f"sin identidad (caso antiguo)   : {report.sin_identidad}",
    ]
    if report.sin_identidad:
        lineas.append(
            "  aviso: ese caso se exportó antes de que la corrección guardara su identidad."
            " Vuelve a exportarlo desde la UI para poder medir transcripción."
        )
    return "\n".join(lineas)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Empareja recortes de OCR con la corrección humana.")
    parser.add_argument("--case", required=True, help="Caso de dataset_eval con la verdad de referencia.")
    parser.add_argument("--crops", required=True, help="Directorio de quality.ocr_crop_debug_dir.")
    parser.add_argument("--output", default="", help="Escribe el emparejamiento como JSON.")
    args = parser.parse_args(argv)

    report = pair_ground_truth_with_crops(load_case_pages(Path(args.case)), index_ocr_crops(Path(args.crops)))
    print(format_report(report))
    if args.output:
        destino = Path(args.output)
        if destino.parent and str(destino.parent):
            destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"emparejamiento escrito en {destino}")
    return 0


__all__ = [
    "CropPair",
    "CropRecord",
    "PairingReport",
    "format_report",
    "index_ocr_crops",
    "load_case_pages",
    "main",
    "pair_ground_truth_with_crops",
]


if __name__ == "__main__":
    raise SystemExit(main())
