"""Convierte trabajos corregidos en la UI en un dataset YOLO-seg de globos.

Es la pieza que faltaba para poder afinar el detector de globos con material propio. Todo
lo demás ya existe: la UI deja las cajas buenas en ``manifest.json`` cada vez que alguien
corrige un manga, y ``dataset_eval`` sabe puntuar el resultado.

Por qué hace falta derivar polígonos
------------------------------------
El pipeline depende de la **máscara** del globo en tres sitios —recorte de OCR, zona segura
y máscara de tinta—, así que un modelo de cajas no sirve: medido, sube el CER de
transcripción de 0.0015 a 0.2584 porque el recorte de OCR se llena de arte vecino. Pero el
manifest sólo guarda cajas.

Aquí el polígono se deriva de la imagen **dentro de una caja que ya validó una persona**.
Eso es generación de etiquetas, no detección: no se inventan globos, se refina la forma de
uno confirmado. Cuando el contorno no sale limpio, la región se descarta en vez de
etiquetarse mal, porque una etiqueta dudosa envenena el entrenamiento más de lo que aporta.

Honestidad sobre el volumen
---------------------------
Con dos trabajos corregidos hay 22 páginas, y son exactamente las de ``dataset_eval``.
Entrenar con ellas sería entrenar sobre el conjunto de test. Por eso los trabajos que
originan un caso del banco se excluyen por defecto: el exportador está pensado para que el
material se acumule según corriges manga, no para exprimir lo que ya hay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.quality.eval_dataset import CASE_FILENAME, detection_bbox
from parallel_manga_translator.quality.text_block_polygon import text_block_polygon

logger = get_logger(__name__)

#: Clase única del dataset. El detector sólo tiene que distinguir globo de no-globo.
CLASS_NAMES = ("bubble",)

#: Tipos de región que son globos. El texto libre y las onomatopeyas no lo son.
BUBBLE_TYPES = frozenset({"dialogue", "narration", "unknown", "manual"})

#: Margen alrededor de la caja humana donde buscar el contorno, como fracción del lado.
SEARCH_MARGIN = 0.12
#: Tolerancia del relleno por inundación al recorrer el interior del globo.
FLOOD_TOLERANCE = 60
#: El contorno debe ocupar entre estas fracciones del área de la caja para ser creíble.
MIN_AREA_RATIO = 0.25
MAX_AREA_RATIO = 1.60

#: A partir de aquí el relleno se escapó del globo y el "contorno" es el recorte entero.
#:
#: ``bubble_polygon`` busca con ``SEARCH_MARGIN`` por lado, así que un recorte totalmente
#: inundado mide siempre 1.24² = 1.5376 veces la caja. Contar vértices **no** detecta esto:
#: medido sobre las 22 páginas del banco, 41 de esas etiquetas tienen ≤4 vértices pero
#: otras 55 tienen más, por ruido en el borde, y son igual de rectangulares. El área sí lo
#: ve. Es la métrica de salud honesta del exportador.
FLOOD_AREA_RATIO = 1.40


@dataclass
class ExportStats:
    """Resumen de lo exportado, para poder juzgar la calidad de las etiquetas."""

    pages: int = 0
    regions: int = 0
    labelled: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)
    #: Etiquetas que siguen siendo el recorte entero: ni el globo ni la tinta dieron forma.
    #: Es la métrica de salud del dataset, medida por área y no por número de vértices.
    rectangular: int = 0
    #: Etiquetas rescatadas por el contorno de la tinta cuando el del globo se desbordó.
    from_ink: int = 0

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


def bubble_polygon(image: np.ndarray, box: Sequence[int]) -> Tuple[Optional[np.ndarray], str]:
    """Contorno del globo dentro de una caja validada por una persona.

    Devuelve ``(None, motivo)`` cuando el resultado no parece un globo, para descartar la
    región en lugar de etiquetarla mal.
    """
    height, width = image.shape[:2]
    x, y, w, h = (int(v) for v in box)
    if w <= 2 or h <= 2:
        return None, "caja degenerada"

    pad_x, pad_y = int(w * SEARCH_MARGIN), int(h * SEARCH_MARGIN)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None, "recorte vacio"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cx, cy = (x + w // 2) - x0, (y + h // 2) - y0
    if not (0 <= cx < gray.shape[1] and 0 <= cy < gray.shape[0]):
        return None, "centro fuera del recorte"

    # El interior del globo es la zona uniforme que contiene el centro de la caja.
    flood = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)
    cv2.floodFill(
        gray.copy(), flood, (cx, cy), 255,
        loDiff=FLOOD_TOLERANCE, upDiff=FLOOD_TOLERANCE,
        flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8),
    )
    interior = cv2.morphologyEx(flood[1:-1, 1:-1] * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "sin contorno"
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / max(1, w * h)
    if area_ratio < MIN_AREA_RATIO:
        return None, "contorno demasiado pequeno"
    if area_ratio > MAX_AREA_RATIO:
        return None, "contorno desbordado"

    approx = cv2.approxPolyDP(largest, 0.006 * cv2.arcLength(largest, True), True)
    points = approx.reshape(-1, 2) + np.array([x0, y0])
    if len(points) < 3:
        return None, "menos de tres vertices"
    return points.astype(np.int32), "ok"


def _area_ratio(points: np.ndarray, box: Sequence[int]) -> float:
    """Área del polígono relativa a la de su caja. Ver ``FLOOD_AREA_RATIO``."""
    if points is None or len(points) < 3:
        return 0.0
    contorno = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.contourArea(contorno) / max(1.0, float(box[2]) * float(box[3]))


def _is_bubble(region: Mapping[str, Any]) -> bool:
    tipo = str(region.get("type") or "").strip().lower()
    if tipo in BUBBLE_TYPES:
        return True
    # Las regiones dibujadas a mano llegan con type "manual"; su estilo dice qué son.
    if region.get("manual"):
        return str(region.get("style") or "").strip().lower() != "onomatopeya"
    return False


def _polygon_line(points: np.ndarray, width: int, height: int) -> str:
    """Una línea de etiqueta YOLO-seg: clase y vértices normalizados."""
    coords: List[str] = []
    for px, py in points:
        coords.append(f"{min(max(px / width, 0.0), 1.0):.6f}")
        coords.append(f"{min(max(py / height, 0.0), 1.0):.6f}")
    return "0 " + " ".join(coords)


def _eval_origin_jobs(dataset_dir: Path) -> set[str]:
    """Trabajos que originan un caso del banco: no deben entrar al entrenamiento."""
    origins: set[str] = set()
    if not dataset_dir.is_dir():
        return origins
    for case_file in dataset_dir.glob(f"*/{CASE_FILENAME}"):
        try:
            meta = json.loads(case_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        origin = str(meta.get("origin_job") or "").strip()
        if origin:
            origins.add(origin)
    return origins


def _split_for(job_id: str, page_index: int, val_ratio: float) -> str:
    """Reparto train/val estable: la misma página cae siempre en el mismo lado.

    Con ``hash()`` no lo sería: para cadenas está aleatorizado por proceso, así que el
    reparto cambiaría en cada ejecución y la validación dejaría de ser comparable.
    """
    if val_ratio <= 0:
        return "train"
    clave = f"{job_id}:{page_index}".encode("utf-8")
    marca = int(hashlib.sha1(clave).hexdigest()[:8], 16) % 1000
    return "val" if marca < val_ratio * 1000 else "train"


def export_jobs(
    jobs_dir: Path,
    out_dir: Path,
    *,
    dataset_dir: Path,
    val_ratio: float = 0.2,
    include_eval_jobs: bool = False,
) -> ExportStats:
    """Escribe un dataset YOLO-seg a partir de los trabajos corregidos en la UI."""
    stats = ExportStats()
    excluded = set() if include_eval_jobs else _eval_origin_jobs(dataset_dir)
    if excluded:
        logger.info("Trabajos excluidos por ser origen del banco de evaluación: %s", sorted(excluded))

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for manifest_path in sorted(jobs_dir.glob("*/manifest.json")):
        job_id = manifest_path.parent.name
        if job_id in excluded:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("No se pudo leer %s: %s", manifest_path, exc)
            continue

        for page in manifest.get("pages", []):
            image_path = Path(str(page.get("original_path") or ""))
            if not image_path.is_file():
                continue
            image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]

            lineas: List[str] = []
            for region in page.get("regions", []):
                if region.get("deleted") or not _is_bubble(region):
                    continue
                stats.regions += 1
                box = detection_bbox(region)
                if box is None:
                    stats.reject("sin caja")
                    continue
                points, reason = bubble_polygon(image, box)

                # Cuando el relleno se desborda, la región casi nunca es un globo: es texto
                # suelto sobre el arte, y ahí el contorno que enseña forma es el de la
                # tinta, no el del globo. Los dos derivadores son complementarios.
                if points is None or _area_ratio(points, box) >= FLOOD_AREA_RATIO:
                    tinta, motivo_tinta = text_block_polygon(image, box)
                    if tinta is not None:
                        points, reason = tinta, motivo_tinta
                        stats.from_ink += 1
                    elif points is None:
                        stats.reject(reason)
                        continue

                if points is None:
                    stats.reject(reason)
                    continue
                if _area_ratio(points, box) >= FLOOD_AREA_RATIO:
                    stats.rectangular += 1
                lineas.append(_polygon_line(points, width, height))
                stats.labelled += 1

            if not lineas:
                continue
            split = _split_for(job_id, int(page.get("index") or 0), val_ratio)
            stem = f"{job_id}_{int(page.get('index') or 0):04d}"
            destino = out_dir / "images" / split / f"{stem}{image_path.suffix.lower()}"
            cv2.imencode(image_path.suffix, image)[1].tofile(str(destino))
            (out_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lineas) + "\n", encoding="utf-8"
            )
            stats.pages += 1

    (out_dir / "data.yaml").write_text(
        "\n".join([
            f"path: {out_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            *[f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)],
            "",
        ]),
        encoding="utf-8",
    )
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Exporta trabajos corregidos de la UI como dataset YOLO-seg de globos."
    )
    parser.add_argument("--jobs", default=".pmt_ui_jobs", help="Carpeta de trabajos de la UI.")
    parser.add_argument("--out", default="dataset_bubbles", help="Carpeta destino del dataset.")
    parser.add_argument("--dataset-dir", default="dataset_eval", help="Banco de evaluación, para excluir sus trabajos.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fracción de páginas para validación.")
    parser.add_argument("--include-eval-jobs", action="store_true",
                        help="Incluye los trabajos que originan casos del banco. Contamina la evaluación.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    stats = export_jobs(
        Path(args.jobs),
        out_dir,
        dataset_dir=Path(args.dataset_dir),
        val_ratio=args.val_ratio,
        include_eval_jobs=bool(args.include_eval_jobs),
    )

    print(f"páginas exportadas:   {stats.pages}")
    print(f"regiones de globo:    {stats.regions}")
    print(f"etiquetas escritas:   {stats.labelled}")
    print(f"  rescatadas por el contorno de la tinta:     {stats.from_ink}")
    porcentaje = (100.0 * stats.rectangular / stats.labelled) if stats.labelled else 0.0
    print(
        f"  aún rectangulares (área >= {FLOOD_AREA_RATIO:.2f}x la caja): "
        f"{stats.rectangular}  ({porcentaje:.1f} %)"
    )
    if stats.rejected:
        print("descartadas:")
        for motivo, n in sorted(stats.rejected.items(), key=lambda kv: -kv[1]):
            print(f"  {motivo}: {n}")
    print(f"\ndataset en {out_dir}  (data.yaml listo para ultralytics)")
    if stats.labelled and stats.pages < 200:
        print(
            f"\nAviso: {stats.pages} páginas es poco para afinar un detector. Sirve para montar\n"
            "el flujo, no para esperar una mejora real; acumula más trabajos corregidos."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
