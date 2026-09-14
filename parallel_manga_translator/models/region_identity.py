"""Identidad estable de una región a lo largo de una ejecución.

Vive en `models/` porque la necesitan dos capas que no deben conocerse: el pipeline, que
numera las regiones y vuelca sus recortes, y la UI, que guarda la corrección humana. Sin
un token común, emparejar una corrección con la región que la originó exige adivinar por
geometría, y eso falla justo en los casos difíciles —globos solapados, regiones borradas,
listas reordenadas por el editor—.
"""

from __future__ import annotations

MANUAL_PREFIX = "manual-"


def run_region_uid(page_no: int, region_index: int) -> str:
    """Identidad de la región `region_index` de la página `page_no` (1-based).

    `region_index` es la posición de la región en `regiones_ordenadas`, que es exactamente
    el `Índice` que el pipeline escribe en `Transcripción.json` y el número con el que se
    vuelcan los recortes de OCR. El `index` del editor no sirve: se reasigna al reordenar,
    borrar o añadir regiones.
    """
    return f"p{int(page_no):04d}r{int(region_index):04d}"


def is_manual_uid(region_uid: str) -> bool:
    """Una región dibujada por el humano no corrige ninguna región de la ejecución."""
    return str(region_uid or "").startswith(MANUAL_PREFIX)


__all__ = ["MANUAL_PREFIX", "is_manual_uid", "run_region_uid"]
