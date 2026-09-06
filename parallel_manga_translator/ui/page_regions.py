"""Las regiones de una página tal como las ve la UI.

El pipeline deja dos JSON —`Transcripción.json` y `Traducción.json`— y el editor manual
deja un tercero con las correcciones del usuario. Este módulo compone los tres en la lista
de regiones que consume el navegador, y reescribe la traducción de una página cuando se
retraduce.

Estaban como seis métodos privados de `JobManager`, pero no usaban nada de él: son
transformaciones de datos. Ahí dentro no se podían probar sin construir el manager entero,
que arranca un worker y toca disco, y la regla de precedencia que implementan —qué gana
cuando una corrección manual y una retraducción tocan la misma región— es justo lo que hay
que poder probar barato.

Orden de precedencia, que es el contrato de verdad de este módulo:

1. `merge_page_regions` funde transcripción y traducción por `Índice`. La transcripción
   pone el texto original y la caja; la traducción pisa texto y estilo.
2. `apply_saved_corrections` pone encima lo que el usuario editó a mano, y **añade** las
   regiones que el usuario creó y no existen en el pipeline.
3. `push_retranslated_page` cambia sólo texto y estilo: el resto de campos del globo
   —"Layout UI" incluido— se arrastra de la ejecución previa, para no perder ajustes de
   posición al retraducir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from parallel_manga_translator.ui.job_state import JobState, PageState
from parallel_manga_translator.ui.queue_adapter import CapturingJsonQueue


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def page_items(data: Dict[str, Any], key: str, page_no: int) -> List[Dict[str, Any]]:
    pages = data.get(key, []) if isinstance(data, dict) else []
    if not isinstance(pages, list):
        return []
    page = next((item for item in pages if isinstance(item, dict) and item.get("Página") == page_no), None)
    items = page.get("Globos de texto", []) if isinstance(page, dict) else []
    return items if isinstance(items, list) else []


def coords_to_bbox(coords: Any) -> List[int]:
    try:
        (x1, y1), (x2, y2) = coords
        return [int(x1), int(y1), max(1, int(x2) - int(x1)), max(1, int(y2) - int(y1))]
    except Exception:
        return [0, 0, 1, 1]


def merge_page_regions(job: JobState,
    page: PageState,
    trans_data: Optional[Dict[str, Any]] = None,
    trad_data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    trans_data = trans_data or read_json(Path(job.output_dir) / "limpieza" / "Transcripción.json")
    trad_data = trad_data or read_json(Path(job.output_dir) / "traduccion" / "Traducción.json")
    page_no = page.index + 1
    originals = page_items(trans_data, "Transcripción", page_no)
    translations = page_items(trad_data, "Traducción", page_no)
    by_index: Dict[int, Dict[str, Any]] = {}

    for item in originals:
        idx = int(item.get("Índice", len(by_index)))
        coords = item.get("Coordenadas") or [[0, 0], [0, 0]]
        bbox = coords_to_bbox(coords)
        by_index.setdefault(idx, {"index": idx})
        by_index[idx].update(
            {
                "bbox": bbox,
                "source_bbox": bbox,
                "original_text": item.get("Texto", ""),
                "style": item.get("Estilo", "dialogo"),
                "type": item.get("Tipo", "dialogue"),
                "confidence": item.get("Confianza", 0),
                "restore_original": False,
                "visible": True,
                "modified": False,
                "deleted": False,
                "auto_font_size": True,
                "font_size": None,
                "rotation_angle": item.get("Ángulo de texto", item.get("rotation_angle", 0.0)),
                "rotation_confidence": item.get("Confianza de inclinación", item.get("rotation_confidence", 0.0)),
                "ui_layout": item.get("Layout UI") or item.get("ui_layout"),
            }
        )
    for item in translations:
        idx = int(item.get("Índice", len(by_index)))
        coords = item.get("Coordenadas") or [[0, 0], [0, 0]]
        bbox = coords_to_bbox(coords)
        by_index.setdefault(idx, {"index": idx})
        by_index[idx].update(
            {
                "bbox": by_index[idx].get("bbox") or bbox,
                "source_bbox": by_index[idx].get("source_bbox") or bbox,
                "translated_text": item.get("Texto", ""),
                "style": item.get("Estilo", by_index[idx].get("style", "dialogo")),
                "type": item.get("Tipo", by_index[idx].get("type", "dialogue")),
                "confidence": item.get("Confianza", by_index[idx].get("confidence", 0)),
                "restore_original": by_index[idx].get("restore_original", False),
                "visible": by_index[idx].get("visible", True),
                "modified": by_index[idx].get("modified", False),
                "deleted": by_index[idx].get("deleted", False),
                "auto_font_size": by_index[idx].get("auto_font_size", True),
                "font_size": by_index[idx].get("font_size"),
                "rotation_angle": item.get("Ángulo de texto", item.get("rotation_angle", by_index[idx].get("rotation_angle", 0.0))),
                "rotation_confidence": item.get("Confianza de inclinación", item.get("rotation_confidence", by_index[idx].get("rotation_confidence", 0.0))),
                "ui_layout": item.get("Layout UI") or item.get("ui_layout") or by_index[idx].get("ui_layout"),
            }
        )
    return [by_index[idx] for idx in sorted(by_index)]


def apply_saved_corrections(regions: List[Dict[str, Any]], corrections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    corrected_by_index = {int(item.get("index", idx)): item for idx, item in enumerate(corrections) if isinstance(item, dict)}
    merged: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for idx, region in enumerate(regions):
        region_index = int(region.get("index", idx))
        correction = corrected_by_index.get(region_index)
        if correction:
            source_bbox = correction.get("source_bbox") or region.get("source_bbox") or region.get("bbox")
            region = {
                **region,
                "bbox": correction.get("bbox", region.get("bbox")),
                "source_bbox": source_bbox,
                "translated_text": correction.get("text", region.get("translated_text", "")),
                "style": correction.get("style", region.get("style", "dialogo")),
                "restore_original": bool(correction.get("restore_original", False)),
                "visible": bool(correction.get("visible", True)),
                "modified": bool(correction.get("modified", True)),
                "manual": bool(correction.get("manual", region.get("manual", False))),
                "deleted": bool(correction.get("deleted", False)),
                "auto_font_size": bool(correction.get("auto_font_size", True)),
                "font_size": correction.get("font_size"),
                "rotation_angle": correction.get("rotation_angle", region.get("rotation_angle", 0.0)),
                "ui_layout": correction.get("ui_layout") or region.get("ui_layout"),
            }
        else:
            region = {
                **region,
                "source_bbox": region.get("source_bbox") or region.get("bbox"),
                "modified": bool(region.get("modified", False)),
                "auto_font_size": region.get("auto_font_size", True),
                "font_size": region.get("font_size"),
                "rotation_angle": region.get("rotation_angle", 0.0),
                "ui_layout": region.get("ui_layout"),
            }
        seen.add(region_index)
        merged.append(region)

    for correction_index, correction in sorted(corrected_by_index.items()):
        if correction_index in seen:
            continue
        bbox = correction.get("bbox") or [0, 0, 1, 1]
        merged.append({
            "index": correction_index,
            "bbox": bbox,
            "source_bbox": correction.get("source_bbox") or bbox,
            "original_text": correction.get("original_text", ""),
            "translated_text": correction.get("text", correction.get("translated_text", "")),
            "style": correction.get("style", "dialogo"),
            "type": correction.get("type", "manual"),
            "confidence": correction.get("confidence", 0),
            "restore_original": bool(correction.get("restore_original", False)),
            "visible": bool(correction.get("visible", True)),
            "modified": bool(correction.get("modified", True)),
            "manual": True,
            "deleted": bool(correction.get("deleted", False)),
            "auto_font_size": bool(correction.get("auto_font_size", True)),
            "font_size": correction.get("font_size"),
            "rotation_angle": correction.get("rotation_angle", 0.0),
            "ui_layout": correction.get("ui_layout"),
        })
    return merged


def push_retranslated_page(trad_queue: CapturingJsonQueue, page: PageState, resultados: Sequence[Any]) -> None:
    """Reescribe los globos de la página en Traducción.json conservando el resto."""
    pagina_no = page.index + 1
    datos = trad_queue.data
    entrada = next(
        (
            dict(item)
            for item in (datos.get("Traducción", []) if isinstance(datos, dict) else [])
            if isinstance(item, dict) and item.get("Página") == pagina_no
        ),
        {"Página": pagina_no},
    )
    previos = {
        int(item.get("Índice", posicion)): dict(item)
        for posicion, item in enumerate(page_items(datos, "Traducción", pagina_no))
    }
    globos: List[Dict[str, Any]] = []
    for resultado in resultados:
        if not resultado.source_language_ok:
            # El pipeline no publica lo que descartó el filtro de idioma de origen.
            continue
        x, y, w, h = resultado.bbox
        elemento = previos.get(resultado.index, {
            "Índice": resultado.index,
            "Coordenadas": [[x, y], [x + w, y + h]],
        })
        # Solo cambian texto y estilo. "Layout UI" y el resto de campos del globo
        # se arrastran tal cual desde la ejecución del pipeline.
        elemento.update({
            "Índice": resultado.index,
            "Texto": resultado.translated_text,
            "Estilo": resultado.style,
        })
        globos.append(elemento)
    entrada["Globos de texto"] = globos
    trad_queue.put({"establecer_elemento_en_lista": {"Traducción": entrada}})
    trad_queue.put({"ordenar_por_paginas": {"tipo": "Traducción"}})

__all__ = [
    "apply_saved_corrections",
    "coords_to_bbox",
    "merge_page_regions",
    "page_items",
    "push_retranslated_page",
    "read_json",
]
