"""Exportar e importar los textos de un trabajo entero, con sus coordenadas.

El editor de la UI corrige página a página, que es lo correcto para ajustar tipografía pero
pésimo para revisar la traducción de un capítulo: para eso se quiere el texto de todo el
trabajo en un archivo, tocarlo fuera (o pasárselo a otra persona) y devolverlo.

Dos decisiones que dan forma a todo lo demás:

- **La exportación lleva la caja de cada globo.** Traducir sin saber cuánto espacio hay es
  traducir a ciegas: una línea que cabe en un globo ancho no cabe en uno de dos palabras.
  Van las dos formas de la misma caja —`bbox` en `[x, y, ancho, alto]` y `coordenadas` en
  `[[x1, y1], [x2, y2]]`, que es como las escribe el pipeline— más el tamaño de la página,
  para poder situarlas.
- **La importación empareja por `region_uid`, nunca por posición.** Es la identidad que el
  pipeline estampa en cada región y que la corrección conserva. Emparejar por el orden de
  la lista parecería funcionar y fallaría justo en los trabajos ya corregidos, donde el
  editor ha reordenado y borrado regiones. Lo que no encuentra pareja se informa; no se
  aplica a ojo.

Aquí no se toca el disco: son transformaciones sobre `JobState`. Quien orquesta —
`JobManager`— es quien decide cuándo volver a renderizar.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from parallel_manga_translator.ui.job_state import JobState, PageState

FORMAT_NAME = "pmt-textos"
FORMAT_VERSION = 1

#: Campos del archivo que la importación aplica. El resto (cajas, tipo, tamaño de página)
#: viaja para dar contexto al que traduce: mover un globo es trabajo del editor, no de un
#: archivo de texto, y aceptar cajas editadas a mano abriría la puerta a romper el render
#: sin ver el resultado.
CAMPOS_IMPORTABLES = ("texto_original", "texto_traducido", "estilo", "borrada")


def _int_box(value: Any) -> Optional[List[int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [int(round(float(v))) for v in list(value)[:4]]
    except (TypeError, ValueError):
        return None


def bbox_to_coords(bbox: Sequence[Any]) -> List[List[int]]:
    """`[x, y, w, h]` → `[[x1, y1], [x2, y2]]`, el formato de los JSON del pipeline."""
    box = _int_box(bbox) or [0, 0, 0, 0]
    x, y, w, h = box
    return [[x, y], [x + w, y + h]]


def region_to_export(region: Mapping[str, Any], position: int) -> Dict[str, Any]:
    """Una región del manifiesto, tal como se le enseña a quien va a traducir."""
    bbox = _int_box(region.get("bbox")) or [0, 0, 0, 0]
    exportada: Dict[str, Any] = {
        "region_uid": str(region.get("region_uid") or ""),
        "indice": int(region.get("index", position)),
        "bbox": bbox,
        "coordenadas": bbox_to_coords(bbox),
        "tipo": str(region.get("type") or ""),
        "estilo": str(region.get("style") or "dialogo"),
        "texto_original": str(region.get("original_text") or ""),
        "texto_traducido": str(region.get("translated_text") or ""),
        "borrada": bool(region.get("deleted")),
        "manual": bool(region.get("manual")),
    }
    caja_run = _int_box(region.get("run_bbox"))
    if caja_run:
        # La caja que detectó la ejecución, antes de que el editor la encogiera al área de
        # texto. Es la que dice cuánto sitio hay de verdad dentro del globo.
        exportada["bbox_deteccion"] = caja_run
    return exportada


def page_to_export(page: PageState, size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    exportada: Dict[str, Any] = {
        "indice": int(page.index),
        "numero": int(page.index) + 1,
        "archivo": page.output_filename or page.source_filename,
        "archivo_original": page.source_filename,
        "estado": page.status,
        "regiones": [region_to_export(region, position) for position, region in enumerate(page.regions or []) if isinstance(region, Mapping)],
    }
    if size:
        exportada["ancho"], exportada["alto"] = int(size[0]), int(size[1])
    return exportada


def build_job_export(job: JobState, page_sizes: Optional[Mapping[int, Tuple[int, int]]] = None) -> Dict[str, Any]:
    """Todo el texto del trabajo con sus coordenadas, listo para editar fuera."""
    page_sizes = page_sizes or {}
    return {
        "formato": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "job_id": job.job_id,
        "titulo": job.title,
        "idioma_origen": job.options.source_language,
        "idioma_destino": job.options.target_language,
        "exportado_en": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paginas": [page_to_export(page, page_sizes.get(page.index)) for page in job.pages],
        "ayuda": (
            "Edita 'texto_traducido' (y 'texto_original' si corrigés la transcripción) y vuelve a "
            "importar este archivo. 'region_uid' identifica cada globo: no lo cambies. Las cajas "
            "son informativas, para saber cuánto espacio hay; moverlas se hace en el editor."
        ),
    }


@dataclass
class PageImport:
    """Lo que hay que aplicar a una página, ya resuelto contra sus regiones."""

    index: int
    regions: List[Dict[str, Any]] = field(default_factory=list)
    changed: int = 0


@dataclass
class ImportPlan:
    pages: List[PageImport] = field(default_factory=list)
    #: `region_uid` (o "página/índice") del archivo que no existe en el trabajo.
    unmatched: List[str] = field(default_factory=list)
    #: Páginas del archivo que el trabajo no tiene.
    unknown_pages: List[int] = field(default_factory=list)

    @property
    def changed_regions(self) -> int:
        return sum(page.changed for page in self.pages)

    @property
    def changed_pages(self) -> List[PageImport]:
        return [page for page in self.pages if page.changed]


def _validate(payload: Any) -> List[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("El archivo no es un JSON de textos de PMT.")
    formato = str(payload.get("formato") or FORMAT_NAME)
    if formato != FORMAT_NAME:
        raise ValueError(f"Formato desconocido: {formato!r}. Se esperaba {FORMAT_NAME!r}.")
    try:
        version = int(payload.get("version", FORMAT_VERSION))
    except (TypeError, ValueError):
        version = FORMAT_VERSION
    if version > FORMAT_VERSION:
        raise ValueError(f"El archivo usa la versión {version} y esta instalación entiende hasta la {FORMAT_VERSION}.")
    paginas = payload.get("paginas")
    if not isinstance(paginas, list) or not paginas:
        raise ValueError("El archivo no trae páginas.")
    return [pagina for pagina in paginas if isinstance(pagina, Mapping)]


def _page_index(pagina: Mapping[str, Any]) -> Optional[int]:
    for clave, ajuste in (("indice", 0), ("numero", -1)):
        if clave in pagina:
            try:
                return int(pagina[clave]) + ajuste
            except (TypeError, ValueError):
                continue
    return None


def _apply_region(destino: Dict[str, Any], origen: Mapping[str, Any]) -> bool:
    """Vuelca los campos importables sobre la región del manifiesto. ¿Cambió algo?"""
    cambios = False
    if "texto_original" in origen:
        texto = str(origen.get("texto_original") or "")
        if texto != str(destino.get("original_text") or ""):
            destino["original_text"] = texto
            cambios = True
    if "texto_traducido" in origen:
        texto = str(origen.get("texto_traducido") or "")
        if texto != str(destino.get("translated_text") or ""):
            destino["translated_text"] = texto
            cambios = True
    if "estilo" in origen:
        estilo = str(origen.get("estilo") or "").strip()
        if estilo and estilo != str(destino.get("style") or ""):
            destino["style"] = estilo
            cambios = True
    if "borrada" in origen:
        borrada = bool(origen.get("borrada"))
        if borrada != bool(destino.get("deleted")):
            destino["deleted"] = borrada
            cambios = True
    if cambios:
        # Sin esto la corrección no se persiste: `write_corrections` solo guarda lo que el
        # humano tocó, y una traducción traída de fuera es exactamente eso.
        destino["modified"] = True
    return cambios


def plan_job_import(job: JobState, payload: Any) -> ImportPlan:
    """Resuelve el archivo contra el trabajo. No escribe nada ni renderiza."""
    paginas = _validate(payload)
    por_indice = {int(page.index): page for page in job.pages}
    plan = ImportPlan()

    for pagina in paginas:
        indice = _page_index(pagina)
        if indice is None or indice not in por_indice:
            if indice is not None:
                plan.unknown_pages.append(indice)
            continue
        page = por_indice[indice]
        regiones = [dict(region) for region in (page.regions or []) if isinstance(region, Mapping)]
        por_uid = {str(region.get("region_uid") or ""): region for region in regiones if region.get("region_uid")}
        por_indice_region = {}
        for position, region in enumerate(regiones):
            try:
                por_indice_region.setdefault(int(region.get("index", position)), region)
            except (TypeError, ValueError):
                continue

        cambiadas = 0
        for position, entrada in enumerate(pagina.get("regiones") or []):
            if not isinstance(entrada, Mapping):
                continue
            uid = str(entrada.get("region_uid") or "")
            if uid:
                # Con identidad no hay segunda oportunidad: si ese globo no está en el
                # trabajo, se informa. Caer al índice aquí colocaría la traducción en otro
                # globo cualquiera y nadie lo notaría hasta ver la página compuesta.
                destino = por_uid.get(uid)
            else:
                # Sin identidad solo queda el índice, que es lo que el editor reasigna.
                # Se acepta para archivos escritos a mano, pero es el camino frágil.
                try:
                    destino = por_indice_region.get(int(entrada.get("indice", position)))
                except (TypeError, ValueError):
                    destino = None
            if destino is None:
                plan.unmatched.append(uid or f"p{indice + 1}/{entrada.get('indice', position)}")
                continue
            if _apply_region(destino, entrada):
                cambiadas += 1

        plan.pages.append(PageImport(index=indice, regions=regiones, changed=cambiadas))

    return plan


def manifest_region_to_payload(region: Mapping[str, Any], position: int) -> Dict[str, Any]:
    """Región del manifiesto → payload de `save_manual_render`.

    El editor del navegador manda este mismo diccionario en cada guardado. Reconstruirlo
    aquí es lo que permite que una importación pase por el camino de corrección manual de
    siempre —correcciones en disco y render regenerado— en vez de inventar uno paralelo.
    """
    bbox = _int_box(region.get("bbox")) or [0, 0, 1, 1]
    return {
        "index": int(region.get("index", position)),
        "bbox": bbox,
        "source_bbox": _int_box(region.get("source_bbox")) or bbox,
        "text": str(region.get("translated_text") or ""),
        "original_text": str(region.get("original_text") or ""),
        "style": str(region.get("style") or "dialogo"),
        "restore_original": bool(region.get("restore_original")),
        "visible": region.get("visible", True) is not False,
        "modified": bool(region.get("modified")),
        "manual": bool(region.get("manual")),
        "deleted": bool(region.get("deleted")),
        "auto_font_size": region.get("auto_font_size", True) is not False,
        "font_size": region.get("font_size"),
        "rotation_angle": float(region.get("rotation_angle") or 0.0),
        "ui_layout": region.get("ui_layout"),
        "region_uid": str(region.get("region_uid") or ""),
        "run_bbox": _int_box(region.get("run_bbox")),
    }


def export_filename(title: str) -> str:
    """Nombre del archivo descargado. Acaba en una cabecera HTTP: solo ASCII."""
    limpio = "".join(c if (c.isalnum() and c.isascii()) or c in "-_ " else "_" for c in str(title or "")).strip()
    return f"{limpio or 'trabajo'}_textos.json"


__all__ = [
    "CAMPOS_IMPORTABLES",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "ImportPlan",
    "PageImport",
    "bbox_to_coords",
    "build_job_export",
    "export_filename",
    "manifest_region_to_payload",
    "page_to_export",
    "plan_job_import",
    "region_to_export",
]
