from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.rendering.text_renderer import TextRenderer

Box = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass
class ManualRegion:
    index: int
    bbox: Box
    text: str
    original_text: str = ""
    style: str = "dialogo"
    restore_original: bool = False
    visible: bool = True
    modified: bool = False
    source_bbox: Optional[Box] = None
    manual: bool = False
    deleted: bool = False
    auto_font_size: bool = True
    font_size: Optional[int] = None
    ui_layout: Optional[Dict[str, Any]] = None


@dataclass
class BrushStroke:
    points: List[Point]
    radius: int = 18
    mode: str = "restore_clean"
    applied: bool = False


def _safe_box(raw: Sequence[Any], image_width: int, image_height: int) -> Box:
    values = [int(round(float(v))) for v in list(raw)[:4]]
    if len(values) != 4:
        raise ValueError("La región debe tener bbox [x, y, w, h].")
    x, y, w, h = values
    x = max(0, min(x, max(0, image_width - 1)))
    y = max(0, min(y, max(0, image_height - 1)))
    w = max(1, min(w, image_width - x))
    h = max(1, min(h, image_height - y))
    return x, y, w, h


def _optional_box(raw: Any, image_width: int, image_height: int) -> Optional[Box]:
    if not raw:
        return None
    try:
        return _safe_box(raw, image_width, image_height)
    except Exception:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "si", "sí", "on", "modified", "corrected", "deleted"}
    return bool(value)


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "si", "sí", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", "none", "null", ""}:
            return False
    return bool(value)


def _optional_font_size(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "none", "null"}:
        return None
    try:
        size = int(round(float(value)))
    except Exception:
        return None
    return max(6, min(160, size))


def parse_manual_regions(payload: Iterable[Dict[str, Any]], image_width: int, image_height: int) -> List[ManualRegion]:
    regions: List[ManualRegion] = []
    for fallback_index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        raw_box = item.get("bbox") or item.get("box") or []
        try:
            bbox = _safe_box(raw_box, image_width, image_height)
        except Exception:
            continue
        source_bbox = _optional_box(
            item.get("source_bbox") or item.get("original_bbox") or item.get("base_bbox"),
            image_width,
            image_height,
        )
        manual = _truthy(item.get("manual", item.get("created_manually", False)))
        deleted = _boolish(item.get("deleted", False), False)
        auto_font_size = _boolish(
            item.get("auto_font_size", item.get("font_auto", item.get("font_size_auto", True))),
            True,
        )
        font_size = _optional_font_size(item.get("font_size"))
        if not auto_font_size and font_size is None:
            font_size = 24
        ui_layout = item.get("ui_layout") or item.get("layout_ui") or item.get("Layout UI")
        if not isinstance(ui_layout, dict):
            ui_layout = None
        regions.append(
            ManualRegion(
                index=int(item.get("index", fallback_index)),
                bbox=bbox,
                text=str(item.get("text", "")),
                original_text=str(item.get("original_text", "")),
                style=str(item.get("style") or "dialogo"),
                restore_original=_boolish(item.get("restore_original", False), False),
                visible=_boolish(item.get("visible", True), True),
                modified=_truthy(item.get("modified", item.get("corrected", item.get("manual", False)))) or manual or deleted,
                source_bbox=source_bbox,
                manual=manual,
                deleted=deleted,
                auto_font_size=auto_font_size,
                font_size=font_size,
                ui_layout=ui_layout,
            )
        )
    return regions


def parse_brush_strokes(payload: Iterable[Dict[str, Any]], image_width: int, image_height: int) -> List[BrushStroke]:
    strokes: List[BrushStroke] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        points: List[Point] = []
        for raw_point in item.get("points", []) or []:
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
                continue
            try:
                x = max(0, min(image_width - 1, int(round(float(raw_point[0])))))
                y = max(0, min(image_height - 1, int(round(float(raw_point[1])))))
            except Exception:
                continue
            points.append((x, y))
        if not points:
            continue
        try:
            radius = int(round(float(item.get("radius", 18))))
        except Exception:
            radius = 18
        strokes.append(
            BrushStroke(
                points=points,
                radius=max(1, min(180, radius)),
                mode=str(item.get("mode") or "restore_clean"),
                applied=_boolish(item.get("applied", False), False),
            )
        )
    return strokes


def _paste_patch(target: np.ndarray, source: np.ndarray, bbox: Box, padding: int = 0) -> None:
    height, width = target.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    if x2 <= x1 or y2 <= y1:
        return
    target[y1:y2, x1:x2] = source[y1:y2, x1:x2]


def _same_box(a: Optional[Box], b: Optional[Box]) -> bool:
    return bool(a and b and tuple(a) == tuple(b))


def _draw_polyline_mask(mask: np.ndarray, points: Sequence[Point], radius: int) -> None:
    if not points:
        return
    thickness = max(1, radius * 2)
    if len(points) == 1:
        cv2.circle(mask, points[0], radius, 255, -1, lineType=cv2.LINE_AA)
        return
    for p1, p2 in zip(points, points[1:]):
        cv2.line(mask, p1, p2, 255, thickness, lineType=cv2.LINE_AA)
    cv2.circle(mask, points[0], radius, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(mask, points[-1], radius, 255, -1, lineType=cv2.LINE_AA)


def _stroke_mask(stroke: BrushStroke, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    _draw_polyline_mask(mask, stroke.points, stroke.radius)
    return mask


def _global_mask_eraser(strokes: Sequence[BrushStroke], height: int, width: int) -> np.ndarray:
    """Máscara global que resta zonas de cualquier máscara/pincel dibujado por el usuario."""
    erase_mask = np.zeros((height, width), dtype=np.uint8)
    for stroke in strokes:
        mode = (stroke.mode or "restore_clean").strip().lower()
        if mode not in {"mask_eraser", "erase_mask", "eraser"}:
            continue
        erase_mask = cv2.bitwise_or(erase_mask, _stroke_mask(stroke, height, width))
    return erase_mask


def _subtract_eraser(mask: np.ndarray, erase_mask: np.ndarray) -> np.ndarray:
    if cv2.countNonZero(mask) == 0 or cv2.countNonZero(erase_mask) == 0:
        return mask
    return cv2.bitwise_and(mask, cv2.bitwise_not(erase_mask))


def _pending_inpaint_mask(strokes: Sequence[BrushStroke], height: int, width: int) -> np.ndarray:
    erase_mask = _global_mask_eraser(strokes, height, width)
    inpaint_mask = np.zeros((height, width), dtype=np.uint8)
    for stroke in strokes:
        mode = (stroke.mode or "restore_clean").strip().lower()
        if mode != "inpaint" or stroke.applied:
            continue
        mask = _subtract_eraser(_stroke_mask(stroke, height, width), erase_mask)
        if cv2.countNonZero(mask) > 0:
            inpaint_mask = cv2.bitwise_or(inpaint_mask, mask)
    return inpaint_mask


def _apply_brush_strokes(base: np.ndarray, *, original_image: np.ndarray, clean_image: np.ndarray, strokes: Sequence[BrushStroke]) -> np.ndarray:
    if not strokes:
        return base
    height, width = base.shape[:2]
    erase_mask = _global_mask_eraser(strokes, height, width)
    inpaint_mask = np.zeros((height, width), dtype=np.uint8)
    for stroke in strokes:
        mode = (stroke.mode or "restore_clean").strip().lower()
        if mode in {"mask_eraser", "erase_mask", "eraser"}:
            continue
        mask = _subtract_eraser(_stroke_mask(stroke, height, width), erase_mask)
        if cv2.countNonZero(mask) == 0:
            continue
        if mode == "inpaint":
            # Los trazos de inpaint ya aplicados quedan registrados solo para auditoría.
            # No se vuelven a ejecutar porque inpaint no es una operación idempotente.
            if not stroke.applied:
                inpaint_mask = cv2.bitwise_or(inpaint_mask, mask)
            continue
        source = clean_image if mode in {"restore_clean", "clean"} else original_image
        base[mask > 0] = source[mask > 0]

    if cv2.countNonZero(inpaint_mask) > 0:
        # Inpaint manual ligero para correcciones dibujadas a pulso. Se aplica sobre
        # la imagen base actual para que el usuario pueda eliminar restos o manchas
        # puntuales sin volver a ejecutar todo el pipeline.
        padded = cv2.dilate(inpaint_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        base[:] = cv2.inpaint(base, padded, 3, cv2.INPAINT_TELEA)
    return base


def apply_pending_inpaint_only(
    *,
    base_path: str | Path,
    output_path: str | Path,
    brush_strokes: Sequence[BrushStroke],
) -> bool:
    """Aplica únicamente máscaras nuevas de inpaint sobre la imagen actual.

    Esta ruta evita redibujar textos/regiones; por eso no recalcula tamaños de fuente.
    Devuelve True cuando hubo una máscara efectiva para inpaint.
    """
    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    if base is None:
        raise ValueError(f"No se pudo leer la imagen base para inpaint: {base_path}")
    height, width = base.shape[:2]
    inpaint_mask = _pending_inpaint_mask(brush_strokes, height, width)
    if cv2.countNonZero(inpaint_mask) == 0:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), base)
        return False
    padded = cv2.dilate(inpaint_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    result = cv2.inpaint(base, padded, 3, cv2.INPAINT_TELEA)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), result)
    return True


def _applied_inpaint_mask(strokes: Sequence[BrushStroke], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for stroke in strokes:
        mode = (stroke.mode or "restore_clean").strip().lower()
        if mode == "inpaint" and stroke.applied:
            stroke_mask = _stroke_mask(stroke, height, width)
            if cv2.countNonZero(stroke_mask) > 0:
                mask = cv2.bitwise_or(mask, stroke_mask)
    return mask


def restore_mask_erased_pixels(
    *,
    base_path: str | Path,
    restore_path: str | Path,
    output_path: str | Path,
    brush_strokes: Sequence[BrushStroke],
) -> bool:
    """Restaura zonas borradas sobre máscaras ya aplicadas.

    El borrador global no debe convertirse en una mancha permanente ni quedarse como
    trazo visual. Cuando cruza una zona de inpaint ya aplicada, se restauran esos
    píxeles desde la copia previa al inpaint. Si no existe máscara aplicada, se usa
    el propio trazo como área de restauración.
    """
    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    if base is None:
        raise ValueError(f"No se pudo leer la imagen base para restaurar máscara: {base_path}")
    restore = cv2.imread(str(restore_path), cv2.IMREAD_COLOR)
    if restore is None:
        raise ValueError(f"No se pudo leer la imagen de restauración: {restore_path}")
    if restore.shape[:2] != base.shape[:2]:
        restore = cv2.resize(restore, (base.shape[1], base.shape[0]))

    height, width = base.shape[:2]
    eraser_mask = _global_mask_eraser(brush_strokes, height, width)
    if cv2.countNonZero(eraser_mask) == 0:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), base)
        return False

    # El usuario espera que todo lo pintado vuelva al manga original, no solo
    # la intersección exacta con la máscara histórica de inpaint. Un margen suave
    # evita bordes duros al restaurar sobre zonas ya reconstruidas.
    target_mask = cv2.dilate(eraser_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    if cv2.countNonZero(target_mask) == 0:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), base)
        return False

    base[target_mask > 0] = restore[target_mask > 0]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), base)
    return True


def render_manual_page(
    *,
    clean_path: str | Path,
    original_path: str | Path,
    translated_path: str | Path,
    output_path: str | Path,
    regions: Sequence[ManualRegion],
    brush_strokes: Sequence[BrushStroke] | None = None,
    base_path: str | Path | None = None,
) -> None:
    """Renderiza cambios manuales sin recalcular regiones no tocadas.

    Parte de la traducción automática, aplica pinceladas manuales de máscara y redibuja
    únicamente regiones modificadas/creadas por el usuario.
    """
    clean_image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
    original_image = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    translated_image = cv2.imread(str(translated_path), cv2.IMREAD_COLOR)
    base_image = cv2.imread(str(base_path), cv2.IMREAD_COLOR) if base_path else None
    if clean_image is None:
        raise ValueError(f"No se pudo leer la imagen limpia: {clean_path}")
    if original_image is None:
        raise ValueError(f"No se pudo leer la imagen original: {original_path}")
    if translated_image is None:
        raise ValueError(f"No se pudo leer la imagen traducida: {translated_path}")

    if original_image.shape[:2] != clean_image.shape[:2]:
        original_image = cv2.resize(original_image, (clean_image.shape[1], clean_image.shape[0]))
    if translated_image.shape[:2] != clean_image.shape[:2]:
        translated_image = cv2.resize(translated_image, (clean_image.shape[1], clean_image.shape[0]))
    if base_image is not None and base_image.shape[:2] != clean_image.shape[:2]:
        base_image = cv2.resize(base_image, (clean_image.shape[1], clean_image.shape[0]))

    modified_regions = [region for region in regions if region.modified or region.deleted]
    base = (base_image if base_image is not None else translated_image).copy()

    base = _apply_brush_strokes(
        base,
        original_image=original_image,
        clean_image=clean_image,
        strokes=brush_strokes or [],
    )

    for region in modified_regions:
        source_bbox = region.source_bbox or region.bbox
        if source_bbox and not _same_box(source_bbox, region.bbox):
            # Si la región se movió, borra la traducción automática de su ubicación anterior.
            _paste_patch(base, clean_image, source_bbox)

        # Limpia o restaura solo el área que el usuario tocó. Las regiones eliminadas
        # siempre se borran con la imagen limpia para evitar manchas.
        background = clean_image if region.deleted else (original_image if region.restore_original else clean_image)
        _paste_patch(base, background, region.bbox)

    drawable = [r for r in modified_regions if (not r.deleted) and r.visible and str(r.text).strip()]
    if drawable:
        renderer = TextRenderer(max_font_size=160)
        base = renderer.render_with_layouts(
            base,
            [r.bbox for r in drawable],
            [r.text for r in drawable],
            text_styles=[r.style for r in drawable],
            font_sizes=[None if r.auto_font_size else r.font_size for r in drawable],
            ui_layouts=[r.ui_layout for r in drawable],
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), base)


def render_manual_region_preview(
    *,
    clean_path: str | Path,
    original_path: str | Path,
    region: ManualRegion,
) -> bytes:
    """Devuelve un PNG de una sola región usando el mismo renderizador final.

    La UI lo usa durante la edición directa para que el texto visible en el
    navegador sea una vista rasterizada por el backend, no una aproximación CSS.
    """
    clean_image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
    original_image = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if clean_image is None:
        raise ValueError(f"No se pudo leer la imagen limpia: {clean_path}")
    if original_image is None:
        raise ValueError(f"No se pudo leer la imagen original: {original_path}")
    if original_image.shape[:2] != clean_image.shape[:2]:
        original_image = cv2.resize(original_image, (clean_image.shape[1], clean_image.shape[0]))

    height, width = clean_image.shape[:2]
    x, y, w, h = _safe_box(region.bbox, width, height)
    background = original_image if region.restore_original else clean_image
    crop = background[y:y + h, x:x + w].copy()

    if region.visible and not region.deleted and str(region.text).strip():
        renderer = TextRenderer(max_font_size=160)
        crop = renderer.render_with_layouts(
            crop,
            [(0, 0, w, h)],
            [region.text],
            text_styles=[region.style],
            font_sizes=[None if region.auto_font_size else region.font_size],
            ui_layouts=[region.ui_layout],
        )

    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("No se pudo codificar la previsualización de región.")
    return encoded.tobytes()


def write_corrections(path: str | Path, regions: Sequence[ManualRegion], brush_strokes: Sequence[BrushStroke] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "regions": [
            {
                "index": region.index,
                "bbox": list(region.bbox),
                "source_bbox": list(region.source_bbox or region.bbox),
                "text": region.text,
                "original_text": region.original_text,
                "style": region.style,
                "restore_original": region.restore_original,
                "visible": region.visible,
                "modified": region.modified,
                "manual": region.manual,
                "deleted": region.deleted,
                "auto_font_size": region.auto_font_size,
                "font_size": region.font_size,
                "ui_layout": region.ui_layout,
            }
            for region in regions
            if region.modified or region.manual or region.deleted
        ],
        "brush_strokes": [
            {
                "points": [list(point) for point in stroke.points],
                "radius": stroke.radius,
                "mode": stroke.mode,
                "applied": stroke.applied,
            }
            for stroke in (brush_strokes or [])
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_corrections_payload(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"regions": [], "brush_strokes": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"regions": [], "brush_strokes": []}
        regions = data.get("regions", []) if isinstance(data.get("regions", []), list) else []
        strokes = data.get("brush_strokes", data.get("mask_strokes", []))
        strokes = strokes if isinstance(strokes, list) else []
        return {"regions": regions, "brush_strokes": strokes}
    except Exception:
        return {"regions": [], "brush_strokes": []}


def read_corrections(path: str | Path) -> List[Dict[str, Any]]:
    return read_corrections_payload(path).get("regions", [])
