from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
    rotation_angle: float = 0.0
    text_align: str = "center"
    vertical_align: str = "middle"
    line_spacing_factor: float = 1.0
    text_offset_x: float = 0.0
    text_offset_y: float = 0.0
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


def _rotation_angle(value: Any) -> float:
    try:
        angle = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    angle = max(-89.0, min(89.0, angle))
    return 0.0 if abs(angle) < 0.65 else round(angle, 3)



def _text_align(value: Any) -> str:
    normalized = str(value or "center").strip().lower()
    aliases = {"izquierda": "left", "centro": "center", "derecha": "right"}
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"left", "center", "right"} else "center"


def _vertical_align(value: Any) -> str:
    normalized = str(value or "middle").strip().lower()
    aliases = {"arriba": "top", "centro": "middle", "medio": "middle", "abajo": "bottom"}
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"top", "middle", "bottom"} else "middle"


def _line_spacing_factor(value: Any) -> float:
    try:
        return max(0.55, min(2.0, float(value if value is not None else 1.0)))
    except (TypeError, ValueError):
        return 1.0


def _text_offset(value: Any) -> float:
    try:
        return max(-1000.0, min(1000.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0

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
                rotation_angle=_rotation_angle(item.get(
                    "rotation_angle",
                    item.get(
                        "text_rotation_angle",
                        ui_layout.get("requested_rotation_angle", ui_layout.get("rotation_angle", 0.0)) if ui_layout else 0.0,
                    ),
                )),
                text_align=_text_align(item.get("text_align", ui_layout.get("text_align", "center") if ui_layout else "center")),
                vertical_align=_vertical_align(item.get("vertical_align", ui_layout.get("vertical_align", "middle") if ui_layout else "middle")),
                line_spacing_factor=_line_spacing_factor(item.get("line_spacing_factor", item.get("line_spacing", ui_layout.get("line_spacing_factor", 1.0) if ui_layout else 1.0))),
                text_offset_x=_text_offset(item.get("text_offset_x", ui_layout.get("text_offset_x", 0.0) if ui_layout else 0.0)),
                text_offset_y=_text_offset(item.get("text_offset_y", ui_layout.get("text_offset_y", 0.0) if ui_layout else 0.0)),
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


def _stroke_intersects_box(stroke: BrushStroke, bbox: Box) -> bool:
    if not stroke.points:
        return False
    x, y, w, h = bbox
    radius = max(1, int(stroke.radius))
    sx1 = min(point[0] for point in stroke.points) - radius
    sy1 = min(point[1] for point in stroke.points) - radius
    sx2 = max(point[0] for point in stroke.points) + radius
    sy2 = max(point[1] for point in stroke.points) + radius
    return sx1 < x + w and sx2 >= x and sy1 < y + h and sy2 >= y


def _global_mask_eraser(strokes: Sequence[BrushStroke], height: int, width: int) -> np.ndarray:
    """Máscara global que resta zonas de cualquier máscara/pincel dibujado por el usuario."""
    erase_mask = np.zeros((height, width), dtype=np.uint8)
    for stroke in strokes:
        mode = (stroke.mode or "restore_clean").strip().lower()
        if mode not in {"mask_eraser", "erase_mask", "eraser"}:
            continue
        erase_mask = cv2.bitwise_or(erase_mask, _stroke_mask(stroke, height, width))
    return erase_mask


def _original_restore_mask(strokes: Sequence[BrushStroke], height: int, width: int) -> np.ndarray:
    """Combina las pinceladas que deben prevalecer como manga original.

    ``mask_eraser`` es el modo expuesto en la UI como "Restaurar manga original".
    También se conserva compatibilidad con el antiguo modo ``restore_original``.
    La máscara se dilata igual que en la ruta de guardado inmediato para que un
    render posterior no vuelva a introducir halos de texto/inpaint en el borde.
    """
    restore_mask = np.zeros((height, width), dtype=np.uint8)
    for stroke in strokes:
        mode = (stroke.mode or "restore_clean").strip().lower()
        if mode not in {"mask_eraser", "erase_mask", "eraser", "restore_original", "original"}:
            continue
        restore_mask = cv2.bitwise_or(restore_mask, _stroke_mask(stroke, height, width))
    if cv2.countNonZero(restore_mask) == 0:
        return restore_mask
    return cv2.dilate(restore_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)


def _subtract_eraser(mask: np.ndarray, erase_mask: np.ndarray) -> np.ndarray:
    if cv2.countNonZero(mask) == 0 or cv2.countNonZero(erase_mask) == 0:
        return mask
    return cv2.bitwise_and(mask, cv2.bitwise_not(erase_mask))


def _pending_inpaint_mask(strokes: Sequence[BrushStroke], height: int, width: int) -> np.ndarray:
    """Construye la máscara pendiente respetando el orden cronológico de los trazos.

    Un borrador/restauración solo cancela máscaras dibujadas *antes* de él. Esto evita
    que una restauración histórica bloquee para siempre un nuevo inpaint en la misma zona.
    """
    inpaint_mask = np.zeros((height, width), dtype=np.uint8)
    for stroke in strokes:
        mode = (stroke.mode or "restore_clean").strip().lower()
        stroke_mask = _stroke_mask(stroke, height, width)
        if cv2.countNonZero(stroke_mask) == 0:
            continue
        if mode == "inpaint" and not stroke.applied:
            inpaint_mask = cv2.bitwise_or(inpaint_mask, stroke_mask)
        elif mode in {"mask_eraser", "erase_mask", "eraser", "restore_original", "original"}:
            inpaint_mask = cv2.bitwise_and(inpaint_mask, cv2.bitwise_not(stroke_mask))
    return inpaint_mask


def _apply_brush_strokes(base: np.ndarray, *, original_image: np.ndarray, clean_image: np.ndarray, strokes: Sequence[BrushStroke]) -> np.ndarray:
    """Aplica acciones raster en orden cronológico.

    Cada trazo nuevo puede sobrescribir el resultado de trazos previos en la misma zona.
    Los inpaints ya aplicados están horneados en ``base`` y se omiten. Los inpaints
    pendientes se dejan para la ruta explícita de ``apply_pending_inpaint_only``.
    """
    if not strokes:
        return base
    height, width = base.shape[:2]
    for stroke in strokes:
        mode = (stroke.mode or "restore_clean").strip().lower()
        mask = _stroke_mask(stroke, height, width)
        if cv2.countNonZero(mask) == 0:
            continue
        if mode == "inpaint":
            continue
        if mode in {"restore_clean", "clean"}:
            base[mask > 0] = clean_image[mask > 0]
        elif mode in {"mask_eraser", "erase_mask", "eraser", "restore_original", "original"}:
            # La restauración es una edición de fondo. El texto se compone después.
            padded = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            base[padded > 0] = original_image[padded > 0]
    return base


def _apply_original_restore_strokes(
    base: np.ndarray,
    *,
    original_image: np.ndarray,
    strokes: Sequence[BrushStroke],
) -> np.ndarray:
    """Aplica restauraciones del manga original como última capa del render.

    Debe ejecutarse después de dibujar el texto. De lo contrario, una región
    modificada podría volver a rasterizarse encima de un área que el usuario ya
    restauró explícitamente con el pincel.
    """
    if not strokes:
        return base
    height, width = base.shape[:2]
    restore_mask = _original_restore_mask(strokes, height, width)
    if cv2.countNonZero(restore_mask) > 0:
        base[restore_mask > 0] = original_image[restore_mask > 0]
    return base


def _ceil_to_multiple(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return ((max(1, int(value)) + multiple - 1) // multiple) * multiple


def _centered_interval(center: float, length: int, limit: int) -> tuple[int, int]:
    """Devuelve un intervalo de ``length`` dentro de [0, limit), centrado si es posible."""
    if limit <= 0:
        return 0, 0
    length = max(1, min(int(length), int(limit)))
    start = int(round(float(center) - (length / 2.0)))
    start = max(0, min(start, limit - length))
    return start, start + length


def _quality_safe_inpaint_crop(
    mask: np.ndarray,
    *,
    context_px: int,
    min_side: int,
    alignment: int = 64,
    max_model_side: int | None = None,
) -> tuple[int, int, int, int]:
    """Calcula un recorte amplio alrededor de la máscara sin tocar su escala.

    Los modelos LaMa/AOT rellenan/padean internamente hasta una entrada cuadrada.
    Darles una ventana aproximadamente cuadrada evita procesar una página completa
    cuando el pincel ocupa una zona pequeña, pero conserva un contexto generoso.

    El recorte *solo* reduce contexto lejano: los píxeles de la zona de trabajo se
    mantienen a resolución nativa. Para LaMa Large esto suele proporcionar incluso
    más detalle efectivo que reducir una página grande completa a 1536 px.
    """
    height, width = mask.shape[:2]
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, width, height

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    mask_w = max(1, x1 - x0)
    mask_h = max(1, y1 - y0)

    # Un lado mínimo grande conserva contexto semántico/local. Si el trazo ya es
    # grande, añadimos contexto a ambos lados antes de alinear a la granularidad
    # natural de estas redes (64 px).
    target_side = max(
        int(min_side),
        max(mask_w, mask_h) + (2 * max(0, int(context_px))),
    )
    target_side = _ceil_to_multiple(target_side, alignment)

    # Si la ventana local ya alcanzaría el tamaño máximo que procesa el modelo,
    # no hay ahorro de inferencia. En ese caso conservamos exactamente la ruta
    # histórica de página completa y todo su contexto global.
    if max_model_side is not None and target_side >= max(1, int(max_model_side)):
        return 0, 0, width, height

    crop_w = min(width, target_side)
    crop_h = min(height, target_side)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    crop_x0, crop_x1 = _centered_interval(center_x, crop_w, width)
    crop_y0, crop_y1 = _centered_interval(center_y, crop_h, height)

    # Si el centrado contra un borde dejara la máscara sin el margen solicitado en
    # el lado donde sí hay espacio, expandimos hacia ese lado hasta target_side.
    # _centered_interval ya garantiza contener la máscara cuando target_side >= bbox.
    return crop_x0, crop_y0, crop_x1, crop_y1


def _run_inpaint_with_optional_crop(
    base: np.ndarray,
    mask: np.ndarray,
    inpaint_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Ejecuta inpaint local cuando el callable declara un perfil seguro.

    Por compatibilidad, callables externos/tests sin perfil siguen recibiendo la
    página completa. La recomposición modifica exclusivamente la máscara solicitada,
    por lo que no puede introducir costuras ni alterar píxeles ajenos al pincel.
    """
    profile = getattr(inpaint_fn, "_pmt_crop_profile", None)
    if not isinstance(profile, dict) or not profile.get("enabled", False):
        return inpaint_fn(base, mask)

    height, width = base.shape[:2]
    x0, y0, x1, y1 = _quality_safe_inpaint_crop(
        mask,
        context_px=int(profile.get("context_px", 384)),
        min_side=int(profile.get("min_side", 1024)),
        alignment=int(profile.get("alignment", 64)),
        max_model_side=(
            int(profile["max_model_side"])
            if profile.get("max_model_side") is not None
            else None
        ),
    )

    # Para páginas pequeñas o máscaras extensas evitamos una copia innecesaria.
    if x0 <= 0 and y0 <= 0 and x1 >= width and y1 >= height:
        return inpaint_fn(base, mask)

    crop_image = base[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].copy()
    crop_result = inpaint_fn(crop_image, crop_mask)
    if crop_result is None or not isinstance(crop_result, np.ndarray):
        raise ValueError("El modelo de inpainting manual no devolvió una imagen válida.")
    if crop_result.shape[:2] != crop_image.shape[:2]:
        crop_result = cv2.resize(
            crop_result,
            (crop_image.shape[1], crop_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    result = base.copy()
    target = result[y0:y1, x0:x1]
    # Incluso si un modelo modifica valores fuera de la máscara durante su forward,
    # esos cambios no se copian. Esto hace la optimización visualmente hermética.
    target[crop_mask > 0] = crop_result[crop_mask > 0]
    return result


def apply_pending_inpaint_only(
    *,
    base_path: str | Path,
    output_path: str | Path,
    brush_strokes: Sequence[BrushStroke],
    inpaint_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
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
    result = (
        _run_inpaint_with_optional_crop(base, padded, inpaint_fn)
        if inpaint_fn is not None
        else cv2.inpaint(base, padded, 3, cv2.INPAINT_TELEA)
    )
    if result is None or not isinstance(result, np.ndarray):
        raise ValueError("El modelo de inpainting manual no devolvió una imagen válida.")
    if result.shape[:2] != base.shape[:2]:
        result = cv2.resize(result, (base.shape[1], base.shape[0]))
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



def apply_background_brush_strokes(
    *,
    base_path: str | Path,
    clean_path: str | Path,
    original_path: str | Path,
    output_path: str | Path,
    brush_strokes: Sequence[BrushStroke],
) -> bool:
    """Aplica pinceladas de fondo sin rasterizar ninguna región de texto.

    La UI mantiene el fondo/limpieza como una capa independiente de las regiones.
    Esto permite que deshacer/rehacer cambie de revisión de fondo y que el texto se
    componga siempre al final, evitando restos rasterizados al mover una región.
    """
    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    clean_image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
    original_image = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if base is None:
        raise ValueError(f"No se pudo leer la capa de fondo: {base_path}")
    if clean_image is None:
        raise ValueError(f"No se pudo leer la imagen limpia: {clean_path}")
    if original_image is None:
        raise ValueError(f"No se pudo leer la imagen original: {original_path}")
    height, width = base.shape[:2]
    if clean_image.shape[:2] != (height, width):
        clean_image = cv2.resize(clean_image, (width, height))
    if original_image.shape[:2] != (height, width):
        original_image = cv2.resize(original_image, (width, height))

    effective = [
        stroke
        for stroke in brush_strokes
        if (stroke.mode or "restore_clean").strip().lower() != "inpaint"
    ]
    result = _apply_brush_strokes(
        base.copy(),
        original_image=original_image,
        clean_image=clean_image,
        strokes=effective,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), result)
    return bool(effective)


def render_manual_composite(
    *,
    background_path: str | Path,
    original_path: str | Path,
    output_path: str | Path,
    regions: Sequence[ManualRegion],
) -> None:
    """Compone todas las regiones sobre una capa de fondo libre de texto editable.

    A diferencia del render incremental legado, este render nunca parte de una imagen
    que ya contenga texto manual rasterizado. Por ello una región puede moverse,
    deshacerse o rehacerse indefinidamente sin dejar copias en posiciones anteriores.
    """
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    original_image = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if background is None:
        raise ValueError(f"No se pudo leer la capa de fondo: {background_path}")
    if original_image is None:
        raise ValueError(f"No se pudo leer la imagen original: {original_path}")
    height, width = background.shape[:2]
    if original_image.shape[:2] != (height, width):
        original_image = cv2.resize(original_image, (width, height))

    base = background.copy()
    # Restaurar original por región también pertenece a la capa de fondo. El texto
    # se dibuja después para mantener la regla: regiones siempre por encima del pincel.
    for region in regions:
        if region.deleted or not region.restore_original:
            continue
        _paste_patch(base, original_image, region.bbox)

    drawable = [
        region
        for region in regions
        if (not region.deleted) and region.visible and str(region.text).strip()
    ]
    if drawable:
        renderer = TextRenderer(max_font_size=160)
        base = renderer.render_with_layouts(
            base,
            [r.bbox for r in drawable],
            [r.text for r in drawable],
            text_styles=[r.style for r in drawable],
            font_sizes=[None if r.auto_font_size else r.font_size for r in drawable],
            rotation_angles=[r.rotation_angle for r in drawable],
            text_aligns=[r.text_align for r in drawable],
            vertical_aligns=[r.vertical_align for r in drawable],
            line_spacing_factors=[r.line_spacing_factor for r in drawable],
            text_offsets_x=[r.text_offset_x for r in drawable],
            text_offsets_y=[r.text_offset_y for r in drawable],
            ui_layouts=[r.ui_layout for r in drawable],
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), base)

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

    brush_strokes = list(brush_strokes or [])
    brush_touched_indices = {
        region.index
        for region in regions
        if any(_stroke_intersects_box(stroke, region.bbox) for stroke in brush_strokes)
    }
    drawable = [
        region
        for region in regions
        if (region.modified or region.index in brush_touched_indices)
        and (not region.deleted)
        and region.visible
        and str(region.text).strip()
    ]
    if drawable:
        renderer = TextRenderer(max_font_size=160)
        base = renderer.render_with_layouts(
            base,
            [r.bbox for r in drawable],
            [r.text for r in drawable],
            text_styles=[r.style for r in drawable],
            font_sizes=[None if r.auto_font_size else r.font_size for r in drawable],
            rotation_angles=[r.rotation_angle for r in drawable],
            text_aligns=[r.text_align for r in drawable],
            vertical_aligns=[r.vertical_align for r in drawable],
            line_spacing_factors=[r.line_spacing_factor for r in drawable],
            text_offsets_x=[r.text_offset_x for r in drawable],
            text_offsets_y=[r.text_offset_y for r in drawable],
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
            rotation_angles=[region.rotation_angle],
            text_aligns=[region.text_align],
            vertical_aligns=[region.vertical_align],
            line_spacing_factors=[region.line_spacing_factor],
            text_offsets_x=[region.text_offset_x],
            text_offsets_y=[region.text_offset_y],
            ui_layouts=[region.ui_layout],
        )

    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("No se pudo codificar la previsualización de región.")
    return encoded.tobytes()



def resolve_manual_region_metrics(
    *,
    clean_path: str | Path,
    region: ManualRegion,
) -> Dict[str, Any]:
    """Devuelve las métricas exactas que usará el render final para una región."""
    clean_image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
    if clean_image is None:
        raise ValueError(f"No se pudo leer la imagen limpia: {clean_path}")
    height, width = clean_image.shape[:2]
    x, y, w, h = _safe_box(region.bbox, width, height)
    renderer = TextRenderer(max_font_size=160)
    resolved = renderer.resolve_manual_layout(
        (x, y, w, h),
        region.text,
        region.style,
        ui_layout=region.ui_layout,
        requested_font_size=None if region.auto_font_size else region.font_size,
        rotation_angle=region.rotation_angle,
        text_align=region.text_align,
        vertical_align=region.vertical_align,
        line_spacing_factor=region.line_spacing_factor,
        text_offset_x=region.text_offset_x,
        text_offset_y=region.text_offset_y,
        image_shape=(height, width),
    )
    resolved["font_available"] = Path(renderer.font_path).exists()
    resolved["font_name"] = Path(renderer.font_path).stem
    return resolved

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
                "rotation_angle": region.rotation_angle,
                "text_align": region.text_align,
                "vertical_align": region.vertical_align,
                "line_spacing_factor": region.line_spacing_factor,
                "text_offset_x": region.text_offset_x,
                "text_offset_y": region.text_offset_y,
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
