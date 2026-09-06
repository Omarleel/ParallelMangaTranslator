from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from parallel_manga_translator.config.constants import RUTA_FUENTE, TAMANIO_MINIMO_FUENTE


from parallel_manga_translator.rendering.font_metrics_mixin import FontMetricsMixin
from parallel_manga_translator.rendering.text_fitting_mixin import TextFittingMixin
from parallel_manga_translator.rendering.mask_text_area_mixin import MaskTextAreaMixin
from parallel_manga_translator.rendering.bubble_slot_layout_mixin import BubbleSlotLayoutMixin
from parallel_manga_translator.rendering.slot_text_splitter_mixin import SlotTextSplitterMixin
class TextRenderer(FontMetricsMixin, TextFittingMixin, MaskTextAreaMixin, BubbleSlotLayoutMixin, SlotTextSplitterMixin):
    """
    Renderiza traducciones dentro de la caja detectada intentando mantenerlas
    legibles y, sobre todo, evitando que el texto se salga del globo.
    """

    def __init__(
        self,
        font_path: str = RUTA_FUENTE,
        min_font_size: int = TAMANIO_MINIMO_FUENTE,
        absolute_min_font_size: int = 7,
        max_font_size: int = 96,
        inner_margin_ratio: float = 0.09,
        smart_typography: bool = True,
        hyphenation: bool = True,
        balance_lines: bool = True,
        line_spacing_factor: float = 1.0,
    ) -> None:
        self.font_path = font_path
        self.min_font_size = max(1, int(min_font_size))
        self.absolute_min_font_size = max(5, min(int(absolute_min_font_size), self.min_font_size))
        self.max_font_size = max(self.min_font_size, int(max_font_size))
        self.inner_margin_ratio = max(0.02, min(0.20, float(inner_margin_ratio)))
        self.smart_typography = bool(smart_typography)
        self.hyphenation = bool(hyphenation)
        self.balance_lines = bool(balance_lines)
        self.line_spacing_factor = max(0.72, min(1.45, float(line_spacing_factor)))





























    @staticmethod
    def _coerce_box(raw_box: Sequence[int], image_width: int | None = None, image_height: int | None = None) -> Tuple[int, int, int, int]:
        values = list(raw_box or (0, 0, 1, 1))[:4]
        while len(values) < 4:
            values.append(1 if len(values) >= 2 else 0)
        x, y, w, h = [int(round(float(value))) for value in values]
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        if image_width is not None:
            x = min(x, max(0, image_width - 1))
            w = min(w, max(1, image_width - x))
        if image_height is not None:
            y = min(y, max(0, image_height - 1))
            h = min(h, max(1, image_height - y))
        return x, y, w, h

    def _fixed_font_size(self, requested_font_size: Optional[int]) -> Optional[int]:
        try:
            if requested_font_size is None:
                return None
            fixed = int(round(float(requested_font_size)))
        except Exception:
            return None
        if fixed <= 0:
            return None
        return max(self.absolute_min_font_size, min(self.max_font_size, fixed))

    @staticmethod
    def _coerce_rotation_angle(value: Any) -> float:
        try:
            angle = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(angle):
            return 0.0
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        angle = max(-89.0, min(89.0, angle))
        return 0.0 if abs(angle) < 0.65 else angle

    @staticmethod
    def _rotation_fit_scale(width: int, height: int, angle: float) -> float:
        if abs(angle) < 0.65 or width <= 1 or height <= 1:
            return 1.0
        radians = math.radians(abs(angle))
        cosine = abs(math.cos(radians))
        sine = abs(math.sin(radians))
        rotated_width = cosine * width + sine * height
        rotated_height = sine * width + cosine * height
        return max(0.18, min(1.0, width / max(1.0, rotated_width), height / max(1.0, rotated_height)))

    @classmethod
    def _rotation_safe_local_box(
        cls,
        box: Tuple[int, int, int, int],
        canvas_width: int,
        canvas_height: int,
        angle: float,
    ) -> Tuple[int, int, int, int]:
        if abs(angle) < 0.65:
            return box
        x, y, w, h = box
        scale = cls._rotation_fit_scale(canvas_width, canvas_height, angle)
        center_x = canvas_width / 2.0
        center_y = canvas_height / 2.0
        box_center_x = x + w / 2.0
        box_center_y = y + h / 2.0
        new_center_x = center_x + (box_center_x - center_x) * scale
        new_center_y = center_y + (box_center_y - center_y) * scale
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        new_x = int(round(new_center_x - new_w / 2.0))
        new_y = int(round(new_center_y - new_h / 2.0))
        new_x = max(0, min(new_x, max(0, canvas_width - new_w)))
        new_y = max(0, min(new_y, max(0, canvas_height - new_h)))
        return new_x, new_y, new_w, new_h

    @staticmethod
    def _rotate_text_layer(layer: Image.Image, angle: float) -> Image.Image:
        if abs(angle) < 0.65:
            return layer
        # PIL usa grados positivos antihorarios; los polígonos OCR usan coordenadas
        # de imagen, donde un ángulo positivo se ve horario en pantalla.
        return layer.rotate(
            -float(angle),
            resample=Image.Resampling.BICUBIC,
            expand=False,
            center=(layer.width / 2.0, layer.height / 2.0),
        )

    @staticmethod
    def _vertical_character_lines(text: Any) -> List[str]:
        """Convierte una única línea en una columna de caracteres legibles."""
        normalized = re.sub(r"\s+", " ", str(text or " ").strip()) or " "
        return [char for char in normalized]

    def _layout_text_for_area(
        self,
        text: str,
        area_width: int,
        area_height: int,
        style: str,
        fixed_font_size: Optional[int],
        spacing_factor: float,
    ):
        if fixed_font_size is not None:
            font = self._get_font(fixed_font_size)
            spacing = self._line_spacing(font) * spacing_factor * (0.82 if style.startswith("onomatopeya") else 1.0)
            lines = self._split_lines(text or " ", font, area_width)
            return font, lines, spacing
        return self._fit_font(
            text or " ",
            area_width,
            area_height,
            style=style,
            line_spacing_factor=spacing_factor,
        )

    def _fit_vertical_character_stack(
        self,
        text: str,
        area_width: int,
        area_height: int,
        style: str,
        spacing_factor: float,
        *,
        fixed_font_size: Optional[int],
        initial_font_size: int,
    ):
        """Ajusta una línea como caracteres apilados sin girar los glifos."""
        lines = self._vertical_character_lines(text)
        if fixed_font_size is not None:
            font = self._get_font(fixed_font_size)
            spacing = self._line_spacing(font) * spacing_factor * (0.82 if style.startswith("onomatopeya") else 1.0)
            return font, lines, spacing

        start_size = max(self.absolute_min_font_size, min(self.max_font_size, int(initial_font_size)))
        for size in range(start_size, self.absolute_min_font_size - 1, -1):
            font = self._get_font(size)
            stroke_width = self._stroke_width_for_style(font, style)
            safe_width = max(1, area_width - stroke_width * 2)
            safe_height = max(1, area_height - stroke_width * 2)
            spacing = self._line_spacing(font) * spacing_factor * (0.82 if style.startswith("onomatopeya") else 1.0)
            if self._fits(lines, font, spacing, safe_width, safe_height):
                return font, lines, spacing

        font = self._get_font(self.absolute_min_font_size)
        spacing = self._line_spacing(font) * spacing_factor * (0.82 if style.startswith("onomatopeya") else 1.0)
        return font, lines, spacing

    @staticmethod
    def _uses_vertical_character_mode(rotation_angle: float, block_texts: Sequence[str]) -> bool:
        """Una sola línea inclinada se escribe vertical; dos o más sí se rotan.

        Cuenta las líneas del **texto de origen**, no las del ajuste tipográfico. Un globo
        estrecho parte una sola palabra en varias líneas, y ése es justo el caso donde
        apilar los caracteres se lee mejor que girar los glifos.
        """
        if abs(float(rotation_angle or 0.0)) < 0.65:
            return False
        total = 0
        for text in block_texts:
            total += len(str(text or "").splitlines()) or 1
            if total > 1:
                return False
        return total == 1

    def _stroke_width_for_style(self, fuente, style: str) -> int:
        if style.startswith("onomatopeya"):
            return max(1, min(5, int(getattr(fuente, "size", self.min_font_size) * 0.11)))
        if style == "narracion":
            return max(1, min(2, int(getattr(fuente, "size", self.min_font_size) * 0.045)))
        return max(1, min(3, int(getattr(fuente, "size", self.min_font_size) * 0.07)))

    @staticmethod
    def _normalize_text_align(value: Any) -> str:
        normalized = str(value or "center").strip().lower()
        aliases = {"izquierda": "left", "centro": "center", "derecha": "right"}
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in {"left", "center", "right"} else "center"

    @staticmethod
    def _normalize_vertical_align(value: Any) -> str:
        normalized = str(value or "middle").strip().lower()
        aliases = {"arriba": "top", "centro": "middle", "medio": "middle", "abajo": "bottom"}
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in {"top", "middle", "bottom"} else "middle"

    @staticmethod
    def _coerce_line_spacing(value: Any) -> float:
        try:
            factor = float(value if value is not None else 1.0)
        except (TypeError, ValueError):
            factor = 1.0
        if not math.isfinite(factor):
            factor = 1.0
        return max(0.55, min(2.0, factor))

    @staticmethod
    def _coerce_text_offset(value: Any) -> float:
        try:
            offset = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return max(-1000.0, min(1000.0, offset)) if math.isfinite(offset) else 0.0

    def resolve_manual_layout(
        self,
        bbox: Sequence[int],
        texto: str,
        style: str = "dialogo",
        *,
        ui_layout: Optional[Dict[str, Any]] = None,
        requested_font_size: Optional[int] = None,
        rotation_angle: Optional[float] = None,
        text_align: Optional[str] = None,
        vertical_align: Optional[str] = None,
        line_spacing_factor: Optional[float] = None,
        text_offset_x: Optional[float] = None,
        text_offset_y: Optional[float] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Resuelve las métricas exactas usadas por la UI y el guardado manual."""
        image_height = image_shape[0] if image_shape else None
        image_width = image_shape[1] if image_shape else None
        x, y, w, h = self._coerce_box(bbox, image_width, image_height)
        layout = ui_layout if isinstance(ui_layout, dict) else None
        style = str(style or (layout or {}).get("style") or "dialogo")
        stored_rotation = (layout or {}).get(
            "requested_rotation_angle",
            (layout or {}).get("rotation_angle", 0.0),
        )
        requested_rotation = self._coerce_rotation_angle(
            rotation_angle if rotation_angle is not None else stored_rotation
        )
        horizontal = self._normalize_text_align(
            text_align if text_align is not None else (layout or {}).get("text_align", "center")
        )
        vertical = self._normalize_vertical_align(
            vertical_align if vertical_align is not None else (layout or {}).get("vertical_align", "middle")
        )
        spacing_factor = self._coerce_line_spacing(
            line_spacing_factor if line_spacing_factor is not None else (layout or {}).get("line_spacing_factor", 1.0)
        )
        offset_x = self._coerce_text_offset(
            text_offset_x if text_offset_x is not None else (layout or {}).get("text_offset_x", 0.0)
        )
        offset_y = self._coerce_text_offset(
            text_offset_y if text_offset_y is not None else (layout or {}).get("text_offset_y", 0.0)
        )
        prepared_text = self._prepare_display_text(texto, style)

        raw_blocks = layout.get("blocks") if layout else None
        block_specs: List[Tuple[Tuple[int, int, int, int], str]] = []
        if isinstance(raw_blocks, list) and raw_blocks:
            _src_x, _src_y, src_w, src_h = self._layout_box_values(layout.get("bbox"), (x, y, w, h))
            block_texts = self._split_text_for_slots(prepared_text, len(raw_blocks)) if len(raw_blocks) > 1 else [prepared_text]
            for block, block_text in zip(raw_blocks, block_texts):
                if not isinstance(block, dict):
                    continue
                area = self._scale_local_box(block.get("area") or block.get("slot"), src_w, src_h, w, h)
                block_specs.append((area, block_text))

        if not block_specs:
            margin_ratio = self.inner_margin_ratio if not style.startswith("onomatopeya") else max(0.035, self.inner_margin_ratio * 0.45)
            margin_x = max(2, int(w * margin_ratio))
            margin_y = max(2, int(h * margin_ratio))
            block_specs = [((margin_x, margin_y, max(1, w - margin_x * 2), max(1, h - margin_y * 2)), prepared_text)]

        fixed_font_size = self._fixed_font_size(requested_font_size)
        base_blocks: List[Dict[str, Any]] = []
        for raw_area, block_text in block_specs:
            area_x, area_y, area_w, area_h = raw_area
            font, lines, line_spacing = self._layout_text_for_area(
                block_text,
                area_w,
                area_h,
                style,
                fixed_font_size,
                spacing_factor,
            )
            base_blocks.append({
                "raw_area": tuple(raw_area),
                "text": block_text,
                "font": font,
                "lines": list(lines),
                "line_spacing": float(line_spacing),
            })

        vertical_character_mode = self._uses_vertical_character_mode(
            requested_rotation,
            [block["text"] for block in base_blocks],
        )
        effective_rotation = 0.0 if vertical_character_mode else requested_rotation

        resolved_blocks: List[Dict[str, Any]] = []
        for base in base_blocks:
            raw_area = base["raw_area"]
            block_text = base["text"]
            if vertical_character_mode:
                area_x, area_y, area_w, area_h = raw_area
                font, lines, line_spacing = self._fit_vertical_character_stack(
                    block_text,
                    area_w,
                    area_h,
                    style,
                    spacing_factor,
                    fixed_font_size=fixed_font_size,
                    initial_font_size=int(getattr(base["font"], "size", self.min_font_size)),
                )
            elif abs(requested_rotation) >= 0.65:
                area_x, area_y, area_w, area_h = self._rotation_safe_local_box(raw_area, w, h, requested_rotation)
                font, lines, line_spacing = self._layout_text_for_area(
                    block_text,
                    area_w,
                    area_h,
                    style,
                    fixed_font_size,
                    spacing_factor,
                )
            else:
                area_x, area_y, area_w, area_h = raw_area
                font = base["font"]
                lines = base["lines"]
                line_spacing = base["line_spacing"]

            stroke_width = self._stroke_width_for_style(font, style)
            draw_area_x = area_x + stroke_width
            draw_area_y = area_y + stroke_width
            draw_area_w = max(1, area_w - stroke_width * 2)
            draw_area_h = max(1, area_h - stroke_width * 2)
            paragraph_height = self._paragraph_height(lines, font, line_spacing)
            if vertical == "top":
                cursor_y = float(draw_area_y)
            elif vertical == "bottom":
                cursor_y = float(draw_area_y + max(0, draw_area_h - paragraph_height))
            else:
                cursor_y = float(draw_area_y + max(0, (draw_area_h - paragraph_height) / 2.0))
            cursor_y += offset_y

            resolved_lines: List[Dict[str, Any]] = []
            for line in lines:
                glyph_box = font.getbbox(line or " ")
                line_height = max(1, glyph_box[3] - glyph_box[1])
                line_width = max(0, glyph_box[2] - glyph_box[0])
                if horizontal == "left":
                    cursor_x = float(draw_area_x)
                elif horizontal == "right":
                    cursor_x = float(draw_area_x + max(0, draw_area_w - line_width))
                else:
                    cursor_x = float(draw_area_x + max(0, (draw_area_w - line_width) / 2.0))
                cursor_x += offset_x
                resolved_lines.append({
                    "text": str(line),
                    "draw_x": float(cursor_x - glyph_box[0]),
                    "draw_y": float(cursor_y - glyph_box[1]),
                    "visual_x": float(cursor_x),
                    "visual_y": float(cursor_y),
                    "width": int(line_width),
                    "height": int(line_height),
                    "glyph_bbox": [int(v) for v in glyph_box],
                })
                cursor_y += line_height + line_spacing

            resolved_blocks.append({
                "area": [int(area_x), int(area_y), int(area_w), int(area_h)],
                "text": str(block_text),
                "font_size": int(getattr(font, "size", fixed_font_size or self.min_font_size)),
                "line_spacing": float(line_spacing),
                "stroke_width": int(stroke_width),
                "paragraph_height": float(paragraph_height),
                "lines": resolved_lines,
            })

        return {
            "version": 3,
            "bbox": [int(x), int(y), int(w), int(h)],
            "style": style,
            "requested_rotation_angle": round(float(requested_rotation), 3),
            "rotation_angle": round(float(effective_rotation), 3),
            "writing_mode": "vertical_chars" if vertical_character_mode else "horizontal",
            "text_align": horizontal,
            "vertical_align": vertical,
            "line_spacing_factor": float(spacing_factor),
            "text_offset_x": float(offset_x),
            "text_offset_y": float(offset_y),
            "font_path": str(self.font_path),
            "blocks": resolved_blocks,
        }

    def build_layout(
        self,
        bbox: Sequence[int],
        texto: str,
        style: str = "dialogo",
        *,
        clip_mask: Optional[np.ndarray] = None,
        requested_font_size: Optional[int] = None,
        rotation_angle: float = 0.0,
        image_shape: Optional[Tuple[int, int]] = None,
        reading_order_right_to_left: bool = False,
        text_align: str = "center",
        vertical_align: str = "middle",
        line_spacing_factor: float = 1.0,
        text_offset_x: float = 0.0,
        text_offset_y: float = 0.0,
    ) -> Dict[str, Any]:
        """Calcula la distribución usada por el render automático y la UI."""
        image_height = image_shape[0] if image_shape else None
        image_width = image_shape[1] if image_shape else None
        x, y, w, h = self._coerce_box(bbox, image_width, image_height)
        style = style or "dialogo"
        requested_rotation = self._coerce_rotation_angle(rotation_angle)
        spacing_factor = self._coerce_line_spacing(line_spacing_factor)
        texto = self._prepare_display_text(texto, style)

        split_slots = self._connected_lobe_slots_from_mask(
            clip_mask,
            w,
            h,
            style,
            right_to_left=reading_order_right_to_left,
        )
        if split_slots:
            slot_texts = self._split_text_for_slots(texto, len(split_slots))
            render_blocks = list(zip(split_slots, slot_texts))
        else:
            safe_x, safe_y, safe_w, safe_h = self._safe_text_area_from_mask(clip_mask, w, h, style)
            render_blocks = [((safe_x, safe_y, safe_w, safe_h), texto)]

        fixed_font_size = self._fixed_font_size(requested_font_size)

        def resolve_slot(slot, block_text):
            safe_x, safe_y, safe_w, safe_h = slot
            margin_ratio = self.inner_margin_ratio if not style.startswith("onomatopeya") else max(0.035, self.inner_margin_ratio * 0.45)
            margin_x = max(2, int(safe_w * margin_ratio))
            margin_y = max(2, int(safe_h * margin_ratio))
            area_x = safe_x + margin_x
            area_y = safe_y + margin_y
            area_w = max(1, safe_w - 2 * margin_x)
            area_h = max(1, safe_h - 2 * margin_y)
            font, lines, line_spacing = self._layout_text_for_area(
                block_text,
                area_w,
                area_h,
                style,
                fixed_font_size,
                spacing_factor,
            )
            return {
                "slot": tuple(slot),
                "area": (area_x, area_y, area_w, area_h),
                "text": block_text,
                "font": font,
                "lines": list(lines),
                "line_spacing": float(line_spacing),
            }

        base_blocks = [resolve_slot(slot, block_text) for slot, block_text in render_blocks]
        vertical_character_mode = self._uses_vertical_character_mode(
            requested_rotation,
            [block["text"] for block in base_blocks],
        )
        effective_rotation = 0.0 if vertical_character_mode else requested_rotation

        blocks: List[Dict[str, Any]] = []
        for base in base_blocks:
            slot = base["slot"]
            block_text = base["text"]
            if vertical_character_mode:
                safe_x, safe_y, safe_w, safe_h = slot
                area_x, area_y, area_w, area_h = base["area"]
                font, lines, line_spacing = self._fit_vertical_character_stack(
                    block_text,
                    area_w,
                    area_h,
                    style,
                    spacing_factor,
                    fixed_font_size=fixed_font_size,
                    initial_font_size=int(getattr(base["font"], "size", self.min_font_size)),
                )
            elif abs(requested_rotation) >= 0.65:
                rotated_slot = self._rotation_safe_local_box(tuple(slot), w, h, requested_rotation)
                resolved = resolve_slot(rotated_slot, block_text)
                safe_x, safe_y, safe_w, safe_h = resolved["slot"]
                area_x, area_y, area_w, area_h = resolved["area"]
                font = resolved["font"]
                lines = resolved["lines"]
                line_spacing = resolved["line_spacing"]
            else:
                safe_x, safe_y, safe_w, safe_h = slot
                area_x, area_y, area_w, area_h = base["area"]
                font = base["font"]
                lines = base["lines"]
                line_spacing = base["line_spacing"]

            blocks.append({
                "slot": [int(safe_x), int(safe_y), int(safe_w), int(safe_h)],
                "area": [int(area_x), int(area_y), int(area_w), int(area_h)],
                "text": block_text,
                "lines": list(lines),
                "font_size": int(getattr(font, "size", fixed_font_size or self.min_font_size)),
                "line_spacing": float(line_spacing),
                "stroke_width": int(self._stroke_width_for_style(font, style)),
            })

        return {
            "version": 2,
            "bbox": [int(x), int(y), int(w), int(h)],
            "style": style,
            "requested_rotation_angle": round(float(requested_rotation), 3),
            "rotation_angle": round(float(effective_rotation), 3),
            "writing_mode": "vertical_chars" if vertical_character_mode else "horizontal",
            "block_count": len(blocks),
            "blocks": blocks,
            "uses_clip_mask": clip_mask is not None,
            "text_align": self._normalize_text_align(text_align),
            "vertical_align": self._normalize_vertical_align(vertical_align),
            "line_spacing_factor": spacing_factor,
            "text_offset_x": self._coerce_text_offset(text_offset_x),
            "text_offset_y": self._coerce_text_offset(text_offset_y),
        }

    @staticmethod
    def _layout_box_values(raw_box: Any, fallback: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        try:
            values = list(raw_box or [])[:4]
            if len(values) != 4:
                raise ValueError
            x, y, w, h = [int(round(float(value))) for value in values]
            return x, y, max(1, w), max(1, h)
        except Exception:
            return fallback

    @staticmethod
    def _scale_local_box(raw_box: Any, src_w: int, src_h: int, dst_w: int, dst_h: int) -> Tuple[int, int, int, int]:
        x, y, w, h = TextRenderer._layout_box_values(raw_box, (0, 0, dst_w, dst_h))
        sx = dst_w / max(1.0, float(src_w))
        sy = dst_h / max(1.0, float(src_h))
        nx = int(round(x * sx))
        ny = int(round(y * sy))
        nw = max(1, int(round(w * sx)))
        nh = max(1, int(round(h * sy)))
        nx = max(0, min(nx, max(0, dst_w - 1)))
        ny = max(0, min(ny, max(0, dst_h - 1)))
        nw = max(1, min(nw, dst_w - nx))
        nh = max(1, min(nh, dst_h - ny))
        return nx, ny, nw, nh

    def render_with_layouts(
        self,
        imagen_limpia: np.ndarray,
        cuadros_delimitadores: Sequence[Tuple[int, int, int, int]],
        textos: Sequence[str],
        *,
        ui_layouts: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        text_styles: Optional[Sequence[str]] = None,
        font_sizes: Optional[Sequence[Optional[int]]] = None,
        rotation_angles: Optional[Sequence[Optional[float]]] = None,
        text_aligns: Optional[Sequence[Optional[str]]] = None,
        vertical_aligns: Optional[Sequence[Optional[str]]] = None,
        line_spacing_factors: Optional[Sequence[Optional[float]]] = None,
        text_offsets_x: Optional[Sequence[Optional[float]]] = None,
        text_offsets_y: Optional[Sequence[Optional[float]]] = None,
    ) -> np.ndarray:
        """Renderiza regiones manuales con una única fuente de métricas.

        Tanto la vista previa como el guardado final pasan por ``resolve_manual_layout``;
        así la fuente, el ajuste, las líneas, la alineación y los offsets son idénticos.
        """
        count = len(textos)
        def padded(values, default):
            if values is None:
                return [default] * count
            return list(values) + [default] * max(0, count - len(values))

        text_styles = padded(text_styles, "dialogo")
        font_sizes = padded(font_sizes, None)
        ui_layouts = padded(ui_layouts, None)
        rotation_angles = padded(rotation_angles, None)
        text_aligns = padded(text_aligns, None)
        vertical_aligns = padded(vertical_aligns, None)
        line_spacing_factors = padded(line_spacing_factors, None)
        text_offsets_x = padded(text_offsets_x, None)
        text_offsets_y = padded(text_offsets_y, None)

        imagen_pil = Image.fromarray(cv2.cvtColor(imagen_limpia, cv2.COLOR_BGR2RGB))
        image_width, image_height = imagen_pil.size

        for bbox, text, style, font_size, layout, rotation, h_align, v_align, spacing, offset_x, offset_y in zip(
            cuadros_delimitadores, textos, text_styles, font_sizes, ui_layouts, rotation_angles,
            text_aligns, vertical_aligns, line_spacing_factors, text_offsets_x, text_offsets_y,
        ):
            x, y, w, h = self._coerce_box(bbox, image_width, image_height)
            if w <= 2 or h <= 2:
                continue
            resolved = self.resolve_manual_layout(
                (x, y, w, h),
                text,
                style or "dialogo",
                ui_layout=layout,
                requested_font_size=font_size,
                rotation_angle=rotation if rotation is not None else (layout or {}).get(
                    "requested_rotation_angle", (layout or {}).get("rotation_angle", 0.0)
                ),
                text_align=h_align if h_align is not None else (layout or {}).get("text_align", "center"),
                vertical_align=v_align if v_align is not None else (layout or {}).get("vertical_align", "middle"),
                line_spacing_factor=spacing if spacing is not None else (layout or {}).get("line_spacing_factor", 1.0),
                text_offset_x=offset_x if offset_x is not None else (layout or {}).get("text_offset_x", 0.0),
                text_offset_y=offset_y if offset_y is not None else (layout or {}).get("text_offset_y", 0.0),
                image_shape=(image_height, image_width),
            )
            color_border, color_text = self._resolve_text_colors(imagen_limpia, x, y, w, h)
            layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            for block in resolved["blocks"]:
                font = self._get_font(int(block["font_size"]))
                for line in block["lines"]:
                    draw.text(
                        (line["draw_x"], line["draw_y"]),
                        line["text"],
                        font=font,
                        fill=self._to_rgba(color_text),
                        stroke_width=int(block["stroke_width"]),
                        stroke_fill=self._to_rgba(color_border),
                    )
            del draw
            layer = self._rotate_text_layer(layer, resolved["rotation_angle"])
            imagen_pil.paste(layer, (x, y), layer)

        return cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)

    def render(
        self,
        imagen_limpia: np.ndarray,
        cuadros_delimitadores: Sequence[Tuple[int, int, int, int]],
        textos: Sequence[str],
        text_styles: Optional[Sequence[str]] = None,
        clip_masks: Optional[Sequence[np.ndarray]] = None,
        font_sizes: Optional[Sequence[Optional[int]]] = None,
        rotation_angles: Optional[Sequence[Optional[float]]] = None,
        *,
        reading_order_right_to_left: bool = False,
    ) -> np.ndarray:
        # OpenCV trabaja en BGR; PIL trabaja en RGB.
        imagen_pil = Image.fromarray(cv2.cvtColor(imagen_limpia, cv2.COLOR_BGR2RGB))
        ancho_img, alto_img = imagen_pil.size
        if text_styles is None:
            text_styles = ["dialogo"] * len(textos)
        if clip_masks is None:
            clip_masks = [None] * len(textos)
        if font_sizes is None:
            font_sizes = [None] * len(textos)
        else:
            font_sizes = list(font_sizes) + [None] * max(0, len(textos) - len(font_sizes))
        if rotation_angles is None:
            rotation_angles = [None] * len(textos)
        else:
            rotation_angles = list(rotation_angles) + [None] * max(0, len(textos) - len(rotation_angles))

        for raw_bbox, texto, style, clip_mask, requested_font_size, requested_rotation in zip(
            cuadros_delimitadores,
            textos,
            text_styles,
            clip_masks,
            font_sizes,
            rotation_angles,
        ):
            style = style or "dialogo"
            x, y, w, h = self._coerce_box(raw_bbox, ancho_img, alto_img)
            if w <= 2 or h <= 2:
                continue

            layout = self.build_layout(
                (x, y, w, h),
                texto,
                style,
                clip_mask=clip_mask,
                requested_font_size=requested_font_size,
                rotation_angle=self._coerce_rotation_angle(requested_rotation),
                image_shape=(alto_img, ancho_img),
                reading_order_right_to_left=reading_order_right_to_left,
                line_spacing_factor=self.line_spacing_factor,
            )
            color_borde, color_texto = self._resolve_text_colors(imagen_limpia, x, y, w, h)
            capa = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(capa)

            for block in layout["blocks"]:
                area_x, area_y, area_w, area_h = [int(value) for value in block["area"]]
                fuente = self._get_font(int(block["font_size"]))
                lineas = list(block["lines"])
                espacio_entre_lineas = float(block["line_spacing"])
                stroke_width = int(block["stroke_width"])
                draw_area_x = area_x + stroke_width
                draw_area_y = area_y + stroke_width
                draw_area_w = max(1, area_w - stroke_width * 2)
                draw_area_h = max(1, area_h - stroke_width * 2)
                alto_parrafo = self._paragraph_height(lineas, fuente, espacio_entre_lineas)
                y_texto = draw_area_y + max(0, (draw_area_h - alto_parrafo) / 2)

                for linea in lineas:
                    glyph_box = fuente.getbbox(linea or " ")
                    alto_linea = max(1, glyph_box[3] - glyph_box[1])
                    ancho_linea = max(0, glyph_box[2] - glyph_box[0])
                    x_texto = draw_area_x + max(0, (draw_area_w - ancho_linea) / 2)
                    draw.text(
                        (x_texto - glyph_box[0], y_texto - glyph_box[1]),
                        linea,
                        font=fuente,
                        fill=self._to_rgba(color_texto),
                        stroke_width=stroke_width,
                        stroke_fill=self._to_rgba(color_borde),
                    )
                    y_texto += alto_linea + espacio_entre_lineas

            del draw
            capa = self._rotate_text_layer(capa, layout["rotation_angle"])

            paste_mask = capa
            if clip_mask is not None and not style.startswith("onomatopeya"):
                local = self._normalize_clip_mask(clip_mask, w, h)
                local = cv2.GaussianBlur(local, (0, 0), sigmaX=0.8, sigmaY=0.8)
                alpha = np.array(capa.getchannel("A"), dtype=np.float32)
                clipped_alpha = np.minimum(alpha, local.astype(np.float32))
                capa.putalpha(Image.fromarray(np.clip(clipped_alpha, 0, 255).astype(np.uint8)))
                paste_mask = capa

            imagen_pil.paste(capa, (x, y), paste_mask)

        return cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
