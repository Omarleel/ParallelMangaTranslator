from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from parallel_manga_translator.config.constants import COLOR_BLANCO, COLOR_NEGRO, FACTOR_ESPACIO, RUTA_FUENTE, TAMANIO_MINIMO_FUENTE


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

    def _stroke_width_for_style(self, fuente, style: str) -> int:
        if style.startswith("onomatopeya"):
            return max(1, min(5, int(getattr(fuente, "size", self.min_font_size) * 0.11)))
        if style == "narracion":
            return max(1, min(2, int(getattr(fuente, "size", self.min_font_size) * 0.045)))
        return max(1, min(3, int(getattr(fuente, "size", self.min_font_size) * 0.07)))

    def build_layout(
        self,
        bbox: Sequence[int],
        texto: str,
        style: str = "dialogo",
        *,
        clip_mask: Optional[np.ndarray] = None,
        requested_font_size: Optional[int] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        reading_order_right_to_left: bool = False,
    ) -> Dict[str, Any]:
        """Calcula la distribución de texto que usa el renderizador.

        La UI guarda este resultado como `ui_layout` para que la edición manual parta
        de la imagen renderizada como fuente de verdad. En especial conserva las
        ranuras internas cuando una sola región lógica contiene dos globos unidos.
        """
        image_height = image_shape[0] if image_shape else None
        image_width = image_shape[1] if image_shape else None
        x, y, w, h = self._coerce_box(bbox, image_width, image_height)
        style = style or "dialogo"
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
        blocks: List[Dict[str, Any]] = []
        for (safe_x, safe_y, safe_w, safe_h), block_text in render_blocks:
            margin_ratio = self.inner_margin_ratio if not style.startswith("onomatopeya") else max(0.035, self.inner_margin_ratio * 0.45)
            margen_x = max(2, int(safe_w * margin_ratio))
            margen_y = max(2, int(safe_h * margin_ratio))
            area_x = safe_x + margen_x
            area_y = safe_y + margen_y
            area_w = max(1, safe_w - 2 * margen_x)
            area_h = max(1, safe_h - 2 * margen_y)

            if fixed_font_size is not None:
                fuente = self._get_font(fixed_font_size)
                espacio_entre_lineas = self._line_spacing(fuente) * getattr(self, "line_spacing_factor", 1.0) * (0.82 if style.startswith("onomatopeya") else 1.0)
                lineas = self._split_lines(block_text or " ", fuente, area_w)
            else:
                fuente, lineas, espacio_entre_lineas = self._fit_font(block_text or " ", area_w, area_h, style=style)

            blocks.append(
                {
                    "slot": [int(safe_x), int(safe_y), int(safe_w), int(safe_h)],
                    "area": [int(area_x), int(area_y), int(area_w), int(area_h)],
                    "text": block_text,
                    "lines": list(lineas),
                    "font_size": int(getattr(fuente, "size", fixed_font_size or self.min_font_size)),
                    "line_spacing": float(espacio_entre_lineas),
                    "stroke_width": int(self._stroke_width_for_style(fuente, style)),
                }
            )

        return {
            "version": 1,
            "bbox": [int(x), int(y), int(w), int(h)],
            "style": style,
            "block_count": len(blocks),
            "blocks": blocks,
            "uses_clip_mask": clip_mask is not None,
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
    ) -> np.ndarray:
        """Renderiza usando layouts congelados por la UI cuando existen.

        Si una región no trae layout válido se delega en `render`, manteniendo el
        comportamiento anterior. Con layout válido se respetan ranuras y áreas
        internas calculadas durante el render automático original.
        """
        if text_styles is None:
            text_styles = ["dialogo"] * len(textos)
        if font_sizes is None:
            font_sizes = [None] * len(textos)
        else:
            font_sizes = list(font_sizes) + [None] * max(0, len(textos) - len(font_sizes))
        if ui_layouts is None:
            ui_layouts = [None] * len(textos)
        else:
            ui_layouts = list(ui_layouts) + [None] * max(0, len(textos) - len(ui_layouts))

        imagen_pil = Image.fromarray(cv2.cvtColor(imagen_limpia, cv2.COLOR_BGR2RGB))
        ancho_img, alto_img = imagen_pil.size

        fallback_boxes: List[Tuple[int, int, int, int]] = []
        fallback_texts: List[str] = []
        fallback_styles: List[str] = []
        fallback_sizes: List[Optional[int]] = []

        for bbox, texto, style, requested_font_size, layout in zip(cuadros_delimitadores, textos, text_styles, font_sizes, ui_layouts):
            x, y, w, h = self._coerce_box(bbox, ancho_img, alto_img)
            if w <= 2 or h <= 2:
                continue
            blocks = layout.get("blocks") if isinstance(layout, dict) else None
            if not isinstance(blocks, list) or not blocks:
                fallback_boxes.append((x, y, w, h))
                fallback_texts.append(texto)
                fallback_styles.append(style or "dialogo")
                fallback_sizes.append(requested_font_size)
                continue

            style = style or layout.get("style") or "dialogo"
            prepared_text = self._prepare_display_text(texto, style)
            src_x, src_y, src_w, src_h = self._layout_box_values(layout.get("bbox"), (x, y, w, h))
            block_texts = self._split_text_for_slots(prepared_text, len(blocks)) if len(blocks) > 1 else [prepared_text]
            color_borde, color_texto = self._resolve_text_colors(imagen_limpia, x, y, w, h)
            capa = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(capa)

            fixed_font_size = self._fixed_font_size(requested_font_size)
            for block, block_text in zip(blocks, block_texts):
                area_x, area_y, area_w, area_h = self._scale_local_box(block.get("area") or block.get("slot"), src_w, src_h, w, h)
                if fixed_font_size is not None:
                    fuente = self._get_font(fixed_font_size)
                    espacio_entre_lineas = self._line_spacing(fuente) * getattr(self, "line_spacing_factor", 1.0) * (0.82 if style.startswith("onomatopeya") else 1.0)
                    lineas = self._split_lines(block_text or " ", fuente, area_w)
                else:
                    fuente, lineas, espacio_entre_lineas = self._fit_font(block_text or " ", area_w, area_h, style=style)

                alto_parrafo = self._paragraph_height(lineas, fuente, espacio_entre_lineas)
                stroke_width = self._stroke_width_for_style(fuente, style)
                draw_area_x = area_x + stroke_width
                draw_area_y = area_y + stroke_width
                draw_area_w = max(1, area_w - stroke_width * 2)
                draw_area_h = max(1, area_h - stroke_width * 2)
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
            imagen_pil.paste(capa, (x, y), capa)

        result = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
        if fallback_boxes:
            result = self.render(
                result,
                fallback_boxes,
                fallback_texts,
                text_styles=fallback_styles,
                font_sizes=fallback_sizes,
            )
        return result

    def render(
        self,
        imagen_limpia: np.ndarray,
        cuadros_delimitadores: Sequence[Tuple[int, int, int, int]],
        textos: Sequence[str],
        text_styles: Optional[Sequence[str]] = None,
        clip_masks: Optional[Sequence[np.ndarray]] = None,
        font_sizes: Optional[Sequence[Optional[int]]] = None,
        *,
        reading_order_right_to_left: bool = False,
    ) -> np.ndarray:
        # OpenCV trabaja en BGR; PIL trabaja en RGB. Convertir explícitamente evita
        # desplazamientos de color en páginas a color.
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

        for (x, y, w, h), texto, style, clip_mask, requested_font_size in zip(cuadros_delimitadores, textos, text_styles, clip_masks, font_sizes):
            style = style or "dialogo"
            texto = self._prepare_display_text(texto, style)
            x = int(max(0, x))
            y = int(max(0, y))
            w = int(min(max(1, w), ancho_img - x))
            h = int(min(max(1, h), alto_img - y))
            if w <= 2 or h <= 2:
                continue

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

            color_borde, color_texto = self._resolve_text_colors(imagen_limpia, x, y, w, h)

            # Dibujamos en una capa del tamaño exacto del globo/caja y la pegamos con máscara.
            # Así se garantiza que ningún píxel de texto quede fuera de la región asignada.
            capa = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(capa)

            for (safe_x, safe_y, safe_w, safe_h), block_text in render_blocks:
                # Margen interno: en diálogos protege esquinas de globos ovalados; en SFX se reduce
                # para que el efecto pueda ocupar más del espacio detectado. Con clip_mask, el margen
                # se aplica sobre la caja interior real del globo, no sobre la bbox rectangular completa.
                margin_ratio = self.inner_margin_ratio if not style.startswith("onomatopeya") else max(0.035, self.inner_margin_ratio * 0.45)
                margen_x = max(2, int(safe_w * margin_ratio))
                margen_y = max(2, int(safe_h * margin_ratio))
                area_x = safe_x + margen_x
                area_y = safe_y + margen_y
                area_w = max(1, safe_w - 2 * margen_x)
                area_h = max(1, safe_h - 2 * margen_y)

                fixed_font_size = self._fixed_font_size(requested_font_size)

                if fixed_font_size is not None:
                    fuente = self._get_font(fixed_font_size)
                    espacio_entre_lineas = self._line_spacing(fuente) * getattr(self, "line_spacing_factor", 1.0) * (0.82 if style.startswith("onomatopeya") else 1.0)
                    # En modo manual no se reduce la fuente para que quepa: el usuario
                    # decide el tamaño y la capa recorta de forma segura dentro de la región.
                    lineas = self._split_lines(block_text or " ", fuente, area_w)
                else:
                    fuente, lineas, espacio_entre_lineas = self._fit_font(block_text or " ", area_w, area_h, style=style)

                alto_parrafo = self._paragraph_height(lineas, fuente, espacio_entre_lineas)
                stroke_width = self._stroke_width_for_style(fuente, style)
                draw_area_x = area_x + stroke_width
                draw_area_y = area_y + stroke_width
                draw_area_w = max(1, area_w - stroke_width * 2)
                draw_area_h = max(1, area_h - stroke_width * 2)
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

            paste_mask = capa
            if clip_mask is not None and not style.startswith("onomatopeya"):
                local = self._normalize_clip_mask(clip_mask, w, h)
                # Suaviza un poco el recorte para que el texto no quede serruchado cerca del borde.
                local = cv2.GaussianBlur(local, (0, 0), sigmaX=0.8, sigmaY=0.8)
                alpha = np.array(capa.getchannel("A"), dtype=np.float32)
                clipped_alpha = np.minimum(alpha, local.astype(np.float32))
                capa.putalpha(Image.fromarray(np.clip(clipped_alpha, 0, 255).astype(np.uint8)))
                paste_mask = capa

            imagen_pil.paste(capa, (x, y), paste_mask)

        return cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
