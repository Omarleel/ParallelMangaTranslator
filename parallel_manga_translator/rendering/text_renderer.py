from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

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

                fixed_font_size = None
                try:
                    if requested_font_size is not None:
                        fixed_font_size = int(round(float(requested_font_size)))
                except Exception:
                    fixed_font_size = None

                if fixed_font_size is not None and fixed_font_size > 0:
                    fixed_font_size = max(self.absolute_min_font_size, min(self.max_font_size, fixed_font_size))
                    fuente = self._get_font(fixed_font_size)
                    espacio_entre_lineas = self._line_spacing(fuente) * getattr(self, "line_spacing_factor", 1.0) * (0.82 if style.startswith("onomatopeya") else 1.0)
                    # En modo manual no se reduce la fuente para que quepa: el usuario
                    # decide el tamaño y la capa recorta de forma segura dentro de la región.
                    lineas = self._split_lines(block_text or " ", fuente, area_w)
                else:
                    fuente, lineas, espacio_entre_lineas = self._fit_font(block_text or " ", area_w, area_h, style=style)

                alto_parrafo = self._paragraph_height(lineas, fuente, espacio_entre_lineas)
                if style.startswith("onomatopeya"):
                    stroke_width = max(1, min(5, int(getattr(fuente, "size", self.min_font_size) * 0.11)))
                elif style == "narracion":
                    stroke_width = max(1, min(2, int(getattr(fuente, "size", self.min_font_size) * 0.045)))
                else:
                    stroke_width = max(1, min(3, int(getattr(fuente, "size", self.min_font_size) * 0.07)))

                y_texto = area_y + max(0, (area_h - alto_parrafo) / 2)
                for linea in lineas:
                    alto_linea = self._text_height(linea, fuente)
                    ancho_linea = self._text_width(linea, fuente)
                    x_texto = area_x + max(0, (area_w - ancho_linea) / 2)
                    draw.text(
                        (x_texto, y_texto),
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
