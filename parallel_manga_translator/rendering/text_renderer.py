from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from parallel_manga_translator.config.constants import RUTA_FUENTE, TAMANIO_MINIMO_FUENTE


from parallel_manga_translator.rendering.text_layout_engine import TextLayoutEngine
from parallel_manga_translator.rendering.text_colors import (
    contorno_por_contraste,
    resolve_text_colors,
    to_rgba,
)
from parallel_manga_translator.rendering.mask_text_area import normalize_clip_mask
class TextRenderer:
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
        # Una sola fuente de verdad: la configuración vive en el motor, no copiada aquí.
        # Duplicarla es como se rompió antes `bubble_fill_strategy`, en silencio.
        self.layout = TextLayoutEngine(
            font_path=font_path,
            min_font_size=min_font_size,
            absolute_min_font_size=absolute_min_font_size,
            max_font_size=max_font_size,
            inner_margin_ratio=inner_margin_ratio,
            smart_typography=smart_typography,
            hyphenation=hyphenation,
            balance_lines=balance_lines,
            line_spacing_factor=line_spacing_factor,
        )

    @property
    def font_path(self) -> str:
        """Lo consulta la UI para decir si la fuente está disponible."""
        return self.layout.font_path

    @property
    def line_spacing_factor(self) -> float:
        return self.layout.line_spacing_factor

    def build_layout(self, *args, **kwargs):
        """Fachada del motor: es API de la UI y del pipeline, no se mueve de sitio."""
        return self.layout.build_layout(*args, **kwargs)

    def resolve_manual_layout(self, *args, **kwargs):
        """Fachada del motor: la usa el editor manual al recomponer una página."""
        return self.layout.resolve_manual_layout(*args, **kwargs)





























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
            x, y, w, h = self.layout._coerce_box(bbox, image_width, image_height)
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
            color_border, color_text = resolve_text_colors(imagen_limpia, x, y, w, h)
            layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            for block in resolved["blocks"]:
                font = self.layout._get_font(int(block["font_size"]))
                for line in block["lines"]:
                    draw.text(
                        (line["draw_x"], line["draw_y"]),
                        line["text"],
                        font=font,
                        fill=to_rgba(color_text),
                        stroke_width=int(block["stroke_width"]),
                        stroke_fill=to_rgba(color_border),
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
        text_colors: Optional[Sequence[Optional[Tuple[int, int, int]]]] = None,
        stroke_colors: Optional[Sequence[Optional[Tuple[int, int, int]]]] = None,
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

        def _rellenar(valores):
            if valores is None:
                return [None] * len(textos)
            return list(valores) + [None] * max(0, len(textos) - len(valores))

        text_colors = _rellenar(text_colors)
        stroke_colors = _rellenar(stroke_colors)

        for (
            raw_bbox,
            texto,
            style,
            clip_mask,
            requested_font_size,
            requested_rotation,
            color_relleno_pedido,
            color_contorno_pedido,
        ) in zip(
            cuadros_delimitadores,
            textos,
            text_styles,
            clip_masks,
            font_sizes,
            rotation_angles,
            text_colors,
            stroke_colors,
        ):
            style = style or "dialogo"
            x, y, w, h = self.layout._coerce_box(raw_bbox, ancho_img, alto_img)
            if w <= 2 or h <= 2:
                continue

            layout = self.build_layout(
                (x, y, w, h),
                texto,
                style,
                clip_mask=clip_mask,
                requested_font_size=requested_font_size,
                rotation_angle=self.layout._coerce_rotation_angle(requested_rotation),
                image_shape=(alto_img, ancho_img),
                reading_order_right_to_left=reading_order_right_to_left,
                line_spacing_factor=self.line_spacing_factor,
            )
            color_borde, color_texto = resolve_text_colors(imagen_limpia, x, y, w, h)
            if color_relleno_pedido is not None:
                # Color estimado del original. El contorno estimado sólo se usa si el
                # estimador lo afirmó; si no, se elige por contraste contra el relleno, que
                # es más seguro que arrastrar el del par por defecto (un relleno claro con
                # borde claro deja el rótulo invisible).
                color_texto = tuple(int(c) for c in color_relleno_pedido)
                color_borde = (
                    tuple(int(c) for c in color_contorno_pedido)
                    if color_contorno_pedido is not None
                    else contorno_por_contraste(color_texto)
                )
            capa = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(capa)

            for block in layout["blocks"]:
                area_x, area_y, area_w, area_h = [int(value) for value in block["area"]]
                fuente = self.layout._get_font(int(block["font_size"]))
                lineas = list(block["lines"])
                espacio_entre_lineas = float(block["line_spacing"])
                stroke_width = int(block["stroke_width"])
                draw_area_x = area_x + stroke_width
                draw_area_y = area_y + stroke_width
                draw_area_w = max(1, area_w - stroke_width * 2)
                draw_area_h = max(1, area_h - stroke_width * 2)
                alto_parrafo = self.layout._paragraph_height(lineas, fuente, espacio_entre_lineas)
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
                        fill=to_rgba(color_texto),
                        stroke_width=stroke_width,
                        stroke_fill=to_rgba(color_borde),
                    )
                    y_texto += alto_linea + espacio_entre_lineas

            del draw
            capa = self._rotate_text_layer(capa, layout["rotation_angle"])

            paste_mask = capa
            if clip_mask is not None and not style.startswith("onomatopeya"):
                local = normalize_clip_mask(clip_mask, w, h)
                local = cv2.GaussianBlur(local, (0, 0), sigmaX=0.8, sigmaY=0.8)
                alpha = np.array(capa.getchannel("A"), dtype=np.float32)
                clipped_alpha = np.minimum(alpha, local.astype(np.float32))
                capa.putalpha(Image.fromarray(np.clip(clipped_alpha, 0, 255).astype(np.uint8)))
                paste_mask = capa

            imagen_pil.paste(capa, (x, y), paste_mask)

        return cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
