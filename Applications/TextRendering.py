from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Utils.Constantes import COLOR_BLANCO, COLOR_NEGRO, FACTOR_ESPACIO, RUTA_FUENTE, TAMANIO_MINIMO_FUENTE


class TextRenderer:
    def __init__(self, font_path: str = RUTA_FUENTE, min_font_size: int = TAMANIO_MINIMO_FUENTE) -> None:
        self.font_path = font_path
        self.min_font_size = min_font_size

    @lru_cache(maxsize=128)
    def _get_font(self, size: int):
        safe_size = max(self.min_font_size, int(size))
        return ImageFont.truetype(self.font_path, safe_size)

    @staticmethod
    def _text_width(texto: str, fuente) -> int:
        bbox = fuente.getbbox(texto)
        return bbox[2] - bbox[0]

    @staticmethod
    def _text_height(texto: str, fuente) -> int:
        bbox = fuente.getbbox(texto)
        return bbox[3] - bbox[1]

    @staticmethod
    def _line_spacing(fuente) -> float:
        return min(TAMANIO_MINIMO_FUENTE + 2, fuente.size * FACTOR_ESPACIO)

    def _split_lines(self, texto: str, fuente, max_width: int):
        palabras = texto.split()
        if not palabras:
            return [""]

        lineas = []
        linea_actual = palabras[0]
        for palabra in palabras[1:]:
            candidata = f"{linea_actual} {palabra}".strip()
            if self._text_width(candidata, fuente) > max_width:
                lineas.append(linea_actual)
                linea_actual = palabra
            else:
                linea_actual = candidata
        lineas.append(linea_actual)
        return lineas

    def _paragraph_height(self, lineas, fuente, espacio_entre_lineas: float) -> float:
        if not lineas:
            return 0
        return sum(self._text_height(linea, fuente) for linea in lineas) + max(0, len(lineas) - 1) * espacio_entre_lineas

    def _fit_font(self, texto: str, box_width: int, box_height: int):
        mejor_fuente = self._get_font(self.min_font_size)
        mejor_lineas = self._split_lines(texto, mejor_fuente, max(1, box_width))
        mejor_espacio = self._line_spacing(mejor_fuente)

        for tamanio in range(self.min_font_size, 101):
            fuente = self._get_font(tamanio)
            espacio = self._line_spacing(fuente)
            lineas = self._split_lines(texto, fuente, max(1, box_width))
            alto = self._paragraph_height(lineas, fuente, espacio)
            ancho_maximo = max((self._text_width(linea, fuente) for linea in lineas), default=0)

            if alto <= box_height and ancho_maximo <= box_width:
                mejor_fuente = fuente
                mejor_lineas = lineas
                mejor_espacio = espacio
            else:
                break

        fuente = mejor_fuente
        lineas = mejor_lineas
        espacio = mejor_espacio

        while any(self._text_width(linea, fuente) > box_width for linea in lineas) and fuente.size > self.min_font_size:
            fuente = self._get_font(fuente.size - 1)
            espacio = self._line_spacing(fuente)
            lineas = self._split_lines(texto, fuente, max(1, box_width))

        while self._paragraph_height(lineas, fuente, espacio) > box_height and fuente.size > self.min_font_size:
            fuente = self._get_font(fuente.size - 1)
            espacio = self._line_spacing(fuente)
            lineas = self._split_lines(texto, fuente, max(1, box_width))

        return fuente, lineas, espacio

    @staticmethod
    def _resolve_text_colors(imagen_limpia: np.ndarray, x: int, y: int, w: int, h: int):
        x_margin = max(0, x - 5)
        y_margin = max(0, y - 5)
        w_margin = min(w + 5, imagen_limpia.shape[1] - x_margin)
        h_margin = min(h + 5, imagen_limpia.shape[0] - y_margin)
        region_alrededor = imagen_limpia[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]
        promedio_color = cv2.mean(region_alrededor)[:3]
        if np.mean(promedio_color) < 128:
            return COLOR_NEGRO, COLOR_BLANCO
        return COLOR_BLANCO, COLOR_NEGRO

    def render(self, imagen_limpia: np.ndarray, cuadros_delimitadores: Sequence[Tuple[int, int, int, int]], textos: Sequence[str]) -> np.ndarray:
        imagen_pil = Image.fromarray(imagen_limpia)
        draw = ImageDraw.Draw(imagen_pil)

        for (x, y, w, h), texto in zip(cuadros_delimitadores, textos):
            fuente, lineas, espacio_entre_lineas = self._fit_font(texto or " ", max(1, w), max(1, h))
            alto_parrafo = self._paragraph_height(lineas, fuente, espacio_entre_lineas)
            color_borde, color_texto = self._resolve_text_colors(imagen_limpia, x, y, w, h)

            margen_superior = max(0, (h - alto_parrafo) / 2)
            y_texto = y + margen_superior
            desplazamiento_bordes = 5

            for linea in lineas:
                alto_linea = self._text_height(linea, fuente)
                ancho_linea = self._text_width(linea, fuente)
                x_texto = x + max(0, (w - ancho_linea) // 2)

                for j in range(desplazamiento_bordes):
                    for dx, dy in [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)]:
                        draw.text((x_texto + j * dx, y_texto + j * dy), linea, font=fuente, fill=color_borde)
                draw.text((x_texto, y_texto), linea, font=fuente, fill=color_texto)
                y_texto += alto_linea + espacio_entre_lineas

        del draw
        return np.array(imagen_pil)