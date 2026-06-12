from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from parallel_manga_translator.config.constants import COLOR_BLANCO, COLOR_NEGRO, FACTOR_ESPACIO, RUTA_FUENTE, TAMANIO_MINIMO_FUENTE


class TextFittingMixin:
    """Normalización, ajuste de texto y decisión de tamaño de fuente."""

    @staticmethod
    def _normalize_text(texto: str) -> str:
        texto = str(texto or " ").replace("\r", "\n")
        texto = re.sub(r"[ \t\f\v]+", " ", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        return texto.strip() or " "

    def _break_long_token(self, token: str, fuente, max_width: int) -> List[str]:
        """Parte palabras muy largas para que no atraviesen la caja."""
        if self._text_width(token, fuente) <= max_width:
            return [token]

        partes: List[str] = []
        actual = ""
        for char in token:
            candidata = actual + char
            if actual and self._text_width(candidata, fuente) > max_width:
                partes.append(actual)
                actual = char
            else:
                actual = candidata
        if actual:
            partes.append(actual)
        return partes or [token]

    def _split_lines(self, texto: str, fuente, max_width: int) -> List[str]:
        max_width = max(1, int(max_width))
        texto = self._normalize_text(texto)
        lineas: List[str] = []

        for parrafo in texto.split("\n"):
            palabras = parrafo.split()
            if not palabras:
                if lineas:
                    lineas.append("")
                continue

            tokens: List[str] = []
            for palabra in palabras:
                tokens.extend(self._break_long_token(palabra, fuente, max_width))

            linea_actual = ""
            for token in tokens:
                candidata = token if not linea_actual else f"{linea_actual} {token}"
                if linea_actual and self._text_width(candidata, fuente) > max_width:
                    lineas.append(linea_actual)
                    linea_actual = token
                else:
                    linea_actual = candidata
            if linea_actual:
                lineas.append(linea_actual)

        return lineas or [" "]

    def _paragraph_height(self, lineas: Sequence[str], fuente, espacio_entre_lineas: float) -> float:
        if not lineas:
            return 0.0
        alturas = [self._text_height(linea, fuente) for linea in lineas]
        return float(sum(alturas) + max(0, len(lineas) - 1) * espacio_entre_lineas)

    def _fits(self, lineas: Sequence[str], fuente, espacio: float, box_width: int, box_height: int) -> bool:
        if not lineas:
            return True
        ancho_maximo = max((self._text_width(linea, fuente) for linea in lineas), default=0)
        alto = self._paragraph_height(lineas, fuente, espacio)
        return ancho_maximo <= box_width and alto <= box_height

    def _ellipsis_line(self, linea: str, fuente, max_width: int) -> str:
        ellipsis = "…"
        linea = linea.strip()
        if self._text_width(linea + ellipsis, fuente) <= max_width:
            return linea + ellipsis
        while linea and self._text_width(linea + ellipsis, fuente) > max_width:
            linea = linea[:-1].rstrip()
        return (linea + ellipsis) if linea else ellipsis

    def _truncate_to_fit(self, lineas: List[str], fuente, box_width: int, box_height: int, espacio: float) -> List[str]:
        """Último recurso: evita desbordes si ni la fuente mínima alcanza."""
        if not lineas:
            return [" "]

        linea_base = max(1, self._text_height("Ag", fuente))
        max_lineas = max(1, int((box_height + espacio) // (linea_base + espacio)))
        recortadas = list(lineas[:max_lineas])

        if len(lineas) > max_lineas:
            recortadas[-1] = self._ellipsis_line(recortadas[-1], fuente, box_width)

        while self._paragraph_height(recortadas, fuente, espacio) > box_height and len(recortadas) > 1:
            recortadas.pop()
            recortadas[-1] = self._ellipsis_line(recortadas[-1], fuente, box_width)

        recortadas = [self._ellipsis_line(linea, fuente, box_width) if self._text_width(linea, fuente) > box_width else linea for linea in recortadas]
        return recortadas or [" "]

    def _fit_font(self, texto: str, box_width: int, box_height: int, style: str = "dialogo"):
        box_width = max(1, int(box_width))
        box_height = max(1, int(box_height))
        texto = self._normalize_text(texto)

        if style.startswith("onomatopeya"):
            # Los efectos de sonido suelen ocupar más espacio visual y admiten letras grandes.
            factor = 0.82 if style == "onomatopeya_subtitle" else 0.92
            start_size = min(self.max_font_size, max(self.absolute_min_font_size, int(box_height * factor)))
        elif style == "narracion":
            start_size = min(self.max_font_size, max(self.absolute_min_font_size, int(box_height * 0.58)))
        else:
            start_size = min(self.max_font_size, max(self.absolute_min_font_size, int(box_height * 0.70)))

        # Intento principal: fuente más grande que quepa.
        for tamanio in range(start_size, self.absolute_min_font_size - 1, -1):
            fuente = self._get_font(tamanio)
            espacio = self._line_spacing(fuente) * (0.82 if style.startswith("onomatopeya") else 1.0)
            lineas = self._split_lines(texto, fuente, box_width)
            if self._fits(lineas, fuente, espacio, box_width, box_height):
                return fuente, lineas, espacio

        # Último recurso: fuente mínima absoluta + recorte seguro.
        fuente = self._get_font(self.absolute_min_font_size)
        espacio = self._line_spacing(fuente) * (0.82 if style.startswith("onomatopeya") else 1.0)
        lineas = self._split_lines(texto, fuente, box_width)
        lineas = self._truncate_to_fit(lineas, fuente, box_width, box_height, espacio)
        return fuente, lineas, espacio

    @staticmethod
    def _resolve_text_colors(imagen_limpia: np.ndarray, x: int, y: int, w: int, h: int):
        x_margin = max(0, x - 5)
        y_margin = max(0, y - 5)
        w_margin = min(w + 10, imagen_limpia.shape[1] - x_margin)
        h_margin = min(h + 10, imagen_limpia.shape[0] - y_margin)
        region_alrededor = imagen_limpia[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]
        promedio_color = cv2.mean(region_alrededor)[:3]
        if np.mean(promedio_color) < 128:
            return COLOR_NEGRO, COLOR_BLANCO
        return COLOR_BLANCO, COLOR_NEGRO

    @staticmethod
    def _to_rgba(color):
        if len(color) == 4:
            return color
        return tuple(color) + (255,)

    @staticmethod
    def _prepare_display_text(texto: str, style: str) -> str:
        texto = str(texto or " ").strip() or " "
        if not style.startswith("onomatopeya"):
            return texto
        if style == "onomatopeya_subtitle" and "\n" in texto:
            original, subtitle = texto.split("\n", 1)
            if re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", subtitle):
                subtitle = subtitle.upper()
            return original.strip() + "\n" + subtitle.strip()
        # En alfabetos latinos las onomatopeyas suelen leerse mejor en mayúsculas.
        if re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", texto):
            texto = texto.upper()
        return texto
