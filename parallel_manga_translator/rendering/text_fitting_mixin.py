from __future__ import annotations

import re
from typing import List, Sequence

import cv2
import numpy as np

from parallel_manga_translator.config.constants import COLOR_BLANCO, COLOR_NEGRO


class TextFittingMixin:
    """Normalización, ajuste de texto y decisión de tamaño de fuente."""

    @staticmethod
    def _normalize_text(texto: str) -> str:
        texto = str(texto or " ").replace("\r", "\n")
        texto = re.sub(r"[ \t\f\v]+", " ", texto)
        return texto.strip() or " "

    @staticmethod
    def _soft_hyphen_points(token: str) -> List[int]:
        """Puntos de corte ligeros para alfabetos latinos, sin dependencia externa."""
        if len(token) < 9 or not re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", token):
            return []
        vowels = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
        points: List[int] = []
        for i in range(3, len(token) - 3):
            prev_c = token[i - 1]
            curr_c = token[i]
            if prev_c in vowels and curr_c not in vowels or prev_c not in vowels and curr_c in vowels and i >= 4:
                points.append(i)
        # Prefiere cortes cerca del centro: menos líneas huérfanas.
        center = len(token) / 2.0
        return sorted(set(points), key=lambda idx: abs(idx - center))

    def _break_long_token(self, token: str, fuente, max_width: int) -> List[str]:
        """Parte palabras muy largas para que no atraviesen la caja."""
        if self._text_width(token, fuente) <= max_width:
            return [token]

        if getattr(self, "hyphenation", True) and getattr(self, "smart_typography", True):
            for split_at in self._soft_hyphen_points(token):
                head = token[:split_at].rstrip() + "-"
                tail = token[split_at:].lstrip()
                if self._text_width(head, fuente) <= max_width and self._text_width(tail, fuente) <= max_width:
                    return [head, tail]

        partes: List[str] = []
        actual = ""
        for char in token:
            candidata = actual + char
            if actual and self._text_width(candidata, fuente) > max_width:
                suffix = "-" if getattr(self, "hyphenation", True) and re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]$", actual) else ""
                partes.append(actual + suffix)
                actual = char
            else:
                actual = candidata
        if actual:
            partes.append(actual)
        return partes or [token]

    def _balance_wrapped_lines(self, lineas: List[str], fuente, max_width: int) -> List[str]:
        if not getattr(self, "balance_lines", True) or not getattr(self, "smart_typography", True):
            return lineas
        if len(lineas) < 2 or len(lineas) > 5:
            return lineas

        balanced = list(lineas)
        for _ in range(8):
            changed = False
            widths = [self._text_width(linea, fuente) for linea in balanced]
            if not widths:
                break
            avg = sum(widths) / len(widths)
            for idx in range(len(balanced) - 1):
                words = balanced[idx].split()
                if len(words) <= 1:
                    continue
                # Si una línea es muy larga y la siguiente muy corta, mueve la última palabra.
                if widths[idx] > avg * 1.18 or widths[idx + 1] < avg * 0.62:
                    moved = words[-1]
                    candidate_a = " ".join(words[:-1])
                    candidate_b = (moved + " " + balanced[idx + 1]).strip()
                    if candidate_a and self._text_width(candidate_a, fuente) <= max_width and self._text_width(candidate_b, fuente) <= max_width:
                        balanced[idx] = candidate_a
                        balanced[idx + 1] = candidate_b
                        changed = True
                        break
            if not changed:
                break
        return balanced

    def _unbreakable_tokens(self, texto: str) -> List[str]:
        """Devuelve palabras que deben caber completas en modo automático.

        El ajuste automático debe reducir la fuente antes de cortar una palabra.
        Solo en el último recurso se permite partir tokens extremos.
        """
        texto = self._normalize_text(texto)
        tokens: List[str] = []
        for parrafo in texto.split("\n"):
            tokens.extend(parrafo.split())
        return tokens or [" "]

    def _all_words_fit(self, texto: str, fuente, max_width: int) -> bool:
        max_width = max(1, int(max_width))
        return all(self._text_width(token, fuente) <= max_width for token in self._unbreakable_tokens(texto))

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

        lineas = lineas or [" "]
        return self._balance_wrapped_lines(lineas, fuente, max_width)

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

    def _fit_font(self, texto: str, box_width: int, box_height: int, style: str = "dialogo", line_spacing_factor: float | None = None):
        box_width = max(1, int(box_width))
        box_height = max(1, int(box_height))
        texto = self._normalize_text(texto)
        spacing_factor = getattr(self, "line_spacing_factor", 1.0) if line_spacing_factor is None else max(0.55, min(2.0, float(line_spacing_factor)))

        if style.startswith("onomatopeya"):
            # Los efectos de sonido suelen ocupar más espacio visual y admiten letras grandes.
            factor = 0.82 if style == "onomatopeya_subtitle" else 0.92
            start_size = min(self.max_font_size, max(self.absolute_min_font_size, int(box_height * factor)))
        elif style == "narracion":
            start_size = min(self.max_font_size, max(self.absolute_min_font_size, int(box_height * 0.58)))
        else:
            start_size = min(self.max_font_size, max(self.absolute_min_font_size, int(box_height * 0.70)))

        # Intento principal: fuente más grande que quepa sin cortar palabras.
        # La comprobación reserva espacio para el borde del texto, porque PIL puede
        # pintar algunos píxeles fuera del bbox tipográfico cuando hay stroke.
        for tamanio in range(start_size, self.absolute_min_font_size - 1, -1):
            fuente = self._get_font(tamanio)
            stroke_width = int(self._stroke_width_for_style(fuente, style)) if hasattr(self, "_stroke_width_for_style") else 0
            safe_width = max(1, box_width - stroke_width * 2)
            safe_height = max(1, box_height - stroke_width * 2)
            if not self._all_words_fit(texto, fuente, safe_width):
                continue
            espacio = self._line_spacing(fuente) * spacing_factor * (0.82 if style.startswith("onomatopeya") else 1.0)
            lineas = self._split_lines(texto, fuente, safe_width)
            if self._fits(lineas, fuente, espacio, safe_width, safe_height):
                return fuente, lineas, espacio

        # Último recurso: fuente mínima absoluta. Antes de partir tokens extremos,
        # vuelve a intentar una composición por palabras completas para evitar
        # que textos normales se vean entrecortados.
        fuente = self._get_font(self.absolute_min_font_size)
        stroke_width = int(self._stroke_width_for_style(fuente, style)) if hasattr(self, "_stroke_width_for_style") else 0
        safe_width = max(1, box_width - stroke_width * 2)
        safe_height = max(1, box_height - stroke_width * 2)
        espacio = self._line_spacing(fuente) * spacing_factor * (0.82 if style.startswith("onomatopeya") else 1.0)
        if self._all_words_fit(texto, fuente, safe_width):
            lineas = self._split_lines(texto, fuente, safe_width)
            if self._fits(lineas, fuente, espacio, safe_width, safe_height):
                return fuente, lineas, espacio
        lineas = self._split_lines(texto, fuente, safe_width)
        lineas = self._truncate_to_fit(lineas, fuente, safe_width, safe_height, espacio)
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
    def _contorno_por_contraste(color_relleno) -> tuple:
        """Blanco o negro, el que más se separe del relleno.

        Se usa cuando se conoce el color del texto original pero no el de su contorno.
        Arrastrar el contorno del par por defecto puede dejar relleno claro sobre borde
        claro, y el rotulo desaparece.
        """
        luminancia = 0.299 * color_relleno[0] + 0.587 * color_relleno[1] + 0.114 * color_relleno[2]
        return COLOR_NEGRO if luminancia > 127 else COLOR_BLANCO

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
