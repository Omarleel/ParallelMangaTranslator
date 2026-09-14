"""Colores del texto rotulado: relleno, contorno y conversión a RGBA.

Vivían en `TextFittingMixin`, que por eso estaba a caballo de la costura de este paquete:
medir texto es decidir **dónde** va, elegir su color es **cómo** se pinta. Las tres son
puras —ni un atributo, ni una llamada a un hermano—, así que aquí son funciones.
"""

from __future__ import annotations

import cv2
import numpy as np

from parallel_manga_translator.config.constants import COLOR_BLANCO, COLOR_NEGRO



def resolve_text_colors(imagen_limpia: np.ndarray, x: int, y: int, w: int, h: int):
    x_margin = max(0, x - 5)
    y_margin = max(0, y - 5)
    w_margin = min(w + 10, imagen_limpia.shape[1] - x_margin)
    h_margin = min(h + 10, imagen_limpia.shape[0] - y_margin)
    region_alrededor = imagen_limpia[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]
    promedio_color = cv2.mean(region_alrededor)[:3]
    if np.mean(promedio_color) < 128:
        return COLOR_NEGRO, COLOR_BLANCO
    return COLOR_BLANCO, COLOR_NEGRO


def contorno_por_contraste(color_relleno) -> tuple:
    """Blanco o negro, el que más se separe del relleno.

    Se usa cuando se conoce el color del texto original pero no el de su contorno.
    Arrastrar el contorno del par por defecto puede dejar relleno claro sobre borde
    claro, y el rotulo desaparece.
    """
    luminancia = 0.299 * color_relleno[0] + 0.587 * color_relleno[1] + 0.114 * color_relleno[2]
    return COLOR_NEGRO if luminancia > 127 else COLOR_BLANCO


def to_rgba(color):
    if len(color) == 4:
        return color
    return tuple(color) + (255,)


__all__ = ['contorno_por_contraste', 'resolve_text_colors', 'to_rgba']
