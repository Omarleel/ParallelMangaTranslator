"""Polígono del bloque de texto dentro de una caja, sin suponer que haya globo.

Por qué no vale ``bubble_polygon``
----------------------------------
``export_bubble_dataset.bubble_polygon`` busca el interior del globo con un relleno por
inundación desde el centro de la caja. Eso funciona cuando hay globo. Pero al mirar las
regiones que el segmentador se pierde en ``dataset_eval`` resulta que **casi ninguna es un
globo**: son columnas de texto vertical dibujadas directamente sobre el arte, sin contorno
que las encierre. Ahí el relleno no encuentra interior cerrado, se escapa e inunda el
recorte entero: medido, devolvía ``ok`` con un área de 1.40-1.53 veces la caja, o sea un
rectángulo. Y un rectángulo llena el recorte de OCR de arte vecino (CER 0.0015 -> 0.2584).

Qué hace este módulo
--------------------
Deriva el polígono de la **tinta**, no del globo, reutilizando ``TextInkMaskRefiner``, que
mide distancia de color al fondo y por tanto cubre tinta oscura, clara y de color sin
ramas de polaridad. El resultado es la envolvente convexa de los trazos, así que **está
contenida en la caja por construcción**: nunca puede ser un rectángulo mayor que ella, y
excluye las zonas de la caja donde solo hay arte.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.geometry.box_geometry import BoxGeometry
from parallel_manga_translator.models.processing_models import Box
from parallel_manga_translator.quality.text_mask_refiner import TextInkMaskRefiner

#: Componentes de tinta más pequeñas que esta fracción del área de la caja se tiran antes
#: de calcular la envolvente: son motas del arte, y una sola en una esquina la infla.
MIN_COMPONENT_RATIO = 0.0008

#: Y también se tiran las mucho más pequeñas que la componente dominante. Los glifos de un
#: mismo bloque tienen tamaños parecidos; una salpicadura del arte no. Este criterio
#: relativo es el que aguanta cuando la caja es grande y el umbral absoluto se queda corto.
MIN_COMPONENT_VS_LARGEST = 0.05

#: La tinta tiene que ocupar al menos esta fracción de la caja. Por debajo no hay texto,
#: hay ruido, y es preferible perder el rescate que emitir una región inventada.
MIN_INK_RATIO = 0.010

#: Y no puede ocuparlo casi todo: eso ya no es texto sobre fondo, es una mancha.
MAX_INK_RATIO = 0.75

#: La envolvente debe cubrir algo de la caja para ser una zona segura utilizable.
MIN_HULL_RATIO = 0.08

#: Se trabaja sobre un recorte, no sobre la página entera: el exportador llama a esto
#: ~1700 veces y las operaciones sobre máscaras del tamaño de la página lo hacían inviable.
#: El margen tiene que superar el ``BACKGROUND_RING_PX`` (12) de ``TextInkMaskRefiner``, que
#: estima el color de fondo con un anillo alrededor de la zona; si el recorte lo cortara, el
#: "fondo" se mediría sobre el propio texto.
CROP_MARGIN_PX = 24


def text_block_polygon(image: np.ndarray, box: Box) -> Tuple[Optional[np.ndarray], str]:
    """Envolvente del bloque de texto contenido en ``box``.

    Devuelve ``(None, motivo)`` cuando no se ve un bloque de texto creíble, para descartar
    la caja en lugar de emitir una región mal formada.
    """
    if image is None or image.size == 0:
        return None, "imagen vacia"

    height, width = image.shape[:2]
    x, y, w, h = BoxGeometry.clip(tuple(int(v) for v in box), width, height)
    if w <= 4 or h <= 4:
        return None, "caja degenerada"

    area_caja = float(w * h)
    x0, y0 = max(0, x - CROP_MARGIN_PX), max(0, y - CROP_MARGIN_PX)
    x1 = min(width, x + w + CROP_MARGIN_PX)
    y1 = min(height, y + h + CROP_MARGIN_PX)
    recorte = image[y0:y1, x0:x1]
    if recorte.size == 0:
        return None, "recorte vacio"

    safe = np.zeros(recorte.shape[:2], dtype=np.uint8)
    cv2.rectangle(safe, (x - x0, y - y0), (x - x0 + w, y - y0 + h), 255, -1)

    # La caja hace de zona segura y de ancla a la vez: es lo que hace el pipeline con el
    # texto libre, y `_estimate_ink_candidates` ya contempla ese caso muestreando el
    # anillo que rodea la zona para estimar el fondo.
    tinta = TextInkMaskRefiner.refine(recorte, safe, safe)
    if tinta is None or cv2.countNonZero(tinta) == 0:
        return None, "sin tinta"

    # Fuera las motas sueltas del arte antes de envolver.
    num, labels, stats, _ = cv2.connectedComponentsWithStats((tinta > 0).astype(np.uint8), 8)
    if num <= 1:
        return None, "sin tinta"
    areas = stats[1:, cv2.CC_STAT_AREA]
    minima = max(4.0, area_caja * MIN_COMPONENT_RATIO, float(areas.max()) * MIN_COMPONENT_VS_LARGEST)
    limpia = np.zeros_like(tinta)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= minima:
            limpia[labels == i] = 255

    pintados = cv2.countNonZero(limpia)
    if pintados == 0:
        return None, "solo motas"
    ratio_tinta = pintados / area_caja
    if ratio_tinta < MIN_INK_RATIO:
        return None, "tinta insuficiente"
    if ratio_tinta > MAX_INK_RATIO:
        return None, "la tinta ocupa la caja entera"

    puntos = cv2.findNonZero(limpia)
    if puntos is None or len(puntos) < 3:
        return None, "menos de tres puntos"

    hull = cv2.convexHull(puntos)
    if cv2.contourArea(hull) < area_caja * MIN_HULL_RATIO:
        return None, "envolvente demasiado pequena"

    aprox = cv2.approxPolyDP(hull, 0.006 * cv2.arcLength(hull, True), True)
    aprox = aprox.reshape(-1, 2)
    if len(aprox) < 3:
        return None, "menos de tres vertices"
    # De vuelta a coordenadas de la página: todo lo anterior corre sobre el recorte.
    return (aprox + np.array([x0, y0], dtype=np.int32)).astype(np.int32), "ok"
