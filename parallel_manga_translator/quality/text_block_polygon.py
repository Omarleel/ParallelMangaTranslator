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

import math
from typing import List, Optional, Tuple

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


def _tinta_en_caja(image: np.ndarray, box: Box):
    """Tinta limpia dentro de ``box``, en coordenadas del recorte.

    Devuelve ``(limpia, (x0, y0), area_caja, motivo)``. Es la parte común de
    `text_block_polygon` (que la envuelve) y `text_block_ink` (que la devuelve tal cual):
    tenerla una sola vez evita que las dos versiones de "qué cuenta como tinta" diverjan.
    """
    if image is None or image.size == 0:
        return None, (0, 0), 0.0, "imagen vacia"

    height, width = image.shape[:2]
    x, y, w, h = BoxGeometry.clip(tuple(int(v) for v in box), width, height)
    if w <= 4 or h <= 4:
        return None, (0, 0), 0.0, "caja degenerada"

    area_caja = float(w * h)
    x0, y0 = max(0, x - CROP_MARGIN_PX), max(0, y - CROP_MARGIN_PX)
    x1 = min(width, x + w + CROP_MARGIN_PX)
    y1 = min(height, y + h + CROP_MARGIN_PX)
    recorte = image[y0:y1, x0:x1]
    if recorte.size == 0:
        return None, (x0, y0), area_caja, "recorte vacio"

    safe = np.zeros(recorte.shape[:2], dtype=np.uint8)
    cv2.rectangle(safe, (x - x0, y - y0), (x - x0 + w, y - y0 + h), 255, -1)

    tinta = TextInkMaskRefiner.refine(recorte, safe, safe)
    if tinta is None or cv2.countNonZero(tinta) == 0:
        return None, (x0, y0), area_caja, "sin tinta"

    num, labels, stats, _ = cv2.connectedComponentsWithStats((tinta > 0).astype(np.uint8), 8)
    if num <= 1:
        return None, (x0, y0), area_caja, "sin tinta"
    areas = stats[1:, cv2.CC_STAT_AREA]
    minima = max(4.0, area_caja * MIN_COMPONENT_RATIO, float(areas.max()) * MIN_COMPONENT_VS_LARGEST)
    limpia = np.zeros_like(tinta)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= minima:
            limpia[labels == i] = 255
    if cv2.countNonZero(limpia) == 0:
        return None, (x0, y0), area_caja, "solo motas"
    return limpia, (x0, y0), area_caja, "ok"


def text_block_ink(image: np.ndarray, box: Box) -> Tuple[Optional[np.ndarray], str]:
    """Máscara de tinta del bloque, a resolución de página.

    A diferencia de `text_block_polygon`, **no** aplica los límites de proporción de tinta:
    esos existen para decidir si la envolvente es creíble como forma de región, no para
    decidir qué píxeles son tinta. Un globo negro con texto blanco llena la caja de tinta y
    su envolvente no sirve, pero su tinta sí.
    """
    limpia, (x0, y0), _area, motivo = _tinta_en_caja(image, box)
    if limpia is None:
        return None, motivo
    pagina = np.zeros(image.shape[:2], dtype=np.uint8)
    alto, ancho = limpia.shape[:2]
    pagina[y0:y0 + alto, x0:x0 + ancho] = limpia
    return pagina, "ok"


#: Glifos mínimos para que un renglón tenga dirección. Con menos no hay hilera.
INK_ANGLE_MIN_GLYPHS = 3

#: Concentración mínima de los saltos entre glifos para aceptar que hay una dirección.
INK_ANGLE_MIN_CONFIDENCE = 0.55

#: Inclinación máxima que se acepta como rotulado. Por encima se asume artefacto: un bloque
#: de líneas apiladas puede parecer una columna, y girar un globo 89° es muy visible.
INK_ANGLE_MAX_ABS = 30.0


def _vecinos_mas_proximos(centroides: np.ndarray) -> List[Tuple[float, float]]:
    """Dirección y longitud del salto de cada glifo a su vecino más próximo.

    Es el paso que hace innecesario segmentar renglones, que era donde se atascaba: dentro
    de una línea los glifos están más cerca entre sí que de la línea de al lado, así que el
    salto al vecino más próximo apunta en la dirección del texto. Y funciona igual con el
    bloque inclinado, que es justo donde la proyección horizontal se entremezclaba: a 18° un
    renglón sube más que lo que separa a dos renglones.
    """
    n = len(centroides)
    if n < 2:
        return []
    difs = centroides[:, None, :] - centroides[None, :, :]
    distancias = np.hypot(difs[:, :, 0], difs[:, :, 1])
    np.fill_diagonal(distancias, np.inf)
    salida: List[Tuple[float, float]] = []
    for i in range(n):
        j = int(np.argmin(distancias[i]))
        d = float(distancias[i, j])
        if not np.isfinite(d) or d <= 0.5:
            continue
        dx, dy = float(centroides[j][0] - centroides[i][0]), float(centroides[j][1] - centroides[i][1])
        salida.append((math.degrees(math.atan2(dy, dx)), d))
    return salida


def text_block_ink_angle(image: np.ndarray, box: Box) -> Optional[Tuple[float, float]]:
    """Inclinación del texto dentro de ``box``, medida glifo a glifo.

    Devuelve ``(angulo, confianza)`` o ``None`` si no hay evidencia creíble.

    Cuatro cosas que costaron una medición cada una, y por qué acaba midiéndose así:

    - No vale la forma de la **caja**: el eje largo de una caja alta es el vertical, y salía
      89° sobre texto horizontal.
    - No vale la forma de la **tinta** en bruto: una sola letra alta (9x23 px) daba 89°,
      porque el glifo es más alto que ancho. Un glifo no es una dirección.
    - No vale la nube de **centroides del bloque**: su eje principal es el del apilado de
      líneas, no el de lectura. Cajas de 138x125 px daban 89°.
    - Y segmentar renglones por proyección horizontal se rompe con el bloque inclinado, que
      es precisamente el caso que importa: a 18° un renglón sube más de lo que separa a dos.

    Lo que queda en pie es local: **el salto de cada glifo a su vecino más próximo**. Dentro
    de una línea los glifos están más juntos que entre líneas, así que esos saltos apuntan a
    donde va el texto, sin necesidad de saber dónde empieza cada renglón.
    """
    limpia, (_x0, _y0), _area, _motivo = _tinta_en_caja(image, box)
    if limpia is None:
        return None

    num, _etiquetas, _stats, centroides = cv2.connectedComponentsWithStats(
        (limpia > 0).astype(np.uint8), 8
    )
    puntos = centroides[1:num].astype(np.float32)
    if len(puntos) < INK_ANGLE_MIN_GLYPHS:
        return None

    saltos = _vecinos_mas_proximos(puntos)
    if len(saltos) < INK_ANGLE_MIN_GLYPHS:
        return None

    # Media axial: el salto al vecino puede ir hacia delante o hacia atrás, y 180° aparte es
    # la misma recta. El peso es la longitud del salto.
    x = sum(peso * math.cos(math.radians(ang * 2.0)) for ang, peso in saltos)
    y = sum(peso * math.sin(math.radians(ang * 2.0)) for ang, peso in saltos)
    total = max(1e-8, sum(peso for _, peso in saltos))
    confianza = min(1.0, math.hypot(x, y) / total)
    if confianza < INK_ANGLE_MIN_CONFIDENCE:
        # Saltos que apuntan a todas partes: no hay una dirección, hay una nube.
        return None
    angulo = math.degrees(math.atan2(y, x)) / 2.0
    angulo = ((angulo + 90.0) % 180.0) - 90.0
    if abs(angulo) > INK_ANGLE_MAX_ABS:
        return None
    return float(angulo), float(confianza)


def text_block_polygon(image: np.ndarray, box: Box) -> Tuple[Optional[np.ndarray], str]:
    """Envolvente del bloque de texto contenido en ``box``.

    Devuelve ``(None, motivo)`` cuando no se ve un bloque de texto creíble, para descartar
    la caja en lugar de emitir una región mal formada.
    """
    limpia, (x0, y0), area_caja, motivo = _tinta_en_caja(image, box)
    if limpia is None:
        return None, motivo

    pintados = cv2.countNonZero(limpia)
    ratio_tinta = pintados / max(1.0, area_caja)
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
