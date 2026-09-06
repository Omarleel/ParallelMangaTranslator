"""Color real de la tinta y del contorno del texto original.

Qué había antes
---------------
`_resolve_text_colors` mira la imagen **ya limpia** alrededor de la caja, calcula el brillo
medio y devuelve blanco-sobre-negro o negro-sobre-blanco. Dos problemas: decide sobre una
imagen de la que el texto original ya fue borrado, y sólo sabe elegir entre dos extremos.
Un manga a color, un rótulo rojo o un SFX con contorno de color se rotulan igual: negro
plano sobre blanco plano.

Qué hace esto
-------------
Estima el color mirando la imagen **original** a través de la máscara de tinta que el
pipeline ya calcula por región (`clean_mask` / `text_mask`, poblada en 119 de 121 regiones
medidas sobre `dataset_eval/ja_01`).

- **Relleno**: mediana por canal de los píxeles de tinta. Mediana y no media porque el
  antialias de los bordes arrastra la media hacia el fondo.
- **Contorno**: mediana del anillo inmediatamente exterior a la tinta. Sólo se acepta si
  se separa **a la vez** del relleno y del fondo; si se parece al fondo es que no hay
  contorno, sólo el papel.

Cuando no hay señal suficiente devuelve `None` y quien llama se queda con la regla de
contraste de siempre. Preferir un fallback conocido a inventar un color es deliberado: un
color mal estimado se ve en la página, y esto **no lo cubre el banco de pruebas**, que se
detiene antes de traducir y rotular.

Nota de formato: OpenCV trabaja en BGR y PIL dibuja en RGB. Los colores salen de aquí ya
en **RGB**, listos para `ImageDraw.text(fill=...)`.

Sobre el contorno, medido
-------------------------
En `dataset_eval/ja_01` **no se afirma ni un solo contorno en 121 regiones**, y está
comprobado que es correcto, no un umbral mal puesto. De las 50 regiones con anillo
medible, la separación entre el anillo y el fondo tiene mediana 0.0 y **máximo 17.3**;
con un umbral de 20 —un tercio del actual— seguirían pasando cero. Ese tomo no tiene texto
perfilado.

La señal contraria confirma que el criterio discrimina: la separación entre el anillo y el
relleno llega a 438 sobre un máximo teórico de 441. Anillo muy distinto de la tinta e
idéntico al fondo es exactamente la firma de "no hay contorno, es papel".

Queda por tanto **sin ejercitar sobre datos reales**: haría falta un tomo a color o con
SFX perfilados para saber si acierta cuando sí hay contorno. El test sintético cubre el
caso, la realidad no.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]

#: Mínimo de píxeles de tinta para fiarse de una mediana.
MIN_PIXELES_TINTA = 40

#: Mínimo de píxeles en el anillo exterior para estimar el contorno.
MIN_PIXELES_ANILLO = 30

#: Separación mínima (distancia euclídea en RGB, 0-441) para considerar dos colores
#: distintos. 60 deja pasar un contorno blanco sobre tinta negra y descarta las
#: variaciones de compresión JPEG dentro de una misma zona plana.
SEPARACION_MINIMA = 60.0

#: Grosor del anillo exterior, en píxeles. Un contorno de rotulación típico ronda los 2-4.
GROSOR_ANILLO = 3

#: A qué distancia se muestrea el fondo, para no confundirlo con el propio contorno.
DISTANCIA_FONDO = 9


@dataclass(frozen=True)
class ColoresDeTexto:
    """Colores estimados de una región, en RGB."""

    relleno: Color
    contorno: Optional[Color]
    #: Cuántos píxeles de tinta sostienen la estimación del relleno.
    pixeles_tinta: int
    #: Diagnóstico: cuánto se separaba el anillo exterior del relleno y del fondo, aunque
    #: se haya rechazado. Sirve para saber si un contorno no se afirma porque no lo hay o
    #: porque el umbral es demasiado estricto. `None` si no había anillo medible.
    separacion_relleno: Optional[float] = None
    separacion_fondo: Optional[float] = None

    def as_tuple(self) -> Tuple[Color, Optional[Color]]:
        return self.relleno, self.contorno


def _mediana_rgb(imagen_bgr: np.ndarray, seleccion: np.ndarray) -> Optional[Color]:
    pixeles = imagen_bgr[seleccion > 0]
    if pixeles.size == 0:
        return None
    b, g, r = (int(np.median(pixeles[:, i])) for i in range(3))
    return (r, g, b)


def _separacion(a: Color, b: Color) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def estimar_colores(
    imagen_original: np.ndarray,
    mascara_tinta: np.ndarray,
    *,
    zona_segura: Optional[np.ndarray] = None,
    separacion_minima: float = SEPARACION_MINIMA,
) -> Optional[ColoresDeTexto]:
    """Colores del texto original, o `None` si no hay señal para afirmarlos.

    `mascara_tinta` y `zona_segura` deben tener el tamaño de `imagen_original`.
    """
    if imagen_original is None or imagen_original.ndim != 3 or mascara_tinta is None:
        return None
    if mascara_tinta.shape[:2] != imagen_original.shape[:2]:
        return None

    tinta = (mascara_tinta > 0).astype(np.uint8)
    if cv2.countNonZero(tinta) < MIN_PIXELES_TINTA:
        return None

    # Se erosiona para quedarse con el interior del trazo: los bordes están mezclados con
    # el fondo por el antialias y sesgan la mediana hacia el papel.
    nucleo = cv2.erode(tinta, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    if cv2.countNonZero(nucleo) < MIN_PIXELES_TINTA:
        nucleo = tinta

    relleno = _mediana_rgb(imagen_original, nucleo)
    if relleno is None:
        return None
    pixeles_tinta = int(cv2.countNonZero(nucleo))

    anillo = cv2.subtract(
        cv2.dilate(tinta, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GROSOR_ANILLO * 2 + 1,) * 2)),
        tinta,
    )
    fondo = cv2.subtract(
        cv2.dilate(tinta, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DISTANCIA_FONDO * 2 + 1,) * 2)),
        cv2.dilate(tinta, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DISTANCIA_FONDO,) * 2)),
    )
    if zona_segura is not None and zona_segura.shape[:2] == imagen_original.shape[:2]:
        segura = (zona_segura > 0).astype(np.uint8)
        anillo = cv2.bitwise_and(anillo, segura)
        fondo = cv2.bitwise_and(fondo, segura)

    contorno: Optional[Color] = None
    sep_relleno: Optional[float] = None
    sep_fondo: Optional[float] = None
    if cv2.countNonZero(anillo) >= MIN_PIXELES_ANILLO:
        candidato = _mediana_rgb(imagen_original, anillo)
        color_fondo = _mediana_rgb(imagen_original, fondo)
        if candidato is not None:
            # Un contorno de verdad se separa del relleno Y del fondo. Si sólo se separa
            # del relleno, lo que se ha medido es el papel, no un contorno.
            sep_relleno = _separacion(candidato, relleno)
            sep_fondo = _separacion(candidato, color_fondo) if color_fondo is not None else None
            if sep_relleno >= separacion_minima and (sep_fondo is None or sep_fondo >= separacion_minima):
                contorno = candidato

    return ColoresDeTexto(
        relleno=relleno,
        contorno=contorno,
        pixeles_tinta=pixeles_tinta,
        separacion_relleno=sep_relleno,
        separacion_fondo=sep_fondo,
    )


__all__ = ["Color", "ColoresDeTexto", "estimar_colores"]
