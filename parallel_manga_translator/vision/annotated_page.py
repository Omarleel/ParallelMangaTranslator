"""Construye lo que ve el VLM: la página con las cajas numeradas y sus recortes.

Trabajo de imagen puro, sin saber de proveedores ni de prompts. Separarlo permite
comprobar en un test qué se le manda al modelo sin tocar la red.

La razón de dibujar los números sobre la página, en vez de pedirle coordenadas al
modelo, es que un VLM al que se le pregunta *dónde* está el texto se inventa cajas. Aquí
la geometría ya está resuelta por el detector y el modelo solo tiene que contestar
"el bloque 7 dice X y es narración".
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.models.processing_models import TextRegion

Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class AnnotatedPage:
    """Página anotada y recortes, ya listos para enviar."""

    image: np.ndarray
    #: (region_id, recorte) solo de las regiones que se decidió enviar aparte.
    crops: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    #: region_id en el orden de lectura, tal como aparecen numerados en la imagen.
    region_ids: List[int] = field(default_factory=list)


class PageAnnotator:
    """Dibuja las cajas numeradas y extrae recortes.

    `max_side` acota lo que se sube: una página de manga ronda 2000x2800 y mandarla
    entera multiplica coste y latencia sin que el modelo lea mejor los números.
    """

    def __init__(
        self,
        *,
        max_side: int = 1024,
        crop_max_side: int = 512,
        crop_padding: int = 8,
        box_color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 3,
    ) -> None:
        self.max_side = int(max_side)
        self.crop_max_side = int(crop_max_side)
        self.crop_padding = int(crop_padding)
        self.box_color = box_color
        self.thickness = int(thickness)

    def build(
        self,
        image: np.ndarray,
        regions: Sequence[TextRegion],
        *,
        crop_ids: Sequence[int] = (),
    ) -> AnnotatedPage:
        if image is None or getattr(image, "size", 0) == 0 or not regions:
            return AnnotatedPage(image=np.zeros((0, 0, 3), dtype=np.uint8))

        wanted_crops = {int(value) for value in crop_ids}
        crops: List[Tuple[int, np.ndarray]] = []
        ids: List[int] = []

        annotated = image.copy()
        for position, region in enumerate(regions):
            region_id = int(region.metadata.get("region_id", position + 1))
            ids.append(region_id)
            self._draw_box(annotated, region.bbox, region_id)
            if region_id in wanted_crops:
                crop = self._crop(image, region.bbox)
                if crop is not None:
                    crops.append((region_id, crop))

        return AnnotatedPage(image=self._fit(annotated, self.max_side), crops=crops, region_ids=ids)

    # -- dibujo ----------------------------------------------------------------

    def _draw_box(self, canvas: np.ndarray, box: Box, region_id: int) -> None:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.box_color, self.thickness)
        label = str(region_id)
        scale = max(0.8, min(canvas.shape[:2]) / 900.0)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        # La etiqueta va sobre un parche sólido: un número encima del dibujo se pierde,
        # y si el modelo no lee el número toda la respuesta es inservible.
        top = max(0, y - text_h - 8)
        cv2.rectangle(canvas, (x, top), (x + text_w + 10, top + text_h + 8), self.box_color, -1)
        cv2.putText(
            canvas, label, (x + 5, top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA,
        )

    def _crop(self, image: np.ndarray, box: Box) -> np.ndarray | None:
        height, width = image.shape[:2]
        x, y, w, h = [int(v) for v in box]
        x1 = max(0, x - self.crop_padding)
        y1 = max(0, y - self.crop_padding)
        x2 = min(width, x + w + self.crop_padding)
        y2 = min(height, y + h + self.crop_padding)
        if x2 <= x1 or y2 <= y1:
            return None
        return self._fit(image[y1:y2, x1:x2], self.crop_max_side)

    @staticmethod
    def _fit(image: np.ndarray, max_side: int) -> np.ndarray:
        if image is None or getattr(image, "size", 0) == 0:
            return image
        longest = max(image.shape[:2])
        if longest <= max_side or max_side <= 0:
            return image
        scale = max_side / float(longest)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def encode_data_url(image: np.ndarray, *, quality: int = 88) -> str:
    """JPEG en base64, que es lo que aceptan las APIs de visión."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("No se pudo codificar la imagen para el VLM.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")
