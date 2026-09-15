"""Localizador de texto basado en RT-DETR-v2, para usar con el detector de globos YOLO.

Por qué aquí y no como fuente de regiones
-----------------------------------------
RT-DETR localiza el texto mucho mejor que el OCR —sobre el subconjunto sin sesgo del banco
empareja 0.447/0.783/0.625/0.692 de las regiones dibujadas a mano, donde el segmentador de
globos saca 0.000 en tres de los cuatro casos— pero devuelve **cajas, no polígonos**, y la
limpieza del pipeline está construida sobre el polígono del globo. Usarlo como fuente de
regiones limpia peor.

Aquí ocupa el sitio del **localizador**, que es el rol que sí le corresponde:

    quality.region_source     -> YOLO: quién decide las regiones y da la forma del globo
    ocr.detection_engine      -> este: quién localiza el texto
    ocr.transcription_engine  -> el OCR de siempre: quién lee el texto

Eso ataca la debilidad medida de la vía YOLO sin tocar la limpieza: hoy el texto fuera de
globo sale de las cajas del OCR global, y ahí la precisión es de 0.15-0.17 frente al 0.92
de dentro del globo.

Qué NO devuelve, y por qué importa
----------------------------------
No devuelve transcripción: es un detector, no un lector. El contrato admite ese hueco —el
texto es "una pista", la lectura final la hace el motor de transcripción— pero **las reglas
de texto libre estaban escritas contando con esa pista**. Por eso `_should_keep_free_text_group`
recibe `localizer_reads_text`: sin él, `only_digits_or_symbols` sería cierto para todo y
rechazaría todas las regiones de texto libre. Ver `bubble_text_rules_mixin`.

La clase ``bubble`` del modelo no se emite: aquí se localiza **texto**, y el globo lo pone
YOLO. Emitirla duplicaría cada globo como si fuera una caja de texto.
"""

from __future__ import annotations

from typing import List

import numpy as np

from parallel_manga_translator.detection.rtdetr_text_detector import (
    RtDetrTextDetector,
    RtDetrTextDetectorWeights,
)
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.ocr.text_detection.base import (
    TextDetection,
    TextDetectionEngineBase,
    TextDetectionSettings,
)

logger = get_logger(__name__)


class RtDetrTextDetectionEngine(TextDetectionEngineBase):
    """Adapta el detector RT-DETR al contrato de localización de texto."""

    #: Devuelve rectángulos alineados a los ejes: no puede aportar inclinación. Quien
    #: estime rotación a partir de estos polígonos obtendría cero o ruido.
    EMITS_ORIENTED_POLYGONS = False
    #: Y cada detección es un bloque entero, no un fragmento de línea.
    EMITS_TEXT_BLOCKS = True

    def __init__(self, settings: TextDetectionSettings) -> None:
        super().__init__(settings)
        # Carga perezosa: elegir el motor no debe descargar 168 MB hasta que se use.
        self._detector = RtDetrTextDetector(weights=RtDetrTextDetectorWeights())

    @property
    def engine_id(self) -> str:
        return "rtdetr"

    def detect_text_boxes(self, image: np.ndarray) -> List[TextDetection]:
        if image is None or image.size == 0:
            return []
        try:
            detection = self._detector.predict(image)
        except Exception as exc:
            # No se cae el pipeline por esto: sin cajas de texto, los globos de YOLO se
            # siguen limpiando con la tinta derivada de su interior.
            logger.warning("RT-DETR falló localizando texto: %s", exc)
            return []

        salida: List[TextDetection] = []
        for caja in detection.text_boxes:
            x, y, w, h = caja.bbox
            # El contrato pide polígono, no rectángulo: cuatro esquinas en el mismo orden
            # que devuelve EasyOCR, porque `BoxGeometry.from_detection` hace boundingRect
            # sobre estos puntos y varios filtros leen la forma.
            poligono = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            # Texto vacío a propósito: este motor no transcribe. Inventar una pista aquí
            # engañaría al diccionario de onomatopeyas y a los filtros de señal textual.
            salida.append((poligono, "", float(caja.score)))

        logger.info("RT-DETR localizó %s cajas de texto.", len(salida))
        return salida
