"""Fuente de regiones basada en el detector RT-DETR-v2 de texto y globos.

Ocupa el mismo sitio que `BubbleDetector` y `CtdRegionSource` —`CleanManga` la recibe
inyectada— y por eso expone los mismos cuatro métodos que el orquestador consume.

De dónde sale cada máscara
--------------------------
    YOLO       globo segmentado -> zona segura; la tinta se deriva dentro del globo.
    CTD        bloque de texto  -> zona segura = la caja; la tinta la da el modelo.
    RT-DETR    bloque de texto  -> dentro de globo, zona segura = el interior del globo;
                                   fuera, el polígono de la tinta. La tinta se deriva
                                   aquí y se pasa como semilla.

**La caja del detector no sirve como zona de búsqueda de tinta, y esto costó un fallo
visible.** Medido sobre `ja_02`: la caja de texto de este modelo cubre solo el 0.66 del
bloque real y el 81 % son más pequeñas que él, así que derivar la tinta dentro de ella
dejaba columnas enteras sin borrar dentro de los globos. La caja del **globo** sí lo cubre
(cobertura 1.00) pero incluye el contorno y las esquinas con arte: usarla tal cual disparó
`bubble_edge_damage` de 4 a 23 y en una página se llevó el dibujo por delante. Lo que
funciona es el **interior** del globo, que excluye el contorno por construcción. Ver
`_bubble_interior`.

La diferencia con CTD importa y está medida. Este modelo da **cajas, no máscaras**, y una
caja como zona segura es exactamente lo que rompió el OCR en los detectores de cajas que
se probaron antes: `_masked_region_crop` blanquea lo que cae fuera de `region.mask`, así
que con un rectángulo el recorte se llena de arte vecino (CER 0.0015 -> 0.2584). Derivar
la envolvente de la tinta dentro de la caja con `text_block_polygon` es lo que arregló eso
en su día (CER 0.2584 -> 0.0231), y es lo que se hace aquí.

La tinta sí se calcula aquí, y esto costó un fallo real. La estrategia de limpieza, para
una región `dialogue`, deduce la zona de texto de `text_bbox`; pero en esta vía `text_bbox`
**es** `bbox`, así que acababa usando un rectángulo expandido como zona de texto y dejaba
media tinta sin borrar dentro de los globos. Pasar `text_mask`/`clean_mask` con la tinta
real quita esa deducción. Y `detections_count` va a **0** porque es la verdad —aquí no hay
localizador de texto por OCR— y es justo la condición con la que esa deducción se desactiva.

Qué no hace todavía
-------------------
El modelo también predice la clase ``bubble`` (el contorno del globo). No se emite como
región —duplicaría cada globo— pero sí se mide su solape con cada bloque de texto y se
guarda en ``metadata["bubble_overlap"]``. Ese es el insumo de la regla espacial
dentro/fuera del futuro `HybridRegionSource`; aquí no se usa para reclasificar porque el
umbral necesita calibrarse con datos y el proyecto ya tiene varias cicatrices de umbrales
elegidos sobre dos casos del banco.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.architecture.ports import RegionSourcePort  # noqa: F401  (contrato documentado)
from parallel_manga_translator.config.app_config import ProcessingConfig, QualityConfig
from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.detection.region_reading_order import order_and_number
from parallel_manga_translator.detection.rtdetr_text_detector import (
    CLASS_TEXT_BUBBLE,
    RtDetrBox,
    RtDetrDetection,
    RtDetrTextDetector,
)
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.layout.panel_order_resolver import PanelAwareReadingOrderResolver, PanelOrderConfig
from parallel_manga_translator.layout.reading_order_resolver import ReadingOrderResolver
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.quality.export_bubble_dataset import FLOOD_AREA_RATIO, bubble_polygon
from parallel_manga_translator.quality.text_block_polygon import text_block_ink, text_block_polygon

logger = get_logger(__name__)

Box = Tuple[int, int, int, int]

#: Misma convención que `CtdRegionSource`: es un *prior* estructural, no la clasificación
#: semántica fina (dialogue/thought/narration/sfx), que la hace luego el VLM.
RTDETR_CLASS_KINDS = {CLASS_TEXT_BUBBLE: "dialogue"}
DEFAULT_KIND = "free_text"


class RtDetrRegionSource:
    """Adapta la salida del detector RT-DETR al dominio del pipeline."""

    def __init__(
        self,
        idioma_entrada: str = "Japonés",
        quality_config: QualityConfig | None = None,
        processing_config: ProcessingConfig | None = None,
        detector: RtDetrTextDetector | None = None,
    ) -> None:
        quality_config = quality_config if quality_config is not None else QualityConfig()
        processing_config = processing_config if processing_config is not None else ProcessingConfig()
        self.idioma_entrada = idioma_entrada
        self.quality_config = quality_config
        self.processing_config = processing_config
        # Inyectable: los tests no deberían necesitar 168 MB de pesos para comprobar cómo
        # se construyen las regiones.
        self.detector = detector if detector is not None else RtDetrTextDetector(
            weights=None,
            conf_threshold=float(getattr(quality_config, "rtdetr_text_conf", 0.30)),
            bubble_conf_threshold=float(getattr(quality_config, "rtdetr_bubble_conf", 0.50)),
        )
        self.derive_polygon = bool(getattr(quality_config, "rtdetr_text_polygon", True))
        self.bubble_search_zone = bool(getattr(quality_config, "rtdetr_bubble_search_zone", True))
        self.bubble_search_min_overlap = float(getattr(quality_config, "rtdetr_bubble_search_min_overlap", 0.80))
        self.reading_order_resolver = ReadingOrderResolver(idioma_entrada)
        self.panel_order_resolver = PanelAwareReadingOrderResolver(
            self.reading_order_resolver,
            PanelOrderConfig.from_quality_config(quality_config),
        )
        self._debug_page_number: Optional[int] = None
        self._debug_source_filename: Optional[str] = None
        self._debug_output_filename: Optional[str] = None

    # -- contexto de depuración (mismo contrato que BubbleDetector) -------------

    def set_debug_page_context(
        self,
        page_index: int,
        *,
        source_filename: str | None = None,
        output_filename: str | None = None,
    ) -> None:
        self._debug_page_number = max(1, int(page_index) + 1)
        self._debug_source_filename = source_filename
        self._debug_output_filename = output_filename

    def clear_debug_page_context(self) -> None:
        self._debug_page_number = None
        self._debug_source_filename = None
        self._debug_output_filename = None

    # -- composición de máscaras -----------------------------------------------
    # Funciones puras sobre regiones: reusar las de `BubbleDetector` evita que las vías
    # compongan la página de formas distintas.

    compose_mask = staticmethod(BubbleDetector.compose_mask)
    compose_clean_mask = staticmethod(BubbleDetector.compose_clean_mask)

    # -- detección --------------------------------------------------------------

    def detect_primary_bubble_regions(self, image: np.ndarray) -> List[TextRegion]:
        detection = self.detector.predict(image)
        regions = self._regions_from_detection(image, detection)
        logger.info(
            "RT-DETR: %s regiones (%s en globo, %s fuera)",
            len(regions),
            sum(1 for r in regions if r.kind == "dialogue"),
            sum(1 for r in regions if r.kind != "dialogue"),
        )
        return regions

    def build_regions_from_bubbles_and_text(
        self,
        image: np.ndarray,
        bubble_regions: Sequence[TextRegion],
        detections: Sequence,
    ) -> List[TextRegion]:
        """Ordena y numera las regiones ya detectadas.

        `detections` son las cajas del OCR global y aquí se ignoran **a propósito**: que
        el OCR deje de crear regiones es el objetivo de esta vía, no un efecto lateral.
        Se registra cuántas se descartan para que el silencio no parezca un olvido.
        """
        if detections:
            logger.debug(
                "RT-DETR es fuente única: %s cajas del OCR global no crean regiones.",
                len(detections),
            )
        return order_and_number(
            image,
            list(bubble_regions),
            panel_resolver=self.panel_order_resolver,
            reading_resolver=self.reading_order_resolver,
            idioma_entrada=self.idioma_entrada,
        )

    # -- construcción interna ---------------------------------------------------

    def _regions_from_detection(self, image: np.ndarray, detection: RtDetrDetection) -> List[TextRegion]:
        height, width = image.shape[:2]
        regions: List[TextRegion] = []
        sin_poligono = 0

        for caja in detection.text_boxes:
            x, y, w, h = caja.bbox
            if w <= 1 or h <= 1:
                continue

            # Dónde buscar la tinta. NO es la caja de la región: medido sobre `ja_02`, la
            # caja de texto de este modelo cubre solo el 0.66 del bloque real (el 81 % son
            # más pequeñas que el bloque), así que derivar la tinta dentro de ella deja
            # columnas enteras sin borrar. El globo que el propio modelo predice sí lo
            # cubre entero (cobertura 1.00) y sale gratis, en el mismo forward.
            zona_busqueda, zona_origen = self._search_zone(image, caja, detection.bubble_boxes)

            if zona_origen == "bubble_interior":
                safe_zone, mask_source = zona_busqueda, "bubble_interior"
                caja_tinta = self._mask_bounds(zona_busqueda)
            else:
                safe_zone, mask_source = self._safe_zone(image, zona_busqueda)
                caja_tinta = zona_busqueda
            if safe_zone is None:
                continue
            if mask_source == "detector_box":
                sin_poligono += 1

            # La tinta real del bloque, no su envolvente. Sin esto la estrategia de limpieza
            # tiene que adivinarla desde `text_bbox`, y como aquí `text_bbox == bbox` acaba
            # usando un RECTÁNGULO expandido como zona de texto: era la máscara cuadrada que
            # dejaba media tinta sin borrar dentro de los globos.
            tinta, _motivo_tinta = text_block_ink(image, caja_tinta)
            if tinta is not None:
                tinta = cv2.bitwise_and(tinta, safe_zone)

            kind = RTDETR_CLASS_KINDS.get(caja.label, DEFAULT_KIND)
            regions.append(
                TextRegion(
                    bbox=(x, y, w, h),
                    text_bbox=(x, y, w, h),
                    mask=safe_zone,
                    kind=kind,
                    confidence=float(caja.score),
                    source_text_hint="",
                    # Cero de verdad: aquí no hubo localizador de texto por OCR. Ponerlo a 1
                    # hacía que `bubble_text_zone` se saltara su propia guarda para el caso
                    # "no hay cajas OCR" y devolviera un rectángulo.
                    detections_count=0,
                    metadata={
                        "region_source": "rtdetr",
                        "rtdetr_class": caja.label,
                        "rtdetr_score": float(caja.score),
                        # Clase estructural del modelo, antes de que el VLM afine el tipo.
                        "structural_kind": "speech_bubble" if kind == "dialogue" else "out_of_bubble",
                        "region_mask_shape": mask_source,
                        # Insumo de la regla espacial dentro/fuera del futuro híbrido.
                        "bubble_overlap": self._bubble_overlap(caja.bbox, detection.bubble_boxes),
                        "ink_pixels": int(cv2.countNonZero(tinta)) if tinta is not None else 0,
                        # De dónde salió la zona donde se buscó la tinta: la caja del
                        # detector, o el globo que la contiene.
                        "search_zone": zona_origen,
                    },
                    text_boxes=[(x, y, w, h)],
                    # Semilla fina y máscara de limpieza de partida. La estrategia las refina
                    # y puede ampliarlas; lo que no puede es inventarlas.
                    text_mask=tinta,
                    clean_mask=tinta,
                )
            )

        if sin_poligono:
            # Un rectángulo como zona segura llena el recorte de OCR de arte vecino. Que
            # el porcentaje sea visible es lo que permite decidir si conviene descartar
            # esas cajas en vez de emitirlas.
            logger.info(
                "RT-DETR: %s de %s bloques sin polígono de tinta; se usa su caja.",
                sin_poligono,
                len(regions),
            )
        return regions

    def _search_zone(self, image: np.ndarray, caja: RtDetrBox, bubbles: Sequence[RtDetrBox]):
        """Dónde buscar la tinta de este bloque. Devuelve ``(zona, origen)``.

        Si el bloque cae holgadamente dentro de un globo, manda el **interior del globo**,
        no su caja. Los dos detalles importan y están medidos sobre `ja_02`:

        - La caja de texto de este modelo cubre solo el 0.66 del bloque real (el 81 % son
          más pequeñas), así que buscar la tinta dentro de ella deja columnas sin borrar.
        - Pero la **caja** del globo incluye su contorno y las esquinas con arte, y la
          tinta del contorno se borra como si fuera texto: probado, el verificador pasó de
          4 a 23 `bubble_edge_damage` y en una página se llevó por delante el dibujo.

        `bubble_polygon` da el interior, que excluye el contorno por construcción. Cuando
        se inunda —no encuentra interior cerrado— se vuelve a la caja de texto, que es el
        comportamiento conservador.

        Esto **no** cambia `region.bbox`: la caja de texto sigue siendo la identidad de la
        región, y es la que localiza mejor (F1 0.767 frente a 0.730 de la del globo).
        """
        if self.bubble_search_zone:
            globo = self._containing_bubble(caja, bubbles)
            if globo is not None:
                mascara = self._bubble_interior(image, globo.bbox, caja.bbox)
                if mascara is not None:
                    return mascara, "bubble_interior"
                # Segundo intento: ensanchar la caja de texto dentro del globo. No hace
                # falta saber la forma exacta del globo para saber que la caja se queda
                # corta —cubre el 0.66 del bloque— y que el interior del globo es sitio
                # seguro. El globo se encoge antes de acotar para no rozar su contorno.
                ampliada = self._expanded_inside_bubble(caja.bbox, globo.bbox, image.shape)
                if ampliada is not None:
                    return ampliada, "expanded_in_bubble"
                # Respaldo: el relleno por inundación del exportador. Acierta en pocos
                # casos (23 de 78 medidos) pero cuando acierta da un contorno más ceñido.
                poligono, _motivo = bubble_polygon(image, globo.bbox)
                if poligono is not None and self._area_ratio(poligono, globo.bbox) < FLOOD_AREA_RATIO:
                    relleno = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(relleno, [poligono.astype(np.int32)], 255)
                    if cv2.countNonZero(relleno) > 0:
                        return relleno, "bubble_interior"
        return caja.bbox, "detector_box"

    #: El interior tiene que ocupar una parte razonable de la caja del globo. Por debajo es
    #: que el relleno se quedó en un hueco entre trazos; por encima, que se comió el
    #: contorno y estamos otra vez con la caja.
    INTERIOR_MIN_RATIO = 0.25
    INTERIOR_MAX_RATIO = 0.94

    def _bubble_interior(self, image: np.ndarray, bubble_box: Box, text_box: Box) -> Optional[np.ndarray]:
        """Interior del globo: lo que el contorno encierra.

        Se inunda **desde fuera**, no desde dentro, y esa es toda la diferencia. El contorno
        de un globo es una curva cerrada, así que un relleno lanzado desde el borde del
        recorte se detiene en él: lo que no alcanza es el interior. Al revés —sembrar dentro
        y crecer— hay que puentear los glifos con un cierre morfológico, y ese cierre también
        puentea el contorno: medido, fundía interior y exterior en una sola componente en
        **30 de 78** globos, con ratio 1.000.

        `bubble_polygon` falla por la misma razón por la que fallaba antes: inunda desde el
        centro de la caja y se escapa por las colas del globo.
        """
        height, width = image.shape[:2]
        bx, by, bw, bh = bubble_box
        if bw <= 8 or bh <= 8:
            return None
        # Margen: el relleno necesita exterior desde donde partir. Sin él, una caja ceñida
        # al globo deja el borde del recorte ya dentro del globo y se inunda el interior.
        margen = max(4, int(round(min(bw, bh) * 0.08)))
        x0, y0 = max(0, bx - margen), max(0, by - margen)
        x1, y1 = min(width, bx + bw + margen), min(height, by + bh + margen)
        recorte = image[y0:y1, x0:x1]
        if recorte.size == 0:
            return None

        gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        _umbral, claro = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # El interior puede ser la capa clara (globo blanco) o la oscura (globo negro con
        # texto blanco). No se asume: se prueban las dos y gana la que deje un interior con
        # tamaño razonable que además contenga al bloque de texto.
        tx, ty, tw, th = text_box
        sx = int(round(tx + tw / 2)) - x0
        sy = int(round(ty + th / 2)) - y0
        if not (0 <= sx < recorte.shape[1] and 0 <= sy < recorte.shape[0]):
            return None

        area_caja = float(bw * bh)
        mejor = None
        for capa in (claro, cv2.bitwise_not(claro)):
            interior = self._enclosed_by_outline(capa)
            if interior is None:
                continue
            if interior[sy, sx] == 0:
                # La semilla cae en un glifo: los glifos son agujeros del interior y ya se
                # rellenan dentro de `_enclosed_by_outline`, así que esto es un interior
                # que sencillamente no contiene al texto.
                continue
            ratio = cv2.countNonZero(interior) / max(1.0, area_caja)
            if not (self.INTERIOR_MIN_RATIO <= ratio <= self.INTERIOR_MAX_RATIO):
                continue
            if mejor is None or cv2.countNonZero(interior) > cv2.countNonZero(mejor):
                mejor = interior

        if mejor is None:
            return None
        pagina = np.zeros((height, width), dtype=np.uint8)
        pagina[y0:y1, x0:x1] = mejor
        return pagina

    @staticmethod
    def _enclosed_by_outline(capa: np.ndarray) -> Optional[np.ndarray]:
        """Lo que la capa encierra: se inunda desde el borde y se devuelve lo no alcanzado.

        Incluye el propio contorno, así que se erosiona para dejarlo fuera: borrar el borde
        negro del globo destroza la viñeta, y es exactamente lo que pasó al usar la caja.
        """
        alto, ancho = capa.shape[:2]
        relleno = capa.copy()
        mascara = np.zeros((alto + 2, ancho + 2), np.uint8)
        # Desde las cuatro esquinas: basta una que caiga fuera del globo.
        for semilla in ((0, 0), (ancho - 1, 0), (0, alto - 1), (ancho - 1, alto - 1)):
            if relleno[semilla[1], semilla[0]] > 0:
                cv2.floodFill(relleno, mascara, semilla, 0)
        exterior = cv2.bitwise_and(capa, cv2.bitwise_not(relleno))
        encerrado = cv2.bitwise_not(exterior)
        if cv2.countNonZero(encerrado) == 0:
            return None
        # Se erosiona para dejar fuera el propio contorno: borrarlo destroza la viñeta.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        interior = cv2.erode(encerrado, kernel, iterations=1)
        return interior if cv2.countNonZero(interior) > 0 else None

    #: Cuanto se ensancha la caja de texto cuando no se pudo derivar el interior. Sale de
    #: medir: la caja cubre el 0.66 del bloque real, con ancho 0.81 y alto 0.87.
    EXPAND_X = 0.25
    EXPAND_Y = 0.15
    #: Y cuanto se encoge el globo antes de acotar, para no rozar su contorno.
    BUBBLE_SHRINK = 0.06

    def _expanded_inside_bubble(self, text_box: Box, bubble_box: Box, shape) -> Optional[Box]:
        """Caja de texto ensanchada, recortada al interior aproximado del globo."""
        height, width = shape[:2]
        tx, ty, tw, th = text_box
        dx, dy = int(round(tw * self.EXPAND_X)), int(round(th * self.EXPAND_Y))
        ex0, ey0 = tx - dx, ty - dy
        ex1, ey1 = tx + tw + dx, ty + th + dy

        bx, by, bw, bh = bubble_box
        mx, my = int(round(bw * self.BUBBLE_SHRINK)), int(round(bh * self.BUBBLE_SHRINK))
        bx0, by0, bx1, by1 = bx + mx, by + my, bx + bw - mx, by + bh - my

        x0, y0 = max(0, max(ex0, bx0)), max(0, max(ey0, by0))
        x1, y1 = min(width, min(ex1, bx1)), min(height, min(ey1, by1))
        if x1 - x0 <= 4 or y1 - y0 <= 4:
            return None
        # Si no gana nada sobre la caja original, no merece cambiar de vía.
        if (x1 - x0) * (y1 - y0) <= tw * th:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def _containing_bubble(self, caja: RtDetrBox, bubbles: Sequence[RtDetrBox]) -> Optional[RtDetrBox]:
        mejor, mejor_solape = None, 0.0
        for bubble in bubbles or ():
            solape = self._contained_fraction(caja.bbox, bubble.bbox)
            if solape > mejor_solape:
                mejor, mejor_solape = bubble, solape
        return mejor if mejor_solape >= self.bubble_search_min_overlap else None

    @staticmethod
    def _area_ratio(polygon: np.ndarray, box: Box) -> float:
        """Área del polígono relativa a la de su caja. Delata un relleno desbordado."""
        x, y, w, h = box
        return float(cv2.contourArea(polygon.astype(np.int32))) / max(1.0, float(w * h))

    @staticmethod
    def _mask_bounds(mask: np.ndarray) -> Box:
        puntos = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(puntos)
        return (int(x), int(y), int(w), int(h))

    @staticmethod
    def _contained_fraction(inner: Box, outer: Box) -> float:
        """Fracción de ``inner`` que cae dentro de ``outer``."""
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        w = max(0, min(ix + iw, ox + ow) - max(ix, ox))
        h = max(0, min(iy + ih, oy + oh) - max(iy, oy))
        return (w * h) / max(1, iw * ih)

    def _safe_zone(self, image: np.ndarray, box: Box) -> Tuple[Optional[np.ndarray], str]:
        """Zona segura de la región: el polígono de la tinta, o la caja si no se deriva."""
        height, width = image.shape[:2]
        if self.derive_polygon:
            poligono, motivo = text_block_polygon(image, box)
            if poligono is not None and len(poligono) >= 3:
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [poligono.astype(np.int32)], 255)
                if cv2.countNonZero(mask) > 0:
                    return mask, "ink_polygon"
            logger.debug("RT-DETR: sin polígono para %s (%s); se usa la caja.", box, motivo)

        x, y, w, h = box
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        return mask, "detector_box"

    @staticmethod
    def _bubble_overlap(box: Box, bubbles: Sequence[RtDetrBox]) -> float:
        """Fracción del bloque de texto cubierta por el globo que más lo cubre."""
        x, y, w, h = box
        area = float(max(1, w * h))
        mejor = 0.0
        for bubble in bubbles or ():
            bx, by, bw, bh = bubble.bbox
            ancho = max(0, min(x + w, bx + bw) - max(x, bx))
            alto = max(0, min(y + h, by + bh) - max(y, by))
            mejor = max(mejor, (ancho * alto) / area)
        return round(mejor, 4)
