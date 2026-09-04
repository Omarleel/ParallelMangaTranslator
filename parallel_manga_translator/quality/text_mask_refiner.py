from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.models.processing_models import Box


#: Mínimo de píxeles para fiarse de una muestra de fondo.
MIN_BACKGROUND_PIXELS = 16
#: Ancho del anillo de fondo alrededor de la zona de tinta.
BACKGROUND_RING_PX = 12
#: Distancia de color mínima para considerar un píxel tinta y no ruido del fondo.
MIN_INK_DISTANCE = 30.0
#: Reserva si no se puede calcular el umbral del halo: fracción del umbral de tinta.
HALO_THRESHOLD_RATIO = 0.5


@dataclass(frozen=True)
class TextMaskRefinementOptions:
    """Parámetros ligeros para refinar máscaras de tinta sin depender de CRF externo."""

    enabled: bool = True
    fine_text_detection: bool = True
    fine_mask_dilate: int = 2
    min_component_area: int = 3
    component_anchor_overlap: float = 0.03
    component_anchor_max_gap_ratio: float = 0.45
    halo_growth_px: int = 0


class TextInkMaskRefiner:
    """Genera y refina máscaras de tinta usando la señal OCR + componentes conectados.

    - la máscara de globo sigue siendo solo zona segura;
    - la máscara OCR/polígono es una semilla fina de texto;
    - los componentes conectados deciden qué tinta real se borra.
    """

    @staticmethod
    def _clip_rect(rect: Box, image_shape) -> Box:
        x, y, w, h = [int(v) for v in rect]
        height, width = image_shape[:2]
        x = max(0, min(width, x))
        y = max(0, min(height, y))
        x2 = max(x, min(width, x + max(0, w)))
        y2 = max(y, min(height, y + max(0, h)))
        return x, y, x2 - x, y2 - y

    @staticmethod
    def _binary(mask: Optional[np.ndarray], image_shape) -> np.ndarray:
        out = np.zeros(image_shape[:2], dtype=np.uint8)
        if mask is None or getattr(mask, "size", 0) == 0:
            return out
        h = min(out.shape[0], mask.shape[0])
        w = min(out.shape[1], mask.shape[1])
        if h > 0 and w > 0:
            out[:h, :w] = (mask[:h, :w] > 0).astype(np.uint8) * 255
        return out

    @staticmethod
    def _box_from_points(points: Sequence[Sequence[float]]) -> Box:
        arr = np.array(points, dtype=np.float32).reshape((-1, 2))
        x, y, w, h = cv2.boundingRect(arr.astype(np.int32))
        return int(x), int(y), int(w), int(h)

    @classmethod
    def detection_boxes(cls, detections: Sequence) -> List[Box]:
        boxes: List[Box] = []
        for det in detections or []:
            try:
                boxes.append(cls._box_from_points(det[0]))
            except Exception:
                continue
        return boxes

    @classmethod
    def mask_from_detections(
        cls,
        image_shape,
        detections: Sequence,
        *,
        dilate_px: int = 2,
        min_pad: int = 1,
    ) -> np.ndarray:
        """Crea una máscara fina desde polígonos OCR, no desde la bbox rectangular."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for det in detections or []:
            try:
                pts = np.array(det[0], dtype=np.float32).reshape((-1, 2))[:4]
                if pts.shape[0] < 4:
                    continue
                cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
            except Exception:
                continue

        if cv2.countNonZero(mask) == 0:
            for box in cls.detection_boxes(detections):
                x, y, w, h = cls._clip_rect(box, image_shape)
                if w > 0 and h > 0:
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        dilation = max(0, int(dilate_px))
        if dilation > 0 and cv2.countNonZero(mask) > 0:
            k = 2 * dilation + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)

        pad = max(0, int(min_pad))
        if pad > 0 and cv2.countNonZero(mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pad + 1, 2 * pad + 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    @classmethod
    def bounding_rect_from_mask(cls, mask: np.ndarray, image_shape, padding: int = 0) -> Box:
        points = cv2.findNonZero((mask > 0).astype(np.uint8)) if mask is not None and getattr(mask, "size", 0) else None
        if points is None:
            return 0, 0, 0, 0
        x, y, w, h = cv2.boundingRect(points)
        p = max(0, int(padding))
        return cls._clip_rect((x - p, y - p, w + 2 * p, h + 2 * p), image_shape)

    @staticmethod
    def _component_gap_to_anchor(box: Box, anchor_mask: np.ndarray) -> float:
        x, y, w, h = box
        pts = cv2.findNonZero((anchor_mask > 0).astype(np.uint8))
        if pts is None:
            return float("inf")
        ax, ay, aw, ah = cv2.boundingRect(pts)
        left = ax + aw < x
        right = x + w < ax
        above = y + h < ay
        below = ay + ah < y
        dx = max(ax - (x + w), x - (ax + aw), 0)
        dy = max(ay - (y + h), y - (ay + ah), 0)
        if (left or right) and (above or below):
            return float((dx * dx + dy * dy) ** 0.5)
        return float(max(dx, dy))

    @staticmethod
    def _surrounding_ring(allowed_mask: np.ndarray) -> np.ndarray:
        """Anillo de fondo alrededor de la zona de tinta, excluyendo la propia zona."""
        zone = (allowed_mask > 0).astype(np.uint8) * 255
        if cv2.countNonZero(zone) == 0:
            return zone
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BACKGROUND_RING_PX + 1,) * 2)
        return cv2.subtract(cv2.dilate(zone, kernel, iterations=1), zone)

    @classmethod
    def _estimate_ink_candidates(
        cls,
        image: np.ndarray,
        allowed_mask: np.ndarray,
        safe_mask: np.ndarray,
        anchor_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Marca como tinta lo que se aleja del color del fondo.

        Devuelve la máscara, el mapa de distancia al fondo y el umbral usado; los dos
        últimos los necesita el crecimiento hacia el halo.

        Se mide distancia de color, no luminancia. Con luminancia sola hay que decidir
        primero si la tinta es más clara o más oscura que el fondo, y esa decisión falla
        justo en los casos frecuentes: rótulos blancos sobre gris medio, y sobre todo
        texto de color, donde parte de los trazos tiene una luma parecida a la del fondo
        y sobrevive a la limpieza. La distancia cubre las tres situaciones a la vez
        —tinta oscura, clara o de color— sin ramas de polaridad.
        """
        vacio = np.zeros(image.shape[:2], dtype=np.uint8)
        allowed = allowed_mask > 0
        if not np.any(allowed):
            return vacio, np.zeros(image.shape[:2], dtype=np.float32), 0.0

        # Muestra de fondo: zona segura menos el ancla OCR dilatada.
        anchor = (anchor_mask > 0).astype(np.uint8) * 255
        if cv2.countNonZero(anchor) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            bg_mask = cv2.bitwise_and(
                (safe_mask > 0).astype(np.uint8) * 255,
                cv2.bitwise_not(cv2.dilate(anchor, kernel, iterations=1)),
            )
        else:
            bg_mask = (safe_mask > 0).astype(np.uint8) * 255

        if cv2.countNonZero(bg_mask) < MIN_BACKGROUND_PIXELS:
            # La zona segura no deja fondo del que muestrear. Pasa siempre en texto libre,
            # donde la máscara segura ES la caja del OCR. Si además la tinta domina la
            # caja, el "color de fondo" sería el de la propia tinta, así que se muestrea
            # el anillo que rodea la zona, que sí es fondo de verdad.
            bg_mask = cls._surrounding_ring(allowed_mask)
        if cv2.countNonZero(bg_mask) < MIN_BACKGROUND_PIXELS:
            bg_mask = (safe_mask > 0).astype(np.uint8) * 255

        color = image.astype(np.float32)
        if color.ndim == 2:
            color = color[..., None]

        bg_pixels = color[bg_mask > 0]
        if bg_pixels.size == 0:
            bg_pixels = color[allowed]
        bg_color = np.median(bg_pixels.reshape(-1, color.shape[-1]), axis=0)

        distance = np.clip(np.linalg.norm(color - bg_color, axis=-1), 0.0, 255.0)
        local = distance[allowed]
        if local.size == 0:
            return vacio, distance, 0.0

        # Otsu parte las dos poblaciones reales de la zona; un percentil fijo se rompe en
        # cuanto la caja del OCR contiene algo más que texto y fondo.
        threshold = max(cls._otsu_threshold(local), MIN_INK_DISTANCE)
        ink = ((distance > threshold) & allowed).astype(np.uint8) * 255
        return ink, distance, threshold

    @classmethod
    def _halo_threshold(cls, distance: np.ndarray, growth_mask: np.ndarray, ink_threshold: float) -> float:
        """Umbral que separa el halo del fondo, por debajo del umbral de tinta.

        Dentro de una región hay tres poblaciones, no dos: los glifos (muy lejos del color
        del fondo), el halo que los rodea (lejos, pero bastante menos) y el fondo. Otsu es
        un corte binario y agrupa el halo con el fondo, así que el umbral de tinta no sirve
        para crecer hacia él. Se vuelve a aplicar Otsu sobre lo que quedó por debajo, que
        es justo la mezcla halo/fondo.
        """
        resto = distance[(growth_mask > 0) & (distance < ink_threshold)]
        if resto.size >= MIN_BACKGROUND_PIXELS:
            umbral = cls._otsu_threshold(resto)
            if MIN_INK_DISTANCE <= umbral < ink_threshold:  # cota exclusiva, ver _otsu_threshold
                return float(umbral)
        return max(MIN_INK_DISTANCE, ink_threshold * HALO_THRESHOLD_RATIO)

    @classmethod
    def _grow_into_halo(
        cls,
        ink: np.ndarray,
        distance: np.ndarray,
        threshold: float,
        growth_mask: np.ndarray,
        max_px: int,
    ) -> np.ndarray:
        """Extiende la máscara al contorno del texto sin agrandar el agujero hacia el arte.

        El texto de manga suele llevar halo —blanco sobre trama, negro sobre fondo claro—
        que queda fuera de la caja del OCR y por tanto fuera de la zona donde se busca
        tinta. Si ese halo se queda sin borrar, el inpainter tiene blanco pegado al borde
        del agujero y lo propaga hacia dentro: el resultado es una mancha con la forma
        exacta de las letras, y da igual qué modelo se use, porque todos rellenan a partir
        de lo que rodea al agujero.

        El crecimiento es por reconstrucción morfológica, no por dilatación: sólo avanza
        hacia píxeles **contiguos** que siguen sin parecerse al fondo, y se detiene solo al
        tocarlo. Dilatar a ciegas el mismo número de píxeles agranda el agujero en todas
        las direcciones y el modelo acaba inventando arte que no estaba.
        """
        if max_px <= 0 or cv2.countNonZero(ink) == 0:
            return ink

        umbral_halo = cls._halo_threshold(distance, growth_mask, threshold)
        alcanzable = ((distance > umbral_halo) & (growth_mask > 0)).astype(np.uint8) * 255
        actual = cv2.bitwise_or(cv2.bitwise_and(ink, alcanzable), ink)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(int(max_px)):
            crecida = cv2.bitwise_and(cv2.dilate(actual, kernel, iterations=1), alcanzable)
            crecida = cv2.bitwise_or(crecida, ink)
            if cv2.countNonZero(crecida) == cv2.countNonZero(actual):
                break
            actual = crecida
        return actual

    @staticmethod
    def _otsu_threshold(values: np.ndarray) -> float:
        """Umbral de Otsu sobre una muestra de distancias de color (0-255).

        Sigue el convenio de OpenCV: el valor devuelto es una cota **exclusiva**, es decir
        se compara con ``>``. Con `>=` una distribución de dos modos limpios devuelve el
        modo bajo y la comparación acaba seleccionando también el fondo.
        """
        sample = np.asarray(values, dtype=np.uint8).reshape(-1, 1)
        if sample.size < 2 or int(sample.min()) == int(sample.max()):
            return float(sample.min()) if sample.size else 0.0
        threshold, _ = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(threshold)

    @classmethod
    def refine(
        cls,
        image: np.ndarray,
        safe_mask: np.ndarray,
        text_zone: np.ndarray,
        *,
        raw_text_mask: Optional[np.ndarray] = None,
        initial_ink_mask: Optional[np.ndarray] = None,
        options: TextMaskRefinementOptions | None = None,
        growth_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        options = options or TextMaskRefinementOptions()
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        safe = cls._binary(safe_mask, image.shape)
        zone = cls._binary(text_zone, image.shape)
        raw = cls._binary(raw_text_mask, image.shape)
        initial = cls._binary(initial_ink_mask, image.shape)
        if cv2.countNonZero(safe) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        anchor = raw if cv2.countNonZero(raw) > 0 else zone
        if cv2.countNonZero(anchor) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # La zona permitida se deriva del ancla OCR/polígono y se recorta por la máscara segura.
        expand_px = max(1, int(options.fine_mask_dilate) + 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand_px + 1, 2 * expand_px + 1))
        allowed = cv2.dilate(anchor, kernel, iterations=1)
        allowed = cv2.bitwise_and(allowed, safe)
        if cv2.countNonZero(allowed) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        candidates, distance, ink_threshold = cls._estimate_ink_candidates(image, allowed, safe, anchor)
        if cv2.countNonZero(initial) > 0:
            candidates = cv2.bitwise_or(candidates, cv2.bitwise_and(initial, allowed))

        num, labels, stats, _ = cv2.connectedComponentsWithStats((candidates > 0).astype(np.uint8) * 255, 8)
        refined = np.zeros_like(candidates)
        anchor_area = max(1, cv2.countNonZero(anchor))
        median_anchor_side = 10.0
        pts = cv2.findNonZero(anchor)
        if pts is not None:
            _ax, _ay, aw, ah = cv2.boundingRect(pts)
            median_anchor_side = float(max(6, min(max(aw, ah), max(12, (aw + ah) / 2))))
        max_gap = median_anchor_side * max(0.05, float(options.component_anchor_max_gap_ratio))

        for idx in range(1, num):
            x, y, w, h, area = [int(v) for v in stats[idx]]
            if area < int(options.min_component_area):
                continue
            comp = (labels[y:y + h, x:x + w] == idx)
            anchor_crop = anchor[y:y + h, x:x + w] > 0
            overlap = int(np.count_nonzero(comp & anchor_crop))
            overlap_ratio = overlap / max(1, min(int(area), anchor_area))
            gap = cls._component_gap_to_anchor((x, y, w, h), anchor)
            if overlap_ratio >= float(options.component_anchor_overlap) or gap <= max_gap or area >= 18:
                refined[y:y + h, x:x + w][comp] = 255

        if cv2.countNonZero(refined) == 0:
            refined = cv2.bitwise_and(candidates, allowed)

        dilation = max(0, int(options.fine_mask_dilate))
        if dilation > 0 and cv2.countNonZero(refined) > 0:
            k = 2 * dilation + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
            refined = cv2.dilate(refined, kernel, iterations=1)

        refined = cv2.bitwise_and(refined, safe)
        if growth_mask is not None and int(options.halo_growth_px) > 0:
            refined = cls._grow_into_halo(
                refined,
                distance,
                ink_threshold,
                cls._binary(growth_mask, image.shape),
                int(options.halo_growth_px),
            )
        return refined
