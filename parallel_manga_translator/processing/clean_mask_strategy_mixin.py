from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.models.processing_models import TextRegion


class CleanMaskStrategyMixin:
    """Construcción y ajuste de máscaras de limpieza."""

    @staticmethod
    def _clip_rect(rect: Tuple[int, int, int, int], image_shape) -> Tuple[int, int, int, int]:
        x, y, w, h = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        height, width = image_shape[:2]
        x = max(0, min(width, x))
        y = max(0, min(height, y))
        x2 = max(x, min(width, x + max(0, w)))
        y2 = max(y, min(height, y + max(0, h)))
        return x, y, x2 - x, y2 - y

    @staticmethod
    def _expand_rect(rect: Tuple[int, int, int, int], px: int, py: int, image_shape) -> Tuple[int, int, int, int]:
        x, y, w, h = rect
        return CleanMaskStrategyMixin._clip_rect((x - px, y - py, w + 2 * px, h + 2 * py), image_shape)

    @staticmethod
    def _rect_mask(rect: Tuple[int, int, int, int], image_shape) -> np.ndarray:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        x, y, w, h = CleanMaskStrategyMixin._clip_rect(rect, image_shape)
        if w > 0 and h > 0:
            mask[y:y + h, x:x + w] = 255
        return mask

    @staticmethod
    def _binary_mask(mask: np.ndarray, image_shape) -> np.ndarray:
        if mask is None or mask.size == 0:
            return np.zeros(image_shape[:2], dtype=np.uint8)
        out = np.zeros(image_shape[:2], dtype=np.uint8)
        h = min(out.shape[0], mask.shape[0])
        w = min(out.shape[1], mask.shape[1])
        if h <= 0 or w <= 0:
            return out
        out[:h, :w] = (mask[:h, :w] > 0).astype(np.uint8) * 255
        return out

    @staticmethod
    def _safe_bubble_mask(mask: np.ndarray, image_shape, margin: int) -> np.ndarray:
        bubble = CleanMaskStrategyMixin._binary_mask(mask, image_shape)
        if cv2.countNonZero(bubble) == 0:
            return bubble
        margin = max(0, int(margin))
        if margin > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            eroded = cv2.erode(bubble, kernel, iterations=1)
            # Si el globo es muy estrecho y la erosión lo destruye, usa una erosión menor.
            if cv2.countNonZero(eroded) < max(8, int(cv2.countNonZero(bubble) * 0.18)):
                small = max(1, margin // 2)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * small + 1, 2 * small + 1))
                eroded = cv2.erode(bubble, kernel, iterations=1)
            bubble = eroded if cv2.countNonZero(eroded) > 0 else bubble
        return bubble

    @staticmethod
    def _mask_rectangularity(mask: np.ndarray) -> float:
        if mask is None or cv2.countNonZero(mask) == 0:
            return 1.0
        points = cv2.findNonZero((mask > 0).astype(np.uint8))
        if points is None:
            return 1.0
        x, y, w, h = cv2.boundingRect(points)
        if w <= 0 or h <= 0:
            return 1.0
        return float(cv2.countNonZero(mask)) / float(w * h)

    @staticmethod
    def _text_ink_mask(imagen: np.ndarray, text_zone: np.ndarray, safe_mask: np.ndarray, dilate_px: int = 2) -> np.ndarray:
        if cv2.countNonZero(text_zone) == 0 or cv2.countNonZero(safe_mask) == 0:
            return np.zeros(imagen.shape[:2], dtype=np.uint8)

        zone = cv2.bitwise_and((text_zone > 0).astype(np.uint8) * 255, (safe_mask > 0).astype(np.uint8) * 255)
        if cv2.countNonZero(zone) == 0:
            return np.zeros(imagen.shape[:2], dtype=np.uint8)

        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        vals = gray[zone > 0]
        if vals.size == 0:
            return np.zeros(imagen.shape[:2], dtype=np.uint8)

        # Decide la polaridad desde el fondo de la región segura, no desde el
        # bbox de texto. El bbox puede estar muy ajustado y quedar dominado por
        # la tinta; en cajas negras con texto blanco eso hacía que se tomara el
        # texto como fondo y se borrara la zona equivocada.
        safe_vals = gray[safe_mask > 0]
        background_median = float(np.median(safe_vals)) if safe_vals.size else float(np.median(vals))

        if background_median < 128:
            # Fondo oscuro: la tinta esperada es clara. Usa un umbral relativo al
            # texto local para aceptar blanco/gris claro sin capturar el fondo.
            percentile_cut = int(np.percentile(vals, 70))
            threshold = max(80, min(245, percentile_cut - 10))
            ink = ((gray >= threshold) & (zone > 0)).astype(np.uint8) * 255
        else:
            # Fondo claro: la tinta esperada es oscura. El umbral alto cubre
            # antialias gris, pero queda por debajo del fondo blanco del globo.
            percentile_cut = int(np.percentile(vals, 30))
            threshold = min(235, max(20, percentile_cut + 10))
            ink = ((gray <= threshold) & (zone > 0)).astype(np.uint8) * 255

        # Evita borrar tramas muy finas sueltas: conserva componentes que parecen trazos de letra.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        filtered = np.zeros_like(ink)
        for idx in range(1, num):
            x, y, w, h, area = stats[idx]
            if area < 3:
                continue
            # Letras japonesas verticales pueden ser altas; puntos de screentone suelen ser minúsculos.
            if area >= 8 or max(w, h) >= 5:
                filtered[labels == idx] = 255

        if cv2.countNonZero(filtered) == 0:
            filtered = ink

        dilate_px = max(0, int(dilate_px))
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            filtered = cv2.dilate(filtered, kernel, iterations=1)
            filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel, iterations=1)

        return cv2.bitwise_and(filtered, safe_mask)

    def _bubble_text_zone(self, region: TextRegion, image_shape) -> np.ndarray:
        # Usa el bbox OCR/text_bbox como guía, pero nunca como máscara de pintado directa.
        # Se expande para cubrir antialias y detecciones parciales de OCR vertical japonés.
        x, y, w, h = self._clip_rect(region.text_bbox, image_shape)
        bx, by, bw, bh = self._clip_rect(region.bbox, image_shape)
        if w <= 0 or h <= 0:
            return np.zeros(image_shape[:2], dtype=np.uint8)

        # Si no hubo OCR y text_bbox == bbox, no inventamos limpieza de todo el globo.
        if int(getattr(region, "detections_count", 0) or 0) <= 0 and (x, y, w, h) == (bx, by, bw, bh):
            return np.zeros(image_shape[:2], dtype=np.uint8)

        pad_x = max(3, min(14, int(round(max(w, h) * 0.08))))
        pad_y = max(4, min(18, int(round(max(w, h) * 0.10))))
        expanded = self._expand_rect((x, y, w, h), pad_x, pad_y, image_shape)
        return self._rect_mask(expanded, image_shape)
    @staticmethod
    def _is_bubble_region(region: TextRegion) -> bool:
        return getattr(region, "kind", "") in {"dialogue", "narration", "unknown"}

    def _build_clean_mask_for_region(self, imagen: np.ndarray, region: TextRegion) -> tuple[np.ndarray, str]:
        """Construye la máscara de limpieza separada de la máscara de región.

        - ``region.mask`` delimita la zona segura del globo/elemento.
        - ``region.clean_mask`` delimita la tinta original que se puede borrar.
        """
        shape = imagen.shape
        if not self._is_bubble_region(region):
            existing_clean = getattr(region, "clean_mask", None)
            raw_mask = self._binary_mask(existing_clean, shape)
            if cv2.countNonZero(raw_mask) == 0 and region.mask is not None:
                raw_mask = self._binary_mask(region.mask, shape)
            return raw_mask, "region_text_mask"

        safe_mask = self._safe_bubble_mask(region.mask, shape, self.bubble_fill_edge_margin)
        if cv2.countNonZero(safe_mask) == 0:
            return np.zeros(shape[:2], dtype=np.uint8), "empty_bubble_mask"

        # Opt-in explícito para limpiar todo el interior del globo. Sigue separado
        # como clean_mask, pero no es el comportamiento por defecto.
        if self.bubble_fill_whole_interior and self._mask_rectangularity(safe_mask) < self.bubble_fill_flat_max_rectangularity:
            return safe_mask, "bubble_safe_interior_opt_in"

        text_zone = self._bubble_text_zone(region, shape)
        clean_mask = self._text_ink_mask(imagen, text_zone, safe_mask, self.bubble_fill_text_dilate)
        return clean_mask, "text_ink_inside_bubble" if cv2.countNonZero(clean_mask) > 0 else "empty_text_ink_inside_bubble"

    def _attach_clean_masks(self, imagen: np.ndarray, regiones: Sequence[TextRegion]) -> List[TextRegion]:
        """Anota cada región con clean_mask sin reemplazar la máscara del globo.

        La máscara del globo queda disponible para OCR/renderizado y restricciones
        espaciales; la máscara de tinta es la que se compone para limpieza.
        """
        prepared: List[TextRegion] = []
        for region in regiones or []:
            clean_mask, source = self._build_clean_mask_for_region(imagen, region)
            region.clean_mask = self._binary_mask(clean_mask, imagen.shape)
            metadata = getattr(region, "metadata", None)
            if isinstance(metadata, dict):
                metadata["region_mask_role"] = "safe_region_for_ocr_and_render"
                metadata["clean_mask_role"] = "ink_or_text_pixels_to_remove"
                metadata["clean_mask_source"] = source
                metadata["clean_mask_pixels"] = int(cv2.countNonZero(region.clean_mask))
                if self._is_bubble_region(region):
                    metadata["bubble_mask_pixels"] = int(cv2.countNonZero(self._binary_mask(region.mask, imagen.shape)))
                    metadata["bubble_clean_mask_separated"] = True
            prepared.append(region)
        return prepared

