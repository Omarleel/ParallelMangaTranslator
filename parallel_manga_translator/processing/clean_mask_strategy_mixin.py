from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.quality.text_mask_refiner import TextInkMaskRefiner, TextMaskRefinementOptions


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

        # Si no hubo OCR y text_bbox == bbox, la caja de texto no aporta nada: el interior
        # del globo se explora aparte en `_bubble_interior_text_zone`, con guardas.
        if int(getattr(region, "detections_count", 0) or 0) <= 0 and (x, y, w, h) == (bx, by, bw, bh):
            return np.zeros(image_shape[:2], dtype=np.uint8)

        pad_x = max(3, min(14, int(round(max(w, h) * 0.08))))
        pad_y = max(4, min(18, int(round(max(w, h) * 0.10))))
        expanded = self._expand_rect((x, y, w, h), pad_x, pad_y, image_shape)
        return self._rect_mask(expanded, image_shape)
    def _bubble_interior_text_zone(self, imagen: np.ndarray, safe_mask: np.ndarray) -> np.ndarray:
        """Zona donde buscar tinta cuando el localizador no encontró texto en el globo.

        El detector de globos y el localizador de texto son independientes: YOLO puede
        encontrar el globo y EasyOCR/Paddle no devolver ninguna caja dentro (texto
        estilizado, contraste bajo, kana pequeño). Antes ese globo se quedaba sin
        máscara de tinta, así que el original nunca se borraba y la traducción acababa
        dibujada encima del texto japonés.

        El interior del globo se usa aquí como zona de **búsqueda**, nunca como máscara
        de borrado: qué se borra lo siguen decidiendo el umbralizado de tinta y los
        componentes conectados. Si la tinta ocupa demasiada parte del interior, la
        máscara no es un globo con texto —un recorte de arte, una viñeta oscura— y se
        prefiere no tocar nada.
        """
        max_ratio = float(getattr(self, "bubble_ink_without_ocr_max_ratio", 0.35) or 0.0)
        if max_ratio <= 0.0:
            return np.zeros(imagen.shape[:2], dtype=np.uint8)

        interior = int(cv2.countNonZero(safe_mask))
        if interior <= 0:
            return np.zeros(imagen.shape[:2], dtype=np.uint8)

        # Sin dilatar: aquí sólo se mide cuánta tinta hay, no se construye la máscara final.
        tinta = self._text_ink_mask(imagen, safe_mask, safe_mask, dilate_px=0)
        ratio = cv2.countNonZero(tinta) / interior
        if ratio <= 0.0 or ratio > max_ratio:
            return np.zeros(imagen.shape[:2], dtype=np.uint8)
        return safe_mask

    def _halo_growth_zone(self, safe_mask: np.ndarray, image_shape) -> np.ndarray:
        """Hasta dónde puede crecer la máscara buscando el halo del texto libre.

        La zona segura del texto libre es la caja del OCR, y el halo casi siempre la
        desborda. Se ensancha lo justo para que quepa el crecimiento; lo que se borre
        dentro de esa zona lo sigue decidiendo la reconstrucción, no este margen.
        """
        margen = max(0, int(getattr(self, "text_halo_growth_px", 10))) + 2
        if margen <= 2:
            return self._binary_mask(safe_mask, image_shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margen + 1, 2 * margen + 1))
        return cv2.dilate(self._binary_mask(safe_mask, image_shape), kernel, iterations=1)

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
            text_mask = self._binary_mask(getattr(region, "text_mask", None), shape)
            if getattr(self, "ink_mask_refinement", True) and cv2.countNonZero(raw_mask) > 0:
                safe = self._binary_mask(region.mask, shape) if region.mask is not None else raw_mask
                # `initial_ink_mask` va vacío a propósito. En texto libre y SFX la máscara de
                # partida es el polígono OCR relleno, es decir la caja del renglón entero, no
                # los trazos. `refine` une la semilla inicial con sus candidatos, así que
                # pasarla aquí devolvía siempre el rectángulo completo y el refinamiento no
                # podía reducir nada: el inpainting recibía un bloque sólido y barría el fondo.
                # Qué tinta se borra lo deciden el umbralizado y los componentes conectados.
                refined = TextInkMaskRefiner.refine(
                    imagen,
                    safe,
                    raw_mask,
                    raw_text_mask=text_mask if cv2.countNonZero(text_mask) > 0 else raw_mask,
                    initial_ink_mask=None,
                    # El halo del texto libre cae fuera de la caja del OCR, así que el
                    # crecimiento necesita permiso para salir de la zona segura.
                    growth_mask=self._halo_growth_zone(safe, shape),
                    options=TextMaskRefinementOptions(
                        enabled=True,
                        fine_text_detection=bool(getattr(self, "fine_text_detection", True)),
                        fine_mask_dilate=int(getattr(self, "fine_text_mask_dilate", self.bubble_fill_text_dilate)),
                        min_component_area=int(getattr(self, "ink_mask_min_component_area", 3)),
                        component_anchor_overlap=float(getattr(self, "ink_mask_component_anchor_overlap", 0.03)),
                        component_anchor_max_gap_ratio=float(getattr(self, "ink_mask_component_anchor_max_gap_ratio", 0.45)),
                        halo_growth_px=int(getattr(self, "text_halo_growth_px", 10)),
                    ),
                )
                if cv2.countNonZero(refined) > 0:
                    return refined, "region_text_ink"
            return raw_mask, "region_text_box"

        safe_mask = self._safe_bubble_mask(region.mask, shape, self.bubble_fill_edge_margin)
        if cv2.countNonZero(safe_mask) == 0:
            return np.zeros(shape[:2], dtype=np.uint8), "empty_bubble_mask"

        # Opt-in explícito para limpiar todo el interior del globo. Sigue separado
        # como clean_mask, pero no es el comportamiento por defecto.
        if self.bubble_fill_whole_interior and self._mask_rectangularity(safe_mask) < self.bubble_fill_flat_max_rectangularity:
            return safe_mask, "bubble_safe_interior_opt_in"

        text_zone = self._bubble_text_zone(region, shape)
        raw_text_mask = self._binary_mask(getattr(region, "text_mask", None), shape)
        if cv2.countNonZero(raw_text_mask) > 0:
            # La máscara de polígonos OCR es una zona más precisa que el bbox unido.
            text_zone = cv2.bitwise_or(text_zone, raw_text_mask)

        zone_source = "text_ink_inside_bubble"
        if cv2.countNonZero(text_zone) == 0:
            text_zone = self._bubble_interior_text_zone(imagen, safe_mask)
            if cv2.countNonZero(text_zone) > 0:
                zone_source = "bubble_interior_ink_without_ocr"

        clean_mask = self._text_ink_mask(imagen, text_zone, safe_mask, self.bubble_fill_text_dilate)
        if getattr(self, "ink_mask_refinement", True):
            clean_mask = TextInkMaskRefiner.refine(
                imagen,
                safe_mask,
                text_zone,
                raw_text_mask=raw_text_mask if cv2.countNonZero(raw_text_mask) > 0 else None,
                initial_ink_mask=clean_mask,
                # En globos el crecimiento no puede salir del interior: el borde negro del
                # globo no es halo de texto y borrarlo destroza la viñeta.
                growth_mask=safe_mask,
                options=TextMaskRefinementOptions(
                    enabled=True,
                    fine_text_detection=bool(getattr(self, "fine_text_detection", True)),
                    fine_mask_dilate=int(getattr(self, "fine_text_mask_dilate", self.bubble_fill_text_dilate)),
                    min_component_area=int(getattr(self, "ink_mask_min_component_area", 3)),
                    component_anchor_overlap=float(getattr(self, "ink_mask_component_anchor_overlap", 0.03)),
                    component_anchor_max_gap_ratio=float(getattr(self, "ink_mask_component_anchor_max_gap_ratio", 0.45)),
                    halo_growth_px=int(getattr(self, "text_halo_growth_px", 10)),
                ),
            )
        return clean_mask, zone_source if cv2.countNonZero(clean_mask) > 0 else "empty_text_ink_inside_bubble"

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
                if getattr(self, "ink_mask_refinement", True):
                    metadata["ink_mask_refinement"] = "connected_components_anchored_to_ocr"
                    metadata["ink_mask_refined"] = True
                if getattr(region, "text_mask", None) is not None and getattr(region.text_mask, "size", 0):
                    metadata["text_mask_pixels"] = int(cv2.countNonZero(self._binary_mask(region.text_mask, imagen.shape)))
                    metadata.setdefault("fine_text_mask_source", "ocr_polygons")
                if self._is_bubble_region(region):
                    metadata["bubble_mask_pixels"] = int(cv2.countNonZero(self._binary_mask(region.mask, imagen.shape)))
                    metadata["bubble_clean_mask_separated"] = True
            prepared.append(region)
        return prepared

