from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.detection.yolo_bubble_detector import YoloBubbleCandidate
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.processing_models import Box, TextRegion

logger = get_logger(__name__)
BUBBLE_SPLIT_DEBUG_VERSION = "v7_bubble_onomatopoeia_translation_2026_06_11"


class FreeTextRecoveryMixin:
    """Recuperación y ordenamiento de texto libre/SFX fuera de globos."""

    def _free_text_regions_from_detections(self, image: np.ndarray, detections: Sequence) -> List[TextRegion]:
        free_regions: List[TextRegion] = []

        grouped_detections = self._group_detections(detections)
        try:
            grouped_detections = self.reading_order_resolver.sort_detection_groups(grouped_detections, self._to_rect)
        except Exception as exc:
            logger.warning("No se pudo ordenar grupos de texto libre por lectura: %s", exc)

        for i, group in enumerate(grouped_detections):
            boxes = [self._to_rect(det) for det in group]
            text_box = boxes[0]
            for box in boxes[1:]:
                text_box = self._union(text_box, box)

            text_hint = " ".join(self._text(det).strip() for det in group if self._text(det).strip())
            conf = float(np.mean([self._confidence(det) for det in group])) if group else 0.0

            free_text_onomatopoeia_metadata = self._free_text_onomatopoeia_metadata(text_hint)
            looks_sfx = self._looks_like_sfx(group) or bool(free_text_onomatopoeia_metadata)
            keep, filter_reason = self._should_keep_free_text_group(
                text_box,
                text_hint,
                conf,
                image.shape,
                looks_sfx,
            )
            if not keep:
                logger.debug(
                    "Texto libre descartado: reason=%s bbox=%s conf=%.3f text=%r",
                    filter_reason,
                    text_box,
                    conf,
                    text_hint[:40],
                )
                continue

            kind = "sfx" if looks_sfx else "free_text"

            if looks_sfx:
                razon = "Texto OCR fuera de globo, detectado por proporciones o diccionario como Onomatopeya (SFX)"
            else:
                razon = "Texto OCR agrupado que quedó huérfano (no está dentro de ningún globo de la IA)"

            mask, bbox, score, source = self._text_box_mask(image.shape, text_box, kind=kind)
            metadata = {
                "mask_source": source,
                "detector": "ocr_free_text",
                "region_flow": "bubble_first_pretrained_only",
                "ocr_scope": "free_text_or_sfx",
                "free_text_filter_reason": filter_reason,
                "free_text_confidence": round(float(conf), 4),
                "free_text_original_kind": "free_text",
                "free_text_sfx_reason": free_text_onomatopoeia_metadata.get("free_text_onomatopoeia_method", "") if free_text_onomatopoeia_metadata else ("shape_candidate" if looks_sfx else ""),
            }
            if filter_reason == "cjk_vertical_bbox_retry_ocr":
                metadata.update({
                    "ocr_global_hint_used_as_bbox_only": True,
                    "ocr_global_hint_untrusted": True,
                    "force_region_ocr": True,
                    "layout_hint": "vertical_cjk",
                    "vertical_text_retry": True,
                })
            metadata.update(free_text_onomatopoeia_metadata)
            free_regions.append(TextRegion(
                bbox=bbox,
                text_bbox=text_box,
                mask=mask,
                clean_mask=mask.copy(),
                kind=kind,
                confidence=max(conf, score),
                source_text_hint=text_hint,
                detections_count=len(group),
                metadata=metadata,
            ))
        return free_regions

    def _text_like_ink_bbox_in_gap(self, image: np.ndarray, gap_box: Box) -> Optional[Box]:
        """Busca texto libre vertical que quedó sin OCR dentro de un hueco.

        Este fallback no traduce por sí solo: solo crea una región para que la
        etapa normal de OCR sobre recorte vuelva a intentarlo y para que la
        máscara limpie el original. Se mantiene deliberadamente conservador y
        se limita a huecos entre textos libres ya encontrados.
        """
        img_h, img_w = image.shape[:2]
        x, y, w, h = self._clip_box_to_image(gap_box, img_w, img_h)
        if w < 16 or h < 42:
            return None

        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Umbral oscuro estricto: evita seleccionar tramas grises/fondos.
        dark = np.uint8(gray <= 120) * 255

        # El texto japonés vertical suele aparecer como caracteres separados;
        # cerramos verticalmente para que una misma columna forme un componente.
        kernel_h = max(7, min(31, int(round(h * 0.055))))
        kernel_w = max(3, min(9, int(round(w * 0.055))))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        grouped = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
        grouped = cv2.dilate(grouped, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)), iterations=1)

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(grouped, connectivity=8)
        candidates: List[Box] = []
        for label_idx in range(1, num_labels):
            cx, cy, cw, ch, area = [int(v) for v in stats[label_idx]]
            if cw < 12 or ch < max(42, int(h * 0.28)):
                continue
            aspect = ch / max(1, cw)
            if aspect < 1.35:
                continue
            # La morfología rellena huecos para conectar caracteres; la densidad
            # se mide sobre la tinta oscura original para no castigar letras con
            # outline ni aceptar bloques sólidos por accidente.
            original_ink = int(cv2.countNonZero(dark[cy:cy + ch, cx:cx + cw]))
            density = original_ink / max(1, cw * ch)
            if density < self.free_text_gap_min_density or density > self.free_text_gap_max_density:
                continue
            # Rechaza líneas de panel muy finas o masas enormes que cubren casi todo el hueco.
            if cw <= 6 or (cw > w * 0.92 and ch > h * 0.92):
                continue
            candidates.append((x + cx, y + cy, cw, ch))

        if not candidates:
            return None

        merged = candidates[0]
        for box in candidates[1:]:
            merged = self._union(merged, box)

        mx, my, mw, mh = merged
        # Validación final contra el tamaño del hueco.
        if mw < 14 or mh < max(48, int(h * 0.30)):
            return None
        if mh / max(1, mw) < 1.25:
            return None

        pad_x = max(4, min(12, int(round(mw * 0.12))))
        pad_y = max(5, min(18, int(round(mh * 0.08))))
        return self._clip_box_to_image((mx - pad_x, my - pad_y, mw + 2 * pad_x, mh + 2 * pad_y), img_w, img_h)

    def _order_regions_for_reading(self, regions: Sequence[TextRegion]) -> List[TextRegion]:
        try:
            ordered = self.reading_order_resolver.sort_regions(list(regions))
        except Exception as exc:
            logger.warning("No se pudo ordenar regiones por lectura; se mantiene orden de detección: %s", exc)
            ordered = list(regions)
        for index, region in enumerate(ordered):
            region.metadata["reading_order_index"] = index
            region.metadata["reading_order_language"] = self.idioma_entrada
            region.metadata["reading_order_flow"] = "rtl_vertical" if self.reading_order_resolver.page_reads_right_to_left else "ltr_horizontal"
        return ordered

    def _recover_free_text_gaps(self, image: np.ndarray, regions: Sequence[TextRegion]) -> List[TextRegion]:
        if not self.free_text_gap_recovery:
            return []
        text_regions = [r for r in regions if r.kind in {"free_text", "sfx"}]
        if len(text_regions) < 2:
            return []

        recovered: List[TextRegion] = []
        seen_boxes: List[Box] = []
        height, width = image.shape[:2]

        ordered = sorted(text_regions, key=lambda r: (r.bbox[1] // 60, r.bbox[0]))
        for i, left in enumerate(ordered):
            lx, ly, lw, lh = left.bbox
            for right in ordered[i + 1:]:
                rx, ry, rw, rh = right.bbox
                # Solo pares izquierda-derecha; si se cruzan, no hay hueco fiable.
                if rx <= lx + lw:
                    continue
                gap_x = rx - (lx + lw)
                if gap_x < 10 or gap_x > self.free_text_gap_max_px:
                    continue

                y_overlap = self._overlap_ratio_1d(ly, ly + lh, ry, ry + rh)
                if y_overlap < self.free_text_gap_min_y_overlap:
                    continue

                y1 = max(0, min(ly, ry) - max(8, int(round(min(lh, rh) * 0.05))))
                y2 = min(height, max(ly + lh, ry + rh) + max(8, int(round(min(lh, rh) * 0.05))))
                # Recorta un poco contra los textos vecinos para no absorber su borde/outline.
                x1 = max(0, lx + lw + max(2, int(round(gap_x * 0.06))))
                x2 = min(width, rx - max(2, int(round(gap_x * 0.06))))
                if x2 <= x1 or y2 <= y1:
                    continue

                candidate_box = self._text_like_ink_bbox_in_gap(image, (x1, y1, x2 - x1, y2 - y1))
                if candidate_box is None:
                    continue

                # Evita duplicar o invadir regiones ya existentes.
                candidate_area = max(1, self._area(candidate_box))
                duplicate = False
                for existing in list(regions) + recovered:
                    overlap = self._intersection_area(candidate_box, existing.bbox) / candidate_area
                    if overlap > 0.20:
                        duplicate = True
                        break
                if duplicate:
                    continue
                if any(self._intersection_area(candidate_box, box) / candidate_area > 0.20 for box in seen_boxes):
                    continue

                mask, bbox, score, source = self._text_box_mask(image.shape, candidate_box, kind="free_text")
                recovered.append(TextRegion(
                    bbox=bbox,
                    text_bbox=candidate_box,
                    mask=mask,
                    clean_mask=mask.copy(),
                    kind="free_text",
                    confidence=score,
                    source_text_hint="",
                    detections_count=0,
                    metadata={
                        "mask_source": source,
                        "detector": "visual_free_text_gap",
                        "region_flow": "bubble_first_pretrained_only",
                        "ocr_scope": "free_text_visual_gap_retry",
                        "free_text_filter_reason": "visual_gap_recovery",
                    },
                ))
                seen_boxes.append(candidate_box)

        return recovered

    def build_regions_from_bubbles_and_text(
        self,
        image: np.ndarray,
        bubble_regions: Sequence[TextRegion],
        detections: Sequence,
    ) -> List[TextRegion]:
        regions = list(bubble_regions or [])
        if regions:
            assigned, detections_by_region = self._assign_text_detections_to_regions(regions, detections)
            regions, debug_records = self._split_merged_bubble_regions(image, regions, detections_by_region)
        else:
            assigned, detections_by_region, debug_records = set(), {}, []
        remaining = []
        for idx, det in enumerate(detections or []):
            if idx in assigned:
                continue
            try:
                text_box = self._to_rect(det)
            except Exception:
                continue
            if regions and self._is_inside_existing_region(text_box, regions):
                continue
            remaining.append(det)
        regions.extend(self._free_text_regions_from_detections(image, remaining))
        regions.extend(self._recover_free_text_gaps(image, regions))
        merged_regions = self._merge_region_masks(regions)
        ordered_regions = self._order_regions_for_reading(merged_regions)
        self._annotate_bubble_visual_expressions(ordered_regions)
        self._save_merge_debug_artifacts(image, ordered_regions, debug_records)
        return ordered_regions

    def detect_regions(self, image: np.ndarray, detections: Sequence) -> List[TextRegion]:
        bubble_regions = self.detect_primary_bubble_regions(image)
        return self.build_regions_from_bubbles_and_text(image, bubble_regions, detections)
