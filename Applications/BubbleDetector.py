from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from Applications.OnomatopoeiaManager import OnomatopoeiaManager
from Applications.ProcessingModels import Box, TextRegion
from Applications.ProfessionalBubbleDetector import ProfessionalBubbleCandidate, ProfessionalBubbleDetector
from .CacheManager import env_flag
from .LoggingConfig import get_logger

logger = get_logger(__name__)


class BubbleDetector:
    """Detector de regiones basado SOLO en modelos preentrenados para globos.

    Ya no existe fallback heurístico para globos de texto. El flujo profesional es:

        modelo preentrenado detecta globos -> OCR dentro de cada globo -> OCR global
        solo aporta pistas y textos libres/SFX fuera de globos.

    Si el modelo no puede cargarse, se lanza error. Las onomatopeyas/textos libres fuera
    de globo siguen usando cajas OCR como máscara propia, pero eso no se usa para inventar
    globos de diálogo.
    """

    def __init__(self, idioma_entrada: str = "Japonés") -> None:
        self.idioma_entrada = idioma_entrada
        self.onomatopoeia_manager = OnomatopoeiaManager()
        self.enabled = env_flag("PMT_BUBBLE_DETECTION", True)
        if not self.enabled:
            raise RuntimeError(
                "PMT_BUBBLE_DETECTION=0 no está permitido en esta versión: la detección de globos "
                "debe hacerse con un modelo preentrenado."
            )
        self.professional_detector = ProfessionalBubbleDetector()

    @staticmethod
    def _to_rect(detection) -> Box:
        points = np.array(detection[0], dtype=np.float32)
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _text(detection) -> str:
        try:
            return str(detection[1] or "")
        except Exception:
            return ""

    @staticmethod
    def _confidence(detection) -> float:
        try:
            return float(detection[2])
        except Exception:
            return 0.0

    @staticmethod
    def _area(box: Box) -> int:
        return max(0, box[2]) * max(0, box[3])

    @staticmethod
    def _union(a: Box, b: Box) -> Box:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax + aw, bx + bw)
        y2 = max(ay + ah, by + bh)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def _intersection_area(a: Box, b: Box) -> int:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _overlap_ratio_1d(a1: int, a2: int, b1: int, b2: int) -> float:
        inter = max(0, min(a2, b2) - max(a1, b1))
        denom = max(1, min(a2 - a1, b2 - b1))
        return inter / denom

    @staticmethod
    def _center(box: Box) -> Tuple[float, float]:
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def _point_inside_box(point: Tuple[float, float], box: Box) -> bool:
        px, py = point
        x, y, w, h = box
        return x <= px <= x + w and y <= py <= y + h

    @staticmethod
    def _point_inside_mask(point: Tuple[float, float], mask: np.ndarray) -> bool:
        if mask is None or mask.size == 0:
            return False
        px, py = int(round(point[0])), int(round(point[1]))
        if py < 0 or px < 0 or py >= mask.shape[0] or px >= mask.shape[1]:
            return False
        return bool(mask[py, px] > 0)

    @staticmethod
    def _label_is_sfx(label: str) -> bool:
        normalized = (label or "").strip().lower().replace("_", " ").replace("-", " ")
        return any(token in normalized for token in {"sfx", "sound", "effect", "onomato", "onomatopoeia", "text free"})

    @staticmethod
    def _label_is_narration(label: str) -> bool:
        normalized = (label or "").strip().lower().replace("_", " ").replace("-", " ")
        return any(token in normalized for token in {"narration", "caption", "box", "thought"})

    def _kind_from_professional_candidate(self, candidate: ProfessionalBubbleCandidate, fallback_sfx: bool = False) -> str:
        if fallback_sfx or self._label_is_sfx(candidate.label):
            return "sfx"
        if self._label_is_narration(candidate.label):
            return "narration"
        return "dialogue"

    def _looks_like_sfx(self, detections: Sequence) -> bool:
        text = "".join(self._text(det) for det in detections).strip()
        if self.onomatopoeia_manager.is_onomatopoeia(text, self.idioma_entrada):
            return True
        boxes = [self._to_rect(det) for det in detections]
        if not boxes:
            return False
        merged = boxes[0]
        for box in boxes[1:]:
            merged = self._union(merged, box)
        _x, _y, w, h = merged
        aspect = max(w, h) / max(1, min(w, h))
        compact_len = len(text.replace(" ", ""))
        return aspect >= 4.0 and compact_len <= 10

    def _should_merge_detection_groups(self, a: Box, b: Box) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        gap_x = max(0, max(bx - ax2, ax - bx2))
        gap_y = max(0, max(by - ay2, ay - by2))
        avg_h = max(1.0, (ah + bh) / 2)
        avg_w = max(1.0, (aw + bw) / 2)
        x_overlap = self._overlap_ratio_1d(ax, ax2, bx, bx2)
        y_overlap = self._overlap_ratio_1d(ay, ay2, by, by2)
        if x_overlap >= 0.20 and gap_y <= max(12, avg_h * 1.55):
            return True
        if self.idioma_entrada in {"Japonés", "Chino", "Coreano"} and y_overlap >= 0.20 and gap_x <= max(12, avg_w * 1.25):
            return True
        if y_overlap >= 0.42 and gap_x <= max(10, avg_h * 0.9):
            return True
        return False

    def _group_detections(self, detections: Sequence) -> List[List]:
        groups: List[Tuple[Box, List]] = []
        for det in detections:
            try:
                rect = self._to_rect(det)
            except Exception:
                continue
            placed = False
            for i, (box, items) in enumerate(groups):
                if self._should_merge_detection_groups(box, rect):
                    groups[i] = (self._union(box, rect), items + [det])
                    placed = True
                    break
            if not placed:
                groups.append((rect, [det]))

        changed = True
        while changed:
            changed = False
            merged: List[Tuple[Box, List]] = []
            used = [False] * len(groups)
            for i, (box, items) in enumerate(groups):
                if used[i]:
                    continue
                current_box = box
                current_items = list(items)
                used[i] = True
                for j in range(i + 1, len(groups)):
                    if used[j]:
                        continue
                    other_box, other_items = groups[j]
                    if self._should_merge_detection_groups(current_box, other_box):
                        current_box = self._union(current_box, other_box)
                        current_items.extend(other_items)
                        used[j] = True
                        changed = True
                merged.append((current_box, current_items))
            groups = merged
        return [items for _box, items in groups]

    @staticmethod
    def _expand_box(box: Box, width: int, height: int, ratio_x: float, ratio_y: float, min_pad: int = 18) -> Box:
        x, y, w, h = box
        pad_x = max(min_pad, int(round(w * ratio_x)))
        pad_y = max(min_pad, int(round(h * ratio_y)))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def _text_box_mask(self, image_shape, box: Box, kind: str = "free_text") -> Tuple[np.ndarray, Box, float, str]:
        """Máscara para texto libre/SFX; no intenta detectar globos."""
        height, width = image_shape[:2]
        if kind == "sfx":
            expanded = self._expand_box(box, width, height, ratio_x=0.16, ratio_y=0.20, min_pad=8)
        else:
            expanded = self._expand_box(box, width, height, ratio_x=0.10, ratio_y=0.12, min_pad=6)
        x, y, w, h = expanded
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        return mask, expanded, 0.30, "ocr_text_box"

    @staticmethod
    def _merge_region_masks(regions: List[TextRegion]) -> List[TextRegion]:
        selected: List[TextRegion] = []
        for region in sorted(regions, key=lambda r: (r.confidence, BubbleDetector._area(r.bbox)), reverse=True):
            duplicate = False
            for existing in selected:
                inter = cv2.bitwise_and(region.mask, existing.mask)
                inter_area = int(cv2.countNonZero(inter))
                min_area = max(1, min(cv2.countNonZero(region.mask), cv2.countNonZero(existing.mask)))
                if inter_area / min_area > 0.82:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(region)
        return selected

    def _match_professional_candidate(
        self,
        text_box: Box,
        candidates: Sequence[ProfessionalBubbleCandidate],
        used: set[int],
        allow_sfx: bool,
    ) -> Tuple[int, ProfessionalBubbleCandidate] | Tuple[None, None]:
        if not candidates:
            return None, None
        text_center = self._center(text_box)
        text_area = max(1, self._area(text_box))
        best_idx = None
        best_score = -1.0
        for idx, candidate in enumerate(candidates):
            if idx in used:
                continue
            label_sfx = self._label_is_sfx(candidate.label)
            if label_sfx and not allow_sfx:
                continue
            inter_box = self._intersection_area(candidate.bbox, text_box)
            text_overlap = inter_box / text_area
            center_in_mask = self._point_inside_mask(text_center, candidate.mask)
            center_in_box = self._point_inside_box(text_center, candidate.bbox)
            if text_overlap < 0.06 and not center_in_mask and not center_in_box:
                continue
            cand_area = max(1, int(cv2.countNonZero(candidate.mask)) or self._area(candidate.bbox))
            relative_size = cand_area / text_area
            size_score = 1.0 if 1.1 <= relative_size <= 42 else 0.55
            score = text_overlap * 2.6 + (1.25 if center_in_mask else 0.0) + (0.55 if center_in_box else 0.0)
            score += min(0.45, candidate.confidence * 0.45) + size_score * 0.25
            if relative_size > 90:
                score *= 0.55
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            return None, None
        return best_idx, candidates[best_idx]

    def _region_from_professional_candidate(
        self,
        candidate: ProfessionalBubbleCandidate,
        text_box: Box,
        text_hint: str,
        ocr_confidence: float,
        detections_count: int,
        fallback_sfx: bool,
    ) -> TextRegion:
        kind = self._kind_from_professional_candidate(candidate, fallback_sfx=fallback_sfx)
        metadata = {
            "mask_source": candidate.source,
            "detector": "professional",
            "label": candidate.label,
            "bubble_model_repo": getattr(self.professional_detector, "repo_id", ""),
            "bubble_model_file": getattr(self.professional_detector, "filename", ""),
        }
        return TextRegion(
            bbox=candidate.bbox,
            text_bbox=text_box,
            mask=candidate.mask,
            kind=kind,
            confidence=max(float(ocr_confidence), float(candidate.confidence)),
            source_text_hint=text_hint,
            detections_count=detections_count,
            metadata=metadata,
        )

    def _get_professional_candidates(self, image: np.ndarray) -> List[ProfessionalBubbleCandidate]:
        candidates = self.professional_detector.detect(image)
        logger.info("Detector profesional: %s candidatos encontrados", len(candidates))
        return candidates

    def detect_primary_bubble_regions(self, image: np.ndarray) -> List[TextRegion]:
        candidates = self._get_professional_candidates(image)
        regions: List[TextRegion] = []
        for i, candidate in enumerate(candidates):
            kind = self._kind_from_professional_candidate(candidate, fallback_sfx=False)
            
            metadata = {
                "mask_source": candidate.source,
                "detector": "professional",
                "label": candidate.label,
                "bubble_model_repo": getattr(self.professional_detector, "repo_id", ""),
                "bubble_model_file": getattr(self.professional_detector, "filename", ""),
                "region_flow": "bubble_first_pretrained_only",
                "ocr_scope": "inside_detected_region",
            }
            regions.append(TextRegion(
                bbox=candidate.bbox,
                text_bbox=candidate.bbox,
                mask=candidate.mask,
                kind=kind,
                confidence=float(candidate.confidence),
                source_text_hint="",
                detections_count=0,
                metadata=metadata,
            ))
        return self._merge_region_masks(regions)

    def _detection_assignment_score(self, text_box: Box, region: TextRegion) -> float:
        text_area = max(1, self._area(text_box))
        inter = self._intersection_area(region.bbox, text_box) / text_area
        center = self._center(text_box)
        center_in_mask = self._point_inside_mask(center, region.mask)
        center_in_box = self._point_inside_box(center, region.bbox)
        if inter < 0.05 and not center_in_mask:
            return -1.0
        return inter * 2.4 + (1.2 if center_in_mask else 0.0) + (0.35 if center_in_box else 0.0)

    def _assign_text_detections_to_regions(self, regions: List[TextRegion], detections: Sequence) -> set[int]:
        assigned: set[int] = set()
        grouped: dict[int, List] = {idx: [] for idx in range(len(regions))}
        for det_idx, det in enumerate(detections or []):
            try:
                text_box = self._to_rect(det)
            except Exception:
                continue
            best_idx = None
            best_score = -1.0
            for idx, region in enumerate(regions):
                if region.kind in {"sfx", "free_text"}:
                    continue
                score = self._detection_assignment_score(text_box, region)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None and best_score >= 0.42:
                assigned.add(det_idx)
                grouped[best_idx].append(det)
        for idx, group in grouped.items():
            if not group:
                continue
            boxes = [self._to_rect(det) for det in group]
            text_box = boxes[0]
            for box in boxes[1:]:
                text_box = self._union(text_box, box)
            regions[idx].text_bbox = text_box
            regions[idx].source_text_hint = " ".join(self._text(det).strip() for det in group if self._text(det).strip())
            regions[idx].detections_count = len(group)
            try:
                regions[idx].confidence = max(regions[idx].confidence, float(np.mean([self._confidence(det) for det in group])))
            except Exception:
                pass
            regions[idx].metadata["ocr_global_hint"] = bool(regions[idx].source_text_hint)
            regions[idx].metadata["assigned_ocr_detections"] = len(group)
        return assigned

    def _is_inside_existing_region(self, text_box: Box, regions: Sequence[TextRegion]) -> bool:
        center = self._center(text_box)
        for region in regions:
            if region.kind in {"sfx", "free_text"}:
                continue
            if self._point_inside_mask(center, region.mask):
                return True
            if self._intersection_area(region.bbox, text_box) / max(1, self._area(text_box)) > 0.18:
                return True
        return False

    def _free_text_regions_from_detections(self, image: np.ndarray, detections: Sequence) -> List[TextRegion]:
        free_regions: List[TextRegion] = []
        
        # Obtener el tamaño total de la página para hacer cálculos de proporción
        img_height, img_width = image.shape[:2]
        img_area = img_height * img_width

        for i, group in enumerate(self._group_detections(detections)):
            boxes = [self._to_rect(det) for det in group]
            text_box = boxes[0]
            for box in boxes[1:]:
                text_box = self._union(text_box, box)
                
            # --- NUEVO FILTRO DE SEGURIDAD PARA FALSOS POSITIVOS ---
            bx, by, bw, bh = text_box
            box_area = bw * bh
            
            # Regla 1: Si "texto" ocupa más del 4% de TODA la página, es basura.
            # (Un texto libre real casi nunca es tan grande)
            if box_area > img_area * 0.04:
                continue
                
            # Regla 2: Si es una franja vertical absurda (más del 25% del alto de la página).
            if bh > img_height * 0.25:
                continue
                
            # Regla 3: Si es una franja horizontal (más del 50% del ancho).
            if bw > img_width * 0.50:
                continue
            # -------------------------------------------------------

            text_hint = " ".join(self._text(det).strip() for det in group if self._text(det).strip())
            conf = float(np.mean([self._confidence(det) for det in group])) if group else 0.0
            
            looks_sfx = self._looks_like_sfx(group)
            kind = "sfx" if looks_sfx else "free_text"
            
            if looks_sfx:
                razon = "Texto OCR fuera de globo, detectado por proporciones o diccionario como Onomatopeya (SFX)"
            else:
                razon = "Texto OCR agrupado que quedó huérfano (no está dentro de ningún globo de la IA)"
                
            mask, bbox, score, source = self._text_box_mask(image.shape, text_box, kind=kind)
            free_regions.append(TextRegion(
                bbox=bbox,
                text_bbox=text_box,
                mask=mask,
                kind=kind,
                confidence=max(conf, score),
                source_text_hint=text_hint,
                detections_count=len(group),
                metadata={
                    "mask_source": source,
                    "detector": "ocr_free_text",
                    "region_flow": "bubble_first_pretrained_only",
                    "ocr_scope": "free_text_or_sfx",
                },
            ))
        return free_regions
    
    def build_regions_from_bubbles_and_text(
        self,
        image: np.ndarray,
        bubble_regions: Sequence[TextRegion],
        detections: Sequence,
    ) -> List[TextRegion]:
        regions = list(bubble_regions or [])
        assigned = self._assign_text_detections_to_regions(regions, detections) if regions else set()
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
        return self._merge_region_masks(regions)

    def detect_regions(self, image: np.ndarray, detections: Sequence) -> List[TextRegion]:
        bubble_regions = self.detect_primary_bubble_regions(image)
        return self.build_regions_from_bubbles_and_text(image, bubble_regions, detections)

    @staticmethod
    def compose_mask(regions: Sequence[TextRegion], image_shape) -> np.ndarray:
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            if region.mask is not None and region.mask.size:
                mask = cv2.bitwise_or(mask, region.mask)
        return mask
