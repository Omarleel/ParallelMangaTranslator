from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.detection.professional_bubble_detector import ProfessionalBubbleCandidate
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.processing_models import Box, TextRegion
from parallel_manga_translator.geometry.box_geometry import BoxGeometry

logger = get_logger(__name__)
BUBBLE_SPLIT_DEBUG_VERSION = "v7_bubble_onomatopoeia_translation_2026_06_11"


class BubbleRegionBuilderMixin:
    """Construcción de regiones a partir del detector profesional y detecciones OCR."""

    def _debug_raw_detection_items(self, detections: Sequence) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        for det_idx, det in enumerate(detections or []):
            try:
                box = self._to_rect(det)
            except Exception:
                continue
            items.append({
                "detection_index": det_idx,
                "bbox": list(map(int, box)),
                "text": self._text(det),
                "confidence": round(float(self._confidence(det)), 4),
            })
        return items

    def _debug_pairwise_raw_merge_decisions(self, detections: Sequence) -> Tuple[List[Dict[str, object]], bool]:
        boxes: List[Tuple[int, Box, object]] = []
        for det_idx, det in enumerate(detections or []):
            try:
                boxes.append((det_idx, self._to_rect(det), det))
            except Exception:
                continue
        decisions: List[Dict[str, object]] = []
        truncated = False
        for local_i, (det_i, box_i, raw_i) in enumerate(boxes):
            for det_j, box_j, raw_j in boxes[local_i + 1:]:
                if len(decisions) >= self.merge_debug_pair_limit:
                    truncated = True
                    return decisions, truncated
                decision = dict(self._detection_group_merge_decision(box_i, box_j))
                decision.update({
                    "detection_a": det_i,
                    "detection_b": det_j,
                    "text_a": self._text(raw_i),
                    "text_b": self._text(raw_j),
                })
                decisions.append(decision)
        return decisions, truncated

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
        for region in sorted(regions, key=lambda r: (r.confidence, BoxGeometry.area(r.bbox)), reverse=True):
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

    def _assign_text_detections_to_regions(self, regions: List[TextRegion], detections: Sequence) -> Tuple[set[int], Dict[int, List]]:
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
        return assigned, grouped

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
