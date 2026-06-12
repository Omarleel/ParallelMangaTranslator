from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.detection.professional_bubble_detector import ProfessionalBubbleCandidate
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.models.processing_models import Box, TextRegion

logger = get_logger(__name__)
BUBBLE_SPLIT_DEBUG_VERSION = "v7_bubble_onomatopoeia_translation_2026_06_11"


class BubbleSplitterMixin:
    """Responsabilidad única: decidir y ejecutar la división de globos fusionados."""

    def _split_group_decisions(
        self,
        group_boxes: Sequence[Box],
        *,
        min_gap_px: Optional[int] = None,
        gap_ratio: Optional[float] = None,
        decision_scope: str = "ocr_groups",
    ) -> List[Dict[str, object]]:
        """Decide si cajas de texto representan globos distintos.

        `ocr_groups` usa umbrales conservadores para no partir columnas dentro de
        un mismo globo. `clusters` usa umbrales más sensibles porque en esa etapa
        las columnas cercanas ya fueron agrupadas y lo que queda suele representar
        globos diferentes dentro de una detección grande.
        """
        decisions: List[Dict[str, object]] = []
        min_gap = int(self.split_min_gap_px if min_gap_px is None else min_gap_px)
        ratio = float(self.split_gap_ratio if gap_ratio is None else gap_ratio)
        for i, a in enumerate(group_boxes):
            for j in range(i + 1, len(group_boxes)):
                b = group_boxes[j]
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
                horizontal_gap_limit = max(float(min_gap), avg_w * ratio)
                vertical_gap_limit = max(float(min_gap), avg_h * ratio)
                separated_horizontal = gap_x >= horizontal_gap_limit and y_overlap >= 0.08
                separated_vertical = gap_y >= vertical_gap_limit and x_overlap >= 0.08
                separated_diagonal = gap_x >= min_gap and gap_y >= min_gap
                split = bool(separated_horizontal or separated_vertical or separated_diagonal)
                if separated_horizontal:
                    reason = "separacion_horizontal_entre_globos"
                elif separated_vertical:
                    reason = "separacion_vertical_entre_globos"
                elif separated_diagonal:
                    reason = "separacion_diagonal_entre_globos"
                else:
                    reason = "distancia_insuficiente_para_dividir"
                decisions.append({
                    "group_a": i,
                    "group_b": j,
                    "split": split,
                    "reason": reason,
                    "gap_x": int(gap_x),
                    "gap_y": int(gap_y),
                    "x_overlap": round(float(x_overlap), 4),
                    "y_overlap": round(float(y_overlap), 4),
                    "thresholds": {
                        "min_gap_px": min_gap,
                        "gap_ratio": ratio,
                        "decision_scope": decision_scope,
                        "horizontal_gap_limit": round(float(horizontal_gap_limit), 3),
                        "vertical_gap_limit": round(float(vertical_gap_limit), 3),
                    },
                })
        return decisions

    def _should_split_region_from_groups(self, region: TextRegion, grouped_detections: Sequence[Sequence]) -> Tuple[bool, List[Dict[str, object]], str]:
        if not self.split_merged_bubbles:
            return False, [], "split_desactivado"
        if region.kind not in {"dialogue", "narration", "unknown"}:
            return False, [], "tipo_no_divisible"
        if len(grouped_detections) < max(2, self.split_min_ocr_groups):
            return False, [], "grupos_ocr_insuficientes"
        group_boxes = [self._detections_box(group) for group in grouped_detections if group]
        if len(group_boxes) < max(2, self.split_min_ocr_groups):
            return False, [], "cajas_ocr_insuficientes"
        decisions = self._split_group_decisions(group_boxes)
        should_split = any(bool(item.get("split")) for item in decisions)
        return should_split, decisions, "division_por_grupos_ocr" if should_split else "grupos_demasiado_cercanos"

    def _cluster_groups_for_split(
        self,
        grouped_detections: Sequence[Sequence],
        pair_decisions: Sequence[Dict[str, object]],
    ) -> List[List[int]]:
        """Agrupa columnas/fragmentos OCR que pertenecen al mismo globo lógico.

        La agrupación OCR previa se mantiene conservadora para no mezclar globos
        distintos antes de tomar la decisión. Pero al momento de dividir una región
        profesional fusionada no debemos crear una subregión por cada columna: si
        dos grupos OCR no activan una separación, se consideran parte del mismo
        globo y se fusionan en un cluster.
        """
        n = len(grouped_detections)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for decision in pair_decisions or []:
            try:
                a = int(decision.get("group_a"))
                b = int(decision.get("group_b"))
            except (TypeError, ValueError):
                continue
            if a < 0 or b < 0 or a >= n or b >= n:
                continue
            if not bool(decision.get("split")):
                union(a, b)

        clusters_by_root: Dict[int, List[int]] = {}
        for idx in range(n):
            clusters_by_root.setdefault(find(idx), []).append(idx)
        return sorted(clusters_by_root.values(), key=lambda cluster: min(cluster))

    def _merge_detection_groups_by_indices(
        self,
        grouped_detections: Sequence[Sequence],
        clusters: Sequence[Sequence[int]],
    ) -> List[List]:
        merged: List[List] = []
        for cluster in clusters:
            items: List = []
            for group_idx in cluster:
                items.extend(grouped_detections[group_idx])
            if items:
                merged.append(items)
        return merged

    def _significant_mask_components(self, region: TextRegion) -> List[Tuple[np.ndarray, Box, int]]:
        if region.mask is None or region.mask.size == 0:
            return []
        mask_bin = np.uint8(region.mask > 0)
        total_area = int(cv2.countNonZero(mask_bin))
        if total_area <= 0:
            return []
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        min_area = max(24, int(total_area * 0.025))
        components: List[Tuple[np.ndarray, Box, int]] = []
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            comp_mask = np.zeros_like(region.mask, dtype=np.uint8)
            comp_mask[labels == label_idx] = 255
            bbox = self._mask_bbox(comp_mask)
            if bbox:
                components.append((comp_mask, bbox, area))
        return sorted(components, key=lambda item: item[2], reverse=True)

    def _expanded_group_box(self, text_box: Box, region_box: Box, image_shape) -> Box:
        height, width = image_shape[:2]
        expanded = self._expand_box(
            text_box,
            width,
            height,
            ratio_x=self.split_group_pad_x,
            ratio_y=self.split_group_pad_y,
            min_pad=self.split_group_min_pad,
        )
        return self._box_intersection(expanded, region_box) or expanded

    def _split_mask_by_group_centers(
        self,
        region: TextRegion,
        group_boxes: Sequence[Box],
        image_shape,
    ) -> List[Tuple[np.ndarray, Box, str]]:
        components = self._significant_mask_components(region)
        centers = [self._center(box) for box in group_boxes]

        if len(components) >= 2:
            used_components: set[int] = set()
            selected: List[Tuple[np.ndarray, Box, str]] = []
            for center in centers:
                best_idx = None
                best_dist = float("inf")
                for idx, (_mask, bbox, _area) in enumerate(components):
                    if idx in used_components:
                        continue
                    cx, cy = self._center(bbox)
                    dist = (center[0] - cx) ** 2 + (center[1] - cy) ** 2
                    if self._point_inside_box(center, bbox):
                        dist *= 0.25
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                if best_idx is None:
                    break
                used_components.add(best_idx)
                comp_mask, comp_bbox, _area = components[best_idx]
                selected.append((comp_mask, comp_bbox, "componentes_mascara"))
            if len(selected) == len(group_boxes) and len({tuple(item[1]) for item in selected}) >= 2:
                return selected

        base_mask = np.uint8(region.mask > 0) * 255
        ys, xs = np.where(base_mask > 0)
        if len(xs) == 0:
            base_mask = np.zeros(image_shape[:2], dtype=np.uint8)
            x, y, w, h = region.bbox
            cv2.rectangle(base_mask, (x, y), (x + w, y + h), 255, -1)
            ys, xs = np.where(base_mask > 0)
        center_array = np.array(centers, dtype=np.float32)
        points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        distances = ((points[:, None, :] - center_array[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(distances, axis=1)

        split_masks: List[Tuple[np.ndarray, Box, str]] = []
        for group_idx, text_box in enumerate(group_boxes):
            mask = np.zeros_like(base_mask, dtype=np.uint8)
            selected = nearest == group_idx
            mask[ys[selected], xs[selected]] = 255
            expanded = self._expanded_group_box(text_box, region.bbox, image_shape)
            ex, ey, ew, eh = expanded
            clip = np.zeros_like(base_mask, dtype=np.uint8)
            cv2.rectangle(clip, (ex, ey), (ex + ew, ey + eh), 255, -1)
            mask = cv2.bitwise_and(mask, clip)
            if cv2.countNonZero(mask) == 0:
                mask = clip
            bbox = self._mask_bbox(mask) or expanded
            split_masks.append((mask, bbox, "ocr_voronoi"))
        return split_masks

    def _split_merged_bubble_regions(
        self,
        image: np.ndarray,
        regions: Sequence[TextRegion],
        detections_by_region: Dict[int, List],
    ) -> Tuple[List[TextRegion], List[Dict[str, object]]]:
        split_regions: List[TextRegion] = []
        debug_records: List[Dict[str, object]] = []

        for idx, region in enumerate(regions):
            assigned_detections = detections_by_region.get(idx, [])
            merge_trace: List[Dict[str, object]] = []
            grouped_detections = self._group_detections(assigned_detections, trace_decisions=merge_trace) if assigned_detections else []
            raw_pair_decisions, raw_pair_truncated = self._debug_pairwise_raw_merge_decisions(assigned_detections)
            group_boxes = [self._detections_box(group) for group in grouped_detections if group]
            should_split, pair_decisions, reason = self._should_split_region_from_groups(region, grouped_detections)

            record: Dict[str, object] = {
                "region_index": idx,
                "kind": region.kind,
                "bbox": list(map(int, region.bbox)),
                "text_bbox": list(map(int, region.text_bbox)),
                "assigned_ocr_detections": len(assigned_detections),
                "raw_ocr_detections": self._debug_raw_detection_items(assigned_detections),
                "raw_ocr_pair_decisions": raw_pair_decisions,
                "raw_ocr_pair_decisions_truncated": raw_pair_truncated,
                "ocr_group_merge_trace": merge_trace,
                "ocr_group_merge_trace_truncated": len(merge_trace) >= self.merge_debug_pair_limit,
                "ocr_groups": [
                    {
                        "group_index": group_idx,
                        "bbox": list(map(int, box)),
                        "text": self._detections_text(grouped_detections[group_idx]),
                        "detections": len(grouped_detections[group_idx]),
                    }
                    for group_idx, box in enumerate(group_boxes)
                ],
                "pair_decisions": pair_decisions,
                "split": should_split,
                "reason": reason,
            }

            if not should_split:
                split_regions.append(region)
                debug_records.append(record)
                continue

            split_clusters = self._cluster_groups_for_split(grouped_detections, pair_decisions)
            clustered_detections = self._merge_detection_groups_by_indices(grouped_detections, split_clusters)
            cluster_boxes = [self._detections_box(group) for group in clustered_detections if group]
            record["split_clusters"] = [
                {
                    "cluster_index": cluster_idx,
                    "group_indices": [int(group_idx) for group_idx in group_indices],
                    "bbox": list(map(int, cluster_boxes[cluster_idx])),
                    "text": self._detections_text(clustered_detections[cluster_idx]),
                    "detections": len(clustered_detections[cluster_idx]),
                }
                for cluster_idx, group_indices in enumerate(split_clusters)
                if cluster_idx < len(cluster_boxes)
            ]

            if len(clustered_detections) < max(2, self.split_min_ocr_groups):
                record["split"] = False
                record["reason"] = "clusters_logicos_insuficientes"
                split_regions.append(region)
                debug_records.append(record)
                continue

            cluster_decisions = self._split_group_decisions(
                cluster_boxes,
                min_gap_px=self.split_cluster_min_gap_px,
                gap_ratio=self.split_cluster_gap_ratio,
                decision_scope="clusters_logicos",
            )
            record["cluster_pair_decisions"] = cluster_decisions
            if not any(bool(item.get("split")) for item in cluster_decisions):
                record["split"] = False
                record["reason"] = "clusters_logicos_demasiado_cercanos"
                split_regions.append(region)
                debug_records.append(record)
                continue

            split_masks = self._split_mask_by_group_centers(region, cluster_boxes, image.shape)
            if len(split_masks) != len(clustered_detections):
                record["split"] = False
                record["reason"] = "fallo_generando_submascaras"
                split_regions.append(region)
                debug_records.append(record)
                continue

            created_regions: List[TextRegion] = []
            for cluster_idx, group in enumerate(clustered_detections):
                mask, bbox, split_method = split_masks[cluster_idx]
                text_box = cluster_boxes[cluster_idx]
                text_hint = self._detections_text(group)
                conf = max(region.confidence, self._detections_confidence(group))
                metadata = dict(region.metadata or {})
                metadata.update({
                    "split_from_merged_bubble": True,
                    "split_parent_bbox": list(map(int, region.bbox)),
                    "split_cluster_index": cluster_idx,
                    "split_cluster_group_indices": [int(i) for i in split_clusters[cluster_idx]],
                    "split_clusters_total": len(clustered_detections),
                    "split_groups_total": len(grouped_detections),
                    "split_method": split_method,
                    "split_reason": reason,
                    "split_cluster_gap_ratio": self.split_cluster_gap_ratio,
                    "split_cluster_min_gap_px": self.split_cluster_min_gap_px,
                    "ocr_group_text": text_hint,
                })
                created_regions.append(TextRegion(
                    bbox=bbox,
                    text_bbox=text_box,
                    mask=mask,
                    kind=region.kind,
                    confidence=conf,
                    source_text_hint=text_hint,
                    detections_count=len(group),
                    metadata=metadata,
                ))
            split_regions.extend(created_regions)
            record["created_regions"] = [
                {
                    "bbox": list(map(int, created.bbox)),
                    "text_bbox": list(map(int, created.text_bbox)),
                    "split_method": created.metadata.get("split_method"),
                    "text": created.source_text_hint,
                }
                for created in created_regions
            ]
            debug_records.append(record)

        return split_regions, debug_records
