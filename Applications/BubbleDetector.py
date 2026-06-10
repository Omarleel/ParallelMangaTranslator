from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from Applications.OnomatopoeiaManager import OnomatopoeiaManager
from Applications.ProcessingModels import Box, TextRegion
from Applications.ProfessionalBubbleDetector import ProfessionalBubbleCandidate, ProfessionalBubbleDetector
from .CacheManager import env_flag
from .LoggingConfig import get_logger

logger = get_logger(__name__)

BUBBLE_SPLIT_DEBUG_VERSION = "v5_cluster_bbox_logic_2026_06_10"


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
        self.split_merged_bubbles = env_flag("PMT_SPLIT_MERGED_BUBBLES", True)
        self.split_min_ocr_groups = self._int_env("PMT_BUBBLE_SPLIT_MIN_OCR_GROUPS", 2)
        self.split_min_gap_px = self._int_env("PMT_BUBBLE_SPLIT_MIN_GAP_PX", 18)
        self.split_gap_ratio = self._float_env("PMT_BUBBLE_SPLIT_GAP_RATIO", 0.70)
        # Una vez que las columnas OCR ya fueron agrupadas en clusters lógicos, la
        # separación entre clusters debe ser más sensible que la separación entre
        # grupos OCR individuales. Si usamos el mismo ratio conservador, dos globos
        # reales muy próximos quedan atrapados dentro de una sola caja grande.
        self.split_cluster_min_gap_px = self._int_env("PMT_BUBBLE_SPLIT_CLUSTER_MIN_GAP_PX", 12)
        self.split_cluster_gap_ratio = self._float_env("PMT_BUBBLE_SPLIT_CLUSTER_GAP_RATIO", 0.35)
        self.split_group_pad_x = self._float_env("PMT_BUBBLE_SPLIT_PAD_X", 0.85)
        self.split_group_pad_y = self._float_env("PMT_BUBBLE_SPLIT_PAD_Y", 1.05)
        self.split_group_min_pad = self._int_env("PMT_BUBBLE_SPLIT_MIN_PAD", 18)
        # Agrupación OCR conservadora. En manga vertical, dos columnas cercanas pueden
        # pertenecer a textos/globos distintos; por defecto NO fusionamos columnas CJK
        # solo por estar cerca horizontalmente. Esto evita que una detección grande
        # termine concatenando textos independientes.
        self.ocr_merge_x_overlap = self._float_env("PMT_OCR_GROUP_MERGE_X_OVERLAP", 0.52)
        self.ocr_merge_y_gap_ratio = self._float_env("PMT_OCR_GROUP_MERGE_Y_GAP_RATIO", 0.55)
        self.ocr_merge_cjk_y_overlap = self._float_env("PMT_OCR_GROUP_MERGE_CJK_Y_OVERLAP", 0.80)
        self.ocr_merge_cjk_x_gap_ratio = self._float_env("PMT_OCR_GROUP_MERGE_CJK_X_GAP_RATIO", 0.35)
        self.ocr_merge_cjk_columns = env_flag("PMT_OCR_GROUP_MERGE_CJK_COLUMNS", False)
        self.ocr_merge_line_y_overlap = self._float_env("PMT_OCR_GROUP_MERGE_LINE_Y_OVERLAP", 0.72)
        self.ocr_merge_line_x_gap_ratio = self._float_env("PMT_OCR_GROUP_MERGE_LINE_X_GAP_RATIO", 0.20)
        self.ocr_merge_line_horizontal_only = env_flag("PMT_OCR_GROUP_MERGE_LINE_HORIZONTAL_ONLY", True)
        # Filtros de texto libre. Antes una línea horizontal que ocupase más del 50%
        # del ancho de la página se descartaba siempre; eso elimina páginas tipo
        # prólogo/afterword/créditos, donde el texto real es precisamente ancho.
        # Ahora el rechazo depende de tamaño + señal OCR + confianza, con límites
        # duros solo para manchas enormes que casi nunca son texto.
        self.free_text_max_area_ratio = self._float_env("PMT_FREE_TEXT_MAX_AREA_RATIO", 0.12)
        self.free_text_hard_max_area_ratio = self._float_env("PMT_FREE_TEXT_HARD_MAX_AREA_RATIO", 0.22)
        self.free_text_max_width_ratio = self._float_env("PMT_FREE_TEXT_MAX_WIDTH_RATIO", 0.96)
        self.free_text_max_height_ratio = self._float_env("PMT_FREE_TEXT_MAX_HEIGHT_RATIO", 0.60)
        self.free_text_min_confidence = self._float_env("PMT_FREE_TEXT_MIN_CONFIDENCE", 0.08)
        self.free_text_large_min_confidence = self._float_env("PMT_FREE_TEXT_LARGE_MIN_CONFIDENCE", 0.16)
        self.merge_debug = env_flag("PMT_BUBBLE_MERGE_DEBUG", False)
        self.merge_debug_pair_limit = self._int_env("PMT_BUBBLE_MERGE_DEBUG_PAIR_LIMIT", 160)
        default_debug_dir = str(Path(os.getenv("PMT_PROJECT_DIR", "Dataset")) / "Outputs" / "DebugGlobos")
        debug_dir_raw = os.getenv("PMT_BUBBLE_MERGE_DEBUG_DIR", "").strip()
        self.merge_debug_dir = Path(debug_dir_raw or default_debug_dir)
        self._debug_page_index = 0
        if self.merge_debug:
            logger.info("Bubble split debug activo: %s", BUBBLE_SPLIT_DEBUG_VERSION)
        self.enabled = env_flag("PMT_BUBBLE_DETECTION", True)
        if not self.enabled:
            raise RuntimeError(
                "PMT_BUBBLE_DETECTION=0 no está permitido en esta versión: la detección de globos "
                "debe hacerse con un modelo preentrenado."
            )
        self.professional_detector = ProfessionalBubbleDetector()

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        try:
            return int(float(os.getenv(name, str(default))))
        except (TypeError, ValueError):
            return default

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
        compact_text = re.sub(r"\s+", "", text)
        compact_len = len(compact_text)
        # Una línea muy horizontal y corta puede ser un SFX, pero en páginas de
        # notas los renglones japoneses largos también son muy anchos. Si contiene
        # hiragana/kanji suficientes para parecer frase, no lo clasifiques como SFX
        # solo por proporción.
        looks_sentence_like = compact_len >= 7 and bool(re.search(r"[\u3040-\u309f\u3400-\u9fff]", compact_text))
        return aspect >= 4.0 and compact_len <= 10 and not looks_sentence_like

    @staticmethod
    def _has_meaningful_text_signal(text: str) -> bool:
        """Devuelve True si el OCR contiene letras/números/CJK reales.

        No exige que la lectura sea perfecta: solo evita que cajas enormes formadas
        por tramas, bordes o ruido pasen como texto libre cuando EasyOCR devuelve
        símbolos sueltos.
        """
        return bool(re.search(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", str(text or "")))

    def _should_keep_free_text_group(
        self,
        text_box: Box,
        text_hint: str,
        confidence: float,
        image_shape,
        looks_sfx: bool,
    ) -> Tuple[bool, str]:
        img_height, img_width = image_shape[:2]
        img_area = max(1, img_height * img_width)
        bx, by, bw, bh = text_box
        box_area = max(1, bw * bh)
        area_ratio = box_area / img_area
        width_ratio = bw / max(1, img_width)
        height_ratio = bh / max(1, img_height)
        has_signal = self._has_meaningful_text_signal(text_hint)

        # Nada que parezca texto y además baja confianza: probablemente ruido.
        if not has_signal and not looks_sfx and confidence < self.free_text_min_confidence:
            return False, "sin_senal_textual_y_baja_confianza"

        # Límites duros para regiones gigantes; estas cajas suelen ser fondos, paneles
        # o dibujos completos detectados como texto.
        if area_ratio > self.free_text_hard_max_area_ratio:
            return False, "area_gigante"

        # Límites blandos: solo se rechazan si la caja grande no tiene señal textual
        # suficiente. Así se conservan líneas horizontales largas de páginas de notas.
        is_large = (
            area_ratio > self.free_text_max_area_ratio
            or width_ratio > self.free_text_max_width_ratio
            or height_ratio > self.free_text_max_height_ratio
        )
        if is_large and (not has_signal or confidence < self.free_text_large_min_confidence):
            return False, "region_grande_sin_senal_ocr_fiable"

        return True, "aceptado"

    def _detection_group_merge_decision(self, a: Box, b: Box) -> Dict[str, object]:
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

        merge = False
        reason = "separados"
        max_vertical_gap = max(8.0, avg_h * self.ocr_merge_y_gap_ratio)
        max_cjk_horizontal_gap = max(6.0, avg_w * self.ocr_merge_cjk_x_gap_ratio)
        # Antes este límite usaba avg_h, lo que en texto vertical permitía saltos enormes
        # entre columnas. Debe depender del ancho de los fragmentos y, por defecto, solo
        # aplica a texto horizontal.
        max_line_gap = max(6.0, avg_w * self.ocr_merge_line_x_gap_ratio)
        a_horizontalish = aw >= ah * 0.90
        b_horizontalish = bw >= bh * 0.90
        both_horizontalish = a_horizontalish and b_horizontalish

        # Reglas conservadoras: evita unir globos/textos cercanos; solo une fragmentos
        # con solape claro y separación pequeña. Los umbrales son ajustables por PMT_*.
        if x_overlap >= self.ocr_merge_x_overlap and gap_y <= max_vertical_gap:
            merge = True
            reason = "fragmentos_verticales_mismo_bloque"
        elif (
            self.ocr_merge_cjk_columns
            and self.idioma_entrada in {"Japonés", "Chino", "Coreano"}
            and y_overlap >= self.ocr_merge_cjk_y_overlap
            and gap_x <= max_cjk_horizontal_gap
        ):
            merge = True
            reason = "columnas_cjk_mismo_bloque"
        elif (
            (not self.ocr_merge_line_horizontal_only or both_horizontalish)
            and y_overlap >= self.ocr_merge_line_y_overlap
            and gap_x <= max_line_gap
        ):
            merge = True
            reason = "fragmentos_horizontales_misma_linea"

        return {
            "merge": merge,
            "reason": reason,
            "a": list(map(int, a)),
            "b": list(map(int, b)),
            "gap_x": int(gap_x),
            "gap_y": int(gap_y),
            "x_overlap": round(float(x_overlap), 4),
            "y_overlap": round(float(y_overlap), 4),
            "thresholds": {
                "x_overlap": self.ocr_merge_x_overlap,
                "vertical_gap": round(float(max_vertical_gap), 3),
                "cjk_y_overlap": self.ocr_merge_cjk_y_overlap,
                "cjk_horizontal_gap": round(float(max_cjk_horizontal_gap), 3),
                "line_y_overlap": self.ocr_merge_line_y_overlap,
                "line_gap": round(float(max_line_gap), 3),
                "cjk_columns_enabled": self.ocr_merge_cjk_columns,
                "line_horizontal_only": self.ocr_merge_line_horizontal_only,
            },
            "orientation": {
                "a_horizontalish": bool(a_horizontalish),
                "b_horizontalish": bool(b_horizontalish),
            },
        }

    def _should_merge_detection_groups(self, a: Box, b: Box) -> bool:
        return bool(self._detection_group_merge_decision(a, b)["merge"])

    def _group_detections(self, detections: Sequence, trace_decisions: Optional[List[Dict[str, object]]] = None) -> List[List]:
        groups: List[Tuple[Box, List]] = []
        for det_idx, det in enumerate(detections):
            try:
                rect = self._to_rect(det)
            except Exception:
                continue
            placed = False
            for i, (box, items) in enumerate(groups):
                decision = self._detection_group_merge_decision(box, rect)
                if trace_decisions is not None and len(trace_decisions) < self.merge_debug_pair_limit:
                    trace_item = dict(decision)
                    trace_item.update({
                        "stage": "asignacion_inicial",
                        "group_index": i,
                        "group_items": len(items),
                        "detection_index": det_idx,
                        "detection_text": self._text(det),
                    })
                    trace_decisions.append(trace_item)
                if decision["merge"]:
                    groups[i] = (self._union(box, rect), items + [det])
                    placed = True
                    break
            if not placed:
                groups.append((rect, [det]))

        changed = True
        pass_index = 0
        while changed:
            pass_index += 1
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
                    decision = self._detection_group_merge_decision(current_box, other_box)
                    if trace_decisions is not None and len(trace_decisions) < self.merge_debug_pair_limit:
                        trace_item = dict(decision)
                        trace_item.update({
                            "stage": "consolidacion",
                            "pass": pass_index,
                            "group_a": i,
                            "group_b": j,
                            "group_a_items": len(current_items),
                            "group_b_items": len(other_items),
                        })
                        trace_decisions.append(trace_item)
                    if decision["merge"]:
                        current_box = self._union(current_box, other_box)
                        current_items.extend(other_items)
                        used[j] = True
                        changed = True
                merged.append((current_box, current_items))
            groups = merged
        return [items for _box, items in groups]

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

    @staticmethod
    def _box_intersection(a: Box, b: Box) -> Optional[Box]:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return None
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> Optional[Box]:
        pts = cv2.findNonZero(mask)
        if pts is None:
            return None
        x, y, w, h = cv2.boundingRect(pts)
        return int(x), int(y), int(w), int(h)

    def _detections_box(self, detections: Sequence) -> Box:
        boxes = [self._to_rect(det) for det in detections]
        merged = boxes[0]
        for box in boxes[1:]:
            merged = self._union(merged, box)
        return merged

    def _detections_text(self, detections: Sequence) -> str:
        return " ".join(self._text(det).strip() for det in detections if self._text(det).strip())

    def _detections_confidence(self, detections: Sequence) -> float:
        try:
            values = [self._confidence(det) for det in detections]
            return float(np.mean(values)) if values else 0.0
        except Exception:
            return 0.0

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

    def _save_merge_debug_artifacts(
        self,
        image: np.ndarray,
        regions: Sequence[TextRegion],
        debug_records: Sequence[Dict[str, object]],
    ) -> None:
        if not self.merge_debug:
            return
        try:
            self.merge_debug_dir.mkdir(parents=True, exist_ok=True)
            self._debug_page_index += 1
            stem = f"pagina_{self._debug_page_index:04d}"
            canvas = image.copy()

            # Primero dibuja datos técnicos de la decisión: detecciones OCR crudas y
            # grupos resultantes. Esto permite ver por qué se fusionó o se separó texto.
            for record in debug_records:
                try:
                    region_idx = int(record.get("region_index", -1))
                except Exception:
                    region_idx = -1
                for det in record.get("raw_ocr_detections", []) or []:
                    try:
                        x, y, w, h = [int(v) for v in det.get("bbox", [])]
                        det_idx = int(det.get("detection_index", -1))
                    except Exception:
                        continue
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 1)
                    cv2.putText(canvas, f"R{region_idx}:D{det_idx}", (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 200), 1, cv2.LINE_AA)
                for group in record.get("ocr_groups", []) or []:
                    try:
                        x, y, w, h = [int(v) for v in group.get("bbox", [])]
                        group_idx = int(group.get("group_index", -1))
                    except Exception:
                        continue
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 80, 80), 1)
                    cv2.putText(canvas, f"R{region_idx}:G{group_idx}", (x, y + h + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 80, 80), 1, cv2.LINE_AA)
                for cluster in record.get("split_clusters", []) or []:
                    try:
                        x, y, w, h = [int(v) for v in cluster.get("bbox", [])]
                        cluster_idx = int(cluster.get("cluster_index", -1))
                    except Exception:
                        continue
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 80, 255), 2)
                    cv2.putText(canvas, f"R{region_idx}:C{cluster_idx}", (x, max(12, y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 80, 255), 1, cv2.LINE_AA)

            for idx, region in enumerate(regions):
                x, y, w, h = region.bbox
                split = bool(region.metadata.get("split_from_merged_bubble"))
                color = (40, 180, 40) if split else (0, 165, 255)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
                cv2.putText(canvas, f"R{idx}:{region.kind}", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                tx, ty, tw, th = region.text_bbox
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (255, 80, 80), 1)
                if region.source_text_hint:
                    label = region.source_text_hint[:24]
                    cv2.putText(canvas, label, (tx, ty + th + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 80, 80), 1, cv2.LINE_AA)

            png_path = self.merge_debug_dir / f"{stem}_globos.png"
            json_path = self.merge_debug_dir / f"{stem}_decisiones.json"
            cv2.imwrite(str(png_path), canvas)
            payload = {
                "bubble_split_debug_version": BUBBLE_SPLIT_DEBUG_VERSION,
                "page_index": self._debug_page_index,
                "legend": {
                    "orange_box": "región final conservada por el detector",
                    "green_box": "subregión final creada al dividir un globo fusionado",
                    "yellow_box": "detección OCR cruda asignada a una región",
                    "blue_box": "grupo OCR después de aplicar reglas de fusión",
                    "magenta_box": "cluster lógico de grupos OCR que se renderizará como un solo globo",
                },
                "thresholds": {
                    "split_min_ocr_groups": self.split_min_ocr_groups,
                    "split_min_gap_px": self.split_min_gap_px,
                    "split_gap_ratio": self.split_gap_ratio,
                    "split_pad_x": self.split_group_pad_x,
                    "split_pad_y": self.split_group_pad_y,
                    "ocr_group_merge_x_overlap": self.ocr_merge_x_overlap,
                    "ocr_group_merge_y_gap_ratio": self.ocr_merge_y_gap_ratio,
                    "ocr_group_merge_cjk_y_overlap": self.ocr_merge_cjk_y_overlap,
                    "ocr_group_merge_cjk_x_gap_ratio": self.ocr_merge_cjk_x_gap_ratio,
                    "ocr_group_merge_cjk_columns": self.ocr_merge_cjk_columns,
                    "ocr_group_merge_line_y_overlap": self.ocr_merge_line_y_overlap,
                    "ocr_group_merge_line_x_gap_ratio": self.ocr_merge_line_x_gap_ratio,
                    "ocr_group_merge_line_horizontal_only": self.ocr_merge_line_horizontal_only,
                    "free_text_max_area_ratio": self.free_text_max_area_ratio,
                    "free_text_hard_max_area_ratio": self.free_text_hard_max_area_ratio,
                    "free_text_max_width_ratio": self.free_text_max_width_ratio,
                    "free_text_max_height_ratio": self.free_text_max_height_ratio,
                    "free_text_min_confidence": self.free_text_min_confidence,
                    "free_text_large_min_confidence": self.free_text_large_min_confidence,
                    "bubble_merge_debug_pair_limit": self.merge_debug_pair_limit,
                },
                "records": list(debug_records),
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Debug de globos guardado: %s | %s", png_path, json_path)
        except Exception as exc:
            logger.warning("No se pudo guardar debug de fusión/división de globos: %s", exc)

    def _free_text_regions_from_detections(self, image: np.ndarray, detections: Sequence) -> List[TextRegion]:
        free_regions: List[TextRegion] = []

        for i, group in enumerate(self._group_detections(detections)):
            boxes = [self._to_rect(det) for det in group]
            text_box = boxes[0]
            for box in boxes[1:]:
                text_box = self._union(text_box, box)

            text_hint = " ".join(self._text(det).strip() for det in group if self._text(det).strip())
            conf = float(np.mean([self._confidence(det) for det in group])) if group else 0.0

            looks_sfx = self._looks_like_sfx(group)
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
                    "free_text_filter_reason": filter_reason,
                    "free_text_confidence": round(float(conf), 4),
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
        merged_regions = self._merge_region_masks(regions)
        self._save_merge_debug_artifacts(image, merged_regions, debug_records)
        return merged_regions

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
