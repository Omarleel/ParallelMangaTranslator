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


class BubbleDebugMixin:
    """Escritura de artefactos visuales/JSON de debug del detector."""

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

            debug_region_rows: List[Dict[str, object]] = []
            for idx, region in enumerate(regions):
                x, y, w, h = region.bbox
                split = bool(region.metadata.get("split_from_merged_bubble"))
                onomato_debug = self._region_onomatopoeia_debug_metadata(region)
                is_onomato_free_text = bool(onomato_debug) and region.kind in {"free_text", "sfx", "onomatopoeia"}
                color = (0, 0, 255) if is_onomato_free_text else ((40, 180, 40) if split else (0, 165, 255))
                thickness = 3 if is_onomato_free_text else 2
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
                label_suffix = ":ONOM" if is_onomato_free_text else ""
                cv2.putText(canvas, f"R{idx}:{region.kind}{label_suffix}", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                tx, ty, tw, th = region.text_bbox
                text_color = (0, 0, 255) if is_onomato_free_text else (255, 80, 80)
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), text_color, 2 if is_onomato_free_text else 1)
                if is_onomato_free_text:
                    cv2.putText(canvas, "ONOMATOPEYA", (tx, ty + th + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA)
                if region.source_text_hint:
                    label = region.source_text_hint[:24]
                    label_y = ty + th + (30 if is_onomato_free_text else 14)
                    cv2.putText(canvas, label, (tx, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)
                debug_row = {
                    "region_index": idx,
                    "kind": region.kind,
                    "bbox": list(map(int, region.bbox)),
                    "text_bbox": list(map(int, region.text_bbox)),
                    "source_text_hint": region.source_text_hint,
                    "is_free_text_onomatopoeia": bool(is_onomato_free_text),
                }
                debug_row.update(onomato_debug)
                debug_region_rows.append(debug_row)

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
                    "red_thick_box": "texto libre/SFX identificado como onomatopeya; se etiqueta ONOMATOPEYA en la imagen debug",
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
                    "free_text_gap_recovery": self.free_text_gap_recovery,
                    "free_text_gap_max_px": self.free_text_gap_max_px,
                    "free_text_gap_min_y_overlap": self.free_text_gap_min_y_overlap,
                    "free_text_gap_min_ink_density": self.free_text_gap_min_density,
                    "free_text_gap_max_ink_density": self.free_text_gap_max_density,
                    "bubble_merge_debug_pair_limit": self.merge_debug_pair_limit,
                },
                "regions": debug_region_rows,
                "records": list(debug_records),
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Debug de globos guardado: %s | %s", png_path, json_path)
        except Exception as exc:
            logger.warning("No se pudo guardar debug de fusión/división de globos: %s", exc)
