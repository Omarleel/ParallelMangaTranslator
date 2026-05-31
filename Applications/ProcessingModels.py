from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Box = Tuple[int, int, int, int]


@dataclass
class TextRegion:
    """Región lógica que viaja por limpieza, OCR, traducción y renderizado.

    bbox/render_bbox cubre el área donde se puede dibujar la traducción.
    text_bbox cubre el texto OCR global asociado, si existe. En el flujo bubble-first puede
    coincidir con bbox aunque todavía no haya OCR.
    mask es una máscara global de la zona a limpiar/renderizar; para globos intenta ser
    el interior real del globo, no solo la caja del texto.
    """

    bbox: Box
    text_bbox: Box
    mask: np.ndarray
    kind: str = "dialogue"  # dialogue | sfx | free_text | narration | unknown
    confidence: float = 0.0
    source_text_hint: str = ""
    detections_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def render_bbox(self) -> Box:
        return self.bbox

    @property
    def ocr_bbox(self) -> Box:
        return self.bbox

    def local_mask(self) -> np.ndarray:
        x, y, w, h = self.bbox
        return self.mask[y:y + h, x:x + w]

    def to_json(self, original: str = "", translation: str = "") -> Dict[str, Any]:
        x, y, w, h = self.bbox
        tx, ty, tw, th = self.text_bbox
        payload = {
            "tipo": self.kind,
            "confianza": round(float(self.confidence), 4),
            "coordenadas": [[x, y], [x + w, y + h]],
            "coordenadas_texto_original": [[tx, ty], [tx + tw, ty + th]],
            "texto_original": original,
            "texto_traducido": translation,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload
