from __future__ import annotations

import re
from typing import Any, List, Protocol, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.layout.reading_order_resolver import ReadingOrderResolver
from parallel_manga_translator.ocr.settings import EASY_OCR_LANGS, PADDLE_LANGS, OcrSettings

logger = get_logger(__name__)

# Alias público conservado para no romper imports existentes.
OcrEngineSettings = OcrSettings


class OcrEngine(Protocol):
    """Contrato mínimo para añadir motores OCR sin cambiar el pipeline."""

    @property
    def engine_id(self) -> str:
        ...

    def extract_text(self, image: np.ndarray) -> str:
        ...


class OcrEngineBase:
    """Utilidades compartidas; cada motor conserva una sola responsabilidad."""

    PADDLE_LANGS = PADDLE_LANGS
    EASY_OCR_LANGS = EASY_OCR_LANGS

    def __init__(self, settings: OcrSettings) -> None:
        self.settings = settings
        self.idioma_entrada = settings.language
        self.reading_order_resolver = ReadingOrderResolver(settings.language)

    @property
    def engine_id(self) -> str:
        return self.__class__.__name__.replace("Engine", "").lower()

    def upscale_if_needed(self, imagen: np.ndarray) -> np.ndarray:
        if imagen is None or imagen.size == 0:
            return imagen
        h, w = imagen.shape[:2]
        min_side = max(1, min(h, w))
        max_side = max(1, max(h, w))
        scale = 1.0
        target_min_side = 72 if self.settings.fast_mode else 90
        target_max_side = 280 if self.settings.fast_mode else 360
        if min_side < target_min_side:
            scale = max(scale, min(3.0, target_min_side / min_side))
        if max_side < target_max_side:
            scale = max(scale, min(2.0, target_max_side / max_side))
        if scale > 1.01:
            return cv2.resize(imagen, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return imagen

    @staticmethod
    def normalize_text(texto: str) -> str:
        texto = str(texto or "")
        texto = texto.replace("\u3000", " ")
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    @staticmethod
    def line_rect(line: Any) -> Tuple[float, float, float, float]:
        if isinstance(line, dict):
            puntos = np.array(line.get("box") or [], dtype=np.float32)
        else:
            puntos = np.array(line[0], dtype=np.float32)
        if puntos.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        xs = puntos[:, 0]
        ys = puntos[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def sort_lines(self, lines: Sequence[Any]) -> List[Any]:
        if not lines:
            return []
        try:
            return self.reading_order_resolver.sort_ocr_items(lines)
        except Exception as exc:
            logger.warning("No se pudo resolver orden de lectura OCR; usando fallback horizontal: %s", exc)
            enriched = []
            for line in lines:
                x1, y1, x2, y2 = self.line_rect(line)
                h = max(1.0, y2 - y1)
                enriched.append((line, x1, y1, x2, y2, h))
            median_h = float(np.median([row[5] for row in enriched])) if enriched else 12.0
            row_step = max(8.0, median_h * 0.70)
            return [row[0] for row in sorted(enriched, key=lambda r: (round(r[2] / row_step), r[1]))]

    @staticmethod
    def paddle_line_text(line: Any) -> Tuple[str, float]:
        try:
            if isinstance(line, dict):
                return str(line.get("text") or ""), float(line.get("confidence") or 0.0)
            text, conf = line[-1]
            return str(text), float(conf)
        except Exception:
            return "", 0.0

    def join_ocr_lines(self, lines: Sequence[Any]) -> str:
        if not lines:
            return ""
        lineas = []
        for line in self.sort_lines(lines):
            linea_actual, confianza = self.paddle_line_text(line)
            linea_actual = self.normalize_text(linea_actual)
            if not linea_actual:
                continue
            if confianza < 0.18 and len(linea_actual) <= 2:
                continue
            if self.idioma_entrada in {"Inglés", "Español"}:
                lineas.append(linea_actual)
            else:
                lineas.append(linea_actual.replace("~", ""))

        if self.idioma_entrada in {"Inglés", "Español"}:
            return self.normalize_text(" ".join(lineas))
        return self.normalize_text("".join(lineas))
