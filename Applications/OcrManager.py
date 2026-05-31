from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Tuple

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")

import cv2
import numpy as np
from PIL import Image

from Applications.CacheManager import PersistentJsonCache
from Applications.PaddleOcrSubprocess import PaddleOcrSubprocess

logger = logging.getLogger(__name__)


class OcrManager:
    """OCR de regiones.

    Importante: este módulo NO importa PaddleOCR al cargar el programa. PaddleOCR puede
    ejecutarse en un subproceso aislado para evitar conflictos CUDA con PyTorch/YOLO.
    """

    PADDLE_LANGS = {
        "Inglés": "en",
        "Coreano": "korean",
        "Chino": "ch",
        "Español": "es",
        "Japonés": "japan",
    }

    EASY_OCR_LANGS = {
        "Japonés": ["ja", "en"],
        "Inglés": ["en"],
        "Coreano": ["ko", "en"],
        "Chino": ["ch_sim", "en"],
        "Español": ["es", "en"],
    }

    def __init__(self, idioma_entrada: str) -> None:
        self.idioma_entrada = idioma_entrada
        self.fast_mode = os.getenv("PMT_FAST_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.use_ocr_gpu = os.getenv("PMT_OCR_GPU", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.ocr_engine = os.getenv("PMT_OCR_ENGINE", "auto").strip().lower()
        self.force_paddle_subprocess = os.getenv("PMT_PADDLE_SUBPROCESS", "auto").strip().lower()
        self._manga_ocr = None
        self._easyocr_reader = None
        self._paddle_ocr = None
        self._paddle_worker = None
        self.cache = PersistentJsonCache("ocr")

    def _engine_for_cache(self) -> str:
        engine = self.ocr_engine
        if engine == "auto":
            return "mangaocr+easyocr" if self.idioma_entrada == "Japonés" else "easyocr"
        if engine == "paddle" and self._should_use_paddle_subprocess():
            return "paddle_subprocess"
        return engine

    def _should_use_paddle_subprocess(self) -> bool:
        if self.force_paddle_subprocess in {"1", "true", "yes", "on", "always"}:
            return True
        if self.force_paddle_subprocess in {"0", "false", "no", "off", "never"}:
            return False
        # Auto: si Paddle usa GPU, aísla para evitar conflicto con PyTorch/YOLO.
        return self.use_ocr_gpu

    def _get_manga_ocr(self):
        if self._manga_ocr is None:
            from manga_ocr import MangaOcr  # type: ignore

            self._manga_ocr = MangaOcr()
        return self._manga_ocr

    def _get_easyocr_reader(self):
        if self._easyocr_reader is None:
            import easyocr  # type: ignore

            langs = self.EASY_OCR_LANGS.get(self.idioma_entrada, ["en"])
            self._easyocr_reader = easyocr.Reader(langs, gpu=self.use_ocr_gpu)
        return self._easyocr_reader

    def _get_paddle_ocr(self):
        if self._paddle_ocr is None:
            from paddleocr import PaddleOCR  # type: ignore

            lang = self.PADDLE_LANGS.get(self.idioma_entrada, "en")
            try:
                self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=self.use_ocr_gpu, show_log=False)
            except TypeError:
                self._paddle_ocr = PaddleOCR(lang=lang)
        return self._paddle_ocr

    def _get_paddle_worker(self) -> PaddleOcrSubprocess:
        if self._paddle_worker is None:
            lang = self.PADDLE_LANGS.get(self.idioma_entrada, "en")
            self._paddle_worker = PaddleOcrSubprocess(lang=lang, use_gpu=self.use_ocr_gpu)
        return self._paddle_worker

    def _upscale_if_needed(self, imagen: np.ndarray) -> np.ndarray:
        if imagen is None or imagen.size == 0:
            return imagen
        h, w = imagen.shape[:2]
        min_side = max(1, min(h, w))
        max_side = max(1, max(h, w))
        scale = 1.0
        target_min_side = 72 if self.fast_mode else 90
        target_max_side = 280 if self.fast_mode else 360
        if min_side < target_min_side:
            scale = max(scale, min(3.0, target_min_side / min_side))
        if max_side < target_max_side:
            scale = max(scale, min(2.0, target_max_side / max_side))
        if scale > 1.01:
            return cv2.resize(imagen, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return imagen

    @staticmethod
    def _normalize_text(texto: str) -> str:
        texto = str(texto or "")
        texto = texto.replace("\u3000", " ")
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    def extract_texts(self, imagenes_interes):
        resultados = []
        engine = self._engine_for_cache()
        for imagen in imagenes_interes:
            key = self.cache.hash_image(imagen, self.idioma_entrada, engine)
            cached = self.cache.get(key)
            if cached is not None:
                resultados.append(str(cached))
                continue
            texto = self._extract_with_selected_engine(imagen)
            self.cache.set(key, texto)
            resultados.append(texto)
        return resultados

    def _extract_with_selected_engine(self, imagen_interes: np.ndarray) -> str:
        engine = self.ocr_engine
        if engine in {"manga", "mangaocr", "manga_ocr"}:
            return self._extract_with_manga_ocr(imagen_interes, fallback_easyocr=True)
        if engine in {"easy", "easyocr"}:
            return self._extract_with_easyocr(imagen_interes)
        if engine in {"paddle", "paddleocr"}:
            return self._extract_with_paddle(imagen_interes)
        if engine in {"paddle_subprocess", "paddle-worker", "paddle_worker"}:
            return self._extract_with_paddle_subprocess(imagen_interes)

        # Auto estable: japonés intenta MangaOCR y cae a EasyOCR; otros idiomas usan EasyOCR.
        if self.idioma_entrada == "Japonés":
            return self._extract_with_manga_ocr(imagen_interes, fallback_easyocr=True)
        return self._extract_with_easyocr(imagen_interes)

    def _extract_with_manga_ocr(self, imagen_interes: np.ndarray, fallback_easyocr: bool = True) -> str:
        if imagen_interes is None or imagen_interes.size == 0:
            return ""
        imagen_interes = self._upscale_if_needed(imagen_interes)
        area_interes_pil = Image.fromarray(cv2.cvtColor(imagen_interes, cv2.COLOR_BGR2RGB))
        try:
            texto = self._get_manga_ocr()(area_interes_pil)
            texto = self._normalize_text(texto)
            if texto:
                return texto
        except Exception as exc:
            logger.warning("MangaOCR falló; se intentará fallback si está habilitado: %s", exc)
        return self._extract_with_easyocr(imagen_interes) if fallback_easyocr else ""

    @staticmethod
    def _line_rect(line) -> Tuple[float, float, float, float]:
        if isinstance(line, dict):
            puntos = np.array(line.get("box") or [], dtype=np.float32)
        else:
            puntos = np.array(line[0], dtype=np.float32)
        if puntos.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        xs = puntos[:, 0]
        ys = puntos[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def _sort_lines(self, lines) -> List:
        if not lines:
            return []
        enriched = []
        for line in lines:
            x1, y1, x2, y2 = self._line_rect(line)
            h = max(1.0, y2 - y1)
            enriched.append((line, x1, y1, x2, y2, h))
        median_h = float(np.median([row[5] for row in enriched])) if enriched else 12.0
        row_step = max(8.0, median_h * 0.70)
        return [row[0] for row in sorted(enriched, key=lambda r: (round(r[2] / row_step), r[1]))]

    @staticmethod
    def _paddle_line_text(line) -> Tuple[str, float]:
        try:
            if isinstance(line, dict):
                return str(line.get("text") or ""), float(line.get("confidence") or 0.0)
            text, conf = line[-1]
            return str(text), float(conf)
        except Exception:
            return "", 0.0

    def _run_paddle_inprocess(self, imagen: np.ndarray):
        area_interes_pil = Image.fromarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        try:
            return self._get_paddle_ocr().ocr(img=np.array(area_interes_pil), cls=True)
        except TypeError:
            return self._get_paddle_ocr().ocr(np.array(area_interes_pil))

    def _extract_with_paddle(self, imagen_interes: np.ndarray) -> str:
        if self._should_use_paddle_subprocess():
            return self._extract_with_paddle_subprocess(imagen_interes)
        if imagen_interes is None or imagen_interes.size == 0:
            return ""
        imagen_interes = self._upscale_if_needed(imagen_interes)
        try:
            resultado_paddle = self._run_paddle_inprocess(imagen_interes)
        except Exception as exc:
            logger.warning("PaddleOCR en proceso principal falló: %s", exc)
            return ""
        lines = resultado_paddle[0] if resultado_paddle and isinstance(resultado_paddle, list) else []
        return self._join_ocr_lines(lines)

    def _extract_with_paddle_subprocess(self, imagen_interes: np.ndarray) -> str:
        if imagen_interes is None or imagen_interes.size == 0:
            return ""
        imagen_interes = self._upscale_if_needed(imagen_interes)
        try:
            lines = self._get_paddle_worker().ocr(imagen_interes)
        except Exception as exc:
            logger.warning("PaddleOCR subproceso falló: %s", exc)
            return ""
        return self._join_ocr_lines(lines)

    def _extract_with_easyocr(self, imagen_interes: np.ndarray) -> str:
        if imagen_interes is None or imagen_interes.size == 0:
            return ""
        imagen_interes = self._upscale_if_needed(imagen_interes)
        try:
            reader = self._get_easyocr_reader()
            lines = reader.readtext(
                imagen_interes,
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                batch_size=6 if self.fast_mode else 4,
                beamWidth=3 if self.fast_mode else 5,
                width_ths=0.35,
                height_ths=0.25,
                canvas_size=1920 if self.fast_mode else 2560,
                mag_ratio=1.2 if self.fast_mode else 1.6,
            )
        except Exception as exc:
            logger.warning("EasyOCR falló en un recorte: %s", exc)
            return ""
        normalized = []
        for line in self._sort_lines(lines):
            try:
                text = line[1]
                conf = float(line[2])
            except Exception:
                text, conf = "", 0.0
            text = self._normalize_text(text)
            if text and not (conf < 0.15 and len(text) <= 2):
                normalized.append(text)
        if self.idioma_entrada in {"Inglés", "Español"}:
            return self._normalize_text(" ".join(normalized))
        return self._normalize_text("".join(t.replace("~", "") for t in normalized))

    def _join_ocr_lines(self, lines) -> str:
        if not lines:
            return ""
        lineas = []
        for line in self._sort_lines(lines):
            linea_actual, confianza = self._paddle_line_text(line)
            linea_actual = self._normalize_text(linea_actual)
            if not linea_actual:
                continue
            if confianza < 0.18 and len(linea_actual) <= 2:
                continue
            if self.idioma_entrada in {"Inglés", "Español"}:
                lineas.append(linea_actual)
            else:
                lineas.append(linea_actual.replace("~", ""))

        if self.idioma_entrada in {"Inglés", "Español"}:
            return self._normalize_text(" ".join(lineas))
        return self._normalize_text("".join(lineas))
