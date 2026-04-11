from __future__ import annotations
import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = '1'

import cv2
import numpy as np
from PIL import Image
from manga_ocr import MangaOcr
from paddleocr import PaddleOCR


class OcrManager:
    PADDLE_LANGS = {
        "Inglés": "en",
        "Coreano": "korean",
        "Chino": "ch",
        "Español": "es",
    }

    def __init__(self, idioma_entrada: str) -> None:
        self.idioma_entrada = idioma_entrada
        self._manga_ocr = None
        self._paddle_ocr = None

    def _get_manga_ocr(self) -> MangaOcr:
        if self._manga_ocr is None:
            self._manga_ocr = MangaOcr()
        return self._manga_ocr

    def _get_paddle_ocr(self) -> PaddleOCR:
        if self._paddle_ocr is None:
            lang = self.PADDLE_LANGS.get(self.idioma_entrada, "en")
            self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        return self._paddle_ocr

    def extract_texts(self, imagenes_interes):
        if self.idioma_entrada == "Japonés":
            return [self._extract_with_manga_ocr(imagen) for imagen in imagenes_interes]
        return [self._extract_with_paddle(imagen) for imagen in imagenes_interes]

    def _extract_with_manga_ocr(self, imagen_interes: np.ndarray) -> str:
        area_interes_pil = Image.fromarray(cv2.cvtColor(imagen_interes, cv2.COLOR_BGR2RGB))
        return self._get_manga_ocr()(area_interes_pil)

    def _extract_with_paddle(self, imagen_interes: np.ndarray) -> str:
        area_interes_pil = Image.fromarray(cv2.cvtColor(imagen_interes, cv2.COLOR_BGR2RGB))
        resultado_paddle = self._get_paddle_ocr().ocr(img=np.array(area_interes_pil), cls=True)

        if not resultado_paddle or not resultado_paddle[0]:
            return ""

        lineas = []
        for line in resultado_paddle[0]:
            linea_actual = line[-1][0]
            if self.idioma_entrada in {"Inglés", "Español"}:
                lineas.append(linea_actual)
            else:
                lineas.append(linea_actual.replace("~", ""))

        if self.idioma_entrada in {"Inglés", "Español"}:
            return " ".join(lineas)
        return "".join(lineas)