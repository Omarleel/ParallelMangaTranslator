from __future__ import annotations

import asyncio

import cv2
import easyocr
import nest_asyncio
import numpy as np
import torch
from PIL import Image

from Applications.inpaint import AOTInpainter, BNInpainter, LamaInpainterMPE, LamaLarge, OpenCVInpainter
from .LoggingConfig import get_logger

nest_asyncio.apply()
logger = get_logger(__name__)


class CleanManga:
    INPAINTER_FACTORIES = {
        "opencv-tela": OpenCVInpainter,
        "lama_mpe": LamaInpainterMPE,
        "lama_large_512px": LamaLarge,
        "aot": AOTInpainter,
        "B/N": BNInpainter,
    }

    def __init__(self, modelo_inpaint: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inpaint_model = modelo_inpaint
        self.inpainter = self._build_inpainter(modelo_inpaint)
        self._ocr_reader = None

    def _build_inpainter(self, model_name: str):
        if model_name not in self.INPAINTER_FACTORIES:
            raise ValueError(f"Modelo de inpainting no soportado: {model_name}")
        return self.INPAINTER_FACTORIES[model_name]()

    def _get_reader(self) -> easyocr.Reader:
        if self._ocr_reader is None:
            self._ocr_reader = easyocr.Reader(["ja", "en"], gpu=self.device == "cuda")
        return self._ocr_reader

    def limpiar_manga(self, imagen: np.ndarray):
        resultados = self.obtener_cuadros_delimitadores(imagen)
        mascara_capa = self.fusionar_cuadros_delimitadores(imagen, resultados)
        res_impainting = self._ejecutar_inpainting(imagen, mascara_capa, resultados)
        imagen_limpia = self.convertir_a_imagen_limpia(res_impainting, imagen)
        return mascara_capa, imagen_limpia

    def _ejecutar_inpainting(self, imagen: np.ndarray, mascara_capa: np.ndarray, resultados):
        if self.inpaint_model == "B/N":
            return self.inpainter.inpaint(imagen, resultados)
        if self.inpaint_model == "opencv-tela":
            return self.inpainter.inpaint(imagen, mascara_capa)
        return self._run_async_inpaint(imagen, mascara_capa)

    def _run_async_inpaint(self, imagen: np.ndarray, mascara_capa: np.ndarray):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.inpaint_async(imagen, mascara_capa))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    async def inpaint_async(self, imagen: np.ndarray, mascara_capa: np.ndarray):
        await self.inpainter._load()
        return await self.inpainter._inpaint(imagen, mascara_capa)

    def convertir_a_imagen_limpia(self, res_impainting: np.ndarray, imagen: np.ndarray) -> np.ndarray:
        pil_image_camuflada_limpieza = Image.fromarray(cv2.cvtColor(res_impainting, cv2.COLOR_BGR2RGB))
        pil_image_limpieza = Image.new("RGB", (imagen.shape[1], imagen.shape[0]))
        pil_image_limpieza.paste(pil_image_camuflada_limpieza, (0, 0))
        imagen_limpia = np.asarray(pil_image_limpieza)
        return cv2.cvtColor(imagen_limpia, cv2.COLOR_RGB2BGR)

    def obtener_cuadros_delimitadores(self, imagen: np.ndarray):
        lector = self._get_reader()
        return lector.readtext(
            imagen,
            paragraph=False,
            decoder="beamsearch",
            batch_size=3,
            beamWidth=3,
            width_ths=0.1,
            height_ths=0.05,
            x_ths=0.1,
            y_ths=0.3,
            min_size=5,
            link_threshold=0.98,
        )

    def fusionar_cuadros_delimitadores(self, imagen: np.ndarray, resultados) -> np.ndarray:
        expansion = 1
        height, width = imagen.shape[:2]
        mascara = np.zeros((height, width), dtype=np.uint8)

        for detection in resultados:
            caja = detection[0]
            puntos = np.array(caja, dtype=np.int32).reshape((-1, 1, 2))
            x, y, w, h = cv2.boundingRect(puntos)
            x_margin = max(0, x - expansion)
            y_margin = max(0, y - expansion)
            w_margin = min(w + expansion, width - x_margin)
            h_margin = min(h + expansion, height - y_margin)
            puntos_margin = np.array(
                [
                    [x_margin, y_margin],
                    [x_margin + w_margin, y_margin],
                    [x_margin + w_margin, y_margin + h_margin],
                    [x_margin, y_margin + h_margin],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mascara, [puntos_margin], 255)

        return mascara