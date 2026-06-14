from __future__ import annotations

import asyncio
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.language.source_language_filter import SourceLanguageFilter
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.ocr.ocr_manager import OcrManager

logger = get_logger(__name__)
Detection = Tuple[Sequence[Sequence[float]], str, float]


class CleanInpaintingPipelineMixin:
    """Orquestación del proceso de limpieza e inpainting."""

    def _build_inpainter(self, model_name: str):
        if model_name not in self.INPAINTER_FACTORIES:
            raise ValueError(f"Modelo de inpainting no soportado: {model_name}")
        return self.INPAINTER_FACTORIES[model_name]()

    def limpiar_manga(self, imagen: np.ndarray):
        # Flujo YOLO: primero detectar todos los globos con el segmentador entrenado.
        # El OCR global se ejecuta después solo para asociar pistas, onomatopeyas y texto libre.
        regiones_primarias = self.bubble_detector.detect_primary_bubble_regions(imagen)
        resultados = self.obtener_cuadros_delimitadores(imagen)
        regiones = self.bubble_detector.build_regions_from_bubbles_and_text(imagen, regiones_primarias, resultados)
        regiones = self._filter_regions_by_source_language(regiones)
        regiones = self._filter_regions_by_specialized_ocr_guard(imagen, regiones)
        regiones = self._attach_clean_masks(imagen, regiones)
        self.last_regions = regiones

        # Esta es la máscara de limpieza/tinta, no la máscara completa de globo.
        # La máscara de globo se conserva en region.mask como zona segura para OCR/render.
        mascara_capa = BubbleDetector.compose_clean_mask(regiones, imagen.shape) if regiones else np.zeros(imagen.shape[:2], dtype=np.uint8)

        imagen_limpia = self._clean_with_regions(imagen, mascara_capa, resultados, regiones)
        return mascara_capa, imagen_limpia, regiones

    def _clean_with_regions(self, imagen: np.ndarray, mascara_capa: np.ndarray, resultados, regiones: Sequence[TextRegion]) -> np.ndarray:
        if self.inpaint_mode == "legacy":
            raise RuntimeError("quality.inpaint_mode=legacy no está disponible en la versión sin detección heurística de globos.")
        if not regiones:
            return imagen.copy()

        # El modo YOLO mantiene dos máscaras distintas: region.mask es la zona segura
        # del globo; region.clean_mask es la tinta/texto original que se borra.
        imagen_base = imagen.copy()
        bubble_regions = [r for r in regiones if r.kind in {"dialogue", "narration", "unknown"} and self._region_matches_source_language(r)]
        # Texto libre y onomatopeyas se limpian con inpainting, no con relleno plano de globo.
        # Si el usuario eligió conservar onomatopeyas, las regiones SFX se dejan intactas
        # para no borrar arte original ni reinsertarlo como fuente plana.
        sfx_regions = [
            r for r in regiones
            if r.kind not in {"dialogue", "narration", "unknown"} and self._region_matches_source_language(r) and self._should_clean_non_bubble_region(r, imagen)
        ]

        if self.bubble_fill and self.inpaint_mode in {"auto", "fast", "bubble_only", "quality", "sfx"}:
            imagen_base = self._fill_bubble_interiors(imagen_base, bubble_regions)

        if self.inpaint_mode == "bubble_only":
            return imagen_base

        # Las onomatopeyas/fx fuera de globo se inpaintan con máscara propia. En modo quality
        # se usa el modelo seleccionado; en auto/fast se prefiere OpenCV por velocidad.
        sfx_mask = BubbleDetector.compose_clean_mask(sfx_regions, imagen.shape) if sfx_regions else np.zeros(mascara_capa.shape, dtype=np.uint8)
        if cv2.countNonZero(sfx_mask) > 0:
            if self.inpaint_mode == "quality" and self.inpaint_model not in {"opencv-tela", "B/N"}:
                res_impainting = self._run_async_inpaint(imagen_base, sfx_mask)
                imagen_base = self.convertir_a_imagen_limpia(res_impainting, imagen_base)
            else:
                imagen_base = cv2.inpaint(imagen_base, sfx_mask, 3, cv2.INPAINT_NS)

        if self.inpaint_mode == "quality" and not self.bubble_fill:
            res_impainting = self._ejecutar_inpainting(imagen, mascara_capa, resultados)
            return self.convertir_a_imagen_limpia(res_impainting, imagen)

        return imagen_base

    @staticmethod
    def _dominant_fill_color(region_img: np.ndarray, local_mask: np.ndarray):
        if region_img.size == 0 or local_mask.size == 0:
            return (255, 255, 255)
        pixels = region_img[local_mask > 0]
        if pixels.size == 0:
            return (255, 255, 255)
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        masked_gray = gray[local_mask > 0]
        if masked_gray.size == 0:
            return (255, 255, 255)
        # Para globos de manga, preferimos el color claro dominante del interior.
        bright = pixels[masked_gray >= max(150, int(np.percentile(masked_gray, 55)))]
        sample = bright if bright.size else pixels
        median = np.median(sample.reshape(-1, 3), axis=0)
        if float(np.mean(median)) > 168:
            return tuple(int(min(255, max(0, c))) for c in median.tolist())
        # Globos oscuros/narración: usa mediana del área detectada.
        return tuple(int(min(255, max(0, c))) for c in median.tolist())

    @staticmethod
    def _apply_solid_fill(imagen: np.ndarray, mask: np.ndarray, fill_color, sigma: float = 0.9) -> np.ndarray:
        if cv2.countNonZero(mask) == 0:
            return imagen
        salida = imagen.copy()
        blur = cv2.GaussianBlur((mask > 0).astype(np.uint8) * 255, (0, 0), sigmaX=sigma, sigmaY=sigma)
        alpha = (blur.astype(np.float32) / 255.0)[..., None]
        color = np.array(fill_color, dtype=np.float32)
        patch = salida.astype(np.float32)
        salida = patch * (1.0 - alpha) + color * alpha
        return np.clip(salida, 0, 255).astype(np.uint8)

    def _fill_bubble_interiors(self, imagen: np.ndarray, regiones: Sequence[TextRegion]) -> np.ndarray:
        salida = imagen.copy()
        for region in regiones:
            safe_mask = self._safe_bubble_mask(region.mask, salida.shape, self.bubble_fill_edge_margin)
            if cv2.countNonZero(safe_mask) == 0:
                continue

            fill_color = self._dominant_fill_color(salida, safe_mask)

            # Comportamiento antiguo, solo opt-in y solo si la máscara no parece una caja.
            # Esto evita que una segmentación rectangular pinte parches cuadrados sobre bordes.
            clean_mask = getattr(region, "clean_mask", None)
            if clean_mask is None or getattr(clean_mask, "size", 0) == 0:
                clean_mask, _source = self._build_clean_mask_for_region(salida, region)
                region.clean_mask = self._binary_mask(clean_mask, salida.shape)
            else:
                clean_mask = self._binary_mask(clean_mask, salida.shape)

            if cv2.countNonZero(clean_mask) == 0:
                continue

            sigma = 1.0 if str((region.metadata or {}).get("clean_mask_source", "")).endswith("opt_in") else 0.65
            salida = self._apply_solid_fill(salida, clean_mask, fill_color, sigma=sigma)
        return salida

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

    def convertir_a_imagen_limpia(self, res_impainting: np.ndarray, imagen: np.ndarray) -> np.ndarray:
        pil_image_camuflada_limpieza = Image.fromarray(cv2.cvtColor(res_impainting, cv2.COLOR_BGR2RGB))
        pil_image_limpieza = Image.new("RGB", (imagen.shape[1], imagen.shape[0]))
        pil_image_limpieza.paste(pil_image_camuflada_limpieza, (0, 0))
        imagen_limpia = np.asarray(pil_image_limpieza)
        return cv2.cvtColor(imagen_limpia, cv2.COLOR_RGB2BGR)
