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

    def _is_color_page(self, image: np.ndarray, threshold: float = 5.0) -> bool:
        """
        Detecta si una página es color o escala de grises.

        Si los canales RGB son prácticamente iguales en toda la imagen,
        se considera B/N.
        """
        if image.ndim != 3 or image.shape[2] < 3:
            return False

        b, g, r = cv2.split(image.astype(np.float32))

        rg = np.mean(np.abs(r - g))
        rb = np.mean(np.abs(r - b))
        gb = np.mean(np.abs(g - b))

        color_score = (rg + rb + gb) / 3.0

        return color_score > threshold
    
    def _resolve_auto_inpaint_model(self, image: np.ndarray) -> str:
        if self._is_color_page(image):
            return "lama_mpe"

        return "B/N"

    def _clean_with_regions(self, imagen: np.ndarray, mascara_capa: np.ndarray, resultados, regiones: Sequence[TextRegion]) -> np.ndarray:
        if not regiones:
            return imagen.copy()

        # El modo YOLO mantiene dos máscaras distintas: region.mask es la zona segura
        # del globo; region.clean_mask es la tinta/texto original que se borra.
        imagen_base = imagen.copy()
        bubble_regions = [r for r in regiones if r.kind in {"dialogue", "narration", "unknown"} and self._region_matches_source_language(r)]
        
        # Texto libre y onomatopeyas se limpian con inpainting, no con relleno plano de globo.
        sfx_regions = [
            r for r in regiones
            if r.kind not in {"dialogue", "narration", "unknown"} and self._region_matches_source_language(r) and self._should_clean_non_bubble_region(r, imagen)
        ]

        if self.bubble_fill and self.inpaint_mode in {"auto", "fast", "bubble_only", "quality", "sfx"}:
            imagen_base = self._fill_bubble_interiors(imagen_base, bubble_regions)

        if self.inpaint_mode == "bubble_only":
            return imagen_base

        # Resolvemos el modelo real si la configuración está en "auto"
        actual_inpaint_model = getattr(self, "inpaint_model", "opencv-tela")
        if actual_inpaint_model == "auto":
            actual_inpaint_model = self._resolve_auto_inpaint_model(imagen)

        # Las onomatopeyas/fx fuera de globo se inpaintan con máscara propia. En modo quality
        # se usa el modelo seleccionado; en auto/fast se prefiere OpenCV por velocidad.
        sfx_mask = BubbleDetector.compose_clean_mask(sfx_regions, imagen.shape) if sfx_regions else np.zeros(mascara_capa.shape, dtype=np.uint8)
        if cv2.countNonZero(sfx_mask) > 0:
            # Aquí usamos actual_inpaint_model en lugar de self.inpaint_model
            if self.inpaint_mode == "quality" and actual_inpaint_model not in {"opencv-tela", "B/N"}:
                res_impainting = self._run_async_inpaint(imagen_base, sfx_mask)
                imagen_base = self.convertir_a_imagen_limpia(res_impainting, imagen_base)
            else:
                imagen_base = cv2.inpaint(imagen_base, sfx_mask, 3, cv2.INPAINT_NS)

        if self.inpaint_mode == "quality" and not self.bubble_fill:
            res_impainting = self._ejecutar_inpainting(imagen, mascara_capa, resultados)
            return self.convertir_a_imagen_limpia(res_impainting, imagen)

        return imagen_base

    @staticmethod
    def _background_sample_mask(local_mask: np.ndarray, exclude_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Devuelve la zona de fondo: región segura menos tinta a borrar."""
        if local_mask is None or getattr(local_mask, "size", 0) == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        base_mask = (local_mask > 0).astype(np.uint8)
        if cv2.countNonZero(base_mask) == 0:
            return base_mask

        if exclude_mask is None or not getattr(exclude_mask, "size", 0):
            return base_mask

        exclusion = (exclude_mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_exclusion = cv2.dilate(exclusion, kernel, iterations=1)
        sample_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(dilated_exclusion))

        if cv2.countNonZero(sample_mask) < max(16, int(cv2.countNonZero(base_mask) * 0.04)):
            sample_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(exclusion))
        if cv2.countNonZero(sample_mask) == 0:
            sample_mask = base_mask
        return sample_mask

    @staticmethod
    def _background_variation_score(region_img: np.ndarray, local_mask: np.ndarray, exclude_mask: Optional[np.ndarray] = None) -> float:
        """Mide variación del fondo seguro."""
        if region_img.size == 0 or local_mask is None or getattr(local_mask, "size", 0) == 0:
            return 0.0
        sample_mask = CleanInpaintingPipelineMixin._background_sample_mask(local_mask, exclude_mask)
        if sample_mask.size == 0 or cv2.countNonZero(sample_mask) == 0:
            return 0.0
        pixels = region_img[sample_mask > 0]
        if pixels.size == 0:
            return 0.0
        pixels = pixels.reshape(-1, 3).astype(np.float32)
        luma = pixels[:, 0] * 0.114 + pixels[:, 1] * 0.587 + pixels[:, 2] * 0.299
        return float(np.std(luma))

    @staticmethod
    def _dominant_fill_color(region_img: np.ndarray, local_mask: np.ndarray, exclude_mask: Optional[np.ndarray] = None):
        """Estima el color de fondo de una región segura."""
        if region_img.size == 0 or local_mask is None or getattr(local_mask, "size", 0) == 0:
            return (255, 255, 255)

        sample_mask = CleanInpaintingPipelineMixin._background_sample_mask(local_mask, exclude_mask)
        if sample_mask.size == 0 or cv2.countNonZero(sample_mask) == 0:
            return (255, 255, 255)

        pixels = region_img[sample_mask > 0]
        if pixels.size == 0:
            return (255, 255, 255)

        median = np.median(pixels.reshape(-1, 3), axis=0)
        return tuple(int(min(255, max(0, round(float(c))))) for c in median.tolist())

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

    @staticmethod
    def _mask_bounding_rect(mask: np.ndarray, image_shape, padding: int = 0) -> Tuple[int, int, int, int]:
        points = cv2.findNonZero((mask > 0).astype(np.uint8)) if mask is not None and getattr(mask, "size", 0) else None
        if points is None:
            return (0, 0, 0, 0)
        x, y, w, h = cv2.boundingRect(points)
        height, width = image_shape[:2]
        padding = max(0, int(padding))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def _run_configured_inpaint_on_mask(self, imagen: np.ndarray, mask: np.ndarray, context_mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, str]:
        """Aplica el inpaint elegido en config.yaml sobre una máscara pequeña."""
        if mask is None or cv2.countNonZero(mask) == 0:
            return imagen, "empty_mask"

        model_name = str(getattr(self, "inpaint_model", "opencv-tela") or "opencv-tela")
        if model_name == "auto":
            model_name = self._resolve_auto_inpaint_model(imagen)
        if model_name == "B/N":
            return imagen, "solid_fallback_bn_model_has_no_mask_api"

        padding_cfg = int(getattr(self, "bubble_fill_inpaint_padding", 18) or 18)
        bbox_mask = context_mask if context_mask is not None and cv2.countNonZero(context_mask) > 0 else mask
        x, y, w, h = self._mask_bounding_rect(bbox_mask, imagen.shape, padding_cfg)
        if w <= 0 or h <= 0:
            return imagen, "empty_crop"

        crop = imagen[y:y + h, x:x + w].copy()
        local_mask = (mask[y:y + h, x:x + w] > 0).astype(np.uint8) * 255
        if cv2.countNonZero(local_mask) == 0:
            return imagen, "empty_local_mask"

        try:
            # Obtenemos la instancia del inpainter, usando la general o instanciando una nueva
            inpainter_inst = getattr(self, "inpainter", None) if self.inpaint_model != "auto" else None
            if inpainter_inst is None:
                inpainter_inst = self._build_inpainter(model_name)

            if model_name == "opencv-tela":
                result_crop = inpainter_inst.inpaint(crop, local_mask)
            else:
                result_crop = self._run_async_inpaint(crop, local_mask, inpainter_instance=inpainter_inst)
        except Exception as exc:  # pragma: no cover
            logger.warning("No se pudo aplicar inpainting configurado '%s' en globo; uso relleno sólido. Error: %s", model_name, exc)
            return imagen, f"solid_fallback_configured_inpaint_failed:{model_name}"

        salida = imagen.copy()
        if result_crop.shape[:2] != crop.shape[:2]:
            result_crop = cv2.resize(result_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        salida[y:y + h, x:x + w] = result_crop
        return salida, f"configured_inpaint:{model_name}"

    def _fill_bubble_interiors(self, imagen: np.ndarray, regiones: Sequence[TextRegion]) -> np.ndarray:
        salida = imagen.copy()
        for region in regiones:
            safe_mask = self._safe_bubble_mask(region.mask, salida.shape, self.bubble_fill_edge_margin)
            if cv2.countNonZero(safe_mask) == 0:
                continue

            clean_mask = getattr(region, "clean_mask", None)
            if clean_mask is None or getattr(clean_mask, "size", 0) == 0:
                clean_mask, _source = self._build_clean_mask_for_region(salida, region)
                region.clean_mask = self._binary_mask(clean_mask, salida.shape)
            else:
                clean_mask = self._binary_mask(clean_mask, salida.shape)

            if cv2.countNonZero(clean_mask) == 0:
                continue

            fill_color = self._dominant_fill_color(salida, safe_mask, clean_mask)
            background_variation = self._background_variation_score(salida, safe_mask, clean_mask)
            variation_threshold = float(getattr(self, "bubble_fill_background_std_threshold", 18.0) or 18.0)
            fill_strategy = str(getattr(self, "bubble_fill_strategy", "inpaint") or "inpaint").strip().lower()
            if fill_strategy in {"configured_inpaint", "config_inpaint", "lama"}:
                fill_strategy = "inpaint"
            if fill_strategy in {"adaptive"}:
                fill_strategy = "auto"
            if fill_strategy not in {"inpaint", "auto", "solid"}:
                fill_strategy = "inpaint"

            metadata = getattr(region, "metadata", None)
            if isinstance(metadata, dict):
                metadata["fill_color_source"] = "safe_region_minus_clean_mask"
                metadata["fill_color_bgr"] = tuple(int(c) for c in fill_color)
                metadata["background_variation_score"] = round(float(background_variation), 3)
                metadata["background_variation_threshold"] = float(variation_threshold)
                metadata["bubble_fill_strategy"] = fill_strategy

            should_use_configured_inpaint = fill_strategy == "inpaint" or (
                fill_strategy == "auto" and background_variation >= variation_threshold
            )
            if should_use_configured_inpaint:
                salida_inpaint, method = self._run_configured_inpaint_on_mask(salida, clean_mask, safe_mask)
                if method.startswith("configured_inpaint"):
                    salida = salida_inpaint
                    if isinstance(metadata, dict):
                        metadata["bubble_fill_method"] = "configured_inpaint"
                        metadata["bubble_fill_inpaint_model"] = str(getattr(self, "inpaint_model", ""))
                    continue
                if isinstance(metadata, dict):
                    metadata["bubble_fill_inpaint_fallback"] = method

            sigma = float(getattr(self, "bubble_fill_feather", 1.0) or 1.0)
            if not str((region.metadata or {}).get("clean_mask_source", "")).endswith("opt_in"):
                sigma = min(sigma, 0.65)
            salida = self._apply_solid_fill(salida, clean_mask, fill_color, sigma=sigma)
            if isinstance(metadata, dict):
                metadata["bubble_fill_method"] = "solid_color"
        return salida

    def _ejecutar_inpainting(self, imagen, mascara_capa, resultados):
        model_name = getattr(self, "inpaint_model", "auto")
        if model_name == "auto":
            model_name = self._resolve_auto_inpaint_model(imagen)
            inpainter = self._build_inpainter(model_name)
        else:
            inpainter = getattr(self, "inpainter", None)
            if inpainter is None:
                inpainter = self._build_inpainter(model_name)

        if model_name == "B/N":
            bn_inpainter = self.INPAINTER_FACTORIES["B/N"]()
            return bn_inpainter.inpaint(imagen, resultados)

        if model_name == "opencv-tela":
            tela_inpainter = self.INPAINTER_FACTORIES["opencv-tela"]()
            return tela_inpainter.inpaint(imagen, mascara_capa)

        return self._run_async_inpaint(imagen, mascara_capa, inpainter_instance=inpainter)

    def _run_async_inpaint(self, imagen: np.ndarray, mascara_capa: np.ndarray, inpainter_instance=None):
        if inpainter_instance is None:
            model_name = getattr(self, "inpaint_model", "auto")
            if model_name == "auto":
                model_name = self._resolve_auto_inpaint_model(imagen)
            inpainter_instance = getattr(self, "inpainter", None)
            if inpainter_instance is None:
                inpainter_instance = self._build_inpainter(model_name)

        async def _do_inpaint():
            # Soporte para inpainters de red neuronal que requieren carga asíncrona
            if hasattr(inpainter_instance, "_load"):
                await inpainter_instance._load()
            
            # Verificamos si usa _inpaint (Modelos neurales) o inpaint clásico
            if hasattr(inpainter_instance, "_inpaint"):
                return await inpainter_instance._inpaint(imagen, mascara_capa)
            elif hasattr(inpainter_instance, "inpaint"):
                res = inpainter_instance.inpaint(imagen, mascara_capa)
                if asyncio.iscoroutine(res):
                    return await res
                return res
            else:
                raise AttributeError(f"El modelo {type(inpainter_instance).__name__} no tiene métodos de inpainting válidos")

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_do_inpaint())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def convertir_a_imagen_limpia(self, res_impainting: np.ndarray, imagen: np.ndarray) -> np.ndarray:
        pil_image_camuflada_limpieza = Image.fromarray(cv2.cvtColor(res_impainting, cv2.COLOR_BGR2RGB))
        pil_image_limpieza = Image.new("RGB", (imagen.shape[1], imagen.shape[0]))
        pil_image_limpieza.paste(pil_image_camuflada_limpieza, (0, 0))
        imagen_limpia = np.asarray(pil_image_limpieza)
        return cv2.cvtColor(imagen_limpia, cv2.COLOR_RGB2BGR)