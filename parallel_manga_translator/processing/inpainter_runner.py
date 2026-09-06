"""Ejecuta el inpainting: resuelve el modelo, cachea instancias y las aplica.

Seis métodos que vivían en `CleanInpaintingPipelineMixin`. Junto con `BubbleFillPolicy`
—que decide *qué* candidato se intenta— completan la separación de esa etapa: aquí está
el *cómo*, y la verificación de si el resultado sirve la hace `VisualInpaintVerifier`.

Se lleva consigo `INPAINTER_FACTORIES` y la caché de instancias de reintento, que sólo
usaban estos métodos.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

import cv2
import numpy as np

from parallel_manga_translator.inpainting import AOTInpainter, BNInpainter, LamaInpainterMPE, LamaLarge, OpenCVInpainter
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.infrastructure.gpu_scheduler import gpu_slot

logger = get_logger(__name__)


class InpainterRunner:
    """Construye y ejecuta inpainters. No decide cuál conviene: eso es `BubbleFillPolicy`."""

    INPAINTER_FACTORIES = {
        "opencv-tela": OpenCVInpainter,
        "lama_mpe": LamaInpainterMPE,
        "lama_large_512px": LamaLarge,
        "aot": AOTInpainter,
        "B/N": BNInpainter,
    }

    def __init__(self, *, inpaint_model: str, fill_policy, inpainter=None, bubble_fill_inpaint_padding: int = 18) -> None:
        self.inpaint_model = inpaint_model
        self.fill_policy = fill_policy
        self.inpainter = inpainter
        self.bubble_fill_inpaint_padding = int(bubble_fill_inpaint_padding)
        self._visual_retry_inpainters = {}

    def build_inpainter(self, model_name: str):
        if model_name not in self.INPAINTER_FACTORIES:
            raise ValueError(f"Modelo de inpainting no soportado: {model_name}")
        return self.INPAINTER_FACTORIES[model_name]()

    def get_inpainter_instance_for_retry(self, model_name: str):
        if model_name == self.fill_policy.normalize_inpaint_candidate(str(getattr(self, "inpaint_model", ""))):
            current = getattr(self, "inpainter", None)
            if current is not None:
                return current

        cache = getattr(self, "_visual_retry_inpainters", None)
        if cache is None:
            cache = {}
            self._visual_retry_inpainters = cache
        if model_name not in cache:
            cache[model_name] = self.build_inpainter(model_name)
        return cache[model_name]

    def run_configured_inpaint_on_mask(
        self,
        imagen: np.ndarray,
        mask: np.ndarray,
        context_mask: Optional[np.ndarray] = None,
        model_name: Optional[str] = None,
    ) -> tuple[np.ndarray, str]:
        """Aplica un modelo de inpainting sobre una máscara pequeña."""
        if mask is None or cv2.countNonZero(mask) == 0:
            return imagen, "empty_mask"

        selected_model = str(model_name or getattr(self, "inpaint_model", "opencv-tela") or "opencv-tela")
        selected_model = self.fill_policy.normalize_inpaint_candidate(selected_model)
        if selected_model == "auto":
            selected_model = self.fill_policy.normalize_inpaint_candidate(self.fill_policy.resolve_auto_inpaint_model(imagen))
        if selected_model == "solid":
            return imagen, "solid_candidate_requires_fill_color"
        if selected_model not in self.INPAINTER_FACTORIES:
            return imagen, f"unsupported_inpaint_model:{selected_model}"

        padding_cfg = int(getattr(self, "bubble_fill_inpaint_padding", 18) or 18)
        bbox_mask = context_mask if context_mask is not None and cv2.countNonZero(context_mask) > 0 else mask
        x, y, w, h = self.mask_bounding_rect(bbox_mask, imagen.shape, padding_cfg)
        if w <= 0 or h <= 0:
            return imagen, "empty_crop"

        crop = imagen[y:y + h, x:x + w].copy()
        local_mask = (mask[y:y + h, x:x + w] > 0).astype(np.uint8) * 255
        if cv2.countNonZero(local_mask) == 0:
            return imagen, "empty_local_mask"

        try:
            inpainter_inst = self.get_inpainter_instance_for_retry(selected_model)
            if selected_model == "opencv-tela":
                result_crop = inpainter_inst.inpaint(crop, local_mask)
            else:
                result_crop = self.run_async_inpaint(crop, local_mask, inpainter_instance=inpainter_inst)
        except Exception as exc:  # pragma: no cover
            logger.warning("No se pudo aplicar inpainting '%s' en globo; se probará otro candidato. Error: %s", selected_model, exc)
            return imagen, f"inpaint_failed:{selected_model}"

        salida = imagen.copy()
        if result_crop.shape[:2] != crop.shape[:2]:
            result_crop = cv2.resize(result_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        salida[y:y + h, x:x + w] = result_crop
        return salida, f"configured_inpaint:{selected_model}"

    def apply_visual_inpaint_candidate(
        self,
        imagen: np.ndarray,
        clean_mask: np.ndarray,
        safe_mask: np.ndarray,
        fill_color,
        candidate: str,
        *,
        sigma: float,
    ) -> tuple[Optional[np.ndarray], str]:
        candidate = self.fill_policy.normalize_inpaint_candidate(candidate)
        if candidate == "auto":
            candidate = self.fill_policy.normalize_inpaint_candidate(self.fill_policy.resolve_auto_inpaint_model(imagen))
        if candidate == "solid":
            return self.apply_solid_fill(imagen, clean_mask, fill_color, sigma=sigma), "solid_color"
        if candidate not in self.INPAINTER_FACTORIES:
            return None, f"unsupported_inpaint_model:{candidate}"
        result, method = self.run_configured_inpaint_on_mask(imagen, clean_mask, safe_mask, model_name=candidate)
        if not method.startswith("configured_inpaint"):
            return None, method
        return result, method

    def ejecutar_inpainting(self, imagen, mascara_capa, resultados):
        model_name = getattr(self, "inpaint_model", "auto")
        if model_name == "auto":
            model_name = self.fill_policy.resolve_auto_inpaint_model(imagen)
            inpainter = self.build_inpainter(model_name)
        else:
            inpainter = getattr(self, "inpainter", None)
            if inpainter is None:
                inpainter = self.build_inpainter(model_name)

        if model_name == "B/N":
            bn_inpainter = self.INPAINTER_FACTORIES["B/N"]()
            return bn_inpainter.inpaint(imagen, resultados)

        if model_name == "opencv-tela":
            tela_inpainter = self.INPAINTER_FACTORIES["opencv-tela"]()
            return tela_inpainter.inpaint(imagen, mascara_capa)

        return self.run_async_inpaint(imagen, mascara_capa, inpainter_instance=inpainter)

    def run_async_inpaint(self, imagen: np.ndarray, mascara_capa: np.ndarray, inpainter_instance=None):
        if inpainter_instance is None:
            model_name = getattr(self, "inpaint_model", "auto")
            if model_name == "auto":
                model_name = self.fill_policy.resolve_auto_inpaint_model(imagen)
            inpainter_instance = getattr(self, "inpainter", None)
            if inpainter_instance is None:
                inpainter_instance = self.build_inpainter(model_name)

        async def _do_inpaint():
            # Los modelos neurales comparten GPU con YOLO/OCR. La compuerta FIFO
            # serializa solo este tramo y deja libre el resto del procesamiento CPU.
            neural_inpainter = hasattr(inpainter_instance, "_load")
            with gpu_slot("inpainting.inference", enabled=neural_inpainter):
                if hasattr(inpainter_instance, "_load"):
                    await inpainter_instance._load()

                if hasattr(inpainter_instance, "_inpaint"):
                    result = inpainter_instance._inpaint(imagen, mascara_capa)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
                if hasattr(inpainter_instance, "inpaint"):
                    result = inpainter_instance.inpaint(imagen, mascara_capa)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
                raise AttributeError(
                    f"El modelo {type(inpainter_instance).__name__} no tiene métodos de inpainting válidos"
                )

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_do_inpaint())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    @staticmethod
    def apply_solid_fill(imagen: np.ndarray, mask: np.ndarray, fill_color, sigma: float = 0.9) -> np.ndarray:
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
    def mask_bounding_rect(mask: np.ndarray, image_shape, padding: int = 0) -> Tuple[int, int, int, int]:
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
