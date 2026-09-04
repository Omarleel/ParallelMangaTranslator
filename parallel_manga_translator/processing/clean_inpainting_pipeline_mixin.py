from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.infrastructure.gpu_scheduler import gpu_slot
from parallel_manga_translator.language.source_language_filter import SourceLanguageFilter
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.ocr.ocr_manager import OcrManager

logger = get_logger(__name__)
Detection = Tuple[Sequence[Sequence[float]], str, float]


class CleanInpaintingPipelineMixin:
    """Orquestación del proceso de limpieza e inpainting."""

    #: Tipos de región que se limpian rellenando el interior del globo.
    BUBBLE_KINDS = frozenset({"dialogue", "narration", "unknown"})

    def regiones_a_limpiar(self, imagen: np.ndarray, regiones: Sequence[TextRegion]) -> Tuple[List[TextRegion], List[TextRegion]]:
        """Separa las regiones que realmente se borran: globos y texto libre/SFX.

        Es la única definición de "esto se borra" y por eso es pública: el arnés de
        evaluación la usa para medir la limpieza sobre exactamente las mismas regiones
        que el pipeline decide limpiar. Medir sobre todas las regiones contaría como
        fallo el arte que se conserva a propósito (onomatopeyas en modo ``keep``).
        """
        globos = [
            region for region in regiones or []
            if region.kind in self.BUBBLE_KINDS and self._region_matches_source_language(region)
        ]
        libres = [
            region for region in regiones or []
            if region.kind not in self.BUBBLE_KINDS
            and self._region_matches_source_language(region)
            and self._should_clean_non_bubble_region(region, imagen)
        ]
        return globos, libres

    def _build_inpainter(self, model_name: str):
        if model_name not in self.INPAINTER_FACTORIES:
            raise ValueError(f"Modelo de inpainting no soportado: {model_name}")
        return self.INPAINTER_FACTORIES[model_name]()

    def limpiar_manga(self, imagen: np.ndarray):
        if self._visual_inpaint_debug_enabled():
            self._visual_inpaint_debug_records = []

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

    def _auto_inpaint_candidate(
        self,
        imagen: np.ndarray,
        background_variation: float,
        variation_threshold: float,
    ) -> str:
        """Resuelve ``inpaint_model: auto`` por región, no por página.

        LaMa solo donde hay que reconstruir estructura: página a color, o fondo con
        textura —trama, degradado, arte— medido por la variación del fondo de esa región.
        Sobre un fondo plano en blanco y negro no aporta nada y cuesta GPU, así que ahí se
        deja el relleno plano, que luego se convierte en ``opencv-tela`` como reserva.

        La resolución anterior era por página y solo miraba si era a color, de modo que
        una página monocroma con trama nunca llegaba a LaMa.
        """
        if self._is_color_page(imagen):
            return "lama_mpe"
        if float(background_variation) >= float(variation_threshold):
            return "lama_mpe"
        return "solid"

    def _clean_with_regions(self, imagen: np.ndarray, mascara_capa: np.ndarray, resultados, regiones: Sequence[TextRegion]) -> np.ndarray:
        if not regiones:
            return imagen.copy()

        # El modo YOLO mantiene dos máscaras distintas: region.mask es la zona segura
        # del globo; region.clean_mask es la tinta/texto original que se borra.
        imagen_base = imagen.copy()
        # Texto libre y onomatopeyas se limpian con inpainting, no con relleno plano de globo.
        # Si el usuario eligió conservar onomatopeyas, las regiones SFX se dejan intactas
        # para no borrar arte original ni reinsertarlo como fuente plana.
        bubble_regions, sfx_regions = self.regiones_a_limpiar(imagen, regiones)

        if self.bubble_fill and self.inpaint_mode in {"auto", "fast", "bubble_only", "quality", "sfx"}:
            imagen_base = self._fill_bubble_interiors(imagen_base, bubble_regions)

        if self.inpaint_mode == "bubble_only":
            return imagen_base

        imagen_base = self._clean_free_text_regions(imagen_base, sfx_regions, debug_index_offset=len(bubble_regions))

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

    @staticmethod
    def _safe_debug_token(value: object, fallback: str = "page") -> str:
        token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
        token = token.strip("._-")
        return token or fallback

    def set_visual_inpaint_debug_context(self, output_root: str, page_index: int, filename: str) -> None:
        """Configura dónde se guardarán los artefactos de debug de inpainting."""
        self.visual_inpaint_debug_output_root = str(output_root or "")
        self.visual_inpaint_debug_page_index = int(page_index)
        self.visual_inpaint_debug_filename = str(filename or "page")
        self._visual_inpaint_debug_records = []

    def clear_visual_inpaint_debug_context(self) -> None:
        self.visual_inpaint_debug_output_root = ""
        self.visual_inpaint_debug_page_index = None
        self.visual_inpaint_debug_filename = ""
        self._visual_inpaint_debug_records = []

    def _visual_inpaint_debug_enabled(self) -> bool:
        return bool(getattr(self, "visual_inpaint_debug", False)) and bool(
            getattr(self, "visual_inpaint_debug_output_root", "")
        )

    def _visual_inpaint_debug_page_dir(self) -> Optional[Path]:
        if not self._visual_inpaint_debug_enabled():
            return None

        output_root = Path(str(getattr(self, "visual_inpaint_debug_output_root", "")))
        page_index = getattr(self, "visual_inpaint_debug_page_index", None)
        filename = str(getattr(self, "visual_inpaint_debug_filename", "page") or "page")
        page_stem = self._safe_debug_token(Path(filename).stem, "page")
        if isinstance(page_index, int):
            page_number = f"{page_index + 1:04d}"
            folder_name = page_number if page_stem == page_number else f"{page_number}_{page_stem}"
        else:
            folder_name = page_stem

        page_dir = output_root / "debug_inpaint" / folder_name
        page_dir.mkdir(parents=True, exist_ok=True)
        return page_dir

    def _visual_inpaint_debug_relpath(self, path: Path) -> str:
        """Ruta relativa con separadores `/`.

        Estas rutas viajan dentro de JSON que consumen la UI y otras herramientas, así que
        no pueden depender del separador del sistema: en Windows saldrían con `\\`.
        """
        output_root = Path(str(getattr(self, "visual_inpaint_debug_output_root", "")))
        try:
            return path.relative_to(output_root).as_posix()
        except Exception:
            return Path(path).as_posix()

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Objeto no serializable: {type(value)!r}")

    def _visual_inpaint_debug_bbox(
        self,
        clean_mask: np.ndarray,
        safe_mask: Optional[np.ndarray],
        image_shape,
        padding: int = 12,
    ) -> Tuple[int, int, int, int]:
        bbox_mask = safe_mask if safe_mask is not None and cv2.countNonZero(safe_mask) > 0 else clean_mask
        return self._mask_bounding_rect(bbox_mask, image_shape, padding=padding)

    def _write_visual_inpaint_debug_image(
        self,
        image_or_mask: np.ndarray,
        clean_mask: np.ndarray,
        safe_mask: Optional[np.ndarray],
        region_index: Optional[int],
        label: str,
    ) -> Optional[str]:
        page_dir = self._visual_inpaint_debug_page_dir()
        if page_dir is None or image_or_mask is None or region_index is None:
            return None

        x, y, w, h = self._visual_inpaint_debug_bbox(clean_mask, safe_mask, image_or_mask.shape, padding=12)
        if w <= 0 or h <= 0:
            return None

        crop = image_or_mask[y:y + h, x:x + w]
        if crop.size == 0:
            return None

        safe_label = self._safe_debug_token(label, "crop")
        path = page_dir / f"r{int(region_index):03d}_{safe_label}.png"
        cv2.imwrite(str(path), crop)
        return self._visual_inpaint_debug_relpath(path)

    def _write_visual_inpaint_debug_manifest(self) -> Optional[str]:
        page_dir = self._visual_inpaint_debug_page_dir()
        if page_dir is None:
            return None

        page_index = getattr(self, "visual_inpaint_debug_page_index", None)
        filename = str(getattr(self, "visual_inpaint_debug_filename", "") or "")
        manifest = {
            "page_index": page_index,
            "page_number": int(page_index) + 1 if isinstance(page_index, int) else None,
            "filename": filename,
            "regions": list(getattr(self, "_visual_inpaint_debug_records", []) or []),
        }
        manifest_path = page_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=self._json_safe), encoding="utf-8")
        return self._visual_inpaint_debug_relpath(manifest_path)

    def _write_visual_inpaint_region_debug_summary(
        self,
        *,
        region_index: int,
        region: TextRegion,
        before_image: Optional[np.ndarray],
        after_image: np.ndarray,
        clean_mask: np.ndarray,
        safe_mask: np.ndarray,
        fill_color,
        fill_strategy: str,
        method: str,
        chosen_candidate: str,
        report,
        attempts: List[dict],
    ) -> dict:
        page_dir = self._visual_inpaint_debug_page_dir()
        if page_dir is None:
            return {}

        files = {
            "before_crop": self._write_visual_inpaint_debug_image(before_image, clean_mask, safe_mask, region_index, "before") if before_image is not None else None,
            "chosen_crop": self._write_visual_inpaint_debug_image(after_image, clean_mask, safe_mask, region_index, "chosen"),
            "clean_mask": self._write_visual_inpaint_debug_image(clean_mask, clean_mask, safe_mask, region_index, "clean_mask"),
            "safe_mask": self._write_visual_inpaint_debug_image(safe_mask, clean_mask, safe_mask, region_index, "safe_mask"),
        }
        files = {key: value for key, value in files.items() if value}

        x, y, w, h = region.bbox
        tx, ty, tw, th = region.text_bbox
        payload = {
            "region_index": int(region_index),
            "kind": str(region.kind),
            "confidence": round(float(region.confidence), 4),
            "bbox": [int(x), int(y), int(w), int(h)],
            "text_bbox": [int(tx), int(ty), int(tw), int(th)],
            "fill_strategy": str(fill_strategy),
            "fill_color_bgr": [int(c) for c in fill_color],
            "method": str(method),
            "chosen_candidate": str(chosen_candidate),
            "passed": bool(getattr(report, "passed", True)) if report is not None else None,
            "score": round(float(getattr(report, "score", 0.0)), 4) if report is not None else None,
            "failed_checks": list(getattr(report, "failed_checks", [])) if report is not None else [],
            "attempts": attempts,
            "report": report.to_dict() if report is not None and hasattr(report, "to_dict") else None,
            "files": files,
        }

        report_path = page_dir / f"r{int(region_index):03d}_report.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=self._json_safe), encoding="utf-8")
        payload["report_file"] = self._visual_inpaint_debug_relpath(report_path)

        records = getattr(self, "_visual_inpaint_debug_records", None)
        if not isinstance(records, list):
            records = []
            self._visual_inpaint_debug_records = records
        records.append(payload)
        manifest_path = self._write_visual_inpaint_debug_manifest()

        return {
            "visual_inpaint_debug_dir": self._visual_inpaint_debug_relpath(page_dir),
            "visual_inpaint_debug_manifest": manifest_path,
            "visual_inpaint_debug_report": payload["report_file"],
            "visual_inpaint_debug_files": files,
        }

    def _normalize_inpaint_candidate(self, model_name: str) -> str:
        model_name = str(model_name or "").strip()
        aliases = {
            "": "opencv-tela",
            "opencv": "opencv-tela",
            "cv2": "opencv-tela",
            "opencv_ns": "opencv-tela",
            "opencv-tela": "opencv-tela",
            "tela": "opencv-tela",
            "solid_color": "solid",
            "solid-fill": "solid",
            "bn": "solid",
            "b/n": "solid",
            "B/N": "solid",
            "lama": "lama_mpe",
            "lama-mpe": "lama_mpe",
            "lama_mpe": "lama_mpe",
            "lama_large": "lama_large_512px",
            "lama_large_512px": "lama_large_512px",
            "aot": "aot",
            "auto": "auto",
        }
        return aliases.get(model_name, aliases.get(model_name.lower(), model_name))

    def _get_inpainter_instance_for_retry(self, model_name: str):
        if model_name == self._normalize_inpaint_candidate(str(getattr(self, "inpaint_model", ""))):
            current = getattr(self, "inpainter", None)
            if current is not None:
                return current

        cache = getattr(self, "_visual_retry_inpainters", None)
        if cache is None:
            cache = {}
            self._visual_retry_inpainters = cache
        if model_name not in cache:
            cache[model_name] = self._build_inpainter(model_name)
        return cache[model_name]

    def _run_configured_inpaint_on_mask(
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
        selected_model = self._normalize_inpaint_candidate(selected_model)
        if selected_model == "auto":
            selected_model = self._normalize_inpaint_candidate(self._resolve_auto_inpaint_model(imagen))
        if selected_model == "solid":
            return imagen, "solid_candidate_requires_fill_color"
        if selected_model not in self.INPAINTER_FACTORIES:
            return imagen, f"unsupported_inpaint_model:{selected_model}"

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
            inpainter_inst = self._get_inpainter_instance_for_retry(selected_model)
            if selected_model == "opencv-tela":
                result_crop = inpainter_inst.inpaint(crop, local_mask)
            else:
                result_crop = self._run_async_inpaint(crop, local_mask, inpainter_instance=inpainter_inst)
        except Exception as exc:  # pragma: no cover
            logger.warning("No se pudo aplicar inpainting '%s' en globo; se probará otro candidato. Error: %s", selected_model, exc)
            return imagen, f"inpaint_failed:{selected_model}"

        salida = imagen.copy()
        if result_crop.shape[:2] != crop.shape[:2]:
            result_crop = cv2.resize(result_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        salida[y:y + h, x:x + w] = result_crop
        return salida, f"configured_inpaint:{selected_model}"

    def _visual_retry_candidates(self, initial_candidate: str) -> List[str]:
        candidates: List[str] = []

        def _add(raw_name: str) -> None:
            name = self._normalize_inpaint_candidate(raw_name)
            if name == "auto":
                # auto se resuelve contra la página/crop al ejecutar; se conserva solo una vez.
                name = "auto"
            if name and name not in candidates:
                candidates.append(name)

        _add(initial_candidate)
        if bool(getattr(self, "visual_inpaint_retry", True)):
            raw_models = str(getattr(self, "visual_inpaint_retry_models", "solid,opencv-tela,lama_mpe,aot") or "")
            for token in raw_models.replace(";", ",").split(","):
                token = token.strip()
                if token:
                    _add(token)

        max_retries = max(0, int(getattr(self, "visual_inpaint_max_retries", 4) or 0))
        max_attempts = 1 + max_retries if bool(getattr(self, "visual_inpaint_retry", True)) else 1
        return candidates[:max_attempts]

    def _apply_visual_inpaint_candidate(
        self,
        imagen: np.ndarray,
        clean_mask: np.ndarray,
        safe_mask: np.ndarray,
        fill_color,
        candidate: str,
        *,
        sigma: float,
    ) -> tuple[Optional[np.ndarray], str]:
        candidate = self._normalize_inpaint_candidate(candidate)
        if candidate == "auto":
            candidate = self._normalize_inpaint_candidate(self._resolve_auto_inpaint_model(imagen))
        if candidate == "solid":
            return self._apply_solid_fill(imagen, clean_mask, fill_color, sigma=sigma), "solid_color"
        if candidate not in self.INPAINTER_FACTORIES:
            return None, f"unsupported_inpaint_model:{candidate}"
        result, method = self._run_configured_inpaint_on_mask(imagen, clean_mask, safe_mask, model_name=candidate)
        if not method.startswith("configured_inpaint"):
            return None, method
        return result, method

    def _apply_bubble_cleaning_with_visual_verifier(
        self,
        imagen: np.ndarray,
        clean_mask: np.ndarray,
        safe_mask: np.ndarray,
        fill_color,
        *,
        fill_strategy: str,
        background_variation: float,
        variation_threshold: float,
        sigma: float,
        debug_region_index: Optional[int] = None,
    ) -> tuple[np.ndarray, str, Optional[object], List[dict], str]:
        should_use_inpaint = fill_strategy == "inpaint" or (
            fill_strategy == "auto" and background_variation >= variation_threshold
        )
        initial_candidate = str(getattr(self, "inpaint_model", "auto") or "auto") if should_use_inpaint else "solid"
        initial_candidate = self._normalize_inpaint_candidate(initial_candidate)
        if initial_candidate == "auto":
            initial_candidate = self._normalize_inpaint_candidate(
                self._auto_inpaint_candidate(imagen, background_variation, variation_threshold)
            )
        if initial_candidate == "solid" and should_use_inpaint:
            initial_candidate = "opencv-tela"

        verifier = getattr(self, "visual_inpaint_verifier", None)
        # Sobre fondo con textura ningún relleno domina: medido en 181 regiones, cada
        # candidato es el mejor en 38-50 de ellas, y quedarse con el primero que aprueba
        # deja un 24 % de score sobre la mesa. Ahí se prueban todos y gana el mejor; sobre
        # fondo plano no compensa multiplicar por cuatro el coste de GPU.
        explorar_todos = bool(getattr(self, "visual_inpaint_best_of_textured", False)) and (
            background_variation >= variation_threshold
        )
        attempts: List[dict] = []
        best_image: Optional[np.ndarray] = None
        best_method = "visual_verifier_no_candidate"
        best_candidate = "unknown"
        best_report = None
        best_score = float("inf")

        for candidate in self._visual_retry_candidates(initial_candidate):
            candidate = self._normalize_inpaint_candidate(candidate)
            candidate_image, method = self._apply_visual_inpaint_candidate(
                imagen, clean_mask, safe_mask, fill_color, candidate, sigma=sigma
            )
            if candidate_image is None:
                attempts.append({"candidate": candidate, "method": method, "accepted": False, "skipped": True})
                continue

            if verifier is None:
                return candidate_image, method, None, attempts, candidate

            report = verifier.evaluate(imagen, candidate_image, clean_mask, context_mask=safe_mask)
            attempt = {
                "candidate": candidate,
                "method": method,
                "accepted": bool(report.passed),
                "score": round(float(report.score), 4),
                "failed_checks": report.failed_checks,
            }
            debug_crop = self._write_visual_inpaint_debug_image(
                candidate_image,
                clean_mask,
                safe_mask,
                debug_region_index,
                f"attempt_{len(attempts) + 1:02d}_{candidate}",
            )
            if debug_crop:
                attempt["debug_crop"] = debug_crop
            attempts.append(attempt)
            if float(report.score) < best_score:
                best_score = float(report.score)
                best_image = candidate_image
                best_method = method
                best_candidate = candidate
                best_report = report
            if report.passed and not explorar_todos:
                return candidate_image, method, report, attempts, candidate

        if best_image is not None:
            sufijo = "best_of_candidates" if bool(getattr(best_report, "passed", False)) else "best_failed_visual_score"
            return best_image, f"{best_method}:{sufijo}", best_report, attempts, best_candidate

        fallback = self._apply_solid_fill(imagen, clean_mask, fill_color, sigma=sigma)
        if verifier is not None:
            best_report = verifier.evaluate(imagen, fallback, clean_mask, context_mask=safe_mask)
        fallback_attempt = {
            "candidate": "solid",
            "method": "solid_color:last_resort",
            "accepted": bool(getattr(best_report, "passed", True)),
            "score": round(float(getattr(best_report, "score", 0.0)), 4),
            "failed_checks": getattr(best_report, "failed_checks", []),
        }
        debug_crop = self._write_visual_inpaint_debug_image(
            fallback,
            clean_mask,
            safe_mask,
            debug_region_index,
            f"attempt_{len(attempts) + 1:02d}_solid_last_resort",
        )
        if debug_crop:
            fallback_attempt["debug_crop"] = debug_crop
        attempts.append(fallback_attempt)
        return fallback, "solid_color:last_resort", best_report, attempts, "solid"

    def _fill_bubble_interiors(self, imagen: np.ndarray, regiones: Sequence[TextRegion]) -> np.ndarray:
        salida = imagen.copy()
        for region_index, region in enumerate(regiones):
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

            salida = self._clean_region_with_masks(salida, region, region_index, clean_mask, safe_mask)
        return salida

    def _normalized_fill_strategy(self) -> str:
        fill_strategy = str(getattr(self, "bubble_fill_strategy", "inpaint") or "inpaint").strip().lower()
        if fill_strategy in {"configured_inpaint", "config_inpaint", "lama"}:
            return "inpaint"
        if fill_strategy in {"adaptive"}:
            return "auto"
        return fill_strategy if fill_strategy in {"inpaint", "auto", "solid"} else "inpaint"

    @staticmethod
    def _free_text_context_mask(clean_mask: np.ndarray, image_shape) -> np.ndarray:
        """Anillo de fondo alrededor de la tinta a borrar, para texto libre y SFX.

        Un globo aporta su interior como zona de referencia; el texto libre no tiene
        ninguna. Sin este anillo, el color de relleno, la medida de variación del fondo
        y el verificador visual se quedan sin muestra y cualquier candidato parece
        igual de bueno. El anillo se mantiene ancho porque el verificador muestrea una
        banda de hasta 28 px alrededor de la máscara.
        """
        ink = (clean_mask > 0).astype(np.uint8) * 255
        points = cv2.findNonZero(ink)
        if points is None:
            return np.zeros(image_shape[:2], dtype=np.uint8)
        _x, _y, width, height = cv2.boundingRect(points)
        ring = max(32, min(72, int(round(min(width, height) * 0.45))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
        return cv2.dilate(ink, kernel, iterations=1)

    def _clean_free_text_regions(
        self,
        imagen: np.ndarray,
        regiones: Sequence[TextRegion],
        *,
        debug_index_offset: int = 0,
    ) -> np.ndarray:
        """Limpia texto libre y onomatopeyas región a región, no de una pasada.

        Antes se componía una única máscara con todas estas regiones y se inpaintaba
        la página entera de golpe. Con manchas grandes y separadas eso arrasa el fondo
        —``cv2.inpaint`` difunde a lo largo de toda la máscara— y además no pasaba
        nunca por el verificador visual, así que un resultado malo se aceptaba igual.
        Aquí cada región usa su propio recorte, sus propios candidatos y su propia
        verificación, exactamente igual que los interiores de globo.
        """
        salida = imagen
        for offset, region in enumerate(regiones or []):
            clean_mask = self._binary_mask(getattr(region, "clean_mask", None), salida.shape)
            if cv2.countNonZero(clean_mask) == 0:
                continue
            context_mask = self._free_text_context_mask(clean_mask, salida.shape)
            salida = self._clean_region_with_masks(
                salida, region, debug_index_offset + offset, clean_mask, context_mask
            )
        return salida

    def _clean_region_with_masks(
        self,
        salida: np.ndarray,
        region: TextRegion,
        region_index: int,
        clean_mask: np.ndarray,
        safe_mask: np.ndarray,
    ) -> np.ndarray:
        """Borra la tinta de una región y verifica el resultado.

        ``clean_mask`` es la tinta que se borra; ``safe_mask`` es la zona de referencia
        de la que se muestrea el fondo (interior del globo, o anillo alrededor del texto
        libre). Nunca se pintan los píxeles de ``safe_mask`` que no estén en
        ``clean_mask``.
        """
        fill_color = self._dominant_fill_color(salida, safe_mask, clean_mask)
        background_variation = self._background_variation_score(salida, safe_mask, clean_mask)
        variation_threshold = float(getattr(self, "bubble_fill_background_std_threshold", 18.0) or 18.0)
        fill_strategy = self._normalized_fill_strategy()

        metadata = getattr(region, "metadata", None)
        if isinstance(metadata, dict):
            metadata["fill_color_source"] = "safe_region_minus_clean_mask"
            metadata["fill_color_bgr"] = tuple(int(c) for c in fill_color)
            metadata["background_variation_score"] = round(float(background_variation), 3)
            metadata["background_variation_threshold"] = float(variation_threshold)
            metadata["bubble_fill_strategy"] = fill_strategy

        sigma = float(getattr(self, "bubble_fill_feather", 1.0) or 1.0)
        if not str((region.metadata or {}).get("clean_mask_source", "")).endswith("opt_in"):
            sigma = min(sigma, 0.65)

        if bool(getattr(self, "visual_inpaint_verifier_enabled", False)):
            debug_before_image = salida.copy() if self._visual_inpaint_debug_enabled() else None
            salida_verificada, method, report, attempts, chosen_candidate = self._apply_bubble_cleaning_with_visual_verifier(
                salida,
                clean_mask,
                safe_mask,
                fill_color,
                fill_strategy=fill_strategy,
                background_variation=background_variation,
                variation_threshold=variation_threshold,
                sigma=sigma,
                debug_region_index=region_index,
            )
            salida = salida_verificada
            if isinstance(metadata, dict):
                metadata["bubble_fill_method"] = method
                metadata["visual_inpaint_candidate"] = chosen_candidate
                metadata["visual_inpaint_retries"] = max(0, len([a for a in attempts if not a.get("skipped")]) - 1)
                if report is not None:
                    metadata["visual_inpaint_passed"] = bool(report.passed)
                    metadata["visual_inpaint_score"] = round(float(report.score), 4)
                    metadata["visual_inpaint_failed_checks"] = report.failed_checks
                    if bool(getattr(self, "visual_inpaint_debug", False)):
                        metadata["visual_inpaint_report"] = report.to_dict()
                if bool(getattr(self, "visual_inpaint_debug", False)):
                    metadata["visual_inpaint_attempts"] = attempts
                    debug_metadata = self._write_visual_inpaint_region_debug_summary(
                        region_index=region_index,
                        region=region,
                        before_image=debug_before_image,
                        after_image=salida,
                        clean_mask=clean_mask,
                        safe_mask=safe_mask,
                        fill_color=fill_color,
                        fill_strategy=fill_strategy,
                        method=method,
                        chosen_candidate=chosen_candidate,
                        report=report,
                        attempts=attempts,
                    )
                    metadata.update(debug_metadata)
            return salida

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
                return salida
            if isinstance(metadata, dict):
                metadata["bubble_fill_inpaint_fallback"] = method

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

    def convertir_a_imagen_limpia(self, res_impainting: np.ndarray, imagen: np.ndarray) -> np.ndarray:
        pil_image_camuflada_limpieza = Image.fromarray(cv2.cvtColor(res_impainting, cv2.COLOR_BGR2RGB))
        pil_image_limpieza = Image.new("RGB", (imagen.shape[1], imagen.shape[0]))
        pil_image_limpieza.paste(pil_image_camuflada_limpieza, (0, 0))
        imagen_limpia = np.asarray(pil_image_limpieza)
        return cv2.cvtColor(imagen_limpia, cv2.COLOR_RGB2BGR)