from __future__ import annotations

import asyncio
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import nest_asyncio
import numpy as np
import torch
from PIL import Image

from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.language.source_language_filter import SourceLanguageFilter
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.ocr.ocr_manager import OcrManager

logger = get_logger(__name__)
Detection = Tuple[Sequence[Sequence[float]], str, float]


class CleanOnomatopoeiaGuardMixin:
    """Reglas de preservación de onomatopeyas y OCR preventivo antes de limpiar."""

    def _onomatopoeia_keep_requested(self) -> bool:
        return (
            self.onomatopoeia_mode in {"keep", "original", "none", "off"}
            or not self.translate_onomatopoeia
            or not self.clean_onomatopoeia
        )

    def _region_text_hints(self, region: TextRegion) -> List[str]:
        hints = [getattr(region, "source_text_hint", "")]
        metadata = getattr(region, "metadata", {}) or {}
        for key in ("text", "ocr_text", "source_text", "source_text_hint", "clean_guard_ocr_text"):
            value = metadata.get(key)
            if value is not None:
                hints.append(str(value))
        return [str(x or "").strip() for x in hints if str(x or "").strip()]

    def _mark_region_as_kept_onomatopoeia(self, region: TextRegion, text: str, *, source: str) -> None:
        metadata = getattr(region, "metadata", None)
        if metadata is None:
            return
        metadata["free_text_onomatopoeia_keep"] = True
        metadata["free_text_onomatopoeia"] = True
        metadata["onomatopoeia"] = True
        metadata["clean_guard_source"] = source
        if text:
            metadata["clean_guard_ocr_text"] = text
        match = self.onomatopoeia_manager.similar_semantic_key(text, self.idioma_entrada)
        if match:
            key, score, matched_source = match
            metadata["onomatopoeia_key"] = key
            metadata["free_text_onomatopoeia_similarity"] = round(float(score), 4)
            metadata["free_text_onomatopoeia_source"] = matched_source

    @staticmethod
    def _prepare_crop_for_clean_guard_ocr(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return crop

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h=8)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        block_size = max(15, (min(gray.shape[:2]) // 8) * 2 + 1)
        binaria = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            9,
        )

        pixeles_blancos = cv2.countNonZero(binaria)
        pixeles_totales = binaria.size
        pixeles_negros = pixeles_totales - pixeles_blancos
        if pixeles_negros > pixeles_blancos:
            binaria = cv2.bitwise_not(binaria)

        border = max(6, min(18, int(round(min(binaria.shape[:2]) * 0.06))))
        binaria = cv2.copyMakeBorder(binaria, border, border, border, border, cv2.BORDER_CONSTANT, value=255)

        h, w = binaria.shape[:2]
        min_side = min(h, w)
        max_side = max(h, w)
        scale = 1.0
        if min_side < 96:
            scale = max(scale, min(3.0, 96 / max(1, min_side)))
        if max_side < 420:
            scale = max(scale, min(2.2, 420 / max(1, max_side)))
        if scale > 1.01:
            binaria = cv2.resize(binaria, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

    def _free_text_crop_for_clean_guard(self, imagen: np.ndarray, region: TextRegion) -> np.ndarray:
        height_img, width_img = imagen.shape[:2]
        x, y, w, h = self.mask_strategy.clip_rect(region.bbox, imagen.shape)
        if w <= 0 or h <= 0:
            return np.empty((0, 0, 3), dtype=imagen.dtype)
        crop = imagen[y:y + h, x:x + w]
        local_mask = region.mask[y:y + h, x:x + w]
        if crop.size and local_mask.size and cv2.countNonZero(local_mask) > 0:
            if local_mask.shape[:2] != crop.shape[:2]:
                local_mask = cv2.resize(local_mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            canvas = np.full_like(crop, 255)
            canvas[local_mask > 0] = crop[local_mask > 0]
            crop = canvas
        return self._prepare_crop_for_clean_guard_ocr(crop)

    def _region_crop_for_specialized_ocr_guard(self, imagen: np.ndarray, region: TextRegion) -> np.ndarray:
        """Prepara el recorte que decide si una región realmente debe procesarse.

        La limpieza borra antes de traducir; por eso no basta con que el detector de
        globos haya propuesto una caja. Esta verificación usa el OCR especializado de
        transcripción (MangaOCR/Paddle/EasyOCR según configuración) sobre la región ya
        enmascarada. Si ese OCR no lee absolutamente nada, la región se preserva.
        """
        x, y, w, h = self.mask_strategy.clip_rect(getattr(region, "ocr_bbox", region.bbox), imagen.shape)
        if w <= 0 or h <= 0:
            return np.empty((0, 0, 3), dtype=imagen.dtype)

        crop = imagen[y:y + h, x:x + w]
        if crop.size == 0:
            return crop

        local_mask = region.mask[y:y + h, x:x + w]
        if local_mask.size and cv2.countNonZero(local_mask) > 0:
            if local_mask.shape[:2] != crop.shape[:2]:
                local_mask = cv2.resize(local_mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            canvas = np.full_like(crop, 255)
            canvas[local_mask > 0] = crop[local_mask > 0]
            crop = canvas

        return self._prepare_crop_for_clean_guard_ocr(crop)

    def _get_clean_guard_ocr_manager(self):
        manager = getattr(self, "_clean_guard_ocr_manager", None)
        if manager is None:
            from parallel_manga_translator.ocr.ocr_manager import OcrManager

            manager = OcrManager(idioma_entrada=self.idioma_entrada, config=getattr(self, "ocr_config", None))
            self._clean_guard_ocr_manager = manager
        return manager

    def _ocr_text_for_processing_guard(self, imagen: Optional[np.ndarray], region: TextRegion) -> Optional[str]:
        """Devuelve el texto OCR del guard, ``""`` si no hay texto y ``None`` si falló.

        ``None`` es fail-open: si el OCR no pudo ejecutarse por una excepción puntual,
        no bloqueamos toda la página. En cambio, una cadena vacía sí significa que el
        OCR especializado inspeccionó el box y no encontró texto procesable.
        """
        if imagen is None:
            return None
        metadata = getattr(region, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("specialized_ocr_guard_attempted"):
            return str(metadata.get("specialized_ocr_guard_text") or metadata.get("region_ocr_text") or "").strip()

        crop = self._region_crop_for_specialized_ocr_guard(imagen, region)
        if crop.size == 0:
            texto = ""
        else:
            try:
                manager = self._get_clean_guard_ocr_manager()
                textos = manager.extract_texts([crop])
                texto = str(textos[0] if textos else "").strip()
            except Exception as exc:
                logger.debug("No se pudo verificar región con OCR especializado antes de limpiar: %s", exc)
                return None

        if isinstance(metadata, dict):
            metadata["specialized_ocr_guard_attempted"] = True
            metadata["specialized_ocr_guard_text"] = texto
            metadata["specialized_ocr_guard_engine"] = getattr(getattr(self, "_clean_guard_ocr_manager", None), "engine_id", "")
            if texto:
                metadata["specialized_ocr_guard_passed"] = True
                metadata["region_ocr_text"] = texto
                metadata["region_ocr_text_source"] = "pre_clean_specialized_ocr_guard"
                metadata["region_ocr_cache_reusable"] = True
                # Conserva compatibilidad con los filtros que ya miraban esta clave.
                metadata["clean_guard_ocr_text"] = texto
                metadata["clean_guard_ocr_cached"] = True
            else:
                metadata["specialized_ocr_guard_passed"] = False
        return texto

    def _mark_region_as_skipped_by_specialized_ocr_guard(self, region: TextRegion) -> None:
        metadata = getattr(region, "metadata", None)
        if not isinstance(metadata, dict):
            return
        metadata["specialized_ocr_guard_empty"] = True
        metadata["processing_skipped"] = True
        metadata["processing_skip_reason"] = "ocr_especializado_sin_texto"
        metadata["source_language_allowed"] = False
        metadata["source_language_filter"] = "ocr_especializado_sin_texto"
        metadata["source_language"] = getattr(self, "idioma_entrada", "")

    def _filter_regions_by_specialized_ocr_guard(self, imagen: np.ndarray, regiones: Sequence[TextRegion]) -> List[TextRegion]:
        filtradas: List[TextRegion] = []
        for region in regiones or []:
            texto = self._ocr_text_for_processing_guard(imagen, region)
            if texto is None:
                filtradas.append(region)
                continue
            if str(texto or "").strip():
                filtradas.append(region)
                continue

            self._mark_region_as_skipped_by_specialized_ocr_guard(region)
            logger.debug(
                "Región omitida porque el OCR especializado no detectó texto: bbox=%s kind=%s",
                getattr(region, "bbox", None),
                getattr(region, "kind", ""),
            )
        return filtradas

    def _ocr_text_for_clean_guard(self, imagen: Optional[np.ndarray], region: TextRegion) -> str:
        if imagen is None or region.kind != "free_text":
            return ""
        metadata = getattr(region, "metadata", {}) or {}
        cached = str(metadata.get("clean_guard_ocr_text") or "").strip()
        if cached:
            return cached
        crop = self._free_text_crop_for_clean_guard(imagen, region)
        if crop.size == 0:
            return ""
        try:
            textos = self._get_clean_guard_ocr_manager().extract_texts([crop])
        except Exception as exc:
            logger.debug("No se pudo verificar texto libre antes de limpiar: %s", exc)
            return ""

        texto = str(textos[0] if textos else "").strip()

        # Evita repetir OCR sobre el mismo recorte en la fase de traducción.
        # OcrManager ya tiene caché persistente por hash de imagen, pero guardar el
        # resultado en metadata cubre también ejecuciones con caché desactivada y
        # deja explícito que esta región ya fue leída localmente.
        if isinstance(metadata, dict):
            metadata["clean_guard_ocr_attempted"] = True
            if texto:
                metadata["clean_guard_ocr_text"] = texto
                metadata["clean_guard_ocr_cached"] = True
                if metadata.get("vertical_text_retry") or metadata.get("ocr_global_hint_used_as_bbox_only"):
                    metadata["region_ocr_text"] = texto
                    metadata["region_ocr_text_source"] = "clean_guard_region_ocr"
                    metadata["region_ocr_cache_reusable"] = True
        return texto

    def _is_kept_onomatopoeia_region(self, region: TextRegion, imagen: Optional[np.ndarray] = None) -> bool:
        if not self._onomatopoeia_keep_requested():
            return False
        metadata = getattr(region, "metadata", {}) or {}
        if metadata.get("free_text_onomatopoeia") or metadata.get("onomatopoeia"):
            return True
        if region.kind in {"sfx", "onomatopoeia"}:
            return True
        if region.kind == "free_text":
            for text in self._region_text_hints(region):
                if self.onomatopoeia_manager.is_free_text_onomatopoeia(text, self.idioma_entrada):
                    self._mark_region_as_kept_onomatopoeia(region, text, source="metadata_or_global_ocr_hint")
                    return True

            # La limpieza ocurre antes de la traducción. A veces el OCR global que creó
            # la región lee una onomatopeya estilizada como basura (por ejemplo
            # "A 、 附A"), mientras que el OCR de traducción sobre el recorte completo
            # sí lee "ハッハッ". Si el modo pide conservar SFX/onomatopeyas, hacemos una
            # verificación OCR local antes de añadir esta región a la máscara de borrado.
            clean_guard_text = self._ocr_text_for_clean_guard(imagen, region)
            if clean_guard_text and self.onomatopoeia_manager.is_free_text_onomatopoeia(clean_guard_text, self.idioma_entrada):
                self._mark_region_as_kept_onomatopoeia(region, clean_guard_text, source="pre_clean_region_ocr")
                return True
        return False

    def _should_clean_non_bubble_region(self, region: TextRegion, imagen: Optional[np.ndarray] = None) -> bool:
        if not self._region_matches_source_language(region):
            return False
        return not self._is_kept_onomatopoeia_region(region, imagen)
