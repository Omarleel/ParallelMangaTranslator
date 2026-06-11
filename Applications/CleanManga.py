from __future__ import annotations

import asyncio
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import easyocr
import nest_asyncio
import numpy as np
import torch
from PIL import Image

from Applications.inpaint import AOTInpainter, BNInpainter, LamaInpainterMPE, LamaLarge, OpenCVInpainter
from Applications.OnomatopoeiaManager import OnomatopoeiaManager
from Applications.BubbleDetector import BubbleDetector
from Applications.CacheManager import env_flag
from Applications.Environment import env_int
from Applications.ProcessingModels import TextRegion
from .LoggingConfig import get_logger

nest_asyncio.apply()
logger = get_logger(__name__)


Detection = Tuple[Sequence[Sequence[float]], str, float]


class CleanManga:
    INPAINTER_FACTORIES = {
        "opencv-tela": OpenCVInpainter,
        "lama_mpe": LamaInpainterMPE,
        "lama_large_512px": LamaLarge,
        "aot": AOTInpainter,
        "B/N": BNInpainter,
    }

    EASY_OCR_LANGS = {
        "Japonés": ["ja", "en"],
        "Inglés": ["en"],
        "Coreano": ["ko", "en"],
        "Chino": ["ch_sim", "en"],
        "Español": ["es", "en"],
    }

    def __init__(self, modelo_inpaint: str, idioma_entrada: str = "Japonés") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inpaint_model = modelo_inpaint
        self.idioma_entrada = idioma_entrada
        self.fast_mode = env_flag("PMT_FAST_MODE", False)
        self.inpaint_mode = os.getenv("PMT_INPAINT_MODE", "auto").strip().lower()
        self.bubble_fill = env_flag("PMT_BUBBLE_FILL", True)
        # Por defecto NO rellenamos plano todo el interior del globo: en páginas reales
        # algunas máscaras YOLO vienen casi rectangulares y eso genera parches cuadrados
        # que cruzan bordes. PMT_BUBBLE_FILL ahora significa "limpiar el texto dentro
        # del globo" usando tinta + recorte por máscara del globo. Si se quiere el
        # comportamiento antiguo se puede activar explícitamente.
        self.bubble_fill_whole_interior = env_flag("PMT_BUBBLE_FILL_WHOLE_INTERIOR", False)
        self.bubble_fill_edge_margin = self._env_int("PMT_BUBBLE_FILL_EDGE_MARGIN", 5)
        self.bubble_fill_text_dilate = self._env_int("PMT_BUBBLE_FILL_TEXT_DILATE", 2)
        self.onomatopoeia_manager = OnomatopoeiaManager()
        self.onomatopoeia_mode = os.getenv("PMT_ONOMATOPOEIA_MODE", "translate").strip().lower()
        self.translate_onomatopoeia = os.getenv("PMT_TRANSLATE_ONOMATOPOEIA", "1").strip().lower() not in {"0", "false", "no", "off"}
        default_clean_onomatopoeia = self.translate_onomatopoeia and self.onomatopoeia_mode not in {"keep", "original", "none", "off"}
        self.clean_onomatopoeia = env_flag("PMT_CLEAN_ONOMATOPOEIA", default_clean_onomatopoeia)
        self.bubble_detector = BubbleDetector(idioma_entrada=idioma_entrada)
        self.inpainter = self._build_inpainter(modelo_inpaint)
        self._ocr_reader = None
        self.last_regions: List[TextRegion] = []

    def _build_inpainter(self, model_name: str):
        if model_name not in self.INPAINTER_FACTORIES:
            raise ValueError(f"Modelo de inpainting no soportado: {model_name}")
        return self.INPAINTER_FACTORIES[model_name]()

    def _get_reader(self) -> easyocr.Reader:
        if self._ocr_reader is None:
            langs = self.EASY_OCR_LANGS.get(self.idioma_entrada, ["ja", "en"])
            self._ocr_reader = easyocr.Reader(langs, gpu=self.device == "cuda")
        return self._ocr_reader

    def limpiar_manga(self, imagen: np.ndarray):
        # Flujo profesional: primero detectar todos los globos con el segmentador entrenado.
        # El OCR global se ejecuta después solo para asociar pistas, onomatopeyas y texto libre.
        regiones_primarias = self.bubble_detector.detect_primary_bubble_regions(imagen)
        resultados = self.obtener_cuadros_delimitadores(imagen)
        regiones = self.bubble_detector.build_regions_from_bubbles_and_text(imagen, regiones_primarias, resultados)
        self.last_regions = regiones

        # Sin globos/modelo no inventamos regiones heurísticas. Si la página no tiene
        # globos ni texto libre/SFX, la máscara queda vacía.
        mascara_capa = BubbleDetector.compose_mask(regiones, imagen.shape) if regiones else np.zeros(imagen.shape[:2], dtype=np.uint8)

        imagen_limpia = self._clean_with_regions(imagen, mascara_capa, resultados, regiones)
        return mascara_capa, imagen_limpia, regiones

    def _clean_with_regions(self, imagen: np.ndarray, mascara_capa: np.ndarray, resultados, regiones: Sequence[TextRegion]) -> np.ndarray:
        if self.inpaint_mode == "legacy":
            raise RuntimeError("PMT_INPAINT_MODE=legacy no está disponible en la versión sin detección heurística de globos.")
        if not regiones:
            return imagen.copy()

        # El modo profesional por defecto limpia el interior completo de globos detectados.
        # Para SFX sobre dibujo conserva inpainting, porque rellenar con blanco destruiría arte.
        imagen_base = imagen.copy()
        bubble_regions = [r for r in regiones if r.kind in {"dialogue", "narration", "unknown"}]
        # Texto libre y onomatopeyas se limpian con inpainting, no con relleno plano de globo.
        # Si el usuario eligió conservar onomatopeyas, las regiones SFX se dejan intactas
        # para no borrar arte original ni reinsertarlo como fuente plana.
        sfx_regions = [
            r for r in regiones
            if r.kind not in {"dialogue", "narration", "unknown"} and self._should_clean_non_bubble_region(r, imagen)
        ]

        if self.bubble_fill and self.inpaint_mode in {"auto", "fast", "bubble_only", "quality", "sfx"}:
            imagen_base = self._fill_bubble_interiors(imagen_base, bubble_regions)

        if self.inpaint_mode == "bubble_only":
            return imagen_base

        # Las onomatopeyas/fx fuera de globo se inpaintan con máscara propia. En modo quality
        # se usa el modelo seleccionado; en auto/fast se prefiere OpenCV por velocidad.
        sfx_mask = BubbleDetector.compose_mask(sfx_regions, imagen.shape) if sfx_regions else np.zeros(mascara_capa.shape, dtype=np.uint8)
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
        x, y, w, h = self._clip_rect(region.bbox, imagen.shape)
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

    def _get_clean_guard_ocr_manager(self):
        manager = getattr(self, "_clean_guard_ocr_manager", None)
        if manager is None:
            from Applications.OcrManager import OcrManager

            manager = OcrManager(idioma_entrada=self.idioma_entrada)
            self._clean_guard_ocr_manager = manager
        return manager

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
        return str(textos[0] if textos else "").strip()

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
        return not self._is_kept_onomatopoeia_region(region, imagen)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        return env_int(name, default)

    @staticmethod
    def _clip_rect(rect: Tuple[int, int, int, int], image_shape) -> Tuple[int, int, int, int]:
        x, y, w, h = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        height, width = image_shape[:2]
        x = max(0, min(width, x))
        y = max(0, min(height, y))
        x2 = max(x, min(width, x + max(0, w)))
        y2 = max(y, min(height, y + max(0, h)))
        return x, y, x2 - x, y2 - y

    @staticmethod
    def _expand_rect(rect: Tuple[int, int, int, int], px: int, py: int, image_shape) -> Tuple[int, int, int, int]:
        x, y, w, h = rect
        return CleanManga._clip_rect((x - px, y - py, w + 2 * px, h + 2 * py), image_shape)

    @staticmethod
    def _rect_mask(rect: Tuple[int, int, int, int], image_shape) -> np.ndarray:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        x, y, w, h = CleanManga._clip_rect(rect, image_shape)
        if w > 0 and h > 0:
            mask[y:y + h, x:x + w] = 255
        return mask

    @staticmethod
    def _binary_mask(mask: np.ndarray, image_shape) -> np.ndarray:
        if mask is None or mask.size == 0:
            return np.zeros(image_shape[:2], dtype=np.uint8)
        out = np.zeros(image_shape[:2], dtype=np.uint8)
        h = min(out.shape[0], mask.shape[0])
        w = min(out.shape[1], mask.shape[1])
        if h <= 0 or w <= 0:
            return out
        out[:h, :w] = (mask[:h, :w] > 0).astype(np.uint8) * 255
        return out

    @staticmethod
    def _safe_bubble_mask(mask: np.ndarray, image_shape, margin: int) -> np.ndarray:
        bubble = CleanManga._binary_mask(mask, image_shape)
        if cv2.countNonZero(bubble) == 0:
            return bubble
        margin = max(0, int(margin))
        if margin > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            eroded = cv2.erode(bubble, kernel, iterations=1)
            # Si el globo es muy estrecho y la erosión lo destruye, usa una erosión menor.
            if cv2.countNonZero(eroded) < max(8, int(cv2.countNonZero(bubble) * 0.18)):
                small = max(1, margin // 2)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * small + 1, 2 * small + 1))
                eroded = cv2.erode(bubble, kernel, iterations=1)
            bubble = eroded if cv2.countNonZero(eroded) > 0 else bubble
        return bubble

    @staticmethod
    def _mask_rectangularity(mask: np.ndarray) -> float:
        if mask is None or cv2.countNonZero(mask) == 0:
            return 1.0
        points = cv2.findNonZero((mask > 0).astype(np.uint8))
        if points is None:
            return 1.0
        x, y, w, h = cv2.boundingRect(points)
        if w <= 0 or h <= 0:
            return 1.0
        return float(cv2.countNonZero(mask)) / float(w * h)

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
    def _text_ink_mask(imagen: np.ndarray, text_zone: np.ndarray, safe_mask: np.ndarray, dilate_px: int = 2) -> np.ndarray:
        if cv2.countNonZero(text_zone) == 0 or cv2.countNonZero(safe_mask) == 0:
            return np.zeros(imagen.shape[:2], dtype=np.uint8)

        zone = cv2.bitwise_and((text_zone > 0).astype(np.uint8) * 255, (safe_mask > 0).astype(np.uint8) * 255)
        if cv2.countNonZero(zone) == 0:
            return np.zeros(imagen.shape[:2], dtype=np.uint8)

        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        vals = gray[zone > 0]
        if vals.size == 0:
            return np.zeros(imagen.shape[:2], dtype=np.uint8)

        # Umbral pensado para letras negras/antialias dentro de globos claros.
        # Usa una mezcla fija + percentil para no capturar todo el fondo del globo.
        percentile_cut = int(np.percentile(vals, 38))
        threshold = min(210, max(115, percentile_cut + 28))
        ink = ((gray <= threshold) & (zone > 0)).astype(np.uint8) * 255

        # Evita borrar tramas muy finas sueltas: conserva componentes que parecen trazos de letra.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        filtered = np.zeros_like(ink)
        for idx in range(1, num):
            x, y, w, h, area = stats[idx]
            if area < 3:
                continue
            # Letras japonesas verticales pueden ser altas; puntos de screentone suelen ser minúsculos.
            if area >= 8 or max(w, h) >= 5:
                filtered[labels == idx] = 255

        if cv2.countNonZero(filtered) == 0:
            filtered = ink

        dilate_px = max(0, int(dilate_px))
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            filtered = cv2.dilate(filtered, kernel, iterations=1)
            filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel, iterations=1)

        return cv2.bitwise_and(filtered, safe_mask)

    def _bubble_text_zone(self, region: TextRegion, image_shape) -> np.ndarray:
        # Usa el bbox OCR/text_bbox como guía, pero nunca como máscara de pintado directa.
        # Se expande para cubrir antialias y detecciones parciales de OCR vertical japonés.
        x, y, w, h = self._clip_rect(region.text_bbox, image_shape)
        bx, by, bw, bh = self._clip_rect(region.bbox, image_shape)
        if w <= 0 or h <= 0:
            return np.zeros(image_shape[:2], dtype=np.uint8)

        # Si no hubo OCR y text_bbox == bbox, no inventamos limpieza de todo el globo.
        if int(getattr(region, "detections_count", 0) or 0) <= 0 and (x, y, w, h) == (bx, by, bw, bh):
            return np.zeros(image_shape[:2], dtype=np.uint8)

        pad_x = max(3, min(14, int(round(max(w, h) * 0.08))))
        pad_y = max(4, min(18, int(round(max(w, h) * 0.10))))
        expanded = self._expand_rect((x, y, w, h), pad_x, pad_y, image_shape)
        return self._rect_mask(expanded, image_shape)

    def _fill_bubble_interiors(self, imagen: np.ndarray, regiones: Sequence[TextRegion]) -> np.ndarray:
        salida = imagen.copy()
        for region in regiones:
            safe_mask = self._safe_bubble_mask(region.mask, salida.shape, self.bubble_fill_edge_margin)
            if cv2.countNonZero(safe_mask) == 0:
                continue

            fill_color = self._dominant_fill_color(salida, safe_mask)

            # Comportamiento antiguo, solo opt-in y solo si la máscara no parece una caja.
            # Esto evita que una segmentación rectangular pinte parches cuadrados sobre bordes.
            if self.bubble_fill_whole_interior and self._mask_rectangularity(safe_mask) < 0.82:
                salida = self._apply_solid_fill(salida, safe_mask, fill_color, sigma=1.0)

            text_zone = self._bubble_text_zone(region, salida.shape)
            ink_mask = self._text_ink_mask(salida, text_zone, safe_mask, self.bubble_fill_text_dilate)
            if cv2.countNonZero(ink_mask) == 0:
                continue
            salida = self._apply_solid_fill(salida, ink_mask, fill_color, sigma=0.65)
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

    async def inpaint_async(self, imagen: np.ndarray, mascara_capa: np.ndarray):
        await self.inpainter._load()
        return await self.inpainter._inpaint(imagen, mascara_capa)

    def convertir_a_imagen_limpia(self, res_impainting: np.ndarray, imagen: np.ndarray) -> np.ndarray:
        pil_image_camuflada_limpieza = Image.fromarray(cv2.cvtColor(res_impainting, cv2.COLOR_BGR2RGB))
        pil_image_limpieza = Image.new("RGB", (imagen.shape[1], imagen.shape[0]))
        pil_image_limpieza.paste(pil_image_camuflada_limpieza, (0, 0))
        imagen_limpia = np.asarray(pil_image_limpieza)
        return cv2.cvtColor(imagen_limpia, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _enhance_for_detection(imagen: np.ndarray) -> np.ndarray:
        """Mejora contraste para textos finos o con tramas de fondo."""
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
        suavizada = cv2.bilateralFilter(clahe, d=5, sigmaColor=35, sigmaSpace=35)
        sharpen = cv2.addWeighted(clahe, 1.45, suavizada, -0.45, 0)
        return cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _to_rect(detection) -> Tuple[int, int, int, int]:
        puntos = np.array(detection[0], dtype=np.float32)
        x, y, w, h = cv2.boundingRect(puntos.astype(np.int32))
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _area(rect: Tuple[int, int, int, int]) -> int:
        return max(0, rect[2]) * max(0, rect[3])

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        union = CleanManga._area(a) + CleanManga._area(b) - inter
        return inter / union if union else 0.0

    @staticmethod
    def _confidence(detection) -> float:
        try:
            return float(detection[2])
        except Exception:
            return 0.0

    @staticmethod
    def _text_from_detection(detection) -> str:
        try:
            return str(detection[1] or "")
        except Exception:
            return ""

    def _is_sound_effect_detection(self, detection) -> bool:
        texto = self._text_from_detection(detection)
        if self.onomatopoeia_manager.is_onomatopoeia_candidate(texto, self.idioma_entrada):
            return True
        try:
            x, y, w, h = self._to_rect(detection)
        except Exception:
            return False
        aspect = max(w, h) / max(1, min(w, h))
        # Muchas onomatopeyas estilizadas aparecen como letras muy alargadas o grandes.
        return aspect >= 4.2 and len(texto.strip()) <= 8

    def _dedupe_detections(self, detections: Iterable, image_shape) -> List:
        """Une resultados de pasadas distintas sin duplicar textos detectados."""
        height, width = image_shape[:2]
        min_area = max(8, int(height * width * 0.000008))
        valid = []
        for det in detections:
            try:
                rect = self._to_rect(det)
            except Exception:
                continue
            x, y, w, h = rect
            if w <= 1 or h <= 1 or self._area(rect) < min_area:
                continue
            if x >= width or y >= height:
                continue
            valid.append((det, rect, self._confidence(det)))

        # Conserva primero los cuadros más confiables y/o grandes.
        valid.sort(key=lambda row: (row[2], self._area(row[1])), reverse=True)
        selected = []
        selected_rects: List[Tuple[int, int, int, int]] = []

        for det, rect, _conf in valid:
            duplicate = False
            for chosen in selected_rects:
                if self._iou(rect, chosen) > 0.62:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(det)
                selected_rects.append(rect)

        return selected

    def _readtext_once(self, lector: easyocr.Reader, imagen: np.ndarray):
        if self.fast_mode:
            canvas_size = 1920
            mag_ratio = 1.15
            beam_width = 3
            batch_size = 8
        else:
            canvas_size = 2880
            mag_ratio = 1.65
            beam_width = 5
            batch_size = 4

        return lector.readtext(
            imagen,
            paragraph=False,
            decoder="beamsearch",
            batch_size=batch_size,
            beamWidth=beam_width,
            width_ths=0.28,
            height_ths=0.18,
            x_ths=0.22,
            y_ths=0.45,
            min_size=4,
            contrast_ths=0.08,
            adjust_contrast=0.65,
            text_threshold=0.45,
            low_text=0.30,
            link_threshold=0.45,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
            add_margin=0.02,
        )

    def obtener_cuadros_delimitadores(self, imagen: np.ndarray):
        lector = self._get_reader()
        detecciones = []

        try:
            detecciones.extend(self._readtext_once(lector, imagen))
        except Exception as exc:
            logger.warning("EasyOCR falló en la imagen original: %s", exc)

        # Segunda pasada de contraste: ayuda en globos con trama, fondo gris o letras finas.
        # En modo rápido se omite porque suele ser la parte más costosa de la detección.
        if not self.fast_mode:
            try:
                imagen_mejorada = self._enhance_for_detection(imagen)
                detecciones.extend(self._readtext_once(lector, imagen_mejorada))
            except Exception as exc:
                logger.warning("EasyOCR falló en la imagen mejorada: %s", exc)

        return self._dedupe_detections(detecciones, imagen.shape)

    def fusionar_cuadros_delimitadores(self, imagen: np.ndarray, resultados) -> np.ndarray:
        height, width = imagen.shape[:2]
        mascara = np.zeros((height, width), dtype=np.uint8)
        base_expansion = max(2, int(round(min(height, width) * 0.0035)))

        for detection in resultados:
            caja = detection[0]
            puntos = np.array(caja, dtype=np.int32).reshape((-1, 1, 2))
            x, y, w, h = cv2.boundingRect(puntos)
            if self._is_sound_effect_detection(detection):
                # Las onomatopeyas suelen tener trazos gruesos/sombras y quedan mal si el inpainting
                # solo cubre la caja OCR mínima. Se expande un poco más, pero con límite.
                expansion = int(min(34, max(base_expansion + 2, round(max(w, h) * 0.13))))
            else:
                expansion = int(min(18, max(base_expansion, round(max(w, h) * 0.08))))
            x_margin = max(0, x - expansion)
            y_margin = max(0, y - expansion)
            x2 = min(width, x + w + expansion)
            y2 = min(height, y + h + expansion)
            if x2 <= x_margin or y2 <= y_margin:
                continue

            puntos_margin = np.array(
                [
                    [x_margin, y_margin],
                    [x2, y_margin],
                    [x2, y2],
                    [x_margin, y2],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mascara, [puntos_margin], 255)

        # Cierra huecos entre trazos y cubre bordes de letras para un inpainting más limpio.
        k = max(2, int(round(min(height, width) * 0.0025)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)
        mascara = cv2.dilate(mascara, kernel, iterations=1)
        return mascara
