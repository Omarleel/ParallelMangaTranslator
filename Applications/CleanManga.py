from __future__ import annotations

import asyncio
import os
from typing import Iterable, List, Sequence, Tuple

import cv2
try:
    import easyocr
except ModuleNotFoundError:  # EasyOCR es opcional si se usa MangaOCR/Paddle o en tests de limpieza.
    easyocr = None  # type: ignore[assignment]
import nest_asyncio
import numpy as np
import torch
from PIL import Image

from Applications.inpaint import AOTInpainter, BNInpainter, LamaInpainterMPE, LamaLarge, OpenCVInpainter
from Applications.OnomatopoeiaManager import OnomatopoeiaManager
from Applications.BubbleDetector import BubbleDetector
from Applications.CacheManager import env_flag
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
        self.bubble_fill_edge_margin = self._int_env("PMT_BUBBLE_FILL_EDGE_MARGIN", 6)
        self.bubble_fill_feather = self._float_env("PMT_BUBBLE_FILL_FEATHER", 1.0)
        self.onomatopoeia_manager = OnomatopoeiaManager()
        self.bubble_detector = BubbleDetector(idioma_entrada=idioma_entrada)
        self.inpainter = self._build_inpainter(modelo_inpaint)
        self._ocr_reader = None
        self.last_regions: List[TextRegion] = []

    def _build_inpainter(self, model_name: str):
        if model_name not in self.INPAINTER_FACTORIES:
            raise ValueError(f"Modelo de inpainting no soportado: {model_name}")
        return self.INPAINTER_FACTORIES[model_name]()

    def _get_reader(self):
        if easyocr is None:
            raise RuntimeError("EasyOCR no está instalado. Instala easyocr o usa PMT_OCR_ENGINE=mangaocr/paddle.")
        if self._ocr_reader is None:
            langs = self.EASY_OCR_LANGS.get(self.idioma_entrada, ["ja", "en"])
            self._ocr_reader = easyocr.Reader(langs, gpu=self.device == "cuda")
        return self._ocr_reader

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            return default

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            return max(0.0, float(raw))
        except ValueError:
            return default

    @staticmethod
    def _safe_bubble_fill_mask(local_mask: np.ndarray, edge_margin_px: int) -> np.ndarray:
        """Devuelve una máscara interior que no toca el borde del globo.

        Las máscaras de segmentación suelen incluir parte del contorno negro del globo o
        quedar demasiado pegadas a él. Para limpiar diálogos por relleno plano no debemos
        pintar sobre ese borde, así que erosionamos la máscara hacia adentro antes de
        mezclar el color de limpieza. Si el globo es muy pequeño, reducimos el margen
        automáticamente para no dejar la máscara vacía.
        """
        if local_mask.size == 0:
            return local_mask

        binary = (local_mask > 0).astype(np.uint8) * 255
        if edge_margin_px <= 0 or cv2.countNonZero(binary) == 0:
            return binary

        max_margin = max(0, (min(binary.shape[:2]) - 3) // 2)
        margin = min(int(edge_margin_px), max_margin)
        while margin > 0:
            kernel_size = margin * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            safe = cv2.erode(binary, kernel, iterations=1)
            if cv2.countNonZero(safe) > 0:
                return safe
            margin -= 1
        return binary

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
        sfx_regions = [r for r in regiones if r.kind not in {"dialogue", "narration", "unknown"}]

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

    def _fill_bubble_interiors(self, imagen: np.ndarray, regiones: Sequence[TextRegion]) -> np.ndarray:
        salida = imagen.copy()
        for region in regiones:
            x, y, w, h = region.bbox
            x = max(0, int(x)); y = max(0, int(y))
            w = min(int(w), salida.shape[1] - x); h = min(int(h), salida.shape[0] - y)
            if w <= 1 or h <= 1:
                continue
            local_mask = region.mask[y:y + h, x:x + w]
            if local_mask.size == 0 or cv2.countNonZero(local_mask) == 0:
                continue

            # No pintamos el borde real del globo: la segmentación puede incluirlo o
            # quedar demasiado pegada a él. Primero hacemos una máscara interior segura
            # y recién sobre esa zona aplicamos un feather suave.
            edge_margin = getattr(self, "bubble_fill_edge_margin", 6)
            safe_mask = self._safe_bubble_fill_mask(local_mask, edge_margin)
            if cv2.countNonZero(safe_mask) == 0:
                continue

            feather = getattr(self, "bubble_fill_feather", 1.0)
            if feather > 0:
                blur = cv2.GaussianBlur(safe_mask, (0, 0), sigmaX=feather, sigmaY=feather)
                # El blur no puede reactivar píxeles fuera de la máscara original.
                blur = cv2.bitwise_and(blur, (local_mask > 0).astype(np.uint8) * 255)
            else:
                blur = safe_mask
            alpha = (blur.astype(np.float32) / 255.0)[..., None]
            fill_color = np.array(self._dominant_fill_color(salida[y:y + h, x:x + w], safe_mask), dtype=np.float32)
            patch = salida[y:y + h, x:x + w].astype(np.float32)
            cleaned = patch * (1.0 - alpha) + fill_color * alpha
            salida[y:y + h, x:x + w] = np.clip(cleaned, 0, 255).astype(np.uint8)
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
        if self.onomatopoeia_manager.is_onomatopoeia(texto, self.idioma_entrada):
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
