from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from Applications.CacheManager import env_flag
from Applications.ProcessingModels import Box
from Applications.LoggingConfig import get_logger

logger = get_logger(__name__)


@dataclass
class ProfessionalBubbleCandidate:
    """Resultado de un detector ML entrenado para globos de cómic/manga."""

    bbox: Box
    mask: np.ndarray
    confidence: float
    label: str = "bubble"
    source: str = "professional_yolo"

    @property
    def area(self) -> int:
        return max(0, self.bbox[2]) * max(0, self.bbox[3])


class ProfessionalBubbleDetector:
    """Adaptador para detectores profesionales de globos de texto.
    """

    def __init__(self) -> None:
        self.mode = os.getenv("PMT_BUBBLE_DETECTOR", "professional").strip().lower()
        if self.mode in {"heuristic", "classic", "opencv", "legacy"}:
            raise RuntimeError("La detección heurística de globos fue eliminada. Usa un modelo preentrenado con PMT_BUBBLE_DETECTOR=professional.")
        self.enabled = self.mode in {"professional", "auto", "ml", "yolo"}
        if not self.enabled:
            raise RuntimeError(f"Detector de globos no soportado: {self.mode}. Usa professional/yolo/ml.")
        # En esta versión el detector profesional es obligatorio: sin modelo no hay fallback heurístico.
        self.required = env_flag("PMT_REQUIRE_PROFESSIONAL_BUBBLE", True)
        self.model_path = os.getenv("PMT_BUBBLE_MODEL_PATH", "").strip()
        self.repo_id = os.getenv("PMT_BUBBLE_MODEL_REPO", "huyvux3005/manga109-segmentation-bubble").strip()
        self.filename = os.getenv("PMT_BUBBLE_MODEL_FILE", "best.pt").strip()
        self.conf = self._float_env("PMT_BUBBLE_CONF", 0.35)
        self.imgsz = self._int_env("PMT_BUBBLE_IMGSZ", 1024)
        self.device = os.getenv("PMT_BUBBLE_DEVICE", "").strip() or None
        self._model = None
        self._load_error: Optional[Exception] = None
        self._warned = False

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        try:
            return int(float(os.getenv(name, str(default))))
        except ValueError:
            return default

    @staticmethod
    def _safe_label(names, cls_id: int) -> str:
        try:
            if isinstance(names, dict):
                return str(names.get(cls_id, "bubble"))
            if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                return str(names[cls_id])
        except Exception:
            pass
        return "bubble"

    def _resolve_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as exc:  # pragma: no cover - depende del entorno del usuario
            raise RuntimeError(
                "Para descargar automáticamente el detector profesional instala `huggingface_hub`, "
                "o define PMT_BUBBLE_MODEL_PATH con un modelo local."
            ) from exc
        return hf_hub_download(repo_id=self.repo_id, filename=self.filename)

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.enabled:
            return None
        if self._load_error is not None:
            if self.required:
                raise self._load_error
            return None
        try:
            from ultralytics import YOLO  # type: ignore
            resolved = self._resolve_model_path()
            if self.model_path and not Path(resolved).exists():
                raise FileNotFoundError(f"No existe PMT_BUBBLE_MODEL_PATH: {resolved}")
            self._model = YOLO(resolved)
            logger.info("Detector profesional de globos cargado: %s", resolved)
            return self._model
        except Exception as exc:  # pragma: no cover - depende de dependencias/modelos externos
            self._load_error = exc
            raise RuntimeError(
                "No se pudo cargar el detector profesional de globos. "
                "Instala ultralytics/huggingface_hub, permite la descarga del modelo o define PMT_BUBBLE_MODEL_PATH con un .pt válido. "
                f"Detalle: {exc}"
            ) from exc

    @staticmethod
    def _box_from_xyxy(xyxy) -> Box:
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[:4]]
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def _clip_box(box: Box, width: int, height: int) -> Box:
        x, y, w, h = box
        x1 = max(0, min(width - 1, x))
        y1 = max(0, min(height - 1, y))
        x2 = max(x1 + 1, min(width, x + w))
        y2 = max(y1 + 1, min(height, y + h))
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _mask_from_box(box: Box, shape) -> np.ndarray:
        height, width = shape[:2]
        x, y, w, h = ProfessionalBubbleDetector._clip_box(box, width, height)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        return mask

    @staticmethod
    def _mask_from_polygon(poly: np.ndarray, shape) -> np.ndarray:
        height, width = shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        if poly is None or len(poly) < 3:
            return mask
        pts = np.asarray(poly, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        return mask

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> Optional[Box]:
        pts = cv2.findNonZero(mask)
        if pts is None:
            return None
        x, y, w, h = cv2.boundingRect(pts)
        return int(x), int(y), int(w), int(h)

    def detect(self, image: np.ndarray) -> List[ProfessionalBubbleCandidate]:
        model = self._load_model()
        if model is None:
            raise RuntimeError("No hay detector profesional de globos cargado.")
        if image is None or image.size == 0:
            return []

        height, width = image.shape[:2]
        try:
            kwargs = {"imgsz": self.imgsz, "conf": self.conf, "verbose": False}
            if self.device:
                kwargs["device"] = self.device
            results = model.predict(image, **kwargs)
        except Exception as exc:  # pragma: no cover - depende de ultralytics/runtime
            raise RuntimeError(f"Falló la inferencia del detector profesional de globos: {exc}") from exc

        candidates: List[ProfessionalBubbleCandidate] = []
        if not results:
            return candidates

        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        polygons = []
        masks_obj = getattr(result, "masks", None)
        if masks_obj is not None:
            polygons = list(getattr(masks_obj, "xy", []) or [])

        if boxes is None:
            return candidates

        for i, box_obj in enumerate(boxes):
            try:
                xyxy = box_obj.xyxy[0].detach().cpu().numpy()
            except Exception:
                try:
                    xyxy = np.asarray(box_obj.xyxy[0])
                except Exception:
                    continue
            bbox = self._clip_box(self._box_from_xyxy(xyxy), width, height)

            mask = None
            if i < len(polygons):
                mask = self._mask_from_polygon(polygons[i], image.shape)
            if mask is None or cv2.countNonZero(mask) == 0:
                mask = self._mask_from_box(bbox, image.shape)
                source = "professional_yolo_box"
            else:
                source = "professional_yolo_seg"
                bbox = self._mask_bbox(mask) or bbox

            # Erosiona un poco para limpiar el interior sin comerse el borde del globo.
            bx, by, bw, bh = bbox
            erode = max(1, min(6, int(round(min(bw, bh) * 0.018))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
            inner_mask = cv2.erode(mask, kernel, iterations=1)
            if cv2.countNonZero(inner_mask) > 0:
                mask = inner_mask
                bbox = self._mask_bbox(mask) or bbox

            try:
                conf = float(box_obj.conf[0])
            except Exception:
                conf = self.conf
            try:
                cls_id = int(box_obj.cls[0])
            except Exception:
                cls_id = 0
            label = self._safe_label(names, cls_id)

            # Descarta detecciones desproporcionadamente grandes que suelen ser viñetas/página.
            if bw * bh > width * height * 0.55:
                continue
            candidates.append(ProfessionalBubbleCandidate(
                bbox=bbox,
                mask=mask,
                confidence=max(0.0, min(1.0, conf)),
                label=label,
                source=source,
            ))

        return candidates
