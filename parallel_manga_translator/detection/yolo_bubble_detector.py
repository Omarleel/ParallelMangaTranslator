from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from parallel_manga_translator.geometry.box_geometry import BoxGeometry
from parallel_manga_translator.models.processing_models import Box
from parallel_manga_translator.infrastructure.logging_config import get_logger
from parallel_manga_translator.infrastructure.gpu_scheduler import gpu_slot
from parallel_manga_translator.config.app_config import QualityConfig
from parallel_manga_translator.config.runtime_config import get_active_config

logger = get_logger(__name__)


@dataclass
class YoloBubbleCandidate:
    """Resultado de un detector ML entrenado para globos/texto de cómic/manga."""

    bbox: Box
    mask: np.ndarray
    confidence: float
    label: str = "bubble"
    source: str = "yolo11_seg_mask"
    detector: str = "yolo11_seg"
    model_family: str = "YOLO11-seg"
    class_id: int = 0

    @property
    def area(self) -> int:
        return max(0, self.bbox[2]) * max(0, self.bbox[3])


class YoloBubbleDetector:
    """Adaptador principal para YOLO11-seg fine-tuned y pesos Ultralytics compatibles."""

    SUPPORTED_MODES = {"yolo11", "yolo11-seg", "yolo11_seg", "yolo11seg", "yolo11-segmentation"}

    def __init__(self, quality_config: QualityConfig | None = None) -> None:
        quality_config = quality_config or get_active_config().quality
        self.mode = str(quality_config.bubble_detector or "yolo11-seg").strip().lower()
        if self.mode not in self.SUPPORTED_MODES:
            raise RuntimeError(
                f"Detector de globos no soportado: {self.mode}. Usa quality.bubble_detector=yolo11-seg."
            )

        self.enabled = True
        self.detector_name = "yolo11_seg"
        self.model_family = "YOLO11-seg"

        # Sin modelo YOLO válido no hay fallback heurístico.
        self.required = bool(quality_config.require_yolo)
        self.model_path = self._first_non_empty(quality_config.bubble_model_path)
        self.repo_id = self._first_non_empty(
            quality_config.bubble_model_repo,
            "huyvux3005/manga109-segmentation-bubble",
        )
        self.filename = self._first_non_empty(
            quality_config.bubble_model_file,
            "best.pt",
        )
        self.conf = float(quality_config.bubble_confidence)
        self.imgsz = int(quality_config.bubble_img_size)
        self.device = self._normalize_device(str(quality_config.bubble_device or "").strip())
        self.retina_masks = bool(getattr(quality_config, "bubble_retina_masks", True))
        self.max_area_ratio = float(getattr(quality_config, "bubble_max_area_ratio", 0.55))
        self.class_ids = self._parse_class_ids(getattr(quality_config, "bubble_model_classes", ""))
        self.include_labels = self._parse_label_tokens(getattr(quality_config, "bubble_include_labels", ""))
        self.exclude_labels = self._parse_label_tokens(
            getattr(quality_config, "bubble_exclude_labels", "ignore_art,panel,page,background")
        )
        self._model = None
        self._load_error: Optional[Exception] = None
        self._warned = False

    @staticmethod
    def _first_non_empty(*values: object) -> str:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _normalize_device(value: str) -> Optional[str]:
        normalized = value.strip().lower()
        if not normalized or normalized in {"auto", "none", "null"}:
            return None
        if normalized in {"gpu", "cuda"}:
            return "cuda:0"
        if normalized.startswith("gpu:"):
            return "cuda:" + normalized.split(":", 1)[1]
        return value.strip()

    @staticmethod
    def _parse_class_ids(raw: object) -> Optional[List[int]]:
        if raw in (None, "", [], ()):
            return None
        if isinstance(raw, (list, tuple, set)):
            values = raw
        else:
            values = str(raw).replace(";", ",").split(",")
        parsed: List[int] = []
        for item in values:
            text = str(item).strip()
            if not text:
                continue
            try:
                parsed.append(int(text))
            except ValueError:
                logger.warning("Ignorando class id no numérico en quality.bubble_model_classes: %s", text)
        return parsed or None

    @staticmethod
    def _parse_label_tokens(raw: object) -> set[str]:
        if raw in (None, "", [], ()):
            return set()
        if isinstance(raw, (list, tuple, set)):
            values = raw
        else:
            values = str(raw).replace(";", ",").split(",")
        return {YoloBubbleDetector._normalize_label_token(str(item)) for item in values if str(item).strip()}

    @staticmethod
    def _normalize_label_token(label: str) -> str:
        return " ".join(str(label or "").strip().lower().replace("_", " ").replace("-", " ").split())

    @staticmethod
    def _label_matches_any(normalized: str, tokens: set[str]) -> bool:
        words = set(normalized.split())
        return any(token == normalized or token in words or token in normalized for token in tokens if token)

    def _label_allowed(self, label: str) -> bool:
        normalized = self._normalize_label_token(label)
        if self.include_labels and not self._label_matches_any(normalized, self.include_labels):
            return False
        if self._label_matches_any(normalized, self.exclude_labels):
            return False
        return True

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
                "Para descargar automáticamente el detector YOLO11-seg instala `huggingface_hub`, "
                "o define quality.bubble_model_path en config.yaml con un modelo local."
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
                raise FileNotFoundError(f"No existe el modelo configurado: {resolved}")
            self._model = YOLO(resolved)
            logger.info("Detector %s cargado: %s", self.model_family, resolved)
            return self._model
        except Exception as exc:  # pragma: no cover - depende de dependencias/modelos externos
            self._load_error = exc
            raise RuntimeError(
                "No se pudo cargar el detector YOLO11-seg de globos. "
                "Instala ultralytics/huggingface_hub, permite la descarga del modelo o define "
                "quality.bubble_model_path en config.yaml con un .pt válido. "
                f"Detalle: {exc}"
            ) from exc

    @staticmethod
    def _box_from_xyxy(xyxy) -> Box:
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[:4]]
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def _clip_box(box: Box, width: int, height: int) -> Box:
        return BoxGeometry.clip(box, width, height)

    @staticmethod
    def _mask_from_box(box: Box, shape) -> np.ndarray:
        height, width = shape[:2]
        x, y, w, h = YoloBubbleDetector._clip_box(box, width, height)
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
    def _mask_from_tensor(mask_tensor, shape) -> np.ndarray:
        """Convierte máscaras binarias de Ultralytics a la resolución de la página."""
        height, width = shape[:2]
        try:
            data = mask_tensor.detach().cpu().numpy()
        except Exception:
            data = np.asarray(mask_tensor)
        if data.ndim > 2:
            data = np.squeeze(data)
        if data.size == 0:
            return np.zeros((height, width), dtype=np.uint8)
        mask = (data > 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.uint8)

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> Optional[Box]:
        pts = cv2.findNonZero(mask)
        if pts is None:
            return None
        x, y, w, h = cv2.boundingRect(pts)
        return int(x), int(y), int(w), int(h)

    def _mask_for_detection(self, result, index: int, bbox: Box, image_shape) -> tuple[np.ndarray, str]:
        masks_obj = getattr(result, "masks", None)
        if masks_obj is not None:
            mask_data = getattr(masks_obj, "data", None)
            if mask_data is not None:
                try:
                    if index < len(mask_data):
                        tensor_mask = self._mask_from_tensor(mask_data[index], image_shape)
                        if cv2.countNonZero(tensor_mask) > 0:
                            return tensor_mask, "yolo11_seg_mask"
                except Exception:
                    pass

            polygons = list(getattr(masks_obj, "xy", []) or [])
            if index < len(polygons):
                poly_mask = self._mask_from_polygon(polygons[index], image_shape)
                if cv2.countNonZero(poly_mask) > 0:
                    return poly_mask, "yolo11_seg_polygon"

        return self._mask_from_box(bbox, image_shape), "yolo11_seg_box"

    def detect(self, image: np.ndarray) -> List[YoloBubbleCandidate]:
        model = self._load_model()
        if model is None:
            raise RuntimeError("No hay detector YOLO11-seg de globos cargado.")
        if image is None or image.size == 0:
            return []

        height, width = image.shape[:2]
        try:
            kwargs = {
                "imgsz": self.imgsz,
                "conf": self.conf,
                "verbose": False,
                "retina_masks": self.retina_masks,
            }
            if self.class_ids is not None:
                kwargs["classes"] = self.class_ids
            if self.device:
                kwargs["device"] = self.device
            use_gpu = torch.cuda.is_available() and str(self.device or "auto").lower() != "cpu"
            with gpu_slot("yolo.predict", enabled=use_gpu):
                results = model.predict(image, **kwargs)
        except Exception as exc:  # pragma: no cover - depende de ultralytics/runtime
            raise RuntimeError(f"Falló la inferencia del detector YOLO11-seg de globos: {exc}") from exc

        candidates: List[YoloBubbleCandidate] = []
        if not results:
            return candidates

        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
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

            try:
                conf = float(box_obj.conf[0])
            except Exception:
                conf = self.conf
            try:
                cls_id = int(box_obj.cls[0])
            except Exception:
                cls_id = 0
            label = self._safe_label(names, cls_id)
            if not self._label_allowed(label):
                continue

            mask, source = self._mask_for_detection(result, i, bbox, image.shape)
            if cv2.countNonZero(mask) == 0:
                continue
            bbox = self._mask_bbox(mask) or bbox

            # Erosiona un poco para limpiar el interior sin comerse el borde del globo.
            bx, by, bw, bh = bbox
            erode = max(1, min(6, int(round(min(bw, bh) * 0.018))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
            inner_mask = cv2.erode(mask, kernel, iterations=1)
            if cv2.countNonZero(inner_mask) > 0:
                mask = inner_mask
                bbox = self._mask_bbox(mask) or bbox
                bx, by, bw, bh = bbox

            # Descarta detecciones desproporcionadamente grandes que suelen ser viñetas/página.
            if bw * bh > width * height * self.max_area_ratio:
                continue
            candidates.append(YoloBubbleCandidate(
                bbox=bbox,
                mask=mask,
                confidence=max(0.0, min(1.0, conf)),
                label=label,
                source=source,
                detector=self.detector_name,
                model_family=self.model_family,
                class_id=cls_id,
            ))

        return candidates
