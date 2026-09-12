"""Detector espacial de texto de manga/cómic (comic-text-detector) sobre ONNX.

Es un detector de **texto**, no de globos: devuelve cajas de bloque y una máscara de
tinta a resolución de página, que es justo lo que el pipeline necesita en tres sitios
(recorte de OCR, zona segura y máscara de tinta). Un detector de solo cajas obliga a
blanquear con un rectángulo y el recorte de OCR se llena de arte vecino.

Corre en `cv2.dnn`, que ya es dependencia: no añade paquetes, no compite por CUDA con
YOLO/OCR/inpainting y cuesta ~1 s de CPU por página.

Las tres salidas del modelo se llaman `blk`, `det` y `seg`, y el nombre engaña. Medido
sobre el .onnx publicado:

    blk: (1, 64512, 7)      cajas [cx, cy, w, h, objectness, cls0, cls1]
    det: (1, 2, 1024, 1024) dos canales de máscara
    seg: (1, 1, 1024, 1024) un canal

La máscara de texto es ``det[0]``: su activación media dentro de las cajas del ground
truth es 28.6x la de fuera, frente a 10.0x de la salida llamada `seg` y 1.1x de
``det[1]``. Escoger `seg` por el nombre da una máscara mucho peor.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from parallel_manga_translator.config.constants import URL_MODELO_COMIC_TEXT_DETECTOR
from parallel_manga_translator.infrastructure.execution_control import execution_checkpoint
from parallel_manga_translator.infrastructure.logging_config import get_logger

logger = get_logger(__name__)

Box = Tuple[int, int, int, int]

#: Ruta por defecto de los pesos, junto al resto de modelos descargados en runtime.
DEFAULT_WEIGHTS_PATH = "models/detection/comictextdetector.pt.onnx"

#: El modelo se exportó a 1024x1024; cambiarlo degrada la detección.
MODEL_INPUT_SIZE = 1024

#: Canal de `det` que contiene la máscara de texto. Ver el docstring del módulo.
TEXT_MASK_CHANNEL = 0


@dataclass(frozen=True)
class ComicTextDetection:
    """Salida cruda del detector, en coordenadas de la página original."""

    boxes: List[Box] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    #: Máscara de tinta 0/255 a resolución de página.
    text_mask: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.boxes)


class ComicTextDetectorWeights:
    """Localiza el .onnx del detector y lo descarga si falta.

    Responsabilidad única a propósito: el detector no debería saber de rutas, descargas
    ni de qué hacer cuando el usuario apunta a un archivo suyo.
    """

    def __init__(self, model_path: str = "", url: str = URL_MODELO_COMIC_TEXT_DETECTOR) -> None:
        self.model_path = str(model_path or "").strip()
        self.url = url

    def resolve(self) -> Path:
        if self.model_path:
            configured = Path(self.model_path)
            if not configured.is_file():
                raise FileNotFoundError(
                    f"No existe el modelo comic-text-detector configurado: {configured}"
                )
            return configured

        target = Path(DEFAULT_WEIGHTS_PATH)
        if target.is_file():
            return target
        return self._download(target)

    def _download(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Descargando comic-text-detector desde %s", self.url)
        try:
            import urllib.request

            # A un archivo temporal primero: una descarga cortada no debe dejar un .onnx
            # a medias que luego falle al cargar con un error incomprensible.
            partial = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(self.url, partial)  # noqa: S310 - URL fija del release
            partial.replace(target)
        except Exception as exc:  # pragma: no cover - depende de la red del usuario
            raise RuntimeError(
                "No se pudo descargar comic-text-detector. Descárgalo a mano en "
                f"{DEFAULT_WEIGHTS_PATH} desde {self.url}, o define "
                "quality.comic_text_detector_model_path en config.yaml. "
                f"Detalle: {exc}"
            ) from exc
        return target


class ComicTextDetector:
    """Ejecuta el modelo y decodifica sus tres cabezas.

    No conoce `TextRegion` ni el pipeline: devuelve cajas y máscara. Convertir eso en
    regiones es responsabilidad del adaptador, que es quien sí depende del dominio.
    """

    def __init__(
        self,
        *,
        weights: ComicTextDetectorWeights | None = None,
        conf_threshold: float = 0.40,
        nms_threshold: float = 0.35,
        mask_threshold: float = 0.30,
        input_size: int = MODEL_INPUT_SIZE,
    ) -> None:
        self.weights = weights if weights is not None else ComicTextDetectorWeights()
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.mask_threshold = float(mask_threshold)
        self.input_size = int(input_size)
        self._net = None
        self._output_names: Sequence[str] = ()
        # cv2.dnn no es reentrante: el productor/consumidor del pipeline puede pedir dos
        # páginas a la vez y compartirían el mismo blob de entrada.
        self._lock = threading.Lock()

    # -- carga -----------------------------------------------------------------

    def _load(self):
        if self._net is not None:
            return self._net
        resolved = self.weights.resolve()
        try:
            net = cv2.dnn.readNetFromONNX(str(resolved))
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo cargar comic-text-detector desde {resolved}. Detalle: {exc}"
            ) from exc
        self._net = net
        self._output_names = tuple(net.getUnconnectedOutLayersNames())
        logger.info("comic-text-detector cargado: %s (salidas: %s)", resolved, ", ".join(self._output_names))
        return net

    # -- inferencia ------------------------------------------------------------

    def predict(self, image: np.ndarray) -> ComicTextDetection:
        """Detecta bloques de texto y devuelve su máscara de tinta."""
        if image is None or getattr(image, "size", 0) == 0:
            return ComicTextDetection(text_mask=np.zeros((0, 0), dtype=np.uint8))
        execution_checkpoint()
        height, width = image.shape[:2]
        blob, scale, pad = self._preprocess(image)

        with self._lock:
            net = self._load()
            net.setInput(blob)
            outputs = net.forward(self._output_names)
        execution_checkpoint()

        named = dict(zip(self._output_names, outputs, strict=True))
        boxes, scores, class_ids = self._decode_boxes(named.get("blk"), scale, pad, width, height)
        text_mask = self._decode_text_mask(named.get("det"), scale, pad, width, height)
        return ComicTextDetection(boxes=boxes, scores=scores, class_ids=class_ids, text_mask=text_mask)

    # -- pre/post proceso ------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Letterbox a la entrada del modelo conservando la relación de aspecto.

        Deformar la página cambiaría la forma de los caracteres, que es justo la señal
        que el modelo aprendió.
        """
        height, width = image.shape[:2]
        scale = min(self.input_size / max(1, width), self.input_size / max(1, height))
        new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        pad_x, pad_y = (self.input_size - new_w) // 2, (self.input_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, swapRB=True)
        return blob, scale, (pad_x, pad_y)

    def _decode_boxes(
        self,
        raw: Optional[np.ndarray],
        scale: float,
        pad: tuple[int, int],
        width: int,
        height: int,
    ) -> tuple[List[Box], List[float], List[int]]:
        if raw is None or getattr(raw, "size", 0) == 0:
            return [], [], []
        predictions = np.asarray(raw)
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2 or predictions.shape[1] < 6:
            logger.warning("Cabeza de cajas inesperada en comic-text-detector: %s", predictions.shape)
            return [], [], []

        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = objectness * class_scores[np.arange(len(class_ids)), class_ids]
        keep = confidences >= self.conf_threshold
        if not np.any(keep):
            return [], [], []

        pad_x, pad_y = pad
        centers = predictions[keep, :4]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # De centro/tamaño en la entrada del modelo a esquina/tamaño en la página.
        xs = (centers[:, 0] - centers[:, 2] / 2 - pad_x) / max(scale, 1e-6)
        ys = (centers[:, 1] - centers[:, 3] / 2 - pad_y) / max(scale, 1e-6)
        ws = centers[:, 2] / max(scale, 1e-6)
        hs = centers[:, 3] / max(scale, 1e-6)

        rects = [[int(round(x)), int(round(y)), int(round(w)), int(round(h))] for x, y, w, h in zip(xs, ys, ws, hs, strict=True)]
        indices = cv2.dnn.NMSBoxes(rects, confidences.astype(float).tolist(), self.conf_threshold, self.nms_threshold)
        if indices is None or len(indices) == 0:
            return [], [], []

        boxes: List[Box] = []
        kept_scores: List[float] = []
        kept_classes: List[int] = []
        for index in np.asarray(indices).reshape(-1):
            x, y, w, h = rects[int(index)]
            clipped = self._clip_box((x, y, w, h), width, height)
            if clipped[2] <= 1 or clipped[3] <= 1:
                continue
            boxes.append(clipped)
            kept_scores.append(float(confidences[int(index)]))
            kept_classes.append(int(class_ids[int(index)]))
        return boxes, kept_scores, kept_classes

    def _decode_text_mask(
        self,
        raw: Optional[np.ndarray],
        scale: float,
        pad: tuple[int, int],
        width: int,
        height: int,
    ) -> np.ndarray:
        empty = np.zeros((height, width), dtype=np.uint8)
        if raw is None or getattr(raw, "size", 0) == 0:
            return empty
        tensor = np.asarray(raw)
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            channel = min(TEXT_MASK_CHANNEL, tensor.shape[0] - 1)
            tensor = tensor[channel]
        if tensor.ndim != 2:
            logger.warning("Cabeza de máscara inesperada en comic-text-detector: %s", np.asarray(raw).shape)
            return empty

        pad_x, pad_y = pad
        inner_w = max(1, int(round(width * scale)))
        inner_h = max(1, int(round(height * scale)))
        # Se recorta el relleno del letterbox antes de devolver la máscara a la página.
        cropped = tensor[pad_y:pad_y + inner_h, pad_x:pad_x + inner_w]
        if cropped.size == 0:
            return empty
        resized = cv2.resize(cropped.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        return ((resized >= self.mask_threshold) * 255).astype(np.uint8)

    @staticmethod
    def _clip_box(box: Box, width: int, height: int) -> Box:
        x, y, w, h = [int(v) for v in box]
        x1 = max(0, min(width, x))
        y1 = max(0, min(height, y))
        x2 = max(x1, min(width, x + max(0, w)))
        y2 = max(y1, min(height, y + max(0, h)))
        return x1, y1, x2 - x1, y2 - y1
