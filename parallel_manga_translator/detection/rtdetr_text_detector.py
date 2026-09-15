"""Detector de bloques de texto de cómic/manga (RT-DETR-v2) sobre onnxruntime.

Qué aporta frente a lo que ya hay
---------------------------------
Es el primer modelo medido que **empareja las regiones que una persona dibujó a mano**.
Sobre el subconjunto ``manual`` de ``dataset_eval`` —el único sin sesgo, porque nadie lo
derivó de un detector— el recall por criterio ``globo`` queda así:

    caso     yolo    comic_text_detector   este (text @0.30)
    ja_01    0.000   0.255                 0.447
    ja_02    0.000   0.022                 0.804
    en_01    0.000   0.250                 0.625
    en_02    0.231   0.538                 0.692

El detector actual saca 0.000 en tres de los cuatro casos: no es que las empareje con la
forma equivocada, es que no las ve. Cuesta ~0.20 s/página en CPU, frente a los ~4.7-5.6
s/página que costó `comic-text-detector` en la misma corrida.

Eso mide **localización, no resultado de página**. Está documentado en el banco que un F1
de detector no predice el del pipeline cuando el pipeline tiene otra fuente de regiones.

Por qué onnxruntime y no cv2.dnn
--------------------------------
`comic-text-detector` corre en `cv2.dnn`, que ya es dependencia. Aquí no se puede: el
importador ONNX de OpenCV 4.10 falla al procesar el nodo ``CumSum`` del encoder. Por eso
`onnxruntime` es un extra opcional (``pip install parallel-manga-translator[onnx]``) y
este módulo da un mensaje accionable si falta, igual que hace PaddleOCR.

Se ejecuta en CPU a propósito: no compite por la GPU con YOLO, OCR e inpainting, que es el
mismo criterio con el que se integró `comic-text-detector`.

Detalles del modelo
-------------------
El .onnx publicado **ya incluye el postproceso**: entra la imagen y ``orig_target_sizes``,
y salen ``labels``/``boxes``/``scores`` en coordenadas de la página original (xyxy). No hay
que decodificar anclas ni aplicar NMS.

Tres clases, y la distinción importa porque no todas son texto:

    0 bubble        el contorno del globo, no su texto
    1 text_bubble   texto dentro de globo
    2 text_free     texto fuera de globo (rótulos, gritos sobre el arte)

El preprocesador publicado usa 640x640 bilineal y ``do_normalize: false``: solo se divide
por 255, sin media/desviación de ImageNet.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from parallel_manga_translator.config.constants import URL_MODELO_RTDETR_COMIC_TEXT
from parallel_manga_translator.infrastructure.logging_config import get_logger

logger = get_logger(__name__)

Box = Tuple[int, int, int, int]

#: Ruta por defecto de los pesos, junto al resto de modelos descargados en runtime.
DEFAULT_WEIGHTS_PATH = "models/detection/comic_text_and_bubble_rtdetr.onnx"

#: El modelo se exportó a 640x640 (preprocessor_config.json). Cambiarlo degrada la
#: detección: las escalas del encoder están fijadas a ese tamaño.
MODEL_INPUT_SIZE = 640

#: Nombres de clase del modelo publicado.
CLASS_BUBBLE = "bubble"
CLASS_TEXT_BUBBLE = "text_bubble"
CLASS_TEXT_FREE = "text_free"
ID2LABEL = {0: CLASS_BUBBLE, 1: CLASS_TEXT_BUBBLE, 2: CLASS_TEXT_FREE}

#: Las clases que son texto. `bubble` describe el contorno, no lo que hay escrito dentro,
#: así que emitirla como región duplicaría cada globo.
TEXT_CLASSES = (CLASS_TEXT_BUBBLE, CLASS_TEXT_FREE)


@dataclass(frozen=True)
class RtDetrBox:
    """Una caja del detector, en coordenadas de la página original."""

    bbox: Box
    score: float
    label: str

    @property
    def is_text(self) -> bool:
        return self.label in TEXT_CLASSES


@dataclass(frozen=True)
class RtDetrDetection:
    """Salida cruda del detector, separada por rol.

    Las cajas de texto son las que se convierten en regiones. Las de globo no: sirven
    para decidir si un bloque de texto cae dentro o fuera, que es una señal del propio
    modelo y sale gratis en el mismo forward.
    """

    text_boxes: List[RtDetrBox] = field(default_factory=list)
    bubble_boxes: List[RtDetrBox] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.text_boxes)


class RtDetrTextDetectorWeights:
    """Localiza el .onnx del detector y lo descarga si falta.

    Mismo reparto que en `comic-text-detector`: el detector no debería saber de rutas,
    descargas ni de qué hacer cuando el usuario apunta a un archivo suyo.
    """

    def __init__(self, model_path: str = "", url: str = URL_MODELO_RTDETR_COMIC_TEXT) -> None:
        self.model_path = str(model_path or "").strip()
        self.url = url

    def resolve(self) -> Path:
        if self.model_path:
            configured = Path(self.model_path)
            if not configured.is_file():
                raise FileNotFoundError(
                    f"No existe el modelo RT-DETR configurado: {configured}"
                )
            return configured

        target = Path(DEFAULT_WEIGHTS_PATH)
        if target.is_file():
            return target
        return self._download(target)

    def _download(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Descargando el detector RT-DETR desde %s", self.url)
        try:
            import urllib.request

            # A un archivo temporal primero: una descarga cortada no debe dejar un .onnx
            # a medias que luego falle al cargar con un error incomprensible.
            partial = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(self.url, partial)  # noqa: S310 - URL fija del repo del modelo
            partial.replace(target)
        except Exception as exc:  # pragma: no cover - depende de la red del usuario
            raise RuntimeError(
                "No se pudo descargar el detector RT-DETR. Descárgalo a mano en "
                f"{DEFAULT_WEIGHTS_PATH} desde {self.url}, o define "
                "quality.rtdetr_text_model_path en config.yaml. "
                f"Detalle: {exc}"
            ) from exc
        return target


class RtDetrTextDetector:
    """Ejecuta el modelo y separa cajas de texto de cajas de globo.

    No conoce `TextRegion` ni el pipeline: devuelve cajas. Convertir eso en regiones es
    responsabilidad del adaptador, que es quien sí depende del dominio.
    """

    def __init__(
        self,
        *,
        weights: RtDetrTextDetectorWeights | None = None,
        conf_threshold: float = 0.30,
        bubble_conf_threshold: float = 0.50,
        input_size: int = MODEL_INPUT_SIZE,
    ) -> None:
        self.weights = weights if weights is not None else RtDetrTextDetectorWeights()
        self.conf_threshold = float(conf_threshold)
        self.bubble_conf_threshold = float(bubble_conf_threshold)
        self.input_size = int(input_size)
        self._session = None
        # El pipeline es productor/consumidor: puede pedir dos páginas a la vez y
        # compartirían la misma sesión.
        self._lock = threading.Lock()

    # -- carga -----------------------------------------------------------------

    def _load(self):
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "El detector RT-DETR necesita onnxruntime, que es un extra opcional. "
                "Instálalo con: pip install onnxruntime  (o 'pip install "
                "parallel-manga-translator[onnx]'). No se puede usar cv2.dnn: el "
                "importador ONNX de OpenCV falla en el nodo CumSum de este modelo."
            ) from exc

        ruta = self.weights.resolve()
        logger.info("Cargando detector RT-DETR: %s", ruta)
        # CPU a propósito: la GPU la reparte `gpu_slot` entre YOLO, OCR e inpainting, y
        # este modelo cuesta 0.2 s por página en CPU.
        self._session = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
        return self._session

    # -- inferencia ------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        # `do_normalize: false` en el preprocesador publicado: solo escala, sin media ni
        # desviación de ImageNet. Restarlas degrada la detección en silencio.
        blob = resized.astype(np.float32) / 255.0
        return np.transpose(blob, (2, 0, 1))[None]

    def predict(self, image: np.ndarray) -> RtDetrDetection:
        if image is None or image.size == 0:
            return RtDetrDetection()

        height, width = image.shape[:2]
        with self._lock:
            session = self._load()
            labels, boxes, scores = session.run(
                None,
                {
                    "images": self._preprocess(image),
                    # El modelo reescala internamente a la página: el orden es (w, h).
                    "orig_target_sizes": np.array([[width, height]], dtype=np.int64),
                },
            )

        text_boxes: List[RtDetrBox] = []
        bubble_boxes: List[RtDetrBox] = []
        # `strict` documenta lo que garantiza el modelo: las tres salidas traen una fila
        # por query. Si alguna vez dejan de cuadrar, mejor romper que desalinear cajas.
        for raw_label, raw_box, raw_score in zip(labels[0], boxes[0], scores[0], strict=True):
            label = ID2LABEL.get(int(raw_label))
            if label is None:
                continue
            score = float(raw_score)
            umbral = self.conf_threshold if label in TEXT_CLASSES else self.bubble_conf_threshold
            if score < umbral:
                continue
            box = self._to_box(raw_box, width, height)
            if box is None:
                continue
            destino = text_boxes if label in TEXT_CLASSES else bubble_boxes
            destino.append(RtDetrBox(bbox=box, score=score, label=label))

        logger.info(
            "RT-DETR: %s bloques de texto (%s en globo, %s fuera), %s globos",
            len(text_boxes),
            sum(1 for b in text_boxes if b.label == CLASS_TEXT_BUBBLE),
            sum(1 for b in text_boxes if b.label == CLASS_TEXT_FREE),
            len(bubble_boxes),
        )
        return RtDetrDetection(text_boxes=text_boxes, bubble_boxes=bubble_boxes)

    @staticmethod
    def _to_box(raw_box, width: int, height: int) -> Optional[Box]:
        """xyxy del modelo -> xywh recortado a la página."""
        x1, y1, x2, y2 = (float(v) for v in raw_box[:4])
        x = max(0, int(round(min(x1, x2))))
        y = max(0, int(round(min(y1, y2))))
        x_max = min(width, int(round(max(x1, x2))))
        y_max = min(height, int(round(max(y1, y2))))
        w, h = x_max - x, y_max - y
        if w <= 1 or h <= 1:
            return None
        return (x, y, w, h)
