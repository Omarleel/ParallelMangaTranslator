from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from parallel_manga_translator.models.processing_models import TextRegion

Box = Tuple[int, int, int, int]


@runtime_checkable
class RegionDetectorPort(Protocol):
    """Contrato para detectores de globos/texto. Permite sustituir YOLO u otro motor."""

    def detect_regions(self, image: np.ndarray, detections: Sequence[Any]) -> list[TextRegion]:
        ...




@runtime_checkable
class TextDetectionPort(Protocol):
    """Contrato para OCR de localización: devuelve bounding boxes de texto.

    Es intencionalmente distinto de `OcrEnginePort`, que transcribe texto desde
    una región ya recortada.
    """

    @property
    def engine_id(self) -> str:
        ...

    def detect_text_boxes(self, image: np.ndarray) -> list[Any]:
        ...


@runtime_checkable
class OcrEnginePort(Protocol):
    """Contrato para un motor OCR que opera sobre una región individual."""

    @property
    def engine_id(self) -> str:
        ...

    def extract_text(self, image: np.ndarray) -> str:
        ...


@runtime_checkable
class OcrPort(Protocol):
    """Contrato batch usado por el pipeline de procesamiento."""

    def extract_texts(self, images: Sequence[np.ndarray]) -> list[str]:
        ...


@runtime_checkable
class TranslatorPort(Protocol):
    """Contrato mínimo para traductores tradicionales o LLM."""

    def traducir_textos(
        self,
        textos_actuales: Sequence[str],
        contexto_previo: Optional[Sequence[Sequence[str]]] = None,
        items_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        character_memory: Optional[Mapping[str, Any]] = None,
    ) -> list[str]:
        ...


@runtime_checkable
class TextRendererPort(Protocol):
    """Contrato para renderizadores de texto sobre imagen."""

    def render(
        self,
        imagen_limpia: np.ndarray,
        cuadros_delimitadores: Sequence[Box],
        textos: Sequence[str],
        text_styles: Optional[Sequence[str]] = None,
        clip_masks: Optional[Sequence[np.ndarray]] = None,
        *,
        reading_order_right_to_left: bool = False,
    ) -> np.ndarray:
        ...


@runtime_checkable
class InpainterPort(Protocol):
    """Contrato para motores de inpainting."""

    async def _load(self) -> None:
        ...

    async def _inpaint(self, imagen: np.ndarray, mascara_capa: np.ndarray) -> np.ndarray:
        ...
