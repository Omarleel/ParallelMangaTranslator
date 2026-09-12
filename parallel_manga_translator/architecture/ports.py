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
class RegionSourcePort(Protocol):
    """Fuente de regiones vista por `CleanManga`.

    Es el contrato que el orquestador ya consumia de hecho, escrito. Existen dos
    implementaciones y ninguna es "la buena por defecto": `BubbleDetector` parte de
    globos segmentados por YOLO y recupera texto libre con el OCR global;
    `CtdRegionSource` parte de un detector de texto que ve las dos cosas.

    `detections` son las cajas del OCR global. Una fuente puede ignorarlas —el detector
    de texto ya cubre lo que ellas aportan— y por eso el parametro es entrada, no
    dependencia: quien implementa decide si las usa.
    """

    def detect_primary_bubble_regions(self, image: np.ndarray) -> list[TextRegion]:
        ...

    def build_regions_from_bubbles_and_text(
        self,
        image: np.ndarray,
        bubble_regions: Sequence[TextRegion],
        detections: Sequence[Any],
    ) -> list[TextRegion]:
        ...

    def set_debug_page_context(
        self,
        page_index: int,
        *,
        source_filename: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> None:
        ...

    def clear_debug_page_context(self) -> None:
        ...


@runtime_checkable
class InkMaskSourcePort(Protocol):
    """Aporte extra a la mascara de tinta que se borra.

    `prepare` corre una vez por pagina; `augment` se llama por region y devuelve
    (mascara, etiqueta). Devolver la mascara recibida sin tocar es una respuesta valida
    y es lo que ocurre cuando la fuente no ve tinta: el metodo derivado manda.
    """

    def prepare(self, image: np.ndarray) -> None:
        ...

    def augment(self, region: TextRegion, derived: np.ndarray) -> Tuple[np.ndarray, str]:
        ...


@runtime_checkable
class RegionSemanticsPort(Protocol):
    """Refinamiento semantico de regiones ya localizadas.

    Deliberadamente NO localiza: recibe regiones con su ID y devuelve, por ID, el tipo
    y opcionalmente una transcripcion corregida. Pedirle coordenadas a un modelo de
    lenguaje es la via rapida a las alucinaciones de cajas.
    """

    def refine(
        self,
        image: np.ndarray,
        regions: Sequence[TextRegion],
        transcriptions: Sequence[str],
    ) -> Mapping[int, Mapping[str, Any]]:
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
    """Contrato para motores de inpainting.

    Es el método público y síncrono que llama el pipeline, no la carga perezosa interna
    de cada motor. `BNInpainter` queda fuera a propósito: recibe la lista de detecciones
    en vez de una máscara, así que no es el mismo contrato.
    """

    def inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class PageCleanerPort(Protocol):
    """Etapa de limpieza vista por el orquestador de páginas.

    El contrato es exactamente lo que `ImageProcessor` necesita, ni más ni menos: limpiar
    una página y acotar el contexto de los artefactos de depuración. Todo lo demás que
    hoy expone `CleanManga` es detalle interno suyo.
    """

    def limpiar_manga(self, imagen: np.ndarray):
        ...

    def set_debug_page_context(
        self,
        page_index: int,
        *,
        source_filename: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> None:
        ...

    def clear_debug_page_context(self) -> None:
        ...

    def set_visual_inpaint_debug_context(self, output_root: str, page_index: int, filename: str) -> None:
        ...

    def clear_visual_inpaint_debug_context(self) -> None:
        ...


@runtime_checkable
class PageTranslatorPort(Protocol):
    """Etapa de OCR + traducción + rotulado vista por el orquestador de páginas."""

    #: Regiones que sobrevivieron a `extraer_regiones`, en orden de lectura. Los pasos
    #: siguientes las leen de aquí, así que es contrato entre etapas, no estado privado.
    ultimas_regiones: Sequence[TextRegion]

    def insertar_json_queue(self, indice_imagen: int, transcripcion_queue: Any, traduccion_queue: Any) -> None:
        ...

    def traducir_manga(
        self,
        imagen: np.ndarray,
        imagen_limpia: np.ndarray,
        mascara_capa: np.ndarray,
        text_regions: Optional[Sequence[TextRegion]] = None,
    ):
        ...

    # Los cuatro pasos que compone `traducir_manga`. Estan en el contrato porque
    # `processing.pipeline` los ejecuta por separado: una composicion parcial —solo-OCR—
    # llama a unos y no a otros, asi que exigir solo el metodo compuesto dejaria fuera
    # justo lo que el orquestador necesita para componer.

    def extraer_regiones(
        self,
        imagen: np.ndarray,
        mascara_capa: Optional[np.ndarray],
        text_regions: Optional[Sequence[TextRegion]] = None,
    ):
        ...

    def obtener_textos(self, imagenes_interes: Sequence[np.ndarray]) -> Sequence[str]:
        ...

    def traducir_textos_de_regiones(
        self, cuadros_delimitadores: Sequence[Any], textos: Sequence[str]
    ) -> Sequence[str]:
        ...

    def rotular(
        self,
        imagen_limpia: np.ndarray,
        cuadros_delimitadores: Sequence[Any],
        textos_para_render: Sequence[str],
    ):
        ...
