from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from parallel_manga_translator.models.page_context import PageContext
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
    una página. Todo lo demás que hoy expone `CleanManga` es detalle interno suyo.

    Tenía además cuatro métodos para acotar el contexto de los artefactos de depuración
    (`set_debug_page_context`, `set_visual_inpaint_debug_context` y sus `clear_`): existían
    solo para empujar estado de la página a un objeto de vida larga antes de cada llamada.
    Ahora eso viaja en el `PageContext`, que es donde vive lo de la página.
    """

    def limpiar_manga(self, ctx: PageContext) -> None:
        ...


@runtime_checkable
class PageTranslatorPort(Protocol):
    """Etapa de OCR + traducción + rotulado vista por el orquestador de páginas."""

    def insertar_json_queue(self, transcripcion_queue: Any, traduccion_queue: Any) -> None:
        """Las colas de salida del trabajo. El índice de página no viene por aquí: es
        estado de la página y viaja en `PageContext.indice_pagina`."""
        ...

    def traducir_manga(
        self,
        imagen: np.ndarray,
        imagen_limpia: np.ndarray,
        mascara_capa: np.ndarray,
        text_regions: Optional[Sequence[TextRegion]] = None,
        indice_pagina: int = 0,
    ):
        ...

    # Los cuatro pasos que compone `traducir_manga`. Estan en el contrato porque
    # `processing.pipeline` los ejecuta por separado: una composicion parcial —solo-OCR—
    # llama a unos y no a otros, asi que exigir solo el metodo compuesto dejaria fuera
    # justo lo que el orquestador necesita para componer.
    #
    # Todos reciben el contexto de la pagina y escriben en el. Antes se pasaban el
    # resultado por atributos del traductor (`ultimas_regiones` y companyia), que este
    # puerto tuvo que declarar como parte del contrato: un objeto de vida larga haciendo
    # de cuaderno de notas de la pagina en curso.

    def extraer_regiones(self, ctx: PageContext) -> None:
        ...

    def obtener_textos(self, ctx: PageContext) -> None:
        ...

    def traducir_textos_de_regiones(self, ctx: PageContext) -> None:
        ...

    def publicar_transcripcion(self, ctx: PageContext) -> None:
        """Escribe la transcripción sin traducir. La usa el modo «limpiar y transcribir»."""
        ...

    def rotular(self, ctx: PageContext) -> None:
        ...
