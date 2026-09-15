"""Vía RT-DETR: decodificación de la salida y construcción de regiones.

El modelo pesa 168 MB y no se carga aquí. Lo que se prueba es lo que puede romperse en
silencio: que la clase `bubble` no cree regiones (duplicaría cada globo), que la zona
segura sea el polígono de la tinta y no la caja —un rectángulo llena el recorte de OCR de
arte vecino, medido en su día como CER 0.0015 -> 0.2584— y que faltar `onnxruntime` dé un
mensaje accionable en vez de un ImportError pelado.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from parallel_manga_translator.architecture.ports import RegionSourcePort
from parallel_manga_translator.config.app_config import ProcessingConfig, QualityConfig
from parallel_manga_translator.detection.region_source_factory import REGION_SOURCES, create_region_source
from parallel_manga_translator.detection.rtdetr_region_source import RtDetrRegionSource
from parallel_manga_translator.detection.rtdetr_text_detector import (
    RtDetrBox,
    RtDetrDetection,
    RtDetrTextDetector,
)


class _FakeSession:
    """Sesión ONNX doble: devuelve las tres salidas del modelo tal cual."""

    def __init__(self, labels, boxes, scores) -> None:
        self.labels = np.asarray([labels], dtype=np.int64)
        self.boxes = np.asarray([boxes], dtype=np.float32)
        self.scores = np.asarray([scores], dtype=np.float32)
        self.last_inputs = None

    def run(self, _outputs, inputs):
        self.last_inputs = inputs
        return self.labels, self.boxes, self.scores


class _FakeDetector:
    def __init__(self, detection: RtDetrDetection) -> None:
        self.detection = detection
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return self.detection


def _page_with_text() -> np.ndarray:
    """Página blanca con trazos oscuros en la mitad izquierda de una caja."""
    image = np.full((300, 400, 3), 255, dtype=np.uint8)
    for x in range(60, 110, 12):
        cv2.line(image, (x, 110), (x, 150), (20, 20, 20), 3)
    return image


# ---------------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------------

def test_the_model_output_becomes_boxes_on_the_page() -> None:
    """xyxy del modelo -> xywh recortado, y la página entra como (ancho, alto)."""
    detector = RtDetrTextDetector(conf_threshold=0.30, bubble_conf_threshold=0.50)
    detector._session = _FakeSession(
        labels=[1, 2, 0],
        boxes=[[10.0, 20.0, 60.0, 80.0], [100.0, 40.0, 150.0, 90.0], [5.0, 5.0, 200.0, 200.0]],
        scores=[0.90, 0.40, 0.80],
    )
    image = np.zeros((300, 400, 3), dtype=np.uint8)

    detection = detector.predict(image)

    assert [b.bbox for b in detection.text_boxes] == [(10, 20, 50, 60), (100, 40, 50, 50)]
    assert [b.label for b in detection.text_boxes] == ["text_bubble", "text_free"]
    assert [b.bbox for b in detection.bubble_boxes] == [(5, 5, 195, 195)]
    ancho, alto = detector._session.last_inputs["orig_target_sizes"][0]
    assert (int(ancho), int(alto)) == (400, 300), "El modelo espera (w, h), no (h, w)."


def test_text_and_bubble_use_their_own_thresholds() -> None:
    """La clase `bubble` solo mide solape, así que puede exigirse más confianza."""
    detector = RtDetrTextDetector(conf_threshold=0.30, bubble_conf_threshold=0.50)
    detector._session = _FakeSession(
        labels=[2, 0],
        boxes=[[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 120.0, 120.0]],
        scores=[0.35, 0.35],
    )

    detection = detector.predict(np.zeros((300, 400, 3), dtype=np.uint8))

    assert len(detection.text_boxes) == 1, "0.35 supera el umbral de texto."
    assert detection.bubble_boxes == [], "0.35 no supera el umbral de globo."


def test_a_degenerate_box_is_dropped_instead_of_emitted() -> None:
    detector = RtDetrTextDetector(conf_threshold=0.10)
    detector._session = _FakeSession(
        labels=[2], boxes=[[10.0, 10.0, 10.4, 40.0]], scores=[0.99]
    )

    assert detector.predict(np.zeros((300, 400, 3), dtype=np.uint8)).text_boxes == []


def test_a_missing_onnxruntime_explains_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin esto el usuario ve un ImportError y no sabe que es un extra opcional."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("no module named onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as error:
        RtDetrTextDetector()._load()

    mensaje = str(error.value)
    assert "onnxruntime" in mensaje
    assert "cv2.dnn" in mensaje, "Debe decir por qué no vale el camino del otro detector."


# ---------------------------------------------------------------------------------
# Fuente de regiones
# ---------------------------------------------------------------------------------

def _source_with(detection: RtDetrDetection, **quality) -> RtDetrRegionSource:
    return RtDetrRegionSource(
        idioma_entrada="Japonés",
        quality_config=replace(QualityConfig(), **quality),
        processing_config=ProcessingConfig(),
        detector=_FakeDetector(detection),
    )


def test_the_bubble_class_does_not_create_regions() -> None:
    """Emitirla duplicaría cada globo: la región es el texto, no el contorno."""
    detection = RtDetrDetection(
        text_boxes=[RtDetrBox(bbox=(50, 100, 70, 60), score=0.9, label="text_bubble")],
        bubble_boxes=[RtDetrBox(bbox=(40, 90, 100, 90), score=0.8, label="bubble")],
    )

    regions = _source_with(detection).detect_primary_bubble_regions(_page_with_text())

    assert len(regions) == 1
    assert regions[0].kind == "dialogue"


def test_the_class_decides_the_kind() -> None:
    detection = RtDetrDetection(
        text_boxes=[
            RtDetrBox(bbox=(50, 100, 70, 60), score=0.9, label="text_bubble"),
            RtDetrBox(bbox=(200, 100, 70, 60), score=0.9, label="text_free"),
        ]
    )

    regions = _source_with(detection).detect_primary_bubble_regions(_page_with_text())

    assert [r.kind for r in regions] == ["dialogue", "free_text"]
    assert [r.metadata["structural_kind"] for r in regions] == ["speech_bubble", "out_of_bubble"]


def test_the_safe_zone_is_the_ink_polygon_not_the_box() -> None:
    """Es lo que evita que el recorte de OCR se llene de arte vecino."""
    detection = RtDetrDetection(
        text_boxes=[RtDetrBox(bbox=(40, 90, 120, 90), score=0.9, label="text_free")]
    )

    region = _source_with(detection).detect_primary_bubble_regions(_page_with_text())[0]

    assert region.metadata["region_mask_shape"] == "ink_polygon"
    pintados = cv2.countNonZero(region.mask)
    assert 0 < pintados < 120 * 90, "El polígono debe estar contenido en la caja, no llenarla."


def test_the_box_is_the_fallback_when_there_is_no_ink() -> None:
    """Sin tinta derivable se emite la caja, pero queda registrado cuál es cuál."""
    detection = RtDetrDetection(
        text_boxes=[RtDetrBox(bbox=(300, 200, 60, 50), score=0.9, label="text_free")]
    )
    blanco = np.full((300, 400, 3), 255, dtype=np.uint8)

    region = _source_with(detection).detect_primary_bubble_regions(blanco)[0]

    assert region.metadata["region_mask_shape"] == "detector_box"
    # `cv2.rectangle` pinta ambos extremos, así que son (w+1)*(h+1). Es la misma
    # convención que usa `CtdRegionSource`: importa más que las dos vías coincidan.
    assert cv2.countNonZero(region.mask) == 61 * 51


def _page_with_balloon() -> np.ndarray:
    """Globo blanco con borde negro y texto dentro, sobre fondo con trama."""
    image = np.full((300, 400, 3), 120, dtype=np.uint8)
    cv2.ellipse(image, (200, 150), (110, 90), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(image, (200, 150), (110, 90), 0, 0, 360, (0, 0, 0), 3)
    for x in range(160, 245, 20):
        cv2.line(image, (x, 110), (x, 190), (10, 10, 10), 4)
    return image


def test_the_ink_is_searched_in_the_balloon_not_in_the_short_text_box() -> None:
    """La caja de texto de este modelo cubre ~0.66 del bloque: buscar solo ahí deja texto.

    El globo que el propio modelo predice sí lo cubre, y su interior excluye el contorno.
    """
    image = _page_with_balloon()
    # Caja de texto deliberadamente corta: solo la mitad izquierda del texto.
    detection = RtDetrDetection(
        text_boxes=[RtDetrBox(bbox=(158, 108, 45, 84), score=0.9, label="text_bubble")],
        bubble_boxes=[RtDetrBox(bbox=(90, 60, 220, 180), score=0.9, label="bubble")],
    )

    region = _source_with(detection).detect_primary_bubble_regions(image)[0]

    assert region.metadata["search_zone"] == "bubble_interior"
    # La zona segura se sale de la caja de texto (por eso alcanza el texto entero)...
    assert cv2.countNonZero(region.mask) > 45 * 84
    # ...pero no toca el borde negro del globo: el pixel del contorno queda fuera.
    assert region.mask[150, 91] == 0, "El contorno del globo no puede entrar en la zona."


def test_when_the_interior_cannot_be_derived_the_box_grows_inside_the_bubble() -> None:
    """Segundo intento antes de rendirse a la caja corta, que es la que deja texto.

    No hace falta conocer la forma del globo para saber dos cosas medidas: que la caja de
    texto se queda corta y que dentro del globo hay sitio seguro.
    """
    source = _source_with(RtDetrDetection())
    texto = (100, 100, 40, 80)
    globo = (60, 60, 160, 160)

    ampliada = source._expanded_inside_bubble(texto, globo, (300, 400, 3))

    assert ampliada is not None
    x, y, w, h = ampliada
    assert w > texto[2] and h > texto[3], "Tiene que ganar area sobre la caja original."
    # Y nunca rozar el contorno: queda dentro del globo encogido.
    margen_x = int(round(globo[2] * source.BUBBLE_SHRINK))
    assert x >= globo[0] + margen_x
    assert x + w <= globo[0] + globo[2] - margen_x


def test_a_bubble_that_gives_no_room_does_not_force_a_worse_box() -> None:
    """Si ensanchar no gana nada, no se cambia de via por cambiar."""
    source = _source_with(RtDetrDetection())
    # Globo practicamente del tamano del texto: al encogerlo no queda margen.
    assert source._expanded_inside_bubble((100, 100, 40, 80), (100, 100, 40, 80), (300, 400, 3)) is None


def test_without_a_containing_bubble_the_search_stays_in_the_box() -> None:
    """Fuera de globo no hay nada que ensanchar sin arriesgar arte."""
    image = _page_with_balloon()
    detection = RtDetrDetection(
        text_boxes=[RtDetrBox(bbox=(158, 108, 45, 84), score=0.9, label="text_free")],
        bubble_boxes=[],
    )

    region = _source_with(detection).detect_primary_bubble_regions(image)[0]

    assert region.metadata["search_zone"] == "detector_box"


def test_the_bubble_overlap_is_recorded_for_the_future_spatial_rule() -> None:
    detection = RtDetrDetection(
        text_boxes=[
            RtDetrBox(bbox=(50, 100, 100, 100), score=0.9, label="text_free"),
            RtDetrBox(bbox=(300, 10, 50, 50), score=0.9, label="text_free"),
        ],
        bubble_boxes=[RtDetrBox(bbox=(50, 100, 50, 100), score=0.8, label="bubble")],
    )

    regions = _source_with(detection).detect_primary_bubble_regions(_page_with_text())

    assert regions[0].metadata["bubble_overlap"] == pytest.approx(0.5)
    assert regions[1].metadata["bubble_overlap"] == 0.0


def test_the_global_ocr_does_not_create_regions_in_this_path() -> None:
    """El objetivo de esta vía es exactamente ese: el OCR deja de proponer cajas."""
    detection = RtDetrDetection(
        text_boxes=[RtDetrBox(bbox=(50, 100, 70, 60), score=0.9, label="text_free")]
    )
    source = _source_with(detection)
    image = _page_with_text()

    primarias = source.detect_primary_bubble_regions(image)
    regiones = source.build_regions_from_bubbles_and_text(image, primarias, ["caja_ocr", "otra"])

    assert len(regiones) == 1
    assert regiones[0].metadata["region_id"] == 1


# ---------------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------------

def test_the_source_is_registered_and_honours_the_port() -> None:
    assert "rtdetr" in REGION_SOURCES
    # Los cuatro métodos que consume el orquestador, explícitos: `__protocol_attrs__` no
    # existe en 3.10 y `isinstance` contra un Protocol no estructural tampoco sirve.
    for metodo in (
        "detect_primary_bubble_regions",
        "build_regions_from_bubbles_and_text",
        "set_debug_page_context",
        "clear_debug_page_context",
    ):
        assert hasattr(RegionSourcePort, metodo), f"{metodo} ya no está en el puerto."
        assert hasattr(RtDetrRegionSource, metodo), f"Falta {metodo} del contrato."


def test_the_factory_builds_it_without_loading_the_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elegir la fuente no debe descargar 168 MB: la carga es perezosa."""
    from parallel_manga_translator.detection import rtdetr_text_detector

    def explota(self):  # pragma: no cover - debe no llamarse
        raise AssertionError("No se deben resolver los pesos al construir la fuente.")

    monkeypatch.setattr(rtdetr_text_detector.RtDetrTextDetectorWeights, "resolve", explota)

    source = create_region_source(
        "Japonés",
        quality_config=replace(QualityConfig(), region_source="rtdetr"),
        processing_config=ProcessingConfig(),
    )

    assert isinstance(source, RtDetrRegionSource)
