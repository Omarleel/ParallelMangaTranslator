"""Vía comic-text-detector: decodificación, regiones y selección de fuente.

El modelo pesa 95 MB y no se carga aquí: lo que se prueba es la aritmética del letterbox
(donde un signo equivocado desplaza todas las cajas) y el contrato de la fuente de
regiones, con un detector doble inyectado.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from parallel_manga_translator.architecture.ports import RegionSourcePort
from parallel_manga_translator.config.app_config import ProcessingConfig, QualityConfig
from parallel_manga_translator.detection.comic_text_detector import (
    ComicTextDetection,
    ComicTextDetector,
    ComicTextDetectorWeights,
)
from parallel_manga_translator.detection.ctd_region_source import CtdRegionSource
from parallel_manga_translator.detection.region_source_factory import REGION_SOURCES, create_region_source


class _FakeDetector:
    """Detector doble: devuelve lo que se le diga, sin tocar el .onnx."""

    def __init__(self, detection: ComicTextDetection) -> None:
        self.detection = detection
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return self.detection


def _detection_with_two_blocks(width: int = 400, height: int = 600) -> ComicTextDetection:
    ink = np.zeros((height, width), dtype=np.uint8)
    ink[60:90, 40:120] = 255      # tinta dentro de la caja izquierda
    ink[300:340, 250:360] = 255   # tinta dentro de la caja derecha
    ink[500:520, 10:30] = 255     # tinta suelta fuera de toda caja
    return ComicTextDetection(
        boxes=[(30, 50, 100, 60), (240, 290, 130, 70)],
        scores=[0.91, 0.77],
        class_ids=[1, 0],
        text_mask=ink,
    )


def test_the_letterbox_is_undone_so_the_boxes_land_on_the_page() -> None:
    """Una caja centrada en la entrada del modelo debe volver al centro de la página."""
    detector = ComicTextDetector(input_size=1024)
    image = np.zeros((600, 400, 3), dtype=np.uint8)
    _, scale, pad = detector._preprocess(image)

    # Caja de 100x50 en el centro exacto del lienzo 1024x1024, en formato cx,cy,w,h.
    raw = np.zeros((1, 1, 7), dtype=np.float32)
    raw[0, 0] = [512.0, 512.0, 100.0, 50.0, 0.9, 0.0, 0.95]

    boxes, scores, class_ids = detector._decode_boxes(raw, scale, pad, 400, 600)

    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    centro_x, centro_y = x + w / 2, y + h / 2
    assert centro_x == pytest.approx(200, abs=2), "El centro horizontal debe caer en el centro de la página."
    assert centro_y == pytest.approx(300, abs=2)
    assert class_ids == [1]
    assert scores[0] == pytest.approx(0.9 * 0.95, abs=1e-3)


def test_the_mask_channel_is_the_documented_one_and_comes_back_page_sized() -> None:
    """`det[0]` es la máscara de texto; `det[1]` es ruido. Escoger mal la arruina."""
    detector = ComicTextDetector(input_size=64, mask_threshold=0.5)
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    _, scale, pad = detector._preprocess(image)

    raw = np.zeros((1, 2, 64, 64), dtype=np.float32)
    raw[0, 0, :, :] = 0.9   # canal de texto: activo
    raw[0, 1, :, :] = 0.0   # el otro canal: vacío

    mask = detector._decode_text_mask(raw, scale, pad, 64, 32)

    assert mask.shape == (32, 64), "La máscara vuelve a resolución de página."
    assert mask.max() == 255 and mask.min() == 255


def test_a_configured_model_that_does_not_exist_fails_with_its_path() -> None:
    weights = ComicTextDetectorWeights(model_path="no/existe/ctd.onnx")
    with pytest.raises(FileNotFoundError) as error:
        weights.resolve()
    assert "ctd.onnx" in str(error.value)


def test_regions_carry_the_ink_clipped_to_their_own_box() -> None:
    """La tinta acotada a la caja es lo que resuelve el recorte de OCR.

    Sin acotar, una región heredaría tinta vecina y el recorte se llenaría de arte.
    """
    source = CtdRegionSource("Inglés", detector=_FakeDetector(_detection_with_two_blocks()))
    image = np.zeros((600, 400, 3), dtype=np.uint8)

    regions = source.detect_primary_bubble_regions(image)

    assert len(regions) == 2
    for region in regions:
        x, y, w, h = region.bbox
        fuera = region.clean_mask.copy()
        fuera[y:y + h, x:x + w] = 0
        assert fuera.max() == 0, "La tinta de una región no puede salirse de su caja."
        assert region.mask[y + h // 2, x + w // 2] == 255, "La zona segura es la caja."
        assert region.text_mask is not None and region.text_boxes


def test_the_structural_class_maps_to_the_kind_the_pipeline_understands() -> None:
    source = CtdRegionSource("Inglés", detector=_FakeDetector(_detection_with_two_blocks()))
    regions = source.detect_primary_bubble_regions(np.zeros((600, 400, 3), dtype=np.uint8))

    dentro = next(r for r in regions if r.metadata["ctd_class_id"] == 1)
    fuera = next(r for r in regions if r.metadata["ctd_class_id"] == 0)
    assert dentro.kind == "dialogue" and dentro.metadata["structural_kind"] == "speech_bubble"
    assert fuera.kind == "free_text" and fuera.metadata["structural_kind"] == "out_of_bubble"


def test_japanese_pages_are_numbered_right_to_left() -> None:
    """El orden de lectura es la entrada del VLM: si los ids van mal, todo va mal."""
    izquierda = (40, 100, 120, 80)
    derecha = (700, 100, 120, 80)
    detection = ComicTextDetection(
        boxes=[izquierda, derecha],
        scores=[0.8, 0.8],
        class_ids=[1, 1],
        text_mask=np.zeros((1000, 900), dtype=np.uint8),
    )
    image = np.zeros((1000, 900, 3), dtype=np.uint8)

    japones = CtdRegionSource("Japonés", detector=_FakeDetector(detection))
    ordenadas = japones.build_regions_from_bubbles_and_text(
        image, japones.detect_primary_bubble_regions(image), []
    )
    assert [r.bbox for r in ordenadas] == [derecha, izquierda]
    assert [r.metadata["region_id"] for r in ordenadas] == [1, 2]
    assert ordenadas[0].metadata["reading_order_flow"] == "rtl_vertical"

    ingles = CtdRegionSource("Inglés", detector=_FakeDetector(detection))
    ordenadas_ltr = ingles.build_regions_from_bubbles_and_text(
        image, ingles.detect_primary_bubble_regions(image), []
    )
    assert [r.bbox for r in ordenadas_ltr] == [izquierda, derecha]


def test_as_the_sole_source_the_global_ocr_does_not_create_regions() -> None:
    """Emitir además texto libre del OCR reintroduciría la fuente de falsos positivos."""
    source = CtdRegionSource("Inglés", detector=_FakeDetector(_detection_with_two_blocks()))
    image = np.zeros((600, 400, 3), dtype=np.uint8)
    primarias = source.detect_primary_bubble_regions(image)

    con_ocr = source.build_regions_from_bubbles_and_text(image, primarias, [((10, 10, 50, 20), "texto")])

    assert len(con_ocr) == len(primarias) == 2


def test_the_source_honours_the_contract_the_orchestrator_consumes() -> None:
    source = CtdRegionSource("Japonés", detector=_FakeDetector(_detection_with_two_blocks()))
    assert isinstance(source, RegionSourcePort)
    # El contexto de depuración es parte del contrato: `CleanManga` lo llama por página.
    source.set_debug_page_context(3, source_filename="p.png", output_filename="0004.png")
    source.clear_debug_page_context()


def test_the_factory_picks_the_source_and_rejects_anything_else() -> None:
    quality = replace(QualityConfig(), region_source="comic_text_detector")
    source = create_region_source("Japonés", quality_config=quality, processing_config=ProcessingConfig())
    assert isinstance(source, CtdRegionSource)

    with pytest.raises(ValueError) as error:
        create_region_source("Japonés", quality_config=replace(QualityConfig(), region_source="inventado"))
    assert "inventado" in str(error.value)


def test_both_yaml_files_expose_the_new_keys() -> None:
    """La trampa conocida del proyecto: el ejemplo y el real se separan."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    for name in ("config.yaml", "config.example.yaml"):
        data = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        # Contra el registro, no contra una copia de la lista: al añadir una fuente, una
        # lista literal aquí se queda vieja y el test pasa a mentir.
        assert data["quality"]["region_source"] in REGION_SOURCES
        assert "comic_text_detector_conf" in data["quality"]
        assert "rtdetr_text_conf" in data["quality"], "El ejemplo y el real no deben divergir."
        assert data["vlm"]["enabled"] is False, "El VLM cuesta por página: apagado por defecto."
        # Sin modelo por defecto: ninguno del catálogo de la cuenta acepta imágenes, y
        # apuntar a uno que devuelve 404 rompería en la primera página.
        assert data["vlm"]["model"] == ""
