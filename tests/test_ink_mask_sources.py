"""Tinta extra para la limpieza: unión con fallback, nunca sustitución.

Las dos reglas que se fijan aquí no son preferencias de diseño, salen de medir el banco:
las dos máscaras aciertan sitios distintos (IoU 0.27–0.42 con áreas casi iguales), así que
sustituir perdería lo que la derivada ve; y el detector no encuentra tinta en el 7 %, 31 %
y 1 % de las regiones de los tres casos, así que sin fallback esas dejarían de limpiarse.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from parallel_manga_translator.config.app_config import QualityConfig
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.processing.clean_mask_strategy import CleanMaskStrategy
from parallel_manga_translator.quality.ink_mask_sources import CtdInkMaskSource, create_ink_mask_source


class _FakeDetector:
    """Devuelve una máscara fija y cuenta cuántas veces se le pregunta."""

    def __init__(self, mask) -> None:
        self.mask = mask
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        from parallel_manga_translator.detection.comic_text_detector import ComicTextDetection

        return ComicTextDetection(text_mask=self.mask)


def _region(box=(10, 10, 40, 40)) -> TextRegion:
    x, y, w, h = box
    safe = np.zeros((100, 100), dtype=np.uint8)
    safe[y:y + h, x:x + w] = 255
    return TextRegion(bbox=box, text_bbox=box, mask=safe, kind="dialogue", metadata={})


def _derived(box=(12, 12, 8, 8)) -> np.ndarray:
    x, y, w, h = box
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    return mask


def test_the_extra_ink_is_added_never_substituted() -> None:
    derived = _derived()
    ctd = np.zeros((100, 100), dtype=np.uint8)
    ctd[30:40, 30:40] = 255  # otra zona dentro de la region

    source = CtdInkMaskSource(_FakeDetector(ctd))
    source.prepare(np.zeros((100, 100, 3), dtype=np.uint8))
    resultado, etiqueta = source.augment(_region(), derived)

    assert etiqueta == "union_ctd"
    assert resultado[14, 14] == 255, "La tinta derivada no puede perderse."
    assert resultado[35, 35] == 255, "La tinta del detector se suma."


def test_when_the_detector_sees_nothing_the_derived_mask_stands() -> None:
    """Es el 31 % de las regiones en en_01: el caso normal, no un error."""
    derived = _derived()
    source = CtdInkMaskSource(_FakeDetector(np.zeros((100, 100), dtype=np.uint8)))
    source.prepare(np.zeros((100, 100, 3), dtype=np.uint8))

    resultado, etiqueta = source.augment(_region(), derived)

    assert etiqueta == "fallback_derivada"
    assert np.array_equal(resultado, derived)


def test_ink_outside_the_region_does_not_leak_in() -> None:
    derived = _derived()
    ctd = np.zeros((100, 100), dtype=np.uint8)
    ctd[80:90, 80:90] = 255  # fuera de la zona segura de la region

    source = CtdInkMaskSource(_FakeDetector(ctd))
    source.prepare(np.zeros((100, 100, 3), dtype=np.uint8))
    resultado, etiqueta = source.augment(_region(), derived)

    assert etiqueta == "fallback_derivada"
    assert resultado[85, 85] == 0, "Otra region no puede heredar esta tinta."


def test_the_detector_runs_once_per_page_not_once_per_region() -> None:
    detector = _FakeDetector(np.full((100, 100), 255, dtype=np.uint8))
    source = CtdInkMaskSource(detector)
    strategy = CleanMaskStrategy(
        bubble_fill_whole_interior=False,
        bubble_fill_edge_margin=5,
        bubble_fill_text_dilate=2,
        bubble_fill_flat_max_rectangularity=0.86,
        ink_source=source,
    )
    imagen = np.full((100, 100, 3), 255, dtype=np.uint8)
    regiones = [_region((10, 10, 20, 20)), _region((40, 40, 20, 20)), _region((70, 10, 20, 20))]

    strategy.attach_clean_masks(imagen, regiones)

    assert detector.calls == 1, "Tres regiones no pueden costar tres inferencias."
    assert all(r.metadata.get("ink_mask_extra_source") for r in regiones)


def test_a_broken_detector_does_not_leave_the_page_uncleaned() -> None:
    class _Roto:
        def predict(self, image):
            raise RuntimeError("pesos corruptos")

    source = CtdInkMaskSource(_Roto())
    source.prepare(np.zeros((100, 100, 3), dtype=np.uint8))
    derived = _derived()

    resultado, etiqueta = source.augment(_region(), derived)

    assert etiqueta == ""
    assert np.array_equal(resultado, derived), "Sin detector se limpia con lo de siempre."


def test_the_strategy_without_a_source_behaves_exactly_as_before() -> None:
    imagen = np.full((100, 100, 3), 255, dtype=np.uint8)
    comun = {
        "bubble_fill_whole_interior": False,
        "bubble_fill_edge_margin": 5,
        "bubble_fill_text_dilate": 2,
        "bubble_fill_flat_max_rectangularity": 0.86,
    }
    sin_fuente = CleanMaskStrategy(**comun)
    region = _region()

    sin_fuente.attach_clean_masks(imagen, [region])

    assert "ink_mask_extra_source" not in region.metadata


def test_the_factory_defaults_to_the_derived_method_and_rejects_the_unknown() -> None:
    assert create_ink_mask_source(QualityConfig()) is None
    con_ctd = create_ink_mask_source(replace(QualityConfig(), ink_mask_source="derivada+ctd"), detector=object())
    assert isinstance(con_ctd, CtdInkMaskSource)
    with pytest.raises(ValueError):
        create_ink_mask_source(replace(QualityConfig(), ink_mask_source="solo_ctd"))


def test_both_yaml_files_default_to_the_derived_method() -> None:
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("config.yaml", "config.example.yaml"):
        data = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert data["quality"]["ink_mask_source"] == "derivada"
