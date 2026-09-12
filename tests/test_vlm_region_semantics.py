"""Refinamiento semántico con VLM: qué se le manda, qué se acepta y qué se ignora.

Nada aquí toca la red. El cliente entra inyectado, que es justo lo que permite fijar por
contrato las dos defensas que importan: que un id inventado no entre en el pipeline y
que el VLM no pise una transcripción que el OCR sí produjo.
"""

from __future__ import annotations

import json

import numpy as np

from parallel_manga_translator.architecture.ports import RegionSemanticsPort
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.vision.annotated_page import PageAnnotator, encode_data_url
from parallel_manga_translator.vision.groq_vision_client import NullVisionClient, build_vision_client
from parallel_manga_translator.vision.vlm_region_semantics import (
    VlmRegionSemantics,
    apply_semantics,
)


class _FakeVisionClient:
    """Doble del cliente: guarda lo que se le manda y devuelve una respuesta fija."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.images: list = []
        self.prompt = ""
        self.calls = 0

    def analyze(self, images, prompt):
        self.calls += 1
        self.images = list(images)
        self.prompt = prompt
        return self.response


def _region(region_id: int, box, kind: str = "dialogue") -> TextRegion:
    x, y, w, h = box
    mask = np.zeros((400, 300), dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    return TextRegion(
        bbox=box,
        text_bbox=box,
        mask=mask,
        kind=kind,
        metadata={"region_id": region_id, "structural_kind": "speech_bubble" if kind == "dialogue" else "out_of_bubble"},
    )


def _page() -> np.ndarray:
    return np.full((400, 300, 3), 240, dtype=np.uint8)


def test_without_a_client_nothing_is_sent_and_nothing_changes() -> None:
    """El default no puede costar dinero ni cambiar la clasificación del detector."""
    semantics = VlmRegionSemantics()
    assert semantics.enabled is False
    assert semantics.refine(_page(), [_region(1, (10, 10, 80, 40))], ["hola"]) == {}
    assert isinstance(build_vision_client(enabled=False, api_key="k", model="m"), NullVisionClient)


def test_the_model_sees_the_numbered_page_and_only_the_hard_crops() -> None:
    client = _FakeVisionClient(json.dumps({"regions": [{"id": 1, "type": "dialogue", "text": "a"}]}))
    semantics = VlmRegionSemantics(client, max_crops=4)
    regiones = [
        _region(1, (10, 10, 80, 40)),                      # globo con texto -> sin recorte
        _region(2, (120, 10, 60, 40)),                     # globo sin texto -> recorte
        _region(3, (10, 200, 90, 50), kind="sfx"),         # fuera de globo   -> recorte
    ]

    semantics.refine(_page(), regiones, ["tiene texto", "", "onomatopeya"])

    assert client.calls == 1
    # Primera imagen: la página anotada. Después, un recorte por bloque difícil.
    assert len(client.images) == 3, "Página + 2 recortes (el globo ya transcrito no viaja)."
    assert "2" in client.prompt and "3" in client.prompt
    assert "No des coordenadas" in client.prompt


def test_hallucinated_ids_and_unknown_types_are_dropped() -> None:
    """Un id que no se dibujó no existe: aceptarlo metería basura en el pipeline."""
    client = _FakeVisionClient(json.dumps({"regions": [
        {"id": 1, "type": "narration"},
        {"id": 99, "type": "dialogue"},      # nunca se dibujó
        {"id": 2, "type": "monólogo"},       # tipo fuera del vocabulario
    ]}))
    semantics = VlmRegionSemantics(client)

    resultado = semantics.refine(_page(), [_region(1, (10, 10, 80, 40)), _region(2, (120, 10, 60, 40))], ["a", "b"])

    assert set(resultado) == {1}
    assert resultado[1]["region_kind"] == "narration"


def test_a_json_wrapped_in_a_code_fence_is_still_read() -> None:
    client = _FakeVisionClient("```json\n{\"regions\": [{\"id\": 1, \"type\": \"sfx\"}]}\n```")
    semantics = VlmRegionSemantics(client)
    resultado = semantics.refine(_page(), [_region(1, (10, 10, 80, 40))], [""])
    assert resultado[1]["region_kind"] == "sfx"


def test_a_failing_vlm_never_takes_the_page_down() -> None:
    class _Roto:
        def analyze(self, images, prompt):
            raise RuntimeError("429 rate limit")

    semantics = VlmRegionSemantics(_Roto())
    assert semantics.refine(_page(), [_region(1, (10, 10, 80, 40))], ["a"]) == {}


def test_thought_lands_as_dialogue_but_keeps_its_nuance_in_the_metadata() -> None:
    regiones = [_region(1, (10, 10, 80, 40))]
    textos = ["texto"]

    cambiadas = apply_semantics(regiones, textos, {1: {"kind": "thought", "region_kind": "dialogue"}})

    assert cambiadas == 1
    # El pipeline solo entiende dialogue/narration/sfx/free_text; el matiz se conserva.
    assert regiones[0].kind == "dialogue"
    assert regiones[0].metadata["vlm_kind"] == "thought"
    assert regiones[0].metadata["kind_before_vlm"] == "dialogue"


def test_the_vlm_only_fills_gaps_it_never_overwrites_the_ocr() -> None:
    """MangaOCR mide CER 0.0015 en el banco: no se pisa con lo que lea un generalista."""
    regiones = [_region(1, (10, 10, 80, 40)), _region(2, (120, 10, 60, 40))]
    textos = ["transcripción buena", ""]
    semantics = {
        1: {"kind": "dialogue", "region_kind": "dialogue", "text": "lo que el VLM creyó leer"},
        2: {"kind": "sfx", "region_kind": "sfx", "text": "DOOON"},
    }

    apply_semantics(regiones, textos, semantics, refine_transcription=True)

    assert textos[0] == "transcripción buena", "Una transcripción existente es intocable."
    assert textos[1] == "DOOON"
    assert regiones[1].metadata.get("vlm_filled_transcription") is True


def test_refinement_off_ignores_the_text_entirely() -> None:
    regiones = [_region(1, (10, 10, 80, 40))]
    textos = [""]
    apply_semantics(regiones, textos, {1: {"kind": "sfx", "region_kind": "sfx", "text": "ZAS"}})
    assert textos == [""]


def test_the_annotator_numbers_every_region_and_shrinks_the_page() -> None:
    annotator = PageAnnotator(max_side=200)
    regiones = [_region(1, (10, 10, 80, 40)), _region(7, (120, 200, 60, 40))]

    page = annotator.build(np.full((400, 300, 3), 255, dtype=np.uint8), regiones, crop_ids=[7])

    assert page.region_ids == [1, 7]
    assert max(page.image.shape[:2]) == 200, "Subir la página entera multiplica coste sin leer mejor."
    assert [region_id for region_id, _ in page.crops] == [7]
    # Se dibujó algo: la página era blanca y ahora tiene tinta de cajas y números.
    assert page.image.min() < 255
    assert encode_data_url(page.image).startswith("data:image/jpeg;base64,")


def test_the_service_satisfies_the_port() -> None:
    assert isinstance(VlmRegionSemantics(), RegionSemanticsPort)
