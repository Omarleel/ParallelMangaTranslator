"""RT-DETR en el rol de localizador de texto, junto al detector de globos YOLO.

El reparto que se prueba aquí es el que resultó correcto tras medirlo sobre material real:

    quality.region_source: yolo    -> el polígono del globo, del que depende la limpieza
    ocr.detection_engine: rtdetr   -> dónde está el texto
    ocr.transcription_engine: ...  -> qué pone

Lo importante no es que el adaptador devuelva cajas: es que **no transcribe**, y las reglas
de texto libre estaban escritas contando con esa transcripción.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from parallel_manga_translator.config.app_config import OcrConfig, ProcessingConfig, QualityConfig
from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.detection.rtdetr_text_detector import RtDetrBox, RtDetrDetection
from parallel_manga_translator.ocr.text_detection.base import TextDetectionSettings
from parallel_manga_translator.ocr.text_detection.factory import TextDetectionFactory
from parallel_manga_translator.ocr.text_detection.rtdetr_detector import RtDetrTextDetectionEngine


class _FakeDetector:
    def __init__(self, detection: RtDetrDetection) -> None:
        self.detection = detection

    def predict(self, image):
        return self.detection


def _engine(detection: RtDetrDetection) -> RtDetrTextDetectionEngine:
    settings = TextDetectionSettings.from_config("Japonés", OcrConfig())
    engine = RtDetrTextDetectionEngine(settings)
    engine._detector = _FakeDetector(detection)
    return engine


# ---------------------------------------------------------------------------------
# El adaptador
# ---------------------------------------------------------------------------------

def test_it_is_registered_as_a_localisation_engine() -> None:
    assert "rtdetr" in TextDetectionFactory.supported_engines()


def test_the_boxes_come_out_in_the_ocr_contract() -> None:
    """(polígono, texto, confianza), con el polígono en cuatro esquinas."""
    detection = RtDetrDetection(
        text_boxes=[RtDetrBox(bbox=(10, 20, 30, 40), score=0.77, label="text_free")]
    )

    salida = _engine(detection).detect_text_boxes(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(salida) == 1
    poligono, texto, confianza = salida[0]
    assert poligono == [[10, 20], [40, 20], [40, 60], [10, 60]]
    assert texto == "", "Este motor no transcribe: inventar una pista engaña a los filtros."
    assert confianza == 0.77


def test_the_bubble_class_is_not_a_text_box() -> None:
    """El globo lo pone YOLO. Emitirlo aquí sería una caja de texto del tamaño del globo."""
    detection = RtDetrDetection(
        text_boxes=[RtDetrBox(bbox=(10, 20, 30, 40), score=0.9, label="text_bubble")],
        bubble_boxes=[RtDetrBox(bbox=(0, 0, 90, 90), score=0.9, label="bubble")],
    )

    salida = _engine(detection).detect_text_boxes(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(salida) == 1
    assert salida[0][0] == [[10, 20], [40, 20], [40, 60], [10, 60]]


def test_a_detector_failure_does_not_take_down_the_page() -> None:
    """Sin cajas de texto, los globos de YOLO se siguen limpiando con su tinta interior."""
    class _Roto:
        def predict(self, image):
            raise RuntimeError("pesos corruptos")

    settings = TextDetectionSettings.from_config("Japonés", OcrConfig())
    engine = RtDetrTextDetectionEngine(settings)
    engine._detector = _Roto()

    assert engine.detect_text_boxes(np.zeros((100, 100, 3), dtype=np.uint8)) == []


# ---------------------------------------------------------------------------------
# La regla que casi se lleva por delante todo el texto libre
# ---------------------------------------------------------------------------------

def _detector() -> BubbleDetector:
    return BubbleDetector(
        idioma_entrada="Japonés",
        quality_config=replace(QualityConfig(), require_yolo=False),
        processing_config=ProcessingConfig(),
    )


def _bubble_region():
    from parallel_manga_translator.models.processing_models import TextRegion
    return TextRegion(
        bbox=(90, 90, 330, 250),
        text_bbox=(90, 90, 330, 250),
        mask=None,
        kind="dialogue",
        confidence=0.9,
        source_text_hint="",
        detections_count=0,
        metadata={},
    )


def test_an_empty_hint_does_not_reject_the_region_when_the_localiser_cannot_read() -> None:
    """Sin esto, RT-DETR como localizador perdería TODAS las regiones de texto libre.

    Con la pista vacía `meaningful_without_digits` vale 0, así que `only_digits_or_symbols`
    es cierto y el filtro de "sólo números o símbolos" rechazaba cada grupo.
    """
    detector = _detector()

    keep, motivo = detector._should_keep_free_text_group(
        (100, 100, 60, 120), "", 0.62, (1200, 900, 3), False, localizer_reads_text=False
    )

    assert keep, f"Rechazada por {motivo!r} pese a que el localizador no transcribe."


def test_the_same_empty_hint_is_still_rejected_when_the_localiser_does_read() -> None:
    """La vía EasyOCR/Paddle no cambia: ahí una caja sin texto sí es ruido."""
    detector = _detector()

    keep, motivo = detector._should_keep_free_text_group(
        (100, 100, 60, 120), "", 0.62, (1200, 900, 3), False, localizer_reads_text=True
    )

    assert not keep
    assert motivo == "solo_numeros_o_simbolos_ocr_ruido"


def test_a_low_score_is_still_rejected_without_a_hint() -> None:
    """Que no lea no significa que valga todo: la confianza del detector sigue filtrando."""
    detector = _detector()

    keep, _motivo = detector._should_keep_free_text_group(
        (100, 100, 60, 120), "", 0.01, (1200, 900, 3), False, localizer_reads_text=False
    )

    assert not keep


# ---------------------------------------------------------------------------------
# Rotación: un rectángulo recto no lleva ángulo
# ---------------------------------------------------------------------------------

def test_axis_aligned_boxes_report_no_rotation_instead_of_zero_degrees() -> None:
    """Cero con confianza cero es "no lo sé"; cero con confianza es una medida falsa.

    El estimador deduce la inclinación de la FORMA del polígono. RT-DETR entrega
    rectángulos rectos, así que estimar de todos modos giraba el texto rotulado.
    """
    from parallel_manga_translator.geometry.text_orientation import estimate_text_rotation

    rectos = [([[10, 10], [60, 10], [60, 30], [10, 30]], "", 0.9)]

    resultado = estimate_text_rotation(rectos)

    assert resultado["angle"] == 0.0
    assert resultado["confidence"] == 0.0
    assert resultado["axis_aligned"] is True
    assert resultado["polygons"], "Los polígonos se conservan: la disposición CJK los usa."


def test_an_inclined_polygon_still_reports_its_angle() -> None:
    """La vía EasyOCR/Paddle no cambia."""
    from parallel_manga_translator.geometry.text_orientation import estimate_text_rotation

    inclinado = [([[10, 20], [60, 10], [62, 30], [12, 40]], "hola", 0.9)]

    resultado = estimate_text_rotation(inclinado)

    assert resultado["axis_aligned"] is False
    assert abs(resultado["angle"]) > 1.0
    assert resultado["confidence"] > 0.0


# ---------------------------------------------------------------------------------
# Partir globos fusionados con un localizador de bloques
# ---------------------------------------------------------------------------------

def test_two_stacked_blocks_split_even_though_the_line_rule_would_refuse() -> None:
    """El hueco relativo está calibrado para líneas y no traslada a bloques.

    Dos bloques apilados de 100 px de alto tendrían que separarse 35 px con la regla
    relativa; dos líneas de 20 px sólo piden 7. Con bloques manda el mínimo absoluto.
    """
    detector = _detector()
    region = _bubble_region()
    # Dos bloques anchos y altos, separados 20 px: por encima del mínimo absoluto (12)
    # pero muy por debajo del 0.35 x alto que pediría la regla relativa.
    grupos = [
        [([[100, 100], [400, 100], [400, 200], [100, 200]], "", 0.9)],
        [([[100, 220], [400, 220], [400, 320], [100, 320]], "", 0.9)],
    ]

    parte, _dec, motivo = detector._should_split_region_from_groups(region, grupos, True)
    assert parte, f"No partió: {motivo}"
    assert motivo == "division_por_bloques_de_texto"

    no_parte, _dec2, motivo2 = detector._should_split_region_from_groups(region, grupos, False)
    assert not no_parte, "Con fragmentos de línea la regla relativa sigue mandando."
    assert motivo2 == "grupos_demasiado_cercanos"


def test_blocks_that_actually_touch_are_not_split() -> None:
    """Que el localizador entregue bloques no convierte cualquier par en dos globos."""
    detector = _detector()
    region = _bubble_region()
    grupos = [
        [([[100, 100], [400, 100], [400, 200], [100, 200]], "", 0.9)],
        [([[100, 203], [400, 203], [400, 300], [100, 300]], "", 0.9)],
    ]

    parte, _dec, motivo = detector._should_split_region_from_groups(region, grupos, True)

    assert not parte
    assert motivo == "bloques_demasiado_cercanos"


def test_two_blocks_placed_diagonally_are_two_blocks() -> None:
    """El caso que se escapaba, con la geometría real de la página que lo destapó.

    Un bloque arriba-derecha y otro abajo-izquierda no cumplen ninguna de las tres reglas
    de siempre: se rozan en horizontal (así que no hay separación horizontal ni diagonal) y
    apenas comparten columna (así que tampoco vertical). Pero sus cajas **no se cortan**, y
    con un localizador de bloques eso basta.
    """
    detector = _detector()
    region = _bubble_region()
    grupos = [
        [([[33, 311], [316, 311], [316, 828], [33, 828]], "", 0.9)],
        [([[314, 76], [466, 76], [466, 285], [314, 285]], "", 0.9)],
    ]

    parte, decisiones, motivo = detector._should_split_region_from_groups(region, grupos, True)

    assert parte, f"No partió: {motivo}"
    assert any(d.get("reason") == "bloques_sin_solape" for d in decisiones)


def test_overlapping_blocks_are_still_one_block() -> None:
    """El criterio es que NO se corten: dos cajas encajadas siguen siendo un bloque."""
    detector = _detector()
    region = _bubble_region()
    grupos = [
        [([[100, 100], [400, 100], [400, 400], [100, 400]], "", 0.9)],
        [([[150, 150], [380, 150], [380, 380], [150, 380]], "", 0.9)],
    ]

    parte, _dec, motivo = detector._should_split_region_from_groups(region, grupos, True)

    assert not parte
    assert motivo == "bloques_demasiado_cercanos"
