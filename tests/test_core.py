import os
import json
import logging
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np

from parallel_manga_translator.detection.bubble_detector import BubbleDetector, BUBBLE_SPLIT_DEBUG_VERSION
from parallel_manga_translator.detection.yolo_bubble_detector import YoloBubbleCandidate, YoloBubbleDetector
from parallel_manga_translator.language.onomatopoeia_manager import OnomatopoeiaManager
from parallel_manga_translator.rendering.text_renderer import TextRenderer
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.layout.reading_order_resolver import ReadingOrderResolver
from parallel_manga_translator.infrastructure.error_handling import PageFailureReport, StageProcessingError, processing_stage, write_failure_report
from parallel_manga_translator.config.app_config import QualityConfig




class Yolo11SegDetectorTests(unittest.TestCase):
    def test_yolo11_seg_is_primary_and_uses_high_resolution_masks(self):
        image = np.full((96, 128, 3), 255, dtype=np.uint8)
        mask_bubble = np.zeros((96, 128), dtype=np.float32)
        mask_bubble[20:62, 18:82] = 1.0
        mask_ignore = np.zeros((96, 128), dtype=np.float32)
        mask_ignore[2:94, 2:126] = 1.0

        class FakeYOLO:
            last = None

            def __init__(self, path):
                self.path = path
                self.calls = []
                FakeYOLO.last = self

            def predict(self, source, **kwargs):
                self.calls.append(kwargs)
                box_bubble = types.SimpleNamespace(
                    xyxy=[np.array([18, 20, 82, 62], dtype=np.float32)],
                    conf=[0.93],
                    cls=[0],
                )
                box_ignore = types.SimpleNamespace(
                    xyxy=[np.array([2, 2, 126, 94], dtype=np.float32)],
                    conf=[0.99],
                    cls=[1],
                )
                return [types.SimpleNamespace(
                    boxes=[box_bubble, box_ignore],
                    masks=types.SimpleNamespace(data=[mask_bubble, mask_ignore], xy=[]),
                    names={0: "speech_bubble", 1: "ignore_art"},
                )]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "manga-yolo11s-seg.pt")
            open(model_path, "wb").close()
            fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
            with unittest.mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                detector = YoloBubbleDetector(QualityConfig(
                    bubble_detector="yolo11-seg",
                    bubble_model_path=model_path,
                    bubble_device="gpu",
                    bubble_model_classes="0,1",
                    bubble_retina_masks=True,
                    bubble_confidence=0.40,
                    bubble_img_size=960,
                ))
                candidates = detector.detect(image)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.detector, "yolo11_seg")
        self.assertEqual(candidate.model_family, "YOLO11-seg")
        self.assertEqual(candidate.label, "speech_bubble")
        self.assertEqual(candidate.source, "yolo11_seg_mask")
        self.assertGreater(cv2.countNonZero(candidate.mask), 0)
        self.assertIsNotNone(FakeYOLO.last)
        self.assertEqual(FakeYOLO.last.path, model_path)
        call_kwargs = FakeYOLO.last.calls[0]
        self.assertTrue(call_kwargs["retina_masks"])
        self.assertEqual(call_kwargs["device"], "cuda:0")
        self.assertEqual(call_kwargs["classes"], [0, 1])
        self.assertEqual(call_kwargs["imgsz"], 960)

    def test_yolo11_seg_metadata_reaches_text_regions(self):
        img = np.full((140, 180, 3), 255, dtype=np.uint8)
        mask = np.zeros((140, 180), dtype=np.uint8)
        cv2.ellipse(mask, (82, 65), (52, 30), 0, 0, 360, 255, -1)
        candidate = YoloBubbleCandidate(
            bbox=(30, 35, 104, 60),
            mask=mask,
            confidence=0.91,
            label="speech_bubble",
            source="yolo11_seg_mask",
            detector="yolo11_seg",
            model_family="YOLO11-seg",
            class_id=0,
        )

        detector = BubbleDetector("Japonés", quality_config=QualityConfig(bubble_detector="yolo11-seg"))
        detector.yolo_detector.detect = lambda _img: [candidate]
        regions = detector.detect_primary_bubble_regions(img)

        self.assertEqual(len(regions), 1)
        metadata = regions[0].metadata
        self.assertEqual(metadata["detector"], "yolo11_seg")
        self.assertEqual(metadata["model_family"], "YOLO11-seg")
        self.assertEqual(metadata["mask_source"], "yolo11_seg_mask")
        self.assertEqual(metadata["label"], "speech_bubble")
        self.assertEqual(metadata["class_id"], 0)

class ReadingOrderResolverTests(unittest.TestCase):
    @staticmethod
    def _region(box, width=400, height=300):
        mask = np.zeros((height, width), dtype=np.uint8)
        x, y, w, h = box
        mask[y:y + h, x:x + w] = 255
        return TextRegion(bbox=box, text_bbox=box, mask=mask, kind="dialogue", confidence=0.8)

    def test_japanese_regions_read_right_to_left_then_down(self):
        resolver = ReadingOrderResolver("Japonés")
        left_top = self._region((40, 20, 50, 42))
        right_top = self._region((240, 22, 50, 42))
        bottom_right = self._region((230, 130, 50, 42))

        ordered = resolver.sort_regions([left_top, bottom_right, right_top])

        self.assertEqual([r.bbox for r in ordered], [right_top.bbox, left_top.bbox, bottom_right.bbox])

    def test_western_regions_read_left_to_right_then_down(self):
        resolver = ReadingOrderResolver("Español")
        left_top = self._region((40, 20, 50, 42))
        right_top = self._region((240, 22, 50, 42))
        bottom_left = self._region((35, 130, 50, 42))

        ordered = resolver.sort_regions([right_top, bottom_left, left_top])

        self.assertEqual([r.bbox for r in ordered], [left_top.bbox, right_top.bbox, bottom_left.bbox])

    def test_japanese_ocr_vertical_columns_read_top_to_bottom_right_to_left(self):
        resolver = ReadingOrderResolver("Japonés")
        right_top = ([[240, 20], [260, 20], [260, 70], [240, 70]], "右上", 0.92)
        right_bottom = ([[240, 82], [260, 82], [260, 132], [240, 132]], "右下", 0.91)
        left_top = ([[120, 20], [140, 20], [140, 70], [120, 70]], "左上", 0.90)
        left_bottom = ([[120, 82], [140, 82], [140, 132], [120, 132]], "左下", 0.89)

        ordered = resolver.sort_ocr_items([left_bottom, right_bottom, left_top, right_top])

        self.assertEqual([item[1] for item in ordered], ["右上", "右下", "左上", "左下"])

    def test_western_ocr_horizontal_rows_read_left_to_right(self):
        resolver = ReadingOrderResolver("Inglés")
        top_left = ([[20, 20], [80, 20], [80, 40], [20, 40]], "Hello", 0.95)
        top_right = ([[100, 20], [170, 20], [170, 40], [100, 40]], "world", 0.95)
        second_row = ([[20, 65], [125, 65], [125, 85], [20, 85]], "again", 0.95)

        ordered = resolver.sort_ocr_items([top_right, second_row, top_left])

        self.assertEqual([item[1] for item in ordered], ["Hello", "world", "again"])

    def test_bubble_detector_returns_reading_order_metadata(self):
        img = np.full((220, 360, 3), 255, dtype=np.uint8)
        right = self._region((235, 40, 70, 70), width=360, height=220)
        left = self._region((55, 42, 70, 70), width=360, height=220)

        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [left, right], [])

        self.assertEqual([r.bbox for r in regions], [right.bbox, left.bbox])
        self.assertEqual([r.metadata.get("reading_order_index") for r in regions], [0, 1])
        self.assertEqual(regions[0].metadata.get("reading_order_flow"), "rtl_vertical")


class ErrorHandlingTests(unittest.TestCase):
    def test_processing_stage_wraps_error_with_stage_context_and_report(self):
        logger = logging.getLogger("pmt_test_error_handling")
        logger.addHandler(logging.NullHandler())

        with self.assertRaises(StageProcessingError) as raised:
            with processing_stage("ocr_traduccion_render", logger=logger, page_index=2, filename="0003.png"):
                raise ValueError("boom OCR")

        error = raised.exception
        self.assertEqual(error.stage, "ocr_traduccion_render")
        self.assertEqual(error.original_error_type, "ValueError")
        self.assertIn("boom OCR", error.original_error_message)

        report = PageFailureReport.from_exception(error, page_index=2, filename="0003.png")
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = write_failure_report(tmpdir, report)
            data = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(data["page_index"], 2)
        self.assertEqual(data["filename"], "0003.png")
        self.assertEqual(data["stage"], "ocr_traduccion_render")
        self.assertEqual(data["error_type"], "ValueError")
        self.assertIn("boom OCR", data["error_message"])
        self.assertIn("ValueError", data["traceback"])


class CoreQualityTests(unittest.TestCase):
    def test_bubble_detector_uses_pretrained_candidate_not_heuristic(self):
        img = np.zeros((240, 240, 3), dtype=np.uint8)
        detection = ([[95, 95], [145, 95], [145, 122], [95, 122]], "行くぞ", 0.91)
        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.ellipse(mask, (120, 110), (72, 48), 0, 0, 360, 255, -1)
        candidate = YoloBubbleCandidate(
            bbox=(49, 63, 143, 95),
            mask=mask,
            confidence=0.88,
            label="speech_bubble",
            source="yolo11_seg_polygon",
        )

        detector = BubbleDetector("Japonés")
        detector.yolo_detector.detect = lambda _img: [candidate]
        regions = detector.detect_regions(img, [detection])

        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertEqual(region.kind, "dialogue")
        self.assertEqual(region.metadata["detector"], "yolo11_seg")
        self.assertGreater(cv2.countNonZero(region.mask), 50 * 27)
        self.assertGreater(region.bbox[2], region.text_bbox[2])
        self.assertGreater(region.bbox[3], region.text_bbox[3])

    def test_yolo_candidate_can_be_associated_to_text_box(self):
        img = np.zeros((240, 240, 3), dtype=np.uint8)
        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.ellipse(mask, (120, 110), (72, 48), 0, 0, 360, 255, -1)
        candidate = YoloBubbleCandidate(
            bbox=(49, 63, 143, 95),
            mask=mask,
            confidence=0.88,
            label="speech_bubble",
            source="yolo11_seg_polygon",
        )
        detector = BubbleDetector("Japonés")
        idx, matched = detector._match_yolo_candidate(
            (95, 95, 50, 27),
            [candidate],
            set(),
            allow_sfx=False,
        )
        self.assertEqual(idx, 0)
        self.assertIs(matched, candidate)


    def test_bubble_first_keeps_bubble_without_global_ocr(self):
        img = np.full((240, 240, 3), 255, dtype=np.uint8)
        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.ellipse(mask, (120, 110), (72, 48), 0, 0, 360, 255, -1)
        bubble = TextRegion(
            bbox=(49, 63, 143, 95),
            text_bbox=(49, 63, 143, 95),
            mask=mask,
            kind="dialogue",
            confidence=0.9,
            metadata={"detector": "yolo11_seg", "region_flow": "bubble_first"},
        )
        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [bubble], [])
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].bbox, bubble.bbox)
        self.assertEqual(regions[0].detections_count, 0)

    def test_bubble_first_splits_free_sfx_from_bubble(self):
        img = np.full((260, 260, 3), 255, dtype=np.uint8)
        mask = np.zeros((260, 260), dtype=np.uint8)
        cv2.ellipse(mask, (90, 90), (55, 38), 0, 0, 360, 255, -1)
        bubble = TextRegion(
            bbox=(35, 52, 110, 76),
            text_bbox=(35, 52, 110, 76),
            mask=mask,
            kind="dialogue",
            confidence=0.9,
            metadata={"detector": "yolo11_seg", "region_flow": "bubble_first"},
        )
        inside = ([[70, 75], [110, 75], [110, 95], [70, 95]], "行くぞ", 0.88)
        outside_sfx = ([[180, 170], [225, 170], [225, 198], [180, 198]], "ドン", 0.85)
        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [bubble], [inside, outside_sfx])
        self.assertEqual(len(regions), 2)
        dialogue = next(r for r in regions if r.kind == "dialogue")
        sfx = next(r for r in regions if r.kind == "sfx")
        self.assertEqual(dialogue.detections_count, 1)
        self.assertIn("行くぞ", dialogue.source_text_hint)
        self.assertIn("ドン", sfx.source_text_hint)

    def test_merged_yolo_bubble_is_split_by_separated_ocr_groups(self):
        img = np.full((260, 360, 3), 255, dtype=np.uint8)
        mask = np.zeros((260, 360), dtype=np.uint8)
        # Simula una mala detección YOLO: una sola región rectangular cubre dos globos cercanos.
        cv2.rectangle(mask, (35, 60), (325, 165), 255, -1)
        merged_bubble = TextRegion(
            bbox=(35, 60, 290, 105),
            text_bbox=(35, 60, 290, 105),
            mask=mask,
            kind="dialogue",
            confidence=0.86,
            metadata={"detector": "yolo11_seg", "region_flow": "bubble_first"},
        )
        left_text = ([[65, 92], [115, 92], [115, 120], [65, 120]], "行くぞ", 0.91)
        right_text = ([[235, 92], [285, 92], [285, 120], [235, 120]], "待て", 0.89)

        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [merged_bubble], [left_text, right_text])
        dialogue_regions = [r for r in regions if r.kind == "dialogue"]

        self.assertEqual(len(dialogue_regions), 2)
        self.assertTrue(all(r.metadata.get("split_from_merged_bubble") for r in dialogue_regions))
        self.assertTrue(any("行くぞ" in r.source_text_hint for r in dialogue_regions))
        self.assertTrue(any("待て" in r.source_text_hint for r in dialogue_regions))

    def test_vertical_cjk_columns_do_not_merge_by_default(self):
        detector = BubbleDetector("Japonés")
        left_column = ([[80, 50], [118, 50], [118, 170], [80, 170]], "そりゃ", 0.90)
        right_column = ([[135, 52], [173, 52], [173, 168], [135, 168]], "俺は", 0.91)

        groups = detector._group_detections([left_column, right_column])

        self.assertEqual(len(groups), 2)

    def test_merged_bubble_with_adjacent_vertical_text_is_split(self):
        img = np.full((260, 360, 3), 255, dtype=np.uint8)
        mask = np.zeros((260, 360), dtype=np.uint8)
        # Simula una región rectangular demasiado amplia que cubre dos bloques verticales cercanos.
        cv2.rectangle(mask, (45, 40), (245, 205), 255, -1)
        merged_bubble = TextRegion(
            bbox=(45, 40, 200, 165),
            text_bbox=(45, 40, 200, 165),
            mask=mask,
            kind="dialogue",
            confidence=0.86,
            metadata={"detector": "yolo11_seg", "region_flow": "bubble_first"},
        )
        first_text = ([[80, 58], [118, 58], [118, 178], [80, 178]], "そりゃ", 0.90)
        second_text = ([[158, 58], [196, 58], [196, 178], [158, 178]], "俺は", 0.91)

        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [merged_bubble], [first_text, second_text])
        dialogue_regions = [r for r in regions if r.kind == "dialogue"]

        self.assertEqual(len(dialogue_regions), 2)
        self.assertTrue(all(r.metadata.get("split_from_merged_bubble") for r in dialogue_regions))

    def test_split_clusters_adjacent_columns_inside_same_bubble(self):
        img = np.full((280, 560, 3), 255, dtype=np.uint8)
        mask = np.zeros((280, 560), dtype=np.uint8)
        # Una detección YOLO demasiado grande cubre dos globos. Cada globo tiene
        # dos columnas OCR cercanas que NO deben transformarse en cuatro globos.
        cv2.rectangle(mask, (30, 35), (520, 235), 255, -1)
        merged_bubble = TextRegion(
            bbox=(30, 35, 490, 200),
            text_bbox=(30, 35, 490, 200),
            mask=mask,
            kind="dialogue",
            confidence=0.86,
            metadata={"detector": "yolo11_seg", "region_flow": "bubble_first"},
        )
        left_a = ([[65, 62], [155, 62], [155, 220], [65, 220]], "そりゃ", 0.91)
        left_b = ([[150, 64], [195, 64], [195, 205], [150, 205]], "そうだろ", 0.89)
        right_a = ([[320, 60], [405, 60], [405, 215], [320, 215]], "俺は", 0.88)
        right_b = ([[400, 62], [445, 62], [445, 220], [400, 220]], "イライラ", 0.87)

        detector = BubbleDetector("Japonés")
        regions, debug_records = detector._split_merged_bubble_regions(
            img,
            [merged_bubble],
            {0: [left_a, left_b, right_a, right_b]},
        )

        self.assertEqual(len(regions), 2)
        self.assertTrue(all(r.metadata.get("split_from_merged_bubble") for r in regions))
        self.assertTrue(any(r.metadata.get("split_cluster_group_indices") == [0, 1] for r in regions))
        self.assertTrue(any(r.metadata.get("split_cluster_group_indices") == [2, 3] for r in regions))
        self.assertEqual(len(debug_records[0].get("split_clusters", [])), 2)

    def test_merge_debug_records_show_raw_ocr_and_pair_decisions(self):
        img = np.full((180, 260, 3), 255, dtype=np.uint8)
        mask = np.zeros((180, 260), dtype=np.uint8)
        cv2.rectangle(mask, (20, 30), (220, 150), 255, -1)
        merged_bubble = TextRegion(
            bbox=(20, 30, 200, 120),
            text_bbox=(20, 30, 200, 120),
            mask=mask,
            kind="dialogue",
            confidence=0.86,
            metadata={"detector": "yolo11_seg", "region_flow": "bubble_first"},
        )
        a = ([[55, 55], [88, 55], [88, 130], [55, 130]], "A", 0.90)
        b = ([[145, 55], [178, 55], [178, 130], [145, 130]], "B", 0.91)
        detector = BubbleDetector("Japonés")
        _regions, debug_records = detector._split_merged_bubble_regions(img, [merged_bubble], {0: [a, b]})

        self.assertEqual(len(debug_records), 1)
        record = debug_records[0]
        self.assertEqual(len(record["raw_ocr_detections"]), 2)
        self.assertGreaterEqual(len(record["raw_ocr_pair_decisions"]), 1)
        self.assertIn("ocr_group_merge_trace", record)



    def test_free_text_long_horizontal_line_is_kept(self):
        # Páginas tipo notas/afterword suelen tener líneas largas fuera de globos.
        # La versión anterior descartaba siempre bw > 50% del ancho de página.
        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        long_line = ([[80, 120], [880, 120], [880, 170], [80, 170]], "こちらはあとがきです", 0.84)

        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [], [long_line])

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].kind, "free_text")
        self.assertIn("あとがき", regions[0].source_text_hint)
        self.assertEqual(regions[0].metadata.get("free_text_filter_reason"), "aceptado")

    def test_huge_free_text_artifact_without_text_signal_is_rejected(self):
        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        huge_artifact = ([[0, 0], [990, 0], [990, 780], [0, 780]], "---", 0.40)

        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [], [huge_artifact])

        self.assertEqual(regions, [])

    def test_single_digit_free_text_artifact_is_rejected(self):
        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        button_or_eye = ([[250, 250], [310, 250], [310, 325], [250, 325]], "7", 0.92)

        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [], [button_or_eye])

        self.assertEqual(regions, [])

    def test_large_noisy_free_text_artifact_with_weak_ocr_signal_is_rejected(self):
        # Simula falsos positivos como una trama de ropa: la caja es grande, el OCR
        # devuelve mezcla de símbolos/dígitos y solo uno o dos caracteres CJK.
        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        noisy_clothes = ([[250, 280], [480, 280], [480, 720], [250, 720]], "( { 忍 ・》 ・ 4 さ", 0.22)

        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [], [noisy_clothes])

        self.assertEqual(regions, [])

    def test_vertical_free_text_taller_than_half_page_is_rejected(self):
        # Falso positivo real: letras cercanas se fusionan en una columna gigante.
        # Aunque el OCR devuelva caracteres japoneses con confianza alta, una caja
        # de texto libre vertical de más de media página no debe limpiarse.
        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        merged_column = ([[420, 120], [470, 120], [470, 760], [420, 760]], "き れ い だ ね", 0.93)

        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [], [merged_column])

        self.assertEqual(regions, [])

    def test_free_text_gap_recovery_adds_missed_vertical_column(self):
        # EasyOCR puede detectar las columnas laterales de texto libre y saltarse
        # una columna central con outline/trama. El fallback debe crear una
        # región extra solo para texto libre; así la limpieza y el OCR por recorte
        # tienen una segunda oportunidad.
        img = np.full((520, 360, 3), 255, dtype=np.uint8)
        left = ([[35, 95], [85, 95], [85, 325], [35, 325]], "なってる", 0.92)
        right = ([[265, 100], [315, 100], [315, 335], [265, 335]], "俺いつの間にか", 0.91)

        # Columna central omitida por el OCR global: caracteres negros separados.
        for cy in [112, 158, 204, 250, 296]:
            cv2.rectangle(img, (160, cy), (199, cy + 31), (0, 0, 0), -1)

        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [], [left, right])
        recovered = [r for r in regions if r.metadata.get("detector") == "visual_free_text_gap"]

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].kind, "free_text")
        self.assertEqual(recovered[0].detections_count, 0)
        self.assertEqual(recovered[0].metadata.get("free_text_filter_reason"), "visual_gap_recovery")
        self.assertTrue(130 <= recovered[0].text_bbox[0] <= 170)

    def test_free_text_gap_recovery_ignores_thin_panel_line(self):
        img = np.full((520, 360, 3), 255, dtype=np.uint8)
        left = ([[35, 95], [85, 95], [85, 325], [35, 325]], "なってる", 0.92)
        right = ([[265, 100], [315, 100], [315, 335], [265, 335]], "俺いつの間にか", 0.91)
        cv2.line(img, (178, 70), (178, 370), (0, 0, 0), 2)

        detector = BubbleDetector("Japonés")
        regions = detector.build_regions_from_bubbles_and_text(img, [], [left, right])

        self.assertFalse(any(r.metadata.get("detector") == "visual_free_text_gap" for r in regions))


    def test_renderer_uses_mask_inner_area_for_dialogue_fit(self):
        # Simula un globo ovalado: la bbox rectangular es más ancha que la zona real
        # disponible cerca de las curvas. El renderer debe ajustar el texto usando un
        # rectángulo interior seguro antes de aplicar la máscara final.
        img = np.full((180, 280, 3), 255, dtype=np.uint8)
        mask = np.zeros((120, 200), dtype=np.uint8)
        cv2.ellipse(mask, (100, 60), (95, 55), 0, 0, 360, 255, -1)

        renderer = TextRenderer(absolute_min_font_size=7, inner_margin_ratio=0.03)
        safe_x, safe_y, safe_w, safe_h = renderer._safe_text_area_from_mask(mask, 200, 120, "dialogo")

        self.assertGreater(safe_x, 0)
        self.assertGreater(safe_y, 0)
        self.assertLess(safe_w, 200)
        self.assertLess(safe_h, 120)

        out = renderer.render(
            img,
            [(40, 30, 200, 120)],
            ["ESTE TEXTO LARGO NO DEBE SER COMIDO POR LA MASCARA DEL GLOBO"],
            text_styles=["dialogo"],
            clip_masks=[mask],
        )
        crop = out[30:150, 40:240]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ink = gray < 245
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1) > 0

        self.assertEqual(int(np.logical_and(ink, ~eroded).sum()), 0)

    def test_renderer_splits_connected_bubble_lobes(self):
        # Dos globos unidos por un cuello pueden llegar como una sola máscara.
        # El renderer debe repartir el texto en ambos lóbulos, no usar sólo el
        # rectángulo interior máximo de uno de ellos.
        img = np.full((180, 300, 3), 255, dtype=np.uint8)
        mask = np.zeros((120, 240), dtype=np.uint8)
        cv2.ellipse(mask, (60, 60), (48, 45), 0, 0, 360, 255, -1)
        cv2.ellipse(mask, (180, 60), (48, 45), 0, 0, 360, 255, -1)
        cv2.rectangle(mask, (108, 55), (132, 65), 255, -1)

        renderer = TextRenderer(absolute_min_font_size=7, inner_margin_ratio=0.03)
        slots = renderer._connected_lobe_slots_from_mask(mask, 240, 120, "dialogo")
        self.assertGreaterEqual(len(slots), 2)

        out = renderer.render(
            img,
            [(30, 30, 240, 120)],
            ["Primera frase para el globo izquierdo. Segunda frase para el globo derecho."],
            text_styles=["dialogo"],
            clip_masks=[mask],
        )
        crop = out[30:150, 30:270]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ink = gray < 245
        left_ink = int(np.logical_and(ink[:, :120], mask[:, :120] > 0).sum())
        right_ink = int(np.logical_and(ink[:, 120:], mask[:, 120:] > 0).sum())

        self.assertGreater(left_ink, 10)
        self.assertGreater(right_ink, 10)

    def test_renderer_splits_diagonal_connected_bubble_lobes(self):
        # Caso parecido a globos unidos vertical/diagonalmente: la erosión simple
        # puede no romper la unión, pero el mapa de distancia sí debe detectar
        # dos centros de globo y repartir el texto.
        img = np.full((620, 540, 3), 255, dtype=np.uint8)
        mask = np.zeros((518, 439), dtype=np.uint8)
        cv2.ellipse(mask, (290, 125), (150, 150), 0, 0, 360, 255, -1)
        cv2.ellipse(mask, (145, 310), (125, 150), 0, 0, 360, 255, -1)
        cv2.rectangle(mask, (200, 180), (270, 270), 255, -1)

        renderer = TextRenderer(absolute_min_font_size=7, inner_margin_ratio=0.03)
        slots = renderer._connected_lobe_slots_from_mask(
            mask,
            439,
            518,
            "dialogo",
            right_to_left=True,
        )
        self.assertGreaterEqual(len(slots), 2)

        out = renderer.render(
            img,
            [(50, 50, 439, 518)],
            ["¿Hay algún truco sobre cómo se siente después? Vi el trabajo anterior y no entiendo la escena."],
            text_styles=["dialogo"],
            clip_masks=[mask],
            reading_order_right_to_left=True,
        )
        crop = out[50:568, 50:489]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ink = gray < 245
        upper_right = int(np.logical_and(ink[:245, 190:], mask[:245, 190:] > 0).sum())
        lower_left = int(np.logical_and(ink[245:, :240], mask[245:, :240] > 0).sum())

        self.assertGreater(upper_right, 10)
        self.assertGreater(lower_left, 10)

    def test_heuristic_bubble_detector_is_rejected(self):
        with self.assertRaises(RuntimeError):
            BubbleDetector("Japonés", quality_config=QualityConfig(bubble_detector="heuristic"))

    def test_ocr_manager_import_does_not_import_paddle(self):
        import sys
        sys.modules.pop("paddleocr", None)
        sys.modules.pop("paddle", None)
        from parallel_manga_translator.ocr.ocr_manager import OcrManager
        _ = OcrManager("Japonés")
        self.assertNotIn("paddleocr", sys.modules)
        self.assertNotIn("paddle", sys.modules)

    def test_onomatopoeia_dictionary(self):
        manager = OnomatopoeiaManager()
        self.assertTrue(manager.is_onomatopoeia("ドン", "Japonés"))
        self.assertEqual(manager.translate("ドン", "Japonés", "Español"), "¡BUM!")

    def test_japanese_dialogue_with_wave_dash_is_not_sfx(self):
        manager = OnomatopoeiaManager()
        self.assertFalse(manager.is_onomatopoeia("...そうですね〜〜", "Japonés"))
        self.assertEqual(manager.render_style("...そうですね〜〜", "Japonés"), "dialogo")
        self.assertFalse(manager.is_onomatopoeia("えー", "Japonés"))

    def test_onomatopoeia_flow_separates_dictionary_similarity_and_heuristic(self):
        manager = OnomatopoeiaManager()
        self.assertTrue(manager.is_onomatopoeia("ドン", "Japonés"))
        self.assertEqual(manager.semantic_key("ドン", "Japonés"), "impact")

        # ドバ no está en el diccionario; sólo debe ser una pista débil de candidato.
        self.assertFalse(manager.is_onomatopoeia("ドバ", "Japonés"))
        self.assertIsNone(manager.semantic_key("ドバ", "Japonés"))
        self.assertFalse(manager.is_free_text_onomatopoeia("ドバ", "Japonés"))
        self.assertEqual(manager.heuristic_semantic_key("ドバ", "Japonés"), "impact")
        self.assertTrue(manager.is_onomatopoeia_candidate("ドバ", "Japonés"))

    def test_renderer_accepts_clip_masks(self):
        img = np.full((120, 200, 3), 255, dtype=np.uint8)
        mask = np.zeros((80, 160), dtype=np.uint8)
        cv2.ellipse(mask, (80, 40), (75, 35), 0, 0, 360, 255, -1)
        renderer = TextRenderer(absolute_min_font_size=7)
        out = renderer.render(
            img,
            [(20, 20, 160, 80)],
            ["Este texto debe quedar dentro del globo sin salirse"],
            text_styles=["dialogo"],
            clip_masks=[mask],
        )
        self.assertEqual(out.shape, img.shape)


if __name__ == "__main__":
    unittest.main()


class TestBubbleSplitDebugVersion(unittest.TestCase):
    def test_debug_version_marker_exists(self):
        self.assertEqual(BUBBLE_SPLIT_DEBUG_VERSION, "v7_bubble_onomatopoeia_translation_2026_06_11")
