import os
import unittest

os.environ.setdefault("PMT_BUBBLE_DETECTOR", "professional")
os.environ.setdefault("PMT_REQUIRE_PROFESSIONAL_BUBBLE", "1")

import cv2
import numpy as np

from Applications.BubbleDetector import BubbleDetector, BUBBLE_SPLIT_DEBUG_VERSION
from Applications.ProfessionalBubbleDetector import ProfessionalBubbleCandidate
from Applications.OnomatopoeiaManager import OnomatopoeiaManager
from Applications.TextRendering import TextRenderer
from Applications.ProcessingModels import TextRegion


class CoreQualityTests(unittest.TestCase):
    def test_bubble_detector_uses_pretrained_candidate_not_heuristic(self):
        img = np.zeros((240, 240, 3), dtype=np.uint8)
        detection = ([[95, 95], [145, 95], [145, 122], [95, 122]], "行くぞ", 0.91)
        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.ellipse(mask, (120, 110), (72, 48), 0, 0, 360, 255, -1)
        candidate = ProfessionalBubbleCandidate(
            bbox=(49, 63, 143, 95),
            mask=mask,
            confidence=0.88,
            label="speech_bubble",
            source="professional_yolo_seg",
        )

        detector = BubbleDetector("Japonés")
        detector.professional_detector.detect = lambda _img: [candidate]
        regions = detector.detect_regions(img, [detection])

        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertEqual(region.kind, "dialogue")
        self.assertEqual(region.metadata["detector"], "professional")
        self.assertGreater(cv2.countNonZero(region.mask), 50 * 27)
        self.assertGreater(region.bbox[2], region.text_bbox[2])
        self.assertGreater(region.bbox[3], region.text_bbox[3])

    def test_professional_candidate_can_be_associated_to_text_box(self):
        img = np.zeros((240, 240, 3), dtype=np.uint8)
        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.ellipse(mask, (120, 110), (72, 48), 0, 0, 360, 255, -1)
        candidate = ProfessionalBubbleCandidate(
            bbox=(49, 63, 143, 95),
            mask=mask,
            confidence=0.88,
            label="speech_bubble",
            source="professional_yolo_seg",
        )
        detector = BubbleDetector("Japonés")
        idx, matched = detector._match_professional_candidate(
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
            metadata={"detector": "professional", "region_flow": "bubble_first"},
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
            metadata={"detector": "professional", "region_flow": "bubble_first"},
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

    def test_merged_professional_bubble_is_split_by_separated_ocr_groups(self):
        img = np.full((260, 360, 3), 255, dtype=np.uint8)
        mask = np.zeros((260, 360), dtype=np.uint8)
        # Simula una mala detección profesional: una sola región rectangular cubre dos globos cercanos.
        cv2.rectangle(mask, (35, 60), (325, 165), 255, -1)
        merged_bubble = TextRegion(
            bbox=(35, 60, 290, 105),
            text_bbox=(35, 60, 290, 105),
            mask=mask,
            kind="dialogue",
            confidence=0.86,
            metadata={"detector": "professional", "region_flow": "bubble_first"},
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
            metadata={"detector": "professional", "region_flow": "bubble_first"},
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
        # Una detección profesional demasiado grande cubre dos globos. Cada globo tiene
        # dos columnas OCR cercanas que NO deben transformarse en cuatro globos.
        cv2.rectangle(mask, (30, 35), (520, 235), 255, -1)
        merged_bubble = TextRegion(
            bbox=(30, 35, 490, 200),
            text_bbox=(30, 35, 490, 200),
            mask=mask,
            kind="dialogue",
            confidence=0.86,
            metadata={"detector": "professional", "region_flow": "bubble_first"},
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
            metadata={"detector": "professional", "region_flow": "bubble_first"},
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

    def test_heuristic_bubble_detector_is_rejected(self):
        previous = os.environ.get("PMT_BUBBLE_DETECTOR")
        os.environ["PMT_BUBBLE_DETECTOR"] = "heuristic"
        try:
            with self.assertRaises(RuntimeError):
                BubbleDetector("Japonés")
        finally:
            if previous is None:
                os.environ.pop("PMT_BUBBLE_DETECTOR", None)
            else:
                os.environ["PMT_BUBBLE_DETECTOR"] = previous

    def test_ocr_manager_import_does_not_import_paddle(self):
        import sys
        sys.modules.pop("paddleocr", None)
        sys.modules.pop("paddle", None)
        from Applications.OcrManager import OcrManager
        _ = OcrManager("Japonés")
        self.assertNotIn("paddleocr", sys.modules)
        self.assertNotIn("paddle", sys.modules)

    def test_onomatopoeia_dictionary(self):
        manager = OnomatopoeiaManager()
        self.assertTrue(manager.is_onomatopoeia("ドン", "Japonés"))
        self.assertEqual(manager.translate("ドン", "Japonés", "Español"), "¡BUM!")

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
        self.assertEqual(BUBBLE_SPLIT_DEBUG_VERSION, "v5_cluster_bbox_logic_2026_06_10")
