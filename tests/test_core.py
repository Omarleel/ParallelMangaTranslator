import os
import unittest

os.environ.setdefault("PMT_BUBBLE_DETECTOR", "professional")
os.environ.setdefault("PMT_REQUIRE_PROFESSIONAL_BUBBLE", "1")

import cv2
import numpy as np

from Applications.BubbleDetector import BubbleDetector
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
