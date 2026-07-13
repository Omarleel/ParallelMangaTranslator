import math
import unittest

import cv2
import numpy as np

from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.geometry.text_orientation import (
    estimate_text_rotation,
    polygon_text_angle,
    text_rotation_metadata,
)
from parallel_manga_translator.rendering.text_renderer import TextRenderer
from parallel_manga_translator.ui.manual_renderer import parse_manual_regions


def rotated_rectangle(cx: float, cy: float, width: float, height: float, angle: float):
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    points = []
    for x, y in (
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ):
        points.append([cx + x * cosine - y * sine, cy + x * sine + y * cosine])
    return points


class TextRotationDetectionTests(unittest.TestCase):
    def test_polygon_angle_is_stable_with_unordered_vertices(self):
        points = rotated_rectangle(100, 90, 120, 26, 23.5)
        shuffled = [points[2], points[0], points[3], points[1]]

        self.assertAlmostEqual(polygon_text_angle(shuffled), 23.5, delta=0.2)

    def test_steep_text_keeps_its_long_axis(self):
        points = rotated_rectangle(100, 90, 120, 26, 68.0)

        self.assertAlmostEqual(polygon_text_angle(points), 68.0, delta=0.2)

    def test_bubble_region_preserves_assigned_ocr_angle(self):
        image = np.full((180, 240, 3), 255, dtype=np.uint8)
        mask = np.zeros((180, 240), dtype=np.uint8)
        cv2.rectangle(mask, (30, 30), (210, 150), 255, -1)
        region = TextRegion(
            bbox=(30, 30, 180, 120),
            text_bbox=(30, 30, 180, 120),
            mask=mask,
            kind="dialogue",
            confidence=0.8,
        )
        detection = (rotated_rectangle(120, 90, 100, 24, -16.0), "texto", 0.92)

        regions = BubbleDetector("Japonés").build_regions_from_bubbles_and_text(image, [region], [detection])

        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0].metadata["text_rotation_angle"], -16.0, delta=0.2)
        self.assertEqual(regions[0].metadata["text_rotation_source"], "assigned_ocr_polygons")

    def test_group_estimation_rejects_a_small_outlier(self):
        detections = [
            (rotated_rectangle(70, 40, 110, 22, 12.0), "a", 0.95),
            (rotated_rectangle(72, 72, 96, 20, 14.0), "b", 0.90),
            (rotated_rectangle(20, 20, 24, 9, -52.0), "noise", 0.15),
        ]

        result = estimate_text_rotation(detections)

        self.assertAlmostEqual(result["angle"], 13.0, delta=1.0)
        self.assertEqual(result["sample_count"], 2)
        self.assertGreater(result["confidence"], 0.95)
        metadata = text_rotation_metadata(detections)
        self.assertEqual(metadata["text_rotation_samples"], 2)
        self.assertEqual(metadata["text_rotation_source"], "ocr_polygons")


class TextRotationPersistenceTests(unittest.TestCase):
    def test_manual_region_reads_rotation_from_saved_payload(self):
        regions = parse_manual_regions(
            [{"index": 0, "bbox": [10, 12, 90, 40], "text": "Hola", "rotation_angle": -17.5}],
            200,
            160,
        )

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].rotation_angle, -17.5)

    def test_renderer_layout_and_pixels_change_with_rotation(self):
        renderer = TextRenderer(absolute_min_font_size=7, inner_margin_ratio=0.03)
        image = np.full((150, 220, 3), 255, dtype=np.uint8)
        bbox = (35, 45, 150, 58)

        plain = renderer.render(image.copy(), [bbox], ["ANGLE"], font_sizes=[28], rotation_angles=[0])
        rotated = renderer.render(image.copy(), [bbox], ["ANGLE"], font_sizes=[28], rotation_angles=[19])
        layout = renderer.build_layout(bbox, "ANGLE", requested_font_size=28, rotation_angle=19)

        self.assertEqual(layout["version"], 2)
        self.assertEqual(layout["rotation_angle"], 19.0)
        self.assertFalse(np.array_equal(plain, rotated))
        changed = np.any(rotated != 255, axis=2)
        ys, xs = np.where(changed)
        self.assertTrue(len(xs) > 0)
        x, y, width, height = bbox
        self.assertGreaterEqual(xs.min(), x)
        self.assertLess(xs.max(), x + width)
        self.assertGreaterEqual(ys.min(), y)
        self.assertLess(ys.max(), y + height)


if __name__ == "__main__":
    unittest.main()
