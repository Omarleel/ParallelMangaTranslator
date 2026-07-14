import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from parallel_manga_translator.detection.bubble_detector import BubbleDetector
from parallel_manga_translator.models.processing_models import TextRegion
from parallel_manga_translator.geometry.text_orientation import (
    effective_text_rotation_angle,
    estimate_text_rotation,
    is_vertical_cjk_layout,
    polygon_text_angle,
    text_rotation_metadata,
)
from parallel_manga_translator.rendering.text_renderer import TextRenderer
from parallel_manga_translator.ui.manual_renderer import (
    ManualRegion,
    parse_manual_regions,
    render_manual_page,
    render_manual_region_preview,
)


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

    def test_vertical_japanese_column_does_not_rotate_translation(self):
        image = np.full((240, 180, 3), 255, dtype=np.uint8)
        mask = np.zeros((240, 180), dtype=np.uint8)
        cv2.rectangle(mask, (30, 20), (150, 220), 255, -1)
        region = TextRegion(
            bbox=(30, 20, 120, 200),
            text_bbox=(30, 20, 120, 200),
            mask=mask,
            kind="dialogue",
            confidence=0.8,
        )
        detection = (rotated_rectangle(90, 120, 150, 28, 88.0), "縦書きです", 0.96)

        regions = BubbleDetector("Japonés").build_regions_from_bubbles_and_text(image, [region], [detection])

        metadata = regions[0].metadata
        self.assertAlmostEqual(metadata["text_rotation_detected_angle"], 88.0, delta=0.2)
        self.assertEqual(metadata["text_rotation_angle"], 0.0)
        self.assertTrue(metadata["text_rotation_suppressed"])
        self.assertEqual(metadata["source_text_layout"], "vertical_cjk")

    def test_vertical_cjk_detection_does_not_affect_non_cjk_text(self):
        detections = [(rotated_rectangle(90, 120, 150, 28, 82.0), "VERTICAL", 0.95)]

        metadata = text_rotation_metadata(detections, source_language="Inglés")

        self.assertFalse(is_vertical_cjk_layout(detections, source_language="Inglés"))
        self.assertAlmostEqual(metadata["text_rotation_angle"], 82.0, delta=0.2)
        self.assertFalse(metadata["text_rotation_suppressed"])

    def test_legacy_vertical_layout_hint_forces_effective_zero_angle(self):
        metadata = {
            "text_rotation_angle": 87.0,
            "layout_hint": "vertical_cjk",
        }

        self.assertEqual(effective_text_rotation_angle(metadata, source_language="Japonés"), 0.0)

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

    def test_single_line_with_rotation_is_written_vertically_without_rotating(self):
        renderer = TextRenderer(absolute_min_font_size=7, inner_margin_ratio=0.03)
        image = np.full((190, 220, 3), 255, dtype=np.uint8)
        bbox = (70, 20, 80, 150)

        plain = renderer.render(image.copy(), [bbox], ["ANGLE"], font_sizes=[28], rotation_angles=[0])
        vertical = renderer.render(image.copy(), [bbox], ["ANGLE"], font_sizes=[28], rotation_angles=[19])
        layout = renderer.build_layout(bbox, "ANGLE", requested_font_size=28, rotation_angle=19)

        self.assertEqual(layout["version"], 2)
        self.assertEqual(layout["requested_rotation_angle"], 19.0)
        self.assertEqual(layout["rotation_angle"], 0.0)
        self.assertEqual(layout["writing_mode"], "vertical_chars")
        self.assertEqual(layout["blocks"][0]["lines"], list("ANGLE"))
        self.assertFalse(np.array_equal(plain, vertical))
        changed = np.any(vertical != 255, axis=2)
        ys, xs = np.where(changed)
        self.assertTrue(len(xs) > 0)
        x, y, width, height = bbox
        self.assertGreaterEqual(xs.min(), x)
        self.assertLess(xs.max(), x + width)
        self.assertGreaterEqual(ys.min(), y)
        self.assertLess(ys.max(), y + height)

    def test_two_or_more_lines_keep_normal_rotation(self):
        renderer = TextRenderer(absolute_min_font_size=7, inner_margin_ratio=0.03)
        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        bbox = (35, 35, 190, 105)
        text = "ANGLE\nTEXT"

        plain = renderer.render(image.copy(), [bbox], [text], font_sizes=[26], rotation_angles=[0])
        rotated = renderer.render(image.copy(), [bbox], [text], font_sizes=[26], rotation_angles=[19])
        layout = renderer.build_layout(bbox, text, requested_font_size=26, rotation_angle=19)

        self.assertEqual(layout["requested_rotation_angle"], 19.0)
        self.assertEqual(layout["rotation_angle"], 19.0)
        self.assertEqual(layout["writing_mode"], "horizontal")
        self.assertGreaterEqual(sum(len(block["lines"]) for block in layout["blocks"]), 2)
        self.assertFalse(np.array_equal(plain, rotated))


if __name__ == "__main__":
    unittest.main()


class ManualTypographyLayoutTests(unittest.TestCase):
    def setUp(self):
        self.renderer = TextRenderer(absolute_min_font_size=7, inner_margin_ratio=0.03)
        self.bbox = (20, 25, 180, 90)

    def _line(self, **kwargs):
        layout = self.renderer.resolve_manual_layout(
            self.bbox,
            "TEXTO",
            requested_font_size=24,
            image_shape=(180, 260),
            **kwargs,
        )
        return layout, layout["blocks"][0]["lines"][0]

    def test_exact_layout_honors_horizontal_and_vertical_alignment(self):
        _left_layout, left = self._line(text_align="left", vertical_align="top")
        _center_layout, center = self._line(text_align="center", vertical_align="middle")
        _right_layout, right = self._line(text_align="right", vertical_align="bottom")

        self.assertLess(left["visual_x"], center["visual_x"])
        self.assertLess(center["visual_x"], right["visual_x"])
        self.assertLess(left["visual_y"], center["visual_y"])
        self.assertLess(center["visual_y"], right["visual_y"])

    def test_exact_layout_uses_offsets_and_line_spacing(self):
        base = self.renderer.resolve_manual_layout(
            self.bbox,
            "UNO DOS TRES CUATRO CINCO SEIS",
            requested_font_size=22,
            text_align="center",
            vertical_align="middle",
            line_spacing_factor=1.0,
            image_shape=(180, 260),
        )
        shifted = self.renderer.resolve_manual_layout(
            self.bbox,
            "UNO DOS TRES CUATRO CINCO SEIS",
            requested_font_size=22,
            text_align="center",
            vertical_align="middle",
            line_spacing_factor=1.6,
            text_offset_x=7,
            text_offset_y=-5,
            image_shape=(180, 260),
        )

        base_block = base["blocks"][0]
        shifted_block = shifted["blocks"][0]
        self.assertGreater(shifted_block["line_spacing"], base_block["line_spacing"])
        self.assertAlmostEqual(
            shifted_block["lines"][0]["visual_x"] - base_block["lines"][0]["visual_x"],
            7,
            delta=0.1,
        )
        self.assertLess(shifted_block["lines"][0]["visual_y"], base_block["lines"][0]["visual_y"])

    def test_saved_ui_layout_is_used_when_no_explicit_override_is_given(self):
        ui_layout = {
            "bbox": list(self.bbox),
            "rotation_angle": 12.5,
            "text_align": "right",
            "vertical_align": "bottom",
            "line_spacing_factor": 1.35,
            "text_offset_x": 4,
            "text_offset_y": -3,
            "blocks": [{"area": [5, 6, 160, 70]}],
        }
        resolved = self.renderer.resolve_manual_layout(
            self.bbox,
            "TEXTO\nDOS",
            ui_layout=ui_layout,
            requested_font_size=24,
            image_shape=(180, 260),
        )

        self.assertEqual(resolved["text_align"], "right")
        self.assertEqual(resolved["vertical_align"], "bottom")
        self.assertEqual(resolved["rotation_angle"], 12.5)
        self.assertEqual(resolved["line_spacing_factor"], 1.35)
        self.assertEqual(resolved["text_offset_x"], 4.0)
        self.assertEqual(resolved["text_offset_y"], -3.0)

    def test_manual_layout_preserves_every_internal_line_break(self):
        resolved = self.renderer.resolve_manual_layout(
            (20, 20, 180, 140),
            "UNO\n\n\nDOS",
            requested_font_size=20,
            image_shape=(220, 260),
        )

        self.assertEqual(
            [line["text"] for line in resolved["blocks"][0]["lines"]],
            ["UNO", "", "", "DOS"],
        )

    def test_manual_single_line_keeps_requested_angle_but_uses_vertical_characters(self):
        resolved = self.renderer.resolve_manual_layout(
            self.bbox,
            "HOLA",
            requested_font_size=24,
            rotation_angle=-17.5,
            image_shape=(180, 260),
        )

        self.assertEqual(resolved["requested_rotation_angle"], -17.5)
        self.assertEqual(resolved["rotation_angle"], 0.0)
        self.assertEqual(resolved["writing_mode"], "vertical_chars")
        self.assertEqual([line["text"] for line in resolved["blocks"][0]["lines"]], list("HOLA"))

    def test_manual_region_persists_precise_typography_controls(self):
        regions = parse_manual_regions(
            [{
                "index": 0,
                "bbox": [10, 12, 120, 70],
                "text": "Hola",
                "text_align": "right",
                "vertical_align": "bottom",
                "line_spacing_factor": 1.4,
                "text_offset_x": 8,
                "text_offset_y": -6,
            }],
            240,
            180,
        )

        region = regions[0]
        self.assertEqual(region.text_align, "right")
        self.assertEqual(region.vertical_align, "bottom")
        self.assertEqual(region.line_spacing_factor, 1.4)
        self.assertEqual(region.text_offset_x, 8.0)
        self.assertEqual(region.text_offset_y, -6.0)


class ExactPreviewParityTests(unittest.TestCase):
    def test_region_preview_pixels_match_the_saved_render_crop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean = np.full((170, 260, 3), 255, dtype=np.uint8)
            original = clean.copy()
            translated = clean.copy()
            clean_path = root / "clean.png"
            original_path = root / "original.png"
            translated_path = root / "translated.png"
            output_path = root / "output.png"
            cv2.imwrite(str(clean_path), clean)
            cv2.imwrite(str(original_path), original)
            cv2.imwrite(str(translated_path), translated)

            region = ManualRegion(
                index=0,
                bbox=(40, 45, 150, 75),
                text="VISTA EXACTA",
                modified=True,
                auto_font_size=False,
                font_size=25,
                rotation_angle=11.5,
                text_align="right",
                vertical_align="bottom",
                line_spacing_factor=1.25,
                text_offset_x=-4,
                text_offset_y=-3,
            )
            preview_bytes = render_manual_region_preview(
                clean_path=clean_path,
                original_path=original_path,
                region=region,
            )
            preview = cv2.imdecode(np.frombuffer(preview_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

            render_manual_page(
                clean_path=clean_path,
                original_path=original_path,
                translated_path=translated_path,
                output_path=output_path,
                regions=[region],
            )
            saved = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            x, y, w, h = region.bbox

            self.assertIsNotNone(preview)
            self.assertTrue(np.array_equal(preview, saved[y:y + h, x:x + w]))
