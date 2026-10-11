import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halisaha.detection_fusion import (  # noqa: E402
    box_iou,
    fuse_full_and_crop_detections,
    non_max_suppression,
    translate_detections,
)


def det(x1, y1, x2, y2, confidence, cls=0):
    return [x1, y1, x2, y2, confidence, cls]


class DetectionFusionTests(unittest.TestCase):
    def test_box_iou_handles_overlap_separation_and_degenerate_box(self):
        candidates = np.asarray(
            [[5, 5, 15, 15], [20, 20, 30, 30], [2, 2, 2, 9]],
            dtype=np.float32,
        )
        actual = box_iou(np.asarray([0, 0, 10, 10]), candidates)
        np.testing.assert_allclose(actual, [25 / 175, 0.0, 0.0])

    def test_nms_keeps_highest_confidence_duplicate(self):
        detections = np.asarray(
            [det(0, 0, 10, 10, 0.70), det(1, 1, 11, 11, 0.90), det(30, 30, 40, 40, 0.50)],
            dtype=np.float32,
        )
        result = non_max_suppression(detections, 0.5)
        np.testing.assert_allclose(result[:, 4], [0.90, 0.50])
        np.testing.assert_allclose(result[0, :4], [1, 1, 11, 11])

    def test_nms_is_stable_for_equal_confidence(self):
        detections = np.asarray(
            [det(0, 0, 10, 10, 0.80), det(1, 1, 11, 11, 0.80)], dtype=np.float32
        )
        result = non_max_suppression(detections, 0.5)
        np.testing.assert_allclose(result[0, :4], detections[0, :4])

    def test_translate_detections_does_not_mutate_input(self):
        original = np.asarray([det(1, 2, 11, 12, 0.75)], dtype=np.float32)
        before = original.copy()
        result = translate_detections(original, offset_x=100, offset_y=200)
        np.testing.assert_array_equal(original, before)
        np.testing.assert_allclose(result[0, :4], [101, 202, 111, 212])

    def test_fusion_maps_crop_box_and_suppresses_duplicate(self):
        full = np.asarray([det(100, 200, 120, 240, 0.65)], dtype=np.float32)
        crop = np.asarray([det(0, 0, 20, 40, 0.90)], dtype=np.float32)
        crop_before = crop.copy()

        mapped, fused = fuse_full_and_crop_detections(
            full, crop, crop_x=100, crop_y=200, iou_threshold=0.5
        )

        np.testing.assert_array_equal(crop, crop_before)
        np.testing.assert_allclose(mapped[0, :4], [100, 200, 120, 240])
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(float(fused[0, 4]), 0.90, places=6)

    def test_fusion_retains_distinct_full_and_crop_players(self):
        full = np.asarray([det(10, 10, 20, 30, 0.80)], dtype=np.float32)
        crop = np.asarray([det(50, 50, 60, 70, 0.70)], dtype=np.float32)
        _, fused = fuse_full_and_crop_detections(
            full, crop, crop_x=100, crop_y=200, iou_threshold=0.5
        )
        self.assertEqual(len(fused), 2)

    def test_fusion_accepts_empty_sources(self):
        empty = np.empty((0, 6), dtype=np.float32)
        one = np.asarray([det(1, 2, 3, 4, 0.75)], dtype=np.float32)
        mapped, fused = fuse_full_and_crop_detections(
            empty, one, crop_x=10, crop_y=20, iou_threshold=0.5
        )
        np.testing.assert_allclose(mapped[0, :4], [11, 22, 13, 24])
        np.testing.assert_array_equal(mapped, fused)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            non_max_suppression(np.empty((0, 4), dtype=np.float32), 0.5)
        with self.assertRaises(ValueError):
            non_max_suppression(np.empty((0, 6), dtype=np.float32), 1.1)
        with self.assertRaises(ValueError):
            non_max_suppression(
                np.asarray([det(0, 0, 1, 1, np.nan)], dtype=np.float32), 0.5
            )


if __name__ == "__main__":
    unittest.main()
