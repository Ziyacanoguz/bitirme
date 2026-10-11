import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from render_multicamera_preview import (  # noqa: E402
    aligned_track_frames,
    build_global_label_lookup,
    choose_label_box,
    color_for_label,
    load_fused_rows,
    observation_label,
    rectangle_overlap_area,
)


def fused_row(status, global_id, frame1="", segment1="", frame2="", segment2=""):
    return {
        "status": status,
        "global_id": global_id,
        "source_frame_camera1": frame1,
        "camera1_id": segment1,
        "source_frame_camera2": frame2,
        "camera2_id": segment2,
    }


class MultiCameraPreviewTests(unittest.TestCase):
    def test_header_only_fusion_renders_camera_local_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fused.csv"
            path.write_text(
                "status,global_id,source_frame_camera1,source_frame_camera2,camera1_id,camera2_id\n",
                encoding="utf-8",
            )
            lookup = build_global_label_lookup(load_fused_rows(path), {(1, 7): 3})
            self.assertEqual(
                observation_label(1, 100, 7, {"local_track_id": 70}, lookup),
                "C1:L70",
            )

    def test_exact_source_frame_and_segment_receive_same_global_label(self):
        mapping = {(1, 7): 3, (2, 12): 3}
        lookup = build_global_label_lookup(
            [fused_row("fused", 3, 100, 7, 102, 12)], mapping
        )
        first = observation_label(1, 100, 7, {"local_track_id": 70}, lookup)
        second = observation_label(2, 102, 12, {"local_track_id": 22}, lookup)
        self.assertEqual(first, "G3")
        self.assertEqual(second, "G3")
        self.assertEqual(color_for_label(first), color_for_label(second))

    def test_mapping_is_not_blindly_applied_to_other_frames(self):
        mapping = {(1, 7): 3}
        lookup = build_global_label_lookup(
            [fused_row("single_view", 3, 100, 7)], mapping
        )
        self.assertEqual(
            observation_label(1, 101, 7, {"local_track_id": 70}, lookup),
            "C1:L70",
        )

    def test_conflict_invalid_and_unmatched_never_show_global_id(self):
        mapping = {(1, 7): 3}
        rows = [
            fused_row("conflict", 3, 100, 7),
            fused_row("invalid", 3, 101, 7),
            fused_row("unmatched", 3, 102, 7),
        ]
        lookup = build_global_label_lookup(rows, mapping)
        for frame in (100, 101, 102):
            self.assertEqual(
                observation_label(1, frame, 7, {"local_track_id": 70}, lookup),
                "C1:L70",
            )

    def test_single_view_can_show_provisional_global_id(self):
        mapping = {(2, 9): 4}
        lookup = build_global_label_lookup(
            [fused_row("single_view", 4, frame2=55, segment2=9)], mapping
        )
        self.assertEqual(observation_label(2, 55, 9, {}, lookup), "G4")

    def test_mapping_disagreement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagrees with mapping"):
            build_global_label_lookup(
                [fused_row("fused", 5, 10, 1)], {(1, 1): 8}
            )

    def test_duplicate_fused_observation_is_rejected(self):
        row = fused_row("unmatched", "", 10, 1)
        with self.assertRaisesRegex(ValueError, "Duplicate fused"):
            build_global_label_lookup([row, dict(row)], {(1, 1): 2})

    def test_aligned_frames_use_fusion_offset_direction(self):
        first = {10: [], 11: [], 12: []}
        second = {12: [], 13: [], 20: []}
        self.assertEqual(aligned_track_frames(first, second, 2), {10, 11})

    def test_local_colors_are_camera_specific(self):
        self.assertNotEqual(color_for_label("C1:L7"), color_for_label("C2:L7"))

    def test_label_box_is_clamped_inside_panel(self):
        box, _displaced = choose_label_box(
            (95, 2, 110, 30), (40, 20), (100, 80), []
        )
        self.assertGreaterEqual(box[0], 0)
        self.assertGreaterEqual(box[1], 0)
        self.assertLessEqual(box[2], 100)
        self.assertLessEqual(box[3], 80)

    def test_label_placement_avoids_header_and_previous_label(self):
        occupied = [(0, 0, 200, 38), (70, 45, 125, 70)]
        first = choose_label_box((75, 72, 115, 125), (55, 25), (200, 150), occupied)
        second = choose_label_box((78, 74, 118, 127), (55, 25), (200, 150), occupied + [first[0]])
        self.assertFalse(any(rectangle_overlap_area(first[0], item) for item in occupied))
        self.assertFalse(any(rectangle_overlap_area(second[0], item) for item in occupied + [first[0]]))
        self.assertNotEqual(first[0], second[0])
        self.assertTrue(first[1] or second[1])

    def test_label_placement_is_deterministic(self):
        arguments = ((10, 40, 30, 80), (50, 20), (120, 100), [(0, 0, 120, 38)])
        self.assertEqual(choose_label_box(*arguments), choose_label_box(*arguments))


if __name__ == "__main__":
    unittest.main()
