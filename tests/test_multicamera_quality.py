import json
import random
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from halisaha.multicamera import associate, build_component_audit, fuse_frames, valid_position


def row(frame, identity, x=0.4, y=0.5, *, local=None, confidence=0.9):
    result = dict(
        frame=frame,
        track_id=identity,
        field_x=x,
        field_y=y,
        calibration_supported=1,
        confidence=confidence,
        x1=100,
        x2=140,
        y1=100,
        y2=220,
    )
    if local is not None:
        result["local_track_id"] = local
    return result


class MultiCameraQualityTests(unittest.TestCase):
    def test_valid_position_rejects_non_finite_and_out_of_range_confidence(self):
        for confidence in (float("nan"), float("inf"), float("-inf"), -0.1, 0.14, 1.01):
            with self.subTest(confidence=confidence):
                self.assertFalse(valid_position(row(0, 1, confidence=confidence)))
        self.assertTrue(valid_position(row(0, 1, confidence=0.15)))
        self.assertTrue(valid_position(row(0, 1, confidence=1.0)))

    def test_association_rejects_non_finite_thresholds(self):
        names = (
            "min_support",
            "max_distance",
            "appearance_gate",
            "margin",
            "min_fraction",
            "min_consecutive_support",
            "max_p95_distance",
            "missing_appearance_min_support",
            "missing_appearance_min_fraction",
            "missing_appearance_max_p95",
            "component_appearance_gate",
        )
        for name in names:
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        associate([], [], {}, {}, **{name: value})

    def test_fusion_rejects_non_finite_runtime_thresholds(self):
        for name in ("fps", "max_disagreement"):
            for value in (float("nan"), float("inf"), float("-inf"), 0.0):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        fuse_frames([], [], {}, set(), **{name: value})

    def test_candidate_records_exact_support_frames_as_json(self):
        frames = [7, 3, 5]
        rows1 = [row(frame, 1) for frame in frames]
        rows2 = [row(frame, 8, x=0.41) for frame in frames]
        _, _, candidates = associate(rows1, rows2, {}, {}, min_support=1)
        self.assertEqual(json.loads(candidates[0]["support_frames"]), sorted(frames))
        self.assertEqual(candidates[0]["support"], len(frames))

    def test_result_is_independent_of_csv_row_order(self):
        rows1 = [row(frame, identity, x=0.2 + identity * 0.2) for frame in range(12) for identity in (1, 2)]
        rows2 = [row(frame, identity, x=0.205 + (identity - 8) * 0.2) for frame in range(12) for identity in (8, 9)]
        expected = associate(rows1, rows2, {}, {}, min_support=4, return_diagnostics=True)
        random.Random(7).shuffle(rows1)
        random.Random(11).shuffle(rows2)
        actual = associate(rows1, rows2, {}, {}, min_support=4, return_diagnostics=True)
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertEqual(expected[2], actual[2])
        self.assertEqual(expected[3], actual[3])

    def test_missing_appearance_requires_stronger_geometry_when_configured(self):
        rows1 = [row(frame, 1) for frame in range(20)]
        rows2 = [row(frame, 8, x=0.41 if frame < 10 else 0.8) for frame in range(20)]
        mapping, shared, candidates = associate(
            rows1,
            rows2,
            {},
            {},
            min_support=5,
            min_fraction=0.4,
            missing_appearance_min_support=8,
            missing_appearance_min_fraction=0.75,
        )
        self.assertNotEqual(mapping[(1, 1)], mapping[(2, 8)])
        self.assertFalse(shared)
        self.assertEqual(candidates[0]["reason"], "appearance_missing_weak_geometry")
        self.assertEqual(candidates[0]["appearance_available"], 0)

    def test_strong_geometry_can_compensate_for_missing_appearance(self):
        rows1 = [row(frame, 1) for frame in range(20)]
        rows2 = [row(frame, 8, x=0.41) for frame in range(20)]
        mapping, shared, candidates = associate(
            rows1,
            rows2,
            {},
            {},
            min_support=5,
            missing_appearance_min_support=15,
            missing_appearance_min_fraction=0.7,
        )
        self.assertEqual(mapping[(1, 1)], mapping[(2, 8)])
        self.assertEqual(len(shared), 1)
        self.assertTrue(candidates[0]["accepted"])
        self.assertGreaterEqual(candidates[0]["evidence_score"], 0)
        self.assertLessEqual(candidates[0]["evidence_score"], 1)

    def test_local_continuity_is_resolved_before_cross_camera_edges(self):
        rows1 = [row(frame, 1, local=7) for frame in range(5)]
        rows1 += [row(frame, 2, local=7) for frame in range(5, 10)]
        rows2 = [row(frame, 8, x=0.4 if frame < 5 else 0.8) for frame in range(10)]
        rows2 += [row(frame, 9) for frame in range(5, 10)]
        mapping, _, candidates, diagnostics = associate(
            rows1,
            rows2,
            {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])},
            {},
            min_support=3,
            return_diagnostics=True,
        )
        self.assertEqual(mapping[(1, 1)], mapping[(1, 2)])
        self.assertEqual(mapping[(1, 1)], mapping[(2, 8)])
        self.assertNotEqual(mapping[(1, 1)], mapping[(2, 9)])
        rejected = next(item for item in candidates if item["camera2_id"] == 9)
        self.assertEqual(rejected["reason"], "same_camera_overlap_conflict")
        self.assertEqual(
            diagnostics["continuity_decisions"].get("continuity_component_link"), 1
        )

    def test_component_wide_appearance_conflict_blocks_transitive_false_merge(self):
        rows1 = [row(frame, 1) for frame in range(5)]
        rows1 += [row(frame, 2) for frame in range(5, 10)]
        rows2 = [row(frame, 8) for frame in range(10)]
        diagonal = np.array([1.0, 1.0])
        mapping, _, candidates = associate(
            rows1,
            rows2,
            {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])},
            {8: diagonal},
            min_support=3,
            component_appearance_gate=0.5,
        )
        self.assertEqual(mapping[(1, 1)], mapping[(2, 8)])
        self.assertNotEqual(mapping[(1, 2)], mapping[(2, 8)])
        rejected = next(item for item in candidates if item["camera1_id"] == 2)
        self.assertEqual(rejected["reason"], "component_appearance_conflict")
        self.assertGreater(rejected["component_max_appearance"], 0.5)

    def test_component_and_frame_outputs_expose_evidence(self):
        rows1 = [row(frame, 1, local=11) for frame in range(8)]
        rows2 = [row(frame, 8, x=0.41, local=19) for frame in range(8)]
        mapping, shared, candidates, diagnostics = associate(
            rows1, rows2, {}, {}, min_support=3, return_diagnostics=True
        )
        components, scores, uids, pair_scores = build_component_audit(
            mapping, shared, candidates, diagnostics, rows1, rows2
        )
        fused, _ = fuse_frames(
            rows1,
            rows2,
            mapping,
            shared,
            pair_evidence=pair_scores,
            component_evidence=scores,
            component_uids=uids,
        )
        shared_component = next(item for item in components if item["shared"] == 1)
        self.assertEqual(len(shared_component["component_uid"]), 16)
        self.assertEqual(shared_component["evidence_semantics"], "weakest_graph_link_heuristic_not_probability")
        self.assertEqual(fused[0]["identity_evidence_kind"], "direct_pair")
        self.assertEqual(fused[0]["component_uid"], shared_component["component_uid"])
        self.assertEqual(fused[0]["camera1_local_track_id"], 11)
        self.assertEqual(fused[0]["camera2_local_track_id"], 19)


if __name__ == "__main__":
    unittest.main()
