import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halisaha.identity_consolidation import (  # noqa: E402
    consolidate_global_identities,
    _cosine_distance,
)


class TestIdentityConsolidation(unittest.TestCase):
    def test_cosine_distance(self):
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(_cosine_distance(u, v), 0.0)

        w = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(_cosine_distance(u, w), 1.0)

    def test_overlapping_tracks_never_merged(self):
        # Two tracks active at the same frame 100
        fused_rows = [
            {"frame": 100, "global_id": 1, "display_label": "G1"},
            {"frame": 100, "global_id": 2, "display_label": "G2"},
        ]
        mapping_rows = [
            {"camera": 1, "track_id": 10, "global_id": 1},
            {"camera": 2, "track_id": 20, "global_id": 2},
        ]
        # Identical features
        feat = np.array([1.0, 0.5, 0.2])
        features1 = {10: feat}
        features2 = {20: feat}

        _, _, cmap = consolidate_global_identities(
            fused_rows, mapping_rows, features1, features2, target_players=1
        )
        # Should remain separate because they temporally overlap at frame 100
        self.assertEqual(len(set(cmap.values())), 2)

    def test_non_overlapping_tracks_merged(self):
        # Track 1 active frame 100..110, Track 2 active frame 120..130
        fused_rows = [
            {"frame": 100, "global_id": 1, "display_label": "G1"},
            {"frame": 120, "global_id": 2, "display_label": "G2"},
        ]
        mapping_rows = [
            {"camera": 1, "track_id": 10, "global_id": 1},
            {"camera": 1, "track_id": 11, "global_id": 2},
        ]
        feat = np.array([1.0, 0.5, 0.2])
        features1 = {10: feat, 11: feat + 0.01}

        c_fused, c_mapping, cmap = consolidate_global_identities(
            fused_rows, mapping_rows, features1, {}, target_players=1, max_appearance_distance=0.4
        )
        # Should merge into 1 cluster (G1)
        self.assertEqual(len(set(cmap.values())), 1)
        self.assertEqual(c_fused[0]["global_id"], 1)
        self.assertEqual(c_fused[1]["global_id"], 1)

    def test_handover_overlap_merged(self):
        # Track 1 active frame 100..150 (51 frames)
        # Track 2 active frame 140..200 (61 frames)
        # Overlap: 11 frames (140..150)
        fused_rows = [{"frame": f, "global_id": 1, "display_label": "G1"} for f in range(100, 151)]
        fused_rows += [{"frame": f, "global_id": 2, "display_label": "G2"} for f in range(140, 201)]

        mapping_rows = [
            {"camera": 1, "track_id": 10, "global_id": 1},
            {"camera": 2, "track_id": 20, "global_id": 2},
        ]
        feat = np.array([1.0, 0.5, 0.2])
        features1 = {10: feat}
        features2 = {20: feat + 0.005}

        # With default max_overlap_tolerance=0, should NOT merge
        _, _, cmap_strict = consolidate_global_identities(
            fused_rows, mapping_rows, features1, features2, target_players=1, max_overlap_tolerance=0
        )
        self.assertEqual(len(set(cmap_strict.values())), 2)

        # With max_overlap_tolerance=20, should merge cleanly into 1 cluster
        c_fused, _, cmap_tol = consolidate_global_identities(
            fused_rows, mapping_rows, features1, features2, target_players=1, max_overlap_tolerance=20
        )
        self.assertEqual(len(set(cmap_tol.values())), 1)
        self.assertEqual(c_fused[0]["global_id"], 1)


if __name__ == "__main__":
    unittest.main()

