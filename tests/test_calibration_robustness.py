import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halisaha.calibration import (
    fit_tps_transform,
    apply_tps_transform,
    load_calibration,
    load_multipoint_calibration,
)
from halisaha.line_calibration import transform
from halisaha.roi import load_roi
from halisaha.multicamera import (
    ConflictSafeComponents,
    _component_appearance_conflict,
)


class TestCalibrationRobustness(unittest.TestCase):
    def test_fit_tps_collinear_points_fallback(self):
        # Collinear source points force a singular kernel matrix in TPS
        source = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]])
        target = np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.2], [1.5, 0.3]])
        
        # Should not raise LinAlgError due to lstsq fallback
        weights, affine = fit_tps_transform(source, target)
        self.assertEqual(weights.shape, (4, 2))
        self.assertEqual(affine.shape, (3, 2))
        
        # Verify transform application works
        projected = apply_tps_transform(source, source, weights, affine)
        self.assertEqual(projected.shape, (4, 2))

    def test_load_calibration_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_file = Path(tmpdir) / "calib.json"
            calib_file.write_text(
                json.dumps({
                    "image_width": 0,
                    "image_height": 1080,
                    "image_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_calibration(calib_file, 1920, 1080)

    def test_load_multipoint_calibration_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_file = Path(tmpdir) / "multi_calib.json"
            calib_file.write_text(
                json.dumps({
                    "image_width": -100,
                    "image_height": 1080,
                    "image_points": [[0, 0]] * 7,
                    "field_points": [[0, 0]] * 7,
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_multipoint_calibration(calib_file)

    def test_line_calibration_transform_zero_z(self):
        # Create a homography matrix where q_z becomes 0
        homography = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # zeros out q_z
        ])
        points = np.array([[10.0, 20.0], [30.0, 40.0]])
        res = transform(points, homography)
        self.assertEqual(res.shape, (2, 2))
        self.assertFalse(np.isnan(res).any())
        self.assertFalse(np.isinf(res).any())

    def test_load_roi_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            roi_file = Path(tmpdir) / "roi.json"
            roi_file.write_text(
                json.dumps({
                    "image_width": 1920,
                    "image_height": 0,
                    "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_roi(roi_file, 1920, 1080)

    def test_component_appearance_conflict_no_distances(self):
        obs = {(1, 101): {1, 2}, (2, 201): {3, 4}}
        csc = ConflictSafeComponents(obs)
        node_features = {}  # No features available
        result = _component_appearance_conflict(
            csc, (1, 101), (2, 201), node_features, gate=0.45
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
