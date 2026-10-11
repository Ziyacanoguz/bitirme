"""Project image-space track foot points into a normalized common field plane."""
import json
from pathlib import Path

import cv2
import numpy as np

from halisaha.calibration import (
    load_calibration,
    load_multipoint_calibration,
    project_points,
    project_points_multipoint,
)
from halisaha.line_calibration import COORDINATE_SYSTEM, project_line_calibration


class FieldProjector:
    """Support legacy calibrations and the line-based global-ID calibration."""

    def __init__(self, path, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Invalid video size")
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.width, self.height = width, height
        self.line_model = data if data.get("coordinate_system") == COORDINATE_SYSTEM else None
        self.multipoint = data.get("coordinate_system") == "normalized_field_tps_v1"
        self.calibration = (None if self.line_model is not None else
                            load_multipoint_calibration(path) if self.multipoint else
                            load_calibration(path, width, height))
        controls = np.asarray(data["image_points"], np.float32).copy()
        controls *= [width / data["image_width"], height / data["image_height"]]
        self.hull = cv2.convexHull(controls)

    def project(self, rows):
        rows = [dict(row) for row in rows]
        if not rows:
            return []
        feet = np.asarray([
            [float(r["foot_x"]), float(r["foot_y"])]
            if "foot_x" in r and "foot_y" in r else
            [(float(r["x1"]) + float(r["x2"])) / 2, float(r["y2"])]
            for r in rows
        ], dtype=float)
        points = (project_line_calibration(feet, self.line_model, self.width, self.height)
                  if self.line_model is not None else
                  project_points_multipoint(feet, self.calibration, self.width, self.height)
                  if self.multipoint else project_points(feet, self.calibration))
        for row, foot, point in zip(rows, feet, points):
            row.update(field_x=float(point[0]), field_y=float(point[1]),
                       calibration_supported=int(
                           cv2.pointPolygonTest(self.hull, tuple(foot), False) >= 0))
        return rows
