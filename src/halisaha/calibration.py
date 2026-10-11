import json
from pathlib import Path

import cv2
import numpy as np


FIELD_POINTS = np.asarray(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ],
    dtype=np.float32,
)

MULTIPOINT_FIELD_POINTS = np.asarray(
    [
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.5, 1.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ],
    dtype=np.float64,
)


def polynomial_features(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack((np.ones(len(points)), x, y, x * x, x * y, y * y))


def fit_polynomial_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    if len(source) < 6 or len(source) != len(target):
        raise ValueError("İkinci derece dönüşüm için en az altı eşleşen nokta gerekir.")
    coefficients, _, _, _ = np.linalg.lstsq(polynomial_features(source), target, rcond=None)
    return coefficients


def apply_polynomial_transform(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return polynomial_features(points) @ np.asarray(coefficients, dtype=np.float64)


def _tps_kernel(distances: np.ndarray) -> np.ndarray:
    squared = np.asarray(distances, dtype=np.float64) ** 2
    return squared * np.log(np.maximum(distances, 1e-12))


def fit_tps_transform(source_points: np.ndarray, target_points: np.ndarray):
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    if len(source) < 3 or len(source) != len(target):
        raise ValueError("TPS dönüşümü için en az üç eşleşen nokta gerekir.")
    differences = source[:, None, :] - source[None, :, :]
    kernel = _tps_kernel(np.linalg.norm(differences, axis=2))
    affine = np.column_stack((np.ones(len(source)), source))
    system = np.block(
        [
            [kernel, affine],
            [affine.T, np.zeros((3, 3), dtype=np.float64)],
        ]
    )
    targets = np.vstack((target, np.zeros((3, 2), dtype=np.float64)))
    try:
        solution = np.linalg.solve(system, targets)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(system, targets, rcond=None)[0]
    return solution[: len(source)], solution[len(source) :]


def apply_tps_transform(
    points: np.ndarray,
    control_points: np.ndarray,
    weights: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    controls = np.asarray(control_points, dtype=np.float64).reshape(-1, 2)
    distances = np.linalg.norm(points[:, None, :] - controls[None, :, :], axis=2)
    radial = _tps_kernel(distances) @ np.asarray(weights, dtype=np.float64)
    linear = np.column_stack((np.ones(len(points)), points)) @ np.asarray(affine, dtype=np.float64)
    return radial + linear


def save_multipoint_calibration(
    path: Path,
    source: str,
    image_width: int,
    image_height: int,
    image_points: np.ndarray,
):
    normalized_image = np.asarray(image_points, dtype=np.float64).copy()
    normalized_image[:, 0] /= image_width
    normalized_image[:, 1] /= image_height
    forward_weights, forward_affine = fit_tps_transform(normalized_image, MULTIPOINT_FIELD_POINTS)
    inverse_weights, inverse_affine = fit_tps_transform(MULTIPOINT_FIELD_POINTS, normalized_image)
    data = {
        "source": source,
        "image_width": image_width,
        "image_height": image_height,
        "coordinate_system": "normalized_field_tps_v1",
        "point_order": [
            "left_upper",
            "center_upper",
            "right_upper",
            "right_lower",
            "center_lower",
            "left_lower",
            "center_spot",
        ],
        "image_points": np.asarray(image_points).round(2).tolist(),
        "field_points": MULTIPOINT_FIELD_POINTS.tolist(),
        "forward_weights": forward_weights.tolist(),
        "forward_affine": forward_affine.tolist(),
        "inverse_weights": inverse_weights.tolist(),
        "inverse_affine": inverse_affine.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_multipoint_calibration(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    width = float(data.get("image_width", 0))
    height = float(data.get("image_height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions in multipoint calibration")
    image_points = np.asarray(data["image_points"], dtype=np.float64)
    image_points[:, 0] /= width
    image_points[:, 1] /= height
    return {
        "image_control_points": image_points,
        "field_control_points": np.asarray(data["field_points"], dtype=np.float64),
        "forward_weights": np.asarray(data["forward_weights"], dtype=np.float64),
        "forward_affine": np.asarray(data["forward_affine"], dtype=np.float64),
        "inverse_weights": np.asarray(data["inverse_weights"], dtype=np.float64),
        "inverse_affine": np.asarray(data["inverse_affine"], dtype=np.float64),
    }


def project_points_multipoint(
    points: np.ndarray, calibration: dict, frame_width: int, frame_height: int
) -> np.ndarray:
    normalized = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    normalized[:, 0] /= frame_width
    normalized[:, 1] /= frame_height
    return apply_tps_transform(
        normalized,
        calibration["image_control_points"],
        calibration["forward_weights"],
        calibration["forward_affine"],
    )


def compute_homography(image_points: np.ndarray) -> np.ndarray:
    points = np.asarray(image_points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("Kalibrasyon için tam olarak dört görüntü noktası gerekir.")
    matrix = cv2.getPerspectiveTransform(points, FIELD_POINTS)
    if not np.isfinite(matrix).all():
        raise ValueError("Geçerli homography hesaplanamadı.")
    return matrix


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(source, homography).reshape(-1, 2)


def save_calibration(
    path: Path,
    source: str,
    image_width: int,
    image_height: int,
    image_points: np.ndarray,
    homography: np.ndarray,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source": source,
        "image_width": image_width,
        "image_height": image_height,
        "coordinate_system": "normalized_field",
        "point_order": [
            "left_goal_upper",
            "right_goal_upper",
            "right_goal_lower",
            "left_goal_lower",
        ],
        "image_points": np.asarray(image_points).round(2).tolist(),
        "field_points": FIELD_POINTS.tolist(),
        "homography": np.asarray(homography).tolist(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_calibration(path: Path, frame_width: int, frame_height: int) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(data["image_points"], dtype=np.float32)
    source_width = float(data.get("image_width", frame_width))
    source_height = float(data.get("image_height", frame_height))
    if source_width <= 0 or source_height <= 0 or frame_width <= 0 or frame_height <= 0:
        raise ValueError("Invalid image dimensions in calibration")
    points[:, 0] *= frame_width / source_width
    points[:, 1] *= frame_height / source_height
    return compute_homography(points)
