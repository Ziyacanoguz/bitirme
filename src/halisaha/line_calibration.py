"""Ground-line calibration: division-model distortion followed by a homography.

The four boundary lines locate unobserved corners by intersection, not by treating
an arbitrary cropped image edge as a physical pitch corner. Units stay normalized.
"""
import numpy as np
import cv2
from scipy.optimize import least_squares


COORDINATE_SYSTEM = "normalized_field_division_homography_v1"


def undistort_division(points, center, scale, coefficient):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    delta = (points - np.asarray(center, dtype=float)) / float(scale)
    denominator = 1 + float(coefficient) * (delta ** 2).sum(axis=1)
    if np.any(denominator <= .05):
        raise ValueError("Division model is singular in this region")
    return delta / denominator[:, None]


def transform(points, homography):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    q = np.column_stack((points, np.ones(len(points)))) @ np.asarray(homography).T
    with np.errstate(divide="ignore", invalid="ignore"):
        res = q[:, :2] / q[:, 2:3]
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


def project_line_calibration(points, data, width=None, height=None):
    points = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    if width is not None:
        points *= [data["image_width"] / width, data["image_height"] / height]
    corrected = undistort_division(points, data["division_center"], data["division_scale"], data["division_k"])
    return transform(corrected, data["homography"])


def unproject_line_calibration(points, data):
    corrected = transform(points, np.linalg.inv(data["homography"]))
    radicand = 1 - 4 * data["division_k"] * (corrected ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = 2 / (1 + np.sqrt(radicand))
    return np.asarray(data["division_center"]) + corrected * factor[:, None] * data["division_scale"]


def fit_image_line(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("At least two points per line")
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center)
    normal = vh[-1]
    return np.r_[normal, -normal @ center]


def intersection(a, b):
    result = np.cross(a, b)
    if abs(result[2]) < 1e-9:
        raise ValueError("Boundary lines are parallel in image; cannot initialize")
    return result[:2] / result[2]


def fit_line_calibration(lines, width, height, *, sample_parity=None):
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    required = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if not required <= {(int(line["axis"]), line["value"]) for line in lines}:
        raise ValueError("All four field boundary lines are required")
    center, scale = np.array([width / 2, height / 2]), float(width)
    selected, axes, values, initial_lines = [], [], [], {}
    for line in lines:
        pixels = np.asarray(line["points"], dtype=float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
            raise ValueError("Malformed line points")
        if sample_parity is not None:
            pixels = pixels[sample_parity::2]
        if len(pixels) < 2:
            raise ValueError("Too few fit samples")
        selected.extend(pixels)
        axes.extend([int(line["axis"])] * len(pixels))
        values.extend([float(line["value"])] * len(pixels))
        initial_lines[(int(line["axis"]), line["value"])] = fit_image_line((pixels - center) / scale)
    image_corners = np.asarray([intersection(initial_lines[(0, x)], initial_lines[(1, y)])
                                for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))])
    h0 = cv2.getPerspectiveTransform(image_corners.astype(np.float32),
                                    np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32))
    points, axes, targets = np.asarray(selected), np.asarray(axes), np.asarray(values)

    def residual(params):
        h = np.r_[params[:8], 1].reshape(3, 3)
        mapped = transform(undistort_division(points, center, scale, params[8]), h)
        error = mapped[np.arange(len(points)), axes] - targets
        return np.nan_to_num(error, nan=1e3, posinf=1e3, neginf=-1e3)

    result = least_squares(residual, np.r_[h0.ravel()[:8], 0.],
                           bounds=(np.r_[np.full(8, -np.inf), -1.5], np.r_[np.full(8, np.inf), 1.5]),
                           loss="soft_l1", f_scale=.01, max_nfev=3000)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Calibration did not converge: {result.message}")
    h = np.r_[result.x[:8], 1].reshape(3, 3)
    if abs(np.linalg.det(h)) < 1e-9:
        raise ValueError("Degenerate homography")
    return dict(coordinate_system=COORDINATE_SYSTEM, image_width=width, image_height=height,
                division_center=center.tolist(), division_scale=scale, division_k=float(result.x[8]),
                homography=h.tolist(), image_points=points.tolist(),
                fit_error_median=float(np.median(np.abs(residual(result.x)))),
                fit_error_p95=float(np.quantile(np.abs(residual(result.x)), .95)),
                point_semantics="Ground lines; x=0 goal A, x=1 goal B, y=0 spectator side",
                status="experimental_not_surveyed", units="normalized_not_meters")


def line_errors(lines, projector, parity=None):
    result = []
    for line in lines:
        points = np.asarray(line["points"], dtype=float)
        if parity is not None:
            points = points[parity::2]
        projected = projector(points)
        result.extend(np.abs(projected[:, int(line["axis"])] - float(line["value"])))
    return np.asarray(result)
