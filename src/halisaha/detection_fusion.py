"""Utilities for combining detections from a full frame and a zoomed crop.

The tracking pipeline runs the detector twice: once on the complete frame and
once on a resized far-field crop.  The crop boxes must first be translated to
full-frame coordinates, after which class-agnostic NMS removes observations of
the same player.  The detector is configured for the person class only, so
class-agnostic suppression is intentional here.
"""

from __future__ import annotations

import numpy as np


def _detection_array(detections, *, name: str) -> np.ndarray:
    """Return a validated, independent float32 ``(N, >=5)`` array."""
    array = np.asarray(detections, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError(f"{name} must have shape (N, >=5), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


def box_iou(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Calculate IoU between one ``xyxy`` box and zero or more boxes."""
    one = np.asarray(reference, dtype=np.float32)
    many = np.asarray(candidates, dtype=np.float32)
    if one.ndim != 1 or one.shape[0] < 4:
        raise ValueError(f"reference must have shape (>=4,), got {one.shape}")
    if many.ndim != 2 or many.shape[1] < 4:
        raise ValueError(f"candidates must have shape (N, >=4), got {many.shape}")

    left = np.maximum(one[:2], many[:, :2])
    right = np.minimum(one[2:4], many[:, 2:4])
    wh = np.maximum(0.0, right - left)
    intersection = wh[:, 0] * wh[:, 1]
    area_one = max(0.0, float(one[2] - one[0])) * max(
        0.0, float(one[3] - one[1])
    )
    area_many = np.maximum(0.0, many[:, 2] - many[:, 0]) * np.maximum(
        0.0, many[:, 3] - many[:, 1]
    )
    return intersection / np.maximum(area_one + area_many - intersection, 1e-6)


def non_max_suppression(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Apply deterministic, confidence-ordered, class-agnostic greedy NMS.

    The output convention matches the original tracking script: boxes whose
    IoU is equal to the threshold are suppressed as duplicates.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    array = _detection_array(detections, name="detections")
    if len(array) == 0:
        return array

    # A stable sort makes equal-confidence results repeatable and retains the
    # input/source ordering (full-frame detections precede crop detections).
    order = np.argsort(-array[:, 4], kind="stable")
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        overlaps = box_iou(array[current], array[remaining])
        order = remaining[overlaps < iou_threshold]
    return array[keep]


def translate_detections(
    detections: np.ndarray, *, offset_x: float, offset_y: float
) -> np.ndarray:
    """Translate crop-local ``xyxy`` boxes without mutating the input array."""
    array = _detection_array(detections, name="detections")
    array[:, [0, 2]] += float(offset_x)
    array[:, [1, 3]] += float(offset_y)
    return array


def fuse_full_and_crop_detections(
    full_detections: np.ndarray,
    crop_detections: np.ndarray,
    *,
    crop_x: float,
    crop_y: float,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map crop detections to the frame and remove full/crop duplicates.

    Returns ``(crop_detections_in_frame, fused_detections)``.  Returning the
    translated crop observations as well as the final set keeps coordinate
    ownership explicit for logging and future diagnostics.
    """
    full = _detection_array(full_detections, name="full_detections")
    crop_in_frame = translate_detections(
        crop_detections, offset_x=crop_x, offset_y=crop_y
    )
    if full.shape[1] != crop_in_frame.shape[1]:
        raise ValueError(
            "full_detections and crop_detections must have the same column count"
        )
    combined = np.concatenate((full, crop_in_frame), axis=0)
    return crop_in_frame, non_max_suppression(combined, iou_threshold)
