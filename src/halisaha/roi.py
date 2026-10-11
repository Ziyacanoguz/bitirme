import json
from pathlib import Path

import cv2
import numpy as np


def load_roi(path: Path, frame_width: int, frame_height: int) -> np.ndarray:
    """ROI JSON dosyasını yükler ve gerekirse video boyutuna ölçekler."""
    data = json.loads(path.read_text(encoding="utf-8"))
    polygon = np.asarray(data["polygon"], dtype=np.float32)
    source_width = float(data.get("image_width", frame_width))
    source_height = float(data.get("image_height", frame_height))
    if source_width <= 0 or source_height <= 0 or frame_width <= 0 or frame_height <= 0:
        raise ValueError("Invalid dimensions for ROI scaling")

    polygon[:, 0] *= frame_width / source_width
    polygon[:, 1] *= frame_height / source_height
    return np.rint(polygon).astype(np.int32)


def create_mask(frame_shape, polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def apply_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(frame, frame, mask=mask)


def save_roi(path: Path, polygon, image_width: int, image_height: int, source: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source": source,
        "image_width": image_width,
        "image_height": image_height,
        "polygon": [[int(x), int(y)] for x, y in polygon],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
