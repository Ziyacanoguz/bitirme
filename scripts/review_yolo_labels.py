import argparse
import csv
import hashlib
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO oyuncu etiketlerini yerel OpenCV penceresinde kontrol et"
    )
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "datasets/halisaha_player"
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Verilmezse ilk kontrol edilmemis goruntuden devam eder",
    )
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=900)
    return parser.parse_args()


def load_manifest(path, split):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indices = [index for index, row in enumerate(rows) if row["split"] == split]
    return rows, indices


def save_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_boxes(label_path, width, height):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        _class_id, center_x, center_y, box_width, box_height = map(float, values)
        x1 = (center_x - box_width / 2) * width
        y1 = (center_y - box_height / 2) * height
        x2 = (center_x + box_width / 2) * width
        y2 = (center_y + box_height / 2) * height
        boxes.append([x1, y1, x2, y2])
    return boxes


def save_boxes(label_path, boxes, width, height):
    lines = []
    for x1, y1, x2, y2 in boxes:
        left, right = sorted((max(0.0, x1), min(float(width - 1), x2)))
        top, bottom = sorted((max(0.0, y1), min(float(height - 1), y2)))
        if right - left < 2 or bottom - top < 2:
            continue
        center_x = (left + right) / 2 / width
        center_y = (top + bottom) / 2 / height
        box_width = (right - left) / width
        box_height = (bottom - top) / height
        lines.append(
            f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class Reviewer:
    def __init__(self, dataset, manifest_rows, manifest_indices, args):
        self.dataset = dataset
        self.rows = manifest_rows
        self.indices = manifest_indices
        self.args = args
        if args.start_index is None:
            first_unreviewed = next(
                (
                    position
                    for position, manifest_index in enumerate(self.indices)
                    if self.rows[manifest_index].get("reviewed") != "1"
                ),
                0,
            )
            self.position = first_unreviewed
        else:
            self.position = max(0, min(args.start_index, len(self.indices) - 1))
        self.image = None
        self.boxes = []
        self.scale = 1.0
        self.drag_start = None
        self.drag_current = None
        self.window = "Hali saha YOLO etiket kontrolu"

    def current_row(self):
        return self.rows[self.indices[self.position]]

    def paths(self):
        row = self.current_row()
        image_path = self.dataset / row["image"]
        relative = Path(row["image"])
        label_path = self.dataset / "labels" / row["split"] / f"{relative.stem}.txt"
        return image_path, label_path

    def load(self):
        image_path, label_path = self.paths()
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise RuntimeError(f"Goruntu acilamadi: {image_path}")
        height, width = self.image.shape[:2]
        self.boxes = load_boxes(label_path, width, height)
        self.scale = min(
            self.args.max_width / width,
            self.args.max_height / height,
            1.0,
        )
        self.drag_start = None
        self.drag_current = None

    def save(self, reviewed=True):
        _image_path, label_path = self.paths()
        height, width = self.image.shape[:2]
        save_boxes(label_path, self.boxes, width, height)
        if reviewed:
            row = self.current_row()
            row["reviewed"] = "1"
            row["prelabel_count"] = str(len(self.boxes))
            if "label_sha256" in row:
                row["label_sha256"] = hashlib.sha256(label_path.read_bytes()).hexdigest()
            save_manifest(self.dataset / "manifest.csv", self.rows)

    def mouse(self, event, x, y, _flags, _parameter):
        point = (x / self.scale, y / self.scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = point
            self.drag_current = point
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = point
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x1, y1 = self.drag_start
            x2, y2 = point
            if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                self.boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            self.drag_start = None
            self.drag_current = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            containing = []
            for index, (x1, y1, x2, y2) in enumerate(self.boxes):
                if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                    containing.append(((x2 - x1) * (y2 - y1), index))
            if containing:
                _area, index = min(containing)
                self.boxes.pop(index)

    def render(self):
        canvas = self.image.copy()
        for index, (x1, y1, x2, y2) in enumerate(self.boxes):
            cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (0, 220, 0), 2)
            cv2.putText(
                canvas,
                str(index + 1),
                (round(x1), max(20, round(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2,
            )
        if self.drag_start is not None and self.drag_current is not None:
            cv2.rectangle(
                canvas,
                tuple(round(value) for value in self.drag_start),
                tuple(round(value) for value in self.drag_current),
                (0, 255, 255),
                2,
            )
        if self.scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale)
        row = self.current_row()
        status = (
            f"{self.position + 1}/{len(self.indices)} | {row['camera']} | "
            f"t={float(row['time_seconds']):.1f}s | kutu={len(self.boxes)} | "
            f"reviewed={row['reviewed']}"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(canvas, status, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(
            canvas,
            "Sol surukle: ekle | Sag tik: sil | N/Sag: kaydet+ileri | P/Sol: geri | Q: cik",
            (15, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        return canvas

    def run(self):
        if not self.indices:
            raise RuntimeError("Secilen split icin goruntu yok.")
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)
        self.load()
        while True:
            cv2.imshow(self.window, self.render())
            key = cv2.waitKeyEx(30)
            if key in (ord("q"), 27):
                self.save(reviewed=False)
                break
            if key in (ord("n"), ord("s"), 13, 63235, 2555904):
                self.save(reviewed=True)
                if self.position < len(self.indices) - 1:
                    self.position += 1
                    self.load()
            elif key in (ord("p"), 63234, 2424832):
                self.save(reviewed=False)
                if self.position > 0:
                    self.position -= 1
                    self.load()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    manifest_path = args.dataset / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    rows, indices = load_manifest(manifest_path, args.split)
    Reviewer(args.dataset, rows, indices, args).run()


if __name__ == "__main__":
    main()
