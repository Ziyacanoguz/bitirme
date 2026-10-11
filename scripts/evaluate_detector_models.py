"""Comparable detector evaluation with a machine-readable result record."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--model", action="append", required=True, help="name=/absolute/model.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--image-size", type=int, default=960)
    args = p.parse_args()
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in args.model:
        name, path = item.split("=", 1)
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        model = YOLO(path)
        metrics = model.val(data=str(args.data.resolve()), split=args.split, imgsz=args.image_size,
                            batch=4, device=args.device, workers=0, verbose=False,
                            project=str(args.output_dir.resolve()), name=name)
        result = dict(name=name, model=str(Path(path).resolve()), data=str(args.data.resolve()),
                      split=args.split, imgsz=args.image_size, precision=float(metrics.box.mp),
                      recall=float(metrics.box.mr), map50=float(metrics.box.map50),
                      map50_95=float(metrics.box.map))
        results.append(result)
        (args.output_dir / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
