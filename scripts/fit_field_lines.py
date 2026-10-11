"""Fit and audit candidate line-based calibrations without overwriting old files."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from halisaha.line_calibration import fit_line_calibration, line_errors, project_line_calibration, unproject_line_calibration
from halisaha.calibration import load_multipoint_calibration, project_points_multipoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    width, height = annotations["image_width"], annotations["image_height"]
    args.output_dir.mkdir(parents=True)
    report, checks = {}, {}
    for camera, data in annotations["cameras"].items():
        cv_fit = fit_line_calibration(data["lines"], width, height, sample_parity=0)
        heldout = line_errors(data["lines"], lambda pts: project_line_calibration(pts, cv_fit), parity=1)
        calibration = fit_line_calibration(data["lines"], width, height)
        calibration.update(annotation_source=str(args.annotations.resolve()), camera=camera,
                           validation_note="Alternating points on manually digitized lines; not an independent camera calibration target")
        old = load_multipoint_calibration(ROOT / "configs" / f"calibration_{camera}_7pt.json")
        old_project = lambda pts: project_points_multipoint(pts, old, width, height)
        old_errors = line_errors(data["lines"], old_project)
        fit_errors = line_errors(data["lines"], lambda pts: project_line_calibration(pts, calibration))
        if camera == "camera2":
            def flipped(pts):
                points = old_project(pts)
                points[:, 0] = 1 - points[:, 0]
                return points
            flipped_errors = line_errors(data["lines"], flipped)
        else:
            flipped_errors = old_errors
        report[camera] = dict(old_median=float(np.median(old_errors)), old_p95=float(np.quantile(old_errors,.95)),
                              old_orientation_fixed_median=float(np.median(flipped_errors)),
                              fit_median=float(np.median(fit_errors)), fit_p95=float(np.quantile(fit_errors,.95)),
                              heldout_median=float(np.median(heldout)), heldout_p95=float(np.quantile(heldout,.95)),
                              division_k=calibration["division_k"])
        points = project_line_calibration(list(data["check_points"].values()), calibration)
        checks[camera] = {name: point.tolist() for name, point in zip(data["check_points"], points)}
        (args.output_dir / f"calibration_{camera}.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        image = cv2.imread(str(args.images_dir / f"{camera}_1800.png"))
        if image is None:
            raise RuntimeError("Missing audit frame")
        for line in data["lines"]:
            pts = np.asarray(line["points"], np.int32)
            cv2.polylines(image, [pts], False, (40, 220, 255), 2, cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(image, (x, y), 4, (40, 220, 255), -1)
            cv2.putText(image, line["name"], tuple(pts[len(pts)//2]), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,0,0), 3)
            cv2.putText(image, line["name"], tuple(pts[len(pts)//2]), cv2.FONT_HERSHEY_SIMPLEX, .65, (40,220,255), 1)
        cv2.imwrite(str(args.output_dir / f"{camera}_line_annotations.png"), image)
        # Analytic inverse of the SAME forward transform, not a separate TPS fit.
        yy, xx = np.mgrid[0:600, 0:1000]
        source = unproject_line_calibration(np.column_stack((xx.ravel()/999, yy.ravel()/599)), calibration)
        clean_image = cv2.imread(str(args.images_dir / f"{camera}_1800.png"))
        warp = cv2.remap(clean_image, source[:,0].reshape(600,1000).astype(np.float32),
                         source[:,1].reshape(600,1000).astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(args.output_dir / f"{camera}_ground_warp.png"), warp)
    report["check_points"] = checks
    report["shared_circle_distance"] = {name: float(np.linalg.norm(np.array(checks["camera1"][name])-checks["camera2"][name]))
                                         for name in ("circle_spectator", "circle_camera")}
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
