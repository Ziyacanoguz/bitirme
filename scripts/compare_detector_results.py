"""Compare detector metrics before and after training and output markdown summary."""
import json
import sys
from pathlib import Path


def compare(baseline_json: Path, trained_json: Path, output_md: Path):
    b_data = json.loads(baseline_json.read_text(encoding="utf-8"))
    t_data = json.loads(trained_json.read_text(encoding="utf-8"))

    b_metric = b_data[0]
    t_metric = t_data[0]

    prec_b, prec_t = b_metric["precision"], t_metric["precision"]
    rec_b, rec_t = b_metric["recall"], t_metric["recall"]
    m50_b, m50_t = b_metric["map50"], t_metric["map50"]
    m95_b, m95_t = b_metric["map50_95"], t_metric["map50_95"]

    prec_diff = (prec_t - prec_b) * 100
    rec_diff = (rec_t - rec_b) * 100
    m50_diff = (m50_t - m50_b) * 100
    m95_diff = (m95_t - m95_b) * 100

    report = f"""# Player Detector Improvement Report (Dataset 2 & 3 Integration)

## 📊 Evaluation Summary

| Metric | Baseline (Before Training) | Fine-Tuned (After Training) | Difference / Gain |
| :--- | :---: | :---: | :---: |
| **Precision** | {prec_b*100:.2f}% | {prec_t*100:.2f}% | {prec_diff:+.2f}% |
| **Recall (Tespit Oranı)** | **{rec_b*100:.2f}%** | **{rec_t*100:.2f}%** | **{rec_diff:+.2f}%** |
| **mAP50** | {m50_b*100:.2f}% | {m50_t*100:.2f}% | {m50_diff:+.2f}% |
| **mAP50-95** | {m95_b*100:.2f}% | {m95_t*100:.2f}% | {m95_diff:+.2f}% |

## 🚀 Key Takeaways & System Performance
- **Recall (Player Detection Rate)**: Changed by **{rec_diff:+.2f}%**.
- **mAP50**: Changed by **{m50_diff:+.2f}%**.
- **Model Weight**: Saved at `{t_metric['model']}`
"""
    output_md.write_text(report, encoding="utf-8")
    print("Report generated successfully:")
    print(report)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        compare(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        print("Usage: python compare_detector_results.py <baseline.json> <trained.json> <output.md>")
