"""Fine-tune YOLO11s on halisaha_merged_v4 at 1280px resolution."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/Users/ziyacan/Desktop/Hali_Saha_Project/project_v3")
DATA_YAML = ROOT / "datasets/halisaha_merged_v4/dataset.yaml"
BASE_MODEL = ROOT / "outputs/training/yolo11s_merged_v3_1280/weights/best.pt"

print(f"Loading champion checkpoint: {BASE_MODEL}")
model = YOLO(str(BASE_MODEL))

print(f"Starting training on {DATA_YAML} (2,807 images, 41,636 boxes)...")
results = model.train(
    data=str(DATA_YAML),
    epochs=30,
    imgsz=1280,
    batch=4,
    device="mps",
    lr0=0.003,
    lrf=0.01,
    patience=10,
    project=str(ROOT / "outputs/training"),
    name="yolo11s_merged_v4_1280",
    exist_ok=True,
    verbose=True,
)
print("Training finished successfully!")
