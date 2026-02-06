#!/usr/bin/env python3
"""Quick Start — Minimal end-to-end example.

This script demonstrates the full pipeline in ~30 lines:
  1. Convert CSV annotations to YOLO format
  2. Train a YOLOv8 model
  3. Run inference

Usage:
    python examples/quick_start.py

Prerequisites:
    - Place images in data/images/
    - Place annotations in data/annotations.csv
    - pip install -r requirements.txt
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yolov8_pipeline import create_yolo_dataset


def main():
    # ── Step 1: Convert CSV → YOLO dataset ───────────────────────
    print("Step 1: Converting CSV annotations to YOLO format...")
    yaml_path = create_yolo_dataset(
        images_dir="data/images",
        csv_path="data/annotations.csv",
        output_dir="dataset",
        split_ratio=0.8,
        seed=42,
    )
    print(f"  data.yaml: {yaml_path}\n")

    # ── Step 2: Train YOLOv8 ─────────────────────────────────────
    print("Step 2: Training YOLOv8...")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.train(data=yaml_path, imgsz=640, epochs=50, batch=16)
    print("  Training complete!\n")

    # ── Step 3: Evaluate ─────────────────────────────────────────
    print("Step 3: Evaluating...")
    best = YOLO("runs/train/weights/best.pt")
    metrics = best.val(data=yaml_path)
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}\n")

    # ── Step 4: Inference ────────────────────────────────────────
    print("Step 4: Running inference on validation images...")
    results = best.predict(source="dataset/images/val", conf=0.25, save=True)
    print(f"  Processed {len(results)} images. Results saved to runs/predict/")


if __name__ == "__main__":
    main()
