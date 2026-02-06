#!/usr/bin/env python3
"""Train a YOLOv8 model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --data dataset/data.yaml --model yolov8s.pt --epochs 100
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# Add project src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --data dataset/data.yaml --model yolov8s.pt --epochs 100
  python scripts/train.py --config configs/default.yaml --batch 32 --device 0
        """,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--data", type=str, help="Path to data.yaml (overrides config)")
    parser.add_argument("--model", type=str, help="Model weights (e.g. yolov8n.pt)")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--device", type=str, help="Device: '' (auto), '0' (GPU), 'cpu'")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    tc = config["training"]
    data_path = args.data or os.path.join(
        config["paths"].get("dataset_dir", "dataset"), "data.yaml"
    )
    model_name = args.model or tc["model"]
    imgsz = args.imgsz or tc["imgsz"]
    epochs = args.epochs or tc["epochs"]
    batch = args.batch or tc["batch"]
    device = args.device if args.device is not None else tc.get("device", "")
    seed = tc.get("seed", 42)

    print("=" * 60)
    print("  YOLOv8 Training")
    print("=" * 60)
    print(f"  Model:   {model_name}")
    print(f"  Data:    {data_path}")
    print(f"  ImgSz:   {imgsz}")
    print(f"  Epochs:  {epochs}")
    print(f"  Batch:   {batch}")
    print(f"  Device:  {device if device else 'auto'}")
    print(f"  Seed:    {seed}")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_name)
    results = model.train(
        data=data_path,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device if device else None,
        seed=seed,
        project="runs",
        name="train",
        exist_ok=True,
    )

    print()
    print("=" * 60)
    print("  Training complete!")
    print(f"  Best weights: runs/train/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
