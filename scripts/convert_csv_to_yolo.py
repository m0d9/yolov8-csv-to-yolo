#!/usr/bin/env python3
"""Convert CSV annotations to YOLO format.

Usage:
    python scripts/convert_csv_to_yolo.py --config configs/default.yaml
    python scripts/convert_csv_to_yolo.py --images-dir data/images --csv-path data/annotations.csv
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add project src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yolov8_pipeline.converter import create_yolo_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV annotations to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_csv_to_yolo.py --config configs/default.yaml
  python scripts/convert_csv_to_yolo.py --images-dir data/images --csv-path data/ann.csv
  python scripts/convert_csv_to_yolo.py --config configs/default.yaml --split-ratio 0.9
        """,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file (default: configs/default.yaml)")
    parser.add_argument("--images-dir", type=str, help="Override images directory")
    parser.add_argument("--csv-path", type=str, help="Override CSV annotation path")
    parser.add_argument("--output-dir", type=str, help="Override dataset output directory")
    parser.add_argument("--split-ratio", type=float, help="Train/val split ratio (e.g. 0.8)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--no-background", action="store_true",
                        help="Do not include unlabeled images as background")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    images_dir = args.images_dir or config["paths"]["images_dir"]
    csv_path = args.csv_path or config["paths"]["csv_path"]
    output_dir = args.output_dir or config["paths"].get("dataset_dir", "dataset")
    split_ratio = args.split_ratio or config["data"]["split_ratio"]
    seed = args.seed or config["data"].get("seed", 42)
    allow_bg = not args.no_background and config["data"].get("allow_background", True)

    print("=" * 60)
    print("  CSV → YOLO Conversion")
    print("=" * 60)
    print(f"  Images dir:  {images_dir}")
    print(f"  CSV path:    {csv_path}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Split ratio: {split_ratio}")
    print(f"  Seed:        {seed}")
    print(f"  Background:  {allow_bg}")
    print("=" * 60)

    yaml_path = create_yolo_dataset(
        images_dir=images_dir,
        csv_path=csv_path,
        output_dir=output_dir,
        split_ratio=split_ratio,
        seed=seed,
        allow_background=allow_bg,
    )

    print("=" * 60)
    print(f"  Done! data.yaml → {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
