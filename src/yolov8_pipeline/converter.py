"""YOLO format conversion utilities.

Converts CSV annotations to YOLO label files and creates the full dataset
directory structure expected by Ultralytics.
"""

import os
import random
import shutil
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm.auto import tqdm

from .parser import detect_csv_format, normalize_columns, build_class_mapping


def convert_to_yolo(row: dict, fmt: str, img_w: int, img_h: int):
    """
    Convert a single annotation row to YOLO format.

    Args:
        row: Dictionary-like row with annotation values.
        fmt: Format identifier ('A', 'B', or 'C').
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Tuple ``(x_center, y_center, width, height)`` normalized to [0, 1],
        or ``None`` if the box is invalid.
    """
    try:
        if fmt == "A":
            xmin, ymin = float(row["xmin"]), float(row["ymin"])
            xmax, ymax = float(row["xmax"]), float(row["ymax"])
        elif fmt == "B":
            xmin, ymin = float(row["xmin"]), float(row["ymin"])
            xmax = xmin + float(row["width"])
            ymax = ymin + float(row["height"])
        elif fmt == "C":
            xc = max(0.0, min(1.0, float(row["x_center"])))
            yc = max(0.0, min(1.0, float(row["y_center"])))
            w = max(0.0, min(1.0, float(row["width"])))
            h = max(0.0, min(1.0, float(row["height"])))
            return (xc, yc, w, h) if w > 0 and h > 0 else None
        else:
            return None

        # Clamp to image bounds
        xmin = max(0, min(img_w, xmin))
        ymin = max(0, min(img_h, ymin))
        xmax = max(0, min(img_w, xmax))
        ymax = max(0, min(img_h, ymax))

        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 0 or bh <= 0:
            return None

        return (
            (xmin + xmax) / 2.0 / img_w,
            (ymin + ymax) / 2.0 / img_h,
            bw / img_w,
            bh / img_h,
        )
    except (ValueError, TypeError):
        return None


def create_yolo_dataset(
    images_dir: str,
    csv_path: str,
    output_dir: str,
    split_ratio: float = 0.8,
    seed: int = 42,
    allow_background: bool = True,
) -> str:
    """
    Create a complete YOLO dataset from images and CSV annotations.

    This function:
      1. Reads and validates the CSV file.
      2. Auto-detects the annotation format.
      3. Builds a class mapping.
      4. Splits images into train/val sets deterministically.
      5. Copies images and writes YOLO label files.
      6. Generates ``data.yaml``.

    Args:
        images_dir: Path to the directory containing images.
        csv_path: Path to the CSV annotation file.
        output_dir: Path where the YOLO dataset will be created.
        split_ratio: Fraction of images used for training (default 0.8).
        seed: Random seed for reproducible splitting.
        allow_background: If True, include unlabeled images as background.

    Returns:
        Absolute path to the generated ``data.yaml`` file.
    """
    # ── Read & validate CSV ──────────────────────────────────────
    df = pd.read_csv(csv_path)
    fmt = detect_csv_format(df)
    print(f"  Detected format: {fmt}")
    df = normalize_columns(df, fmt)

    # Coerce numeric columns
    for col in df.columns:
        if col not in ("filename", "class"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # ── Class mapping ────────────────────────────────────────────
    class_mapping, class_names = build_class_mapping(df["class"])
    print(f"  Classes ({len(class_names)}): {class_names}")

    # ── Discover available images ────────────────────────────────
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    available = {}
    for f in os.listdir(images_dir):
        if Path(f).suffix.lower() in image_extensions:
            available[f] = os.path.join(images_dir, f)

    df = df[df["filename"].isin(available)]
    assert len(df) > 0, "No matching images found between CSV and images directory."

    # ── Create directory structure ───────────────────────────────
    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # ── Deterministic split ──────────────────────────────────────
    all_images = sorted(df["filename"].unique())
    random.seed(seed)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * split_ratio)
    train_set = set(all_images[:split_idx])
    val_set = set(all_images[split_idx:])

    if allow_background:
        bg = set(available.keys()) - set(df["filename"].unique())
        bg_list = sorted(bg)
        random.shuffle(bg_list)
        bg_split = int(len(bg_list) * split_ratio)
        train_set |= set(bg_list[:bg_split])
        val_set |= set(bg_list[bg_split:])

    print(f"  Train: {len(train_set)} images | Val: {len(val_set)} images")

    # ── Convert & copy ───────────────────────────────────────────
    total_boxes = 0
    for img_name in tqdm(sorted(train_set | val_set), desc="  Converting"):
        split = "train" if img_name in train_set else "val"
        src = available.get(img_name)
        if not src:
            continue

        shutil.copy2(src, os.path.join(output_dir, "images", split, img_name))
        img_w, img_h = Image.open(src).size

        img_df = df[df["filename"] == img_name]
        lines = []
        for _, row in img_df.iterrows():
            result = convert_to_yolo(row, fmt, img_w, img_h)
            if result:
                cls_id = class_mapping[str(row["class"])]
                xc, yc, w, h = result
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                total_boxes += 1

        label_name = Path(img_name).stem + ".txt"
        with open(os.path.join(output_dir, "labels", split, label_name), "w") as f:
            f.write("\n".join(lines))

    print(f"  Total boxes written: {total_boxes}")

    # ── Generate data.yaml ───────────────────────────────────────
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    return yaml_path
