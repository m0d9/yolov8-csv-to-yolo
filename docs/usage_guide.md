# Usage Guide

A step-by-step guide to using the YOLOv8 CSV-to-YOLO pipeline for object detection.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Preparing Your Data](#3-preparing-your-data)
4. [Converting Annotations](#4-converting-annotations)
5. [Training](#5-training)
6. [Evaluation](#6-evaluation)
7. [Inference](#7-inference)
8. [Exporting Models](#8-exporting-models)
9. [Using as a Python Package](#9-using-as-a-python-package)
10. [Google Colab Workflow](#10-google-colab-workflow)

---

## 1. Prerequisites

| Requirement | Minimum Version |
|-------------|-----------------|
| Python | 3.8+ |
| pip | 21.0+ |
| GPU (optional) | CUDA-capable NVIDIA GPU recommended |
| RAM | 8 GB+ (16 GB recommended for larger datasets) |

---

## 2. Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-username/yolov8-csv-to-yolo.git
cd yolov8-csv-to-yolo

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as a package (includes src/ as importable library)
pip install -e .
```

### Quick Install

```bash
pip install -r requirements.txt
```

---

## 3. Preparing Your Data

Organize your data in the following structure:

```
data/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   ├── img003.png
│   └── ...
└── annotations.csv
```

Your CSV file should follow one of the three supported formats described in [`dataset_format.md`](dataset_format.md). The format is detected automatically.

### Quick Validation

Before running the full pipeline, you can verify your CSV:

```python
import pandas as pd
from yolov8_pipeline import detect_csv_format

df = pd.read_csv("data/annotations.csv")
fmt = detect_csv_format(df)
print(f"Detected format: {fmt}")
print(f"Rows: {len(df)}, Unique images: {df.iloc[:, 0].nunique()}")
print(f"Classes: {df.iloc[:, -1].unique()}")
```

---

## 4. Converting Annotations

### Using the CLI Script

```bash
# With default config
python scripts/convert_csv_to_yolo.py --config configs/default.yaml

# With custom paths
python scripts/convert_csv_to_yolo.py \
    --images-dir /path/to/images \
    --csv-path /path/to/annotations.csv \
    --output-dir /path/to/dataset \
    --split-ratio 0.85 \
    --seed 123
```

### Using the Python API

```python
from yolov8_pipeline import create_yolo_dataset

yaml_path = create_yolo_dataset(
    images_dir="data/images",
    csv_path="data/annotations.csv",
    output_dir="dataset",
    split_ratio=0.8,
    seed=42,
    allow_background=True,
)
print(f"Dataset created! data.yaml: {yaml_path}")
```

### Output Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   └── val/
│       ├── img050.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   └── ...
│   └── val/
│       ├── img050.txt
│       └── ...
└── data.yaml
```

---

## 5. Training

### Using the CLI Script

```bash
# Default config
python scripts/train.py --config configs/default.yaml

# With overrides
python scripts/train.py \
    --data dataset/data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 32 \
    --imgsz 640 \
    --device 0
```

### Model Size Options

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n.pt` | Nano | Fastest | Lower |
| `yolov8s.pt` | Small | Fast | Good |
| `yolov8m.pt` | Medium | Moderate | Better |
| `yolov8l.pt` | Large | Slower | High |
| `yolov8x.pt` | Extra-Large | Slowest | Highest |

### Training Tips

- Start with `yolov8n.pt` for quick experiments
- Use `yolov8s.pt` or `yolov8m.pt` for production
- Increase `epochs` if metrics are still improving at the end
- Reduce `batch` if you encounter GPU memory errors
- Use `imgsz=640` as a good default; increase to 1280 for small objects

---

## 6. Evaluation

After training, evaluate on the validation set:

```python
from ultralytics import YOLO

model = YOLO("runs/train/weights/best.pt")
metrics = model.val(data="dataset/data.yaml")

print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
```

### Understanding Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **mAP50** | Mean AP at IoU 0.50 | > 0.70 |
| **mAP50-95** | Mean AP at IoU 0.50–0.95 | > 0.50 |
| **Precision** | Fraction of correct detections | > 0.80 |
| **Recall** | Fraction of objects found | > 0.70 |

---

## 7. Inference

### CLI Script

```bash
# Single image
python scripts/predict.py --weights runs/train/weights/best.pt --source image.jpg

# Directory of images
python scripts/predict.py --weights runs/train/weights/best.pt --source images/

# With custom confidence
python scripts/predict.py --weights runs/train/weights/best.pt --source images/ --conf 0.5
```

### Python API

```python
from ultralytics import YOLO

model = YOLO("runs/train/weights/best.pt")
results = model.predict("image.jpg", conf=0.25)

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"Class: {cls_id}, Conf: {confidence:.2f}, Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

---

## 8. Exporting Models

```python
from ultralytics import YOLO

model = YOLO("runs/train/weights/best.pt")

# ONNX (cross-platform)
model.export(format="onnx")

# TorchScript (C++ deployment)
model.export(format="torchscript")

# TensorFlow Lite (mobile)
model.export(format="tflite")

# TensorRT (NVIDIA GPU optimization)
model.export(format="engine")
```

### Export Format Comparison

| Format | Use Case | Requires |
|--------|----------|----------|
| ONNX | Cross-platform inference | `onnx`, `onnxruntime` |
| TorchScript | C++ / production | PyTorch |
| TFLite | Mobile / edge devices | TensorFlow |
| TensorRT | NVIDIA GPU optimization | TensorRT SDK |

---

## 9. Using as a Python Package

The `src/yolov8_pipeline/` directory is an installable Python package:

```python
from yolov8_pipeline import (
    detect_csv_format,
    normalize_columns,
    build_class_mapping,
    convert_to_yolo,
    create_yolo_dataset,
    visualize_yolo_labels,
)
```

### Available Functions

| Function | Description |
|----------|-------------|
| `detect_csv_format(df)` | Auto-detect CSV format (returns 'A', 'B', or 'C') |
| `normalize_columns(df, fmt)` | Standardize column names |
| `build_class_mapping(series)` | Create class-to-ID mapping |
| `convert_to_yolo(row, fmt, w, h)` | Convert one box to YOLO format |
| `create_yolo_dataset(...)` | Full pipeline: CSV → YOLO dataset |
| `visualize_yolo_labels(...)` | Draw boxes on an image |

---

## 10. Google Colab Workflow

The easiest way to use this pipeline is through the included Colab notebook:

1. Open `notebooks/YOLOv8_Professional_Pipeline.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Upload your data or mount Google Drive
4. Edit the **Configuration** cell
5. Run all cells

The notebook handles everything: installation, conversion, training, evaluation, export, and repository generation.
