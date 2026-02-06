# 🚀 YOLOv8 CSV-to-YOLO — Professional Training Pipeline

A professional, end-to-end pipeline for training **YOLOv8** object detection models using **CSV-formatted annotations**. Supports multiple annotation formats with automatic detection, deterministic train/val splitting, model evaluation, multi-format export, and a complete repository structure ready for **GitHub** and **Hugging Face**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.2-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-orange)

---

## 📋 Table of Contents

- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Data Format](#-data-format)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Export](#-export)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Documentation](#-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

- **Auto-detect CSV format** — Supports pixel boxes (`xmin/ymin/xmax/ymax`), width/height format (`xmin/ymin/w/h`), and normalized YOLO format
- **Robust data validation** — Checks for missing images, invalid boxes, NaN values, and out-of-bound coordinates with automatic clamping
- **Deterministic splitting** — Reproducible train/val splits with configurable seed
- **Background image support** — Optionally include unlabeled images as negative samples
- **Full YOLOv8 training** — Powered by Ultralytics with metrics logging and visualization
- **Multi-format export** — ONNX, TorchScript, and PyTorch weights
- **Google Colab notebook** — Run the entire pipeline in one click
- **Reusable Python package** — `src/yolov8_pipeline/` installable as a library
- **CLI scripts** — Standalone conversion, training, and prediction scripts
- **Hugging Face ready** — Model card and folder layout included

---

## 📁 Repository Structure

```
yolov8-csv-to-yolo/
├── README.md                        # This file
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # MIT License
├── requirements.txt                  # Pinned Python dependencies
├── pyproject.toml                    # Project metadata & packaging
├── Makefile                          # Common commands (install, train, etc.)
├── .gitignore                        # Git ignore rules
│
├── notebooks/                        # Jupyter / Colab notebooks
│   └── YOLOv8_Professional_Pipeline.ipynb
│
├── docs/                             # Documentation
│   ├── dataset_format.md             # Supported CSV annotation formats
│   └── usage_guide.md               # Step-by-step usage guide
│
├── examples/                         # Example data & quick-start scripts
│   ├── sample_annotations.csv        # Tiny sample CSV (Format A)
│   ├── sample_annotations_b.csv      # Tiny sample CSV (Format B)
│   ├── sample_annotations_c.csv      # Tiny sample CSV (Format C)
│   └── quick_start.py               # Minimal end-to-end example
│
├── src/yolov8_pipeline/             # Reusable Python package
│   ├── __init__.py
│   ├── parser.py                     # CSV parsing & format detection
│   ├── converter.py                  # YOLO format conversion
│   └── visualizer.py                # Visualization utilities
│
├── scripts/                          # Standalone CLI tools
│   ├── convert_csv_to_yolo.py        # CSV → YOLO conversion
│   ├── train.py                      # Model training
│   └── predict.py                    # Inference / prediction
│
├── configs/                          # Configuration files
│   └── default.yaml                  # Default training config
│
├── artifacts/                        # Trained model outputs (populated after training)
│   └── .gitkeep
│
├── assets/                           # Images for README / docs
│   └── .gitkeep
│
└── huggingface/                      # Hugging Face model card
    └── README.md
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Open `notebooks/YOLOv8_Professional_Pipeline.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Edit the **Configuration** cell with your data paths
4. Run all cells sequentially

### Option 2: Command Line

```bash
# Clone the repository
git clone https://github.com/m0d9/yolov8-csv-to-yolo.git
cd yolov8-csv-to-yolo

# Install dependencies
pip install -r requirements.txt

# Convert CSV annotations to YOLO format
python scripts/convert_csv_to_yolo.py --config configs/default.yaml

# Train the model
python scripts/train.py --config configs/default.yaml

# Run predictions
python scripts/predict.py --weights artifacts/best.pt --source path/to/images
```

### Option 3: Using Makefile

```bash
make install    # Install dependencies
make convert    # Convert CSV → YOLO
make train      # Train the model
make predict    # Run inference
```

---

## 📊 Data Format

Place your data as follows:

```
data/
├── images/              # Image files (jpg, png, bmp, tiff, webp)
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── annotations.csv      # Bounding box annotations
```

### Supported CSV Formats

| Format | Columns | Example |
|--------|---------|---------|
| **A** — Pixel corners | `filename, xmin, ymin, xmax, ymax, class` | `img001.jpg, 100, 50, 300, 200, cat` |
| **B** — Pixel top-left + size | `filename, xmin, ymin, width, height, class` | `img001.jpg, 100, 50, 200, 150, cat` |
| **C** — Normalized YOLO | `filename, x_center, y_center, width, height, class, normalized` | `img001.jpg, 0.31, 0.26, 0.31, 0.31, cat, true` |

> **The format is auto-detected** from column names. See [`docs/dataset_format.md`](docs/dataset_format.md) for full details including accepted column name variants.

---

## 🏋️ Training

### Using Config File

```bash
python scripts/train.py --config configs/default.yaml
```

### With Command-Line Overrides

```bash
python scripts/train.py \
    --data dataset/data.yaml \
    --model yolov8s.pt \
    --imgsz 640 \
    --epochs 100 \
    --batch 32
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `yolov8n.pt` | Model size: `n` / `s` / `m` / `l` / `x` |
| `imgsz` | `640` | Input image size |
| `epochs` | `50` | Number of training epochs |
| `batch` | `16` | Batch size |
| `seed` | `42` | Random seed for reproducibility |
| `device` | `auto` | Device (`cuda` / `cpu` / `0`) |

---

## 📈 Evaluation

```python
from ultralytics import YOLO

model = YOLO("artifacts/best.pt")
metrics = model.val(data="artifacts/data.yaml")

print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
```

---

## 🔍 Inference

```bash
# Single image
python scripts/predict.py --weights artifacts/best.pt --source image.jpg

# Directory of images
python scripts/predict.py --weights artifacts/best.pt --source images/ --conf 0.5
```

### Python API

```python
from ultralytics import YOLO

model = YOLO("artifacts/best.pt")
results = model.predict("image.jpg", conf=0.25)

for r in results:
    for box in r.boxes:
        print(f"Class: {box.cls}, Conf: {box.conf:.2f}, Box: {box.xyxy}")
```

---

## 📦 Export

```python
from ultralytics import YOLO
model = YOLO("artifacts/best.pt")

model.export(format="onnx")         # ONNX
model.export(format="torchscript")  # TorchScript
model.export(format="tflite")       # TensorFlow Lite
model.export(format="engine")       # TensorRT
```

---

## ⚙️ Configuration

All parameters are centralized in `configs/default.yaml`:

```yaml
paths:
  images_dir: data/images
  csv_path: data/annotations.csv
  output_dir: artifacts

training:
  model: yolov8n.pt
  imgsz: 640
  epochs: 50
  batch: 16
  seed: 42

data:
  split_ratio: 0.8
  allow_background: true
```

---

## 💡 Examples

See the [`examples/`](examples/) directory for:

- **Sample CSV files** in all three supported formats
- **`quick_start.py`** — a minimal script that converts data and trains a model in ~20 lines

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [`docs/dataset_format.md`](docs/dataset_format.md) | Detailed guide on all supported CSV formats and column variants |
| [`docs/usage_guide.md`](docs/usage_guide.md) | Step-by-step guide: installation, data prep, training, evaluation, export |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute to this project |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `No matching images found` | Verify `images_dir` path and that CSV filenames match actual image files |
| `Cannot detect CSV format` | Ensure CSV has proper column headers (see [Data Format](#-data-format)) |
| `CUDA out of memory` | Reduce `batch` size or use a smaller model (`yolov8n`) |
| `Invalid bounding boxes` | Check that coordinates are within image dimensions |
| `Low mAP scores` | Increase epochs, try a larger model, or review annotation quality |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |

---

## 🤝 Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on how to submit issues, feature requests, and pull requests.

---

## 📄 License

This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

> **Note:** The Ultralytics YOLOv8 framework is licensed under AGPL-3.0. If you use their pretrained weights, please review the [Ultralytics license terms](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — State-of-the-art object detection framework
- [YOLO Documentation](https://docs.ultralytics.com/) — Comprehensive guides and API reference
