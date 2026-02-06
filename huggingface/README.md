---
license: mit
tags:
  - object-detection
  - yolov8
  - ultralytics
  - computer-vision
  - pytorch
library_name: ultralytics
pipeline_tag: object-detection
---

# YOLOv8 Object Detection Model

## Model Description

This is a **YOLOv8** object detection model trained using the [Ultralytics](https://github.com/ultralytics/ultralytics) framework. The model was trained on a custom dataset with CSV-formatted annotations, converted to YOLO format using an automated pipeline.

### Model Details

- **Architecture**: YOLOv8 (configurable: nano / small / medium / large / x-large)
- **Input Size**: 640x640 (configurable)
- **Framework**: Ultralytics / PyTorch

## Training Details

| Parameter | Default Value |
|-----------|---------------|
| Model | yolov8n.pt |
| Image Size | 640 |
| Epochs | 50 |
| Batch Size | 16 |
| Seed | 42 |
| Train/Val Split | 0.8 / 0.2 |

> Update this table with your actual training parameters and metrics after training.

## Intended Use

This model is intended for object detection tasks on images similar to the training data. It can detect and localize objects with bounding boxes and class labels.

### Primary Use Cases

- Real-time object detection in images and video
- Batch processing of image datasets
- Integration into computer vision pipelines

### Limitations

- Performance may degrade on images significantly different from the training distribution
- The model is optimized for the specific classes it was trained on
- Real-time performance depends on hardware (GPU recommended)

## How to Use

### Python API

```python
from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")

# Run inference
results = model.predict("image.jpg", conf=0.25)

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"Class: {cls}, Confidence: {conf:.2f}, Box: {xyxy}")
```

### Command Line

```bash
yolo predict model=best.pt source=image.jpg conf=0.25
```

## Export Formats

| Format | File | Notes |
|--------|------|-------|
| PyTorch | `best.pt` | Default format |
| ONNX | `best.onnx` | Cross-platform inference |
| TorchScript | `best.torchscript` | C++ deployment |
| TensorFlow Lite | `best.tflite` | Mobile deployment |
| TensorRT | `best.engine` | NVIDIA GPU optimization |

## Metrics

> Fill in after training:

| Metric | Value |
|--------|-------|
| mAP50 | — |
| mAP50-95 | — |
| Precision | — |
| Recall | — |

## Citation

```bibtex
@software{ultralytics_yolov8,
  author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  title = {Ultralytics YOLOv8},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

## License

This model is released under the MIT license. See the LICENSE file for details.

> **Note**: The Ultralytics YOLOv8 framework is licensed under AGPL-3.0. If you use their pretrained weights, please review the [Ultralytics license terms](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
