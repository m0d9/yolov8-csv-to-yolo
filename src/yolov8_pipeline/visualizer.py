"""Visualization utilities for YOLO datasets.

Provides functions to draw bounding boxes on images using YOLO label files.
"""

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def visualize_yolo_labels(
    image_path: str,
    label_path: str,
    class_names: list,
    ax=None,
    show: bool = False,
):
    """
    Draw YOLO bounding boxes on an image.

    Args:
        image_path: Path to the image file.
        label_path: Path to the YOLO label ``.txt`` file.
        class_names: List of class names ordered by ID.
        ax: Matplotlib axes object (created automatically if ``None``).
        show: If ``True``, call ``plt.show()`` after drawing.

    Returns:
        The matplotlib axes with the visualization.
    """
    img = Image.open(image_path)
    w, h = img.size

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.imshow(img)
    ax.set_title(os.path.basename(image_path), fontsize=9)
    ax.axis("off")

    colors = plt.cm.Set3(np.linspace(0, 1, max(len(class_names), 1)))

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, bw, bh = [float(x) for x in parts[1:]]
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                rect_w = bw * w
                rect_h = bh * h
                color = colors[cls_id % len(colors)]
                rect = patches.Rectangle(
                    (x1, y1),
                    rect_w,
                    rect_h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)
                label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                ax.text(
                    x1,
                    y1 - 4,
                    label,
                    fontsize=8,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
                )

    if show:
        plt.tight_layout()
        plt.show()

    return ax
