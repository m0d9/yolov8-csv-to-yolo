# Dataset Format Guide

## Overview

This pipeline accepts image datasets with bounding box annotations provided in CSV format. The CSV format is **automatically detected** based on column names, so you do not need to specify it manually.

---

## Image Requirements

- **Supported formats**: JPG, JPEG, PNG, BMP, TIFF, WebP
- **Location**: Place all images in a single directory (subdirectories are not scanned)
- **Naming**: The `filename` column in the CSV must match the image file basenames exactly

---

## CSV Annotation Formats

### Format A: Pixel Coordinates (Corners)

Bounding boxes defined by their top-left and bottom-right corners in pixel coordinates.

| Column | Type | Description |
|--------|------|-------------|
| `filename` | string | Image filename (e.g., `img001.jpg`) |
| `xmin` | float | Left edge of bounding box (pixels) |
| `ymin` | float | Top edge of bounding box (pixels) |
| `xmax` | float | Right edge of bounding box (pixels) |
| `ymax` | float | Bottom edge of bounding box (pixels) |
| `class` | string or int | Class label or numeric ID |

**Example:**

```csv
filename,xmin,ymin,xmax,ymax,class
img001.jpg,100,50,300,200,cat
img001.jpg,400,100,550,350,dog
img002.jpg,50,50,200,180,cat
```

---

### Format B: Pixel Coordinates (Top-Left + Size)

Bounding boxes defined by the top-left corner plus width and height in pixels.

| Column | Type | Description |
|--------|------|-------------|
| `filename` | string | Image filename |
| `xmin` | float | Left edge (pixels) |
| `ymin` | float | Top edge (pixels) |
| `width` | float | Box width (pixels) |
| `height` | float | Box height (pixels) |
| `class` | string or int | Class label or numeric ID |

**Example:**

```csv
filename,xmin,ymin,width,height,class
img001.jpg,100,50,200,150,cat
img001.jpg,400,100,150,250,dog
img002.jpg,50,50,150,130,cat
```

---

### Format C: Normalized YOLO Format

Bounding boxes already in YOLO-style normalized coordinates. The presence of a `normalized` column triggers this format.

| Column | Type | Description |
|--------|------|-------------|
| `filename` | string | Image filename |
| `x_center` | float | Center X (normalized 0–1) |
| `y_center` | float | Center Y (normalized 0–1) |
| `width` | float | Box width (normalized 0–1) |
| `height` | float | Box height (normalized 0–1) |
| `class` | string or int | Class label or numeric ID |
| `normalized` | any | Flag column — its presence triggers Format C detection |

**Example:**

```csv
filename,x_center,y_center,width,height,class,normalized
img001.jpg,0.3125,0.2604,0.3125,0.3125,cat,true
img001.jpg,0.7422,0.4688,0.2344,0.5208,dog,true
img002.jpg,0.1953,0.2396,0.2344,0.2708,cat,true
```

---

## Class Labels

The pipeline handles both string and integer class labels:

- **String labels**: Sorted alphabetically and mapped to sequential integer IDs starting from 0
- **Integer labels**: Validated for contiguous range `[0, N-1]`; remapped if gaps are found

The resulting class mapping is saved in `data.yaml` and `classes.json`.

---

## Multiple Objects Per Image

Multiple rows with the same `filename` represent multiple objects in that image. Each row becomes one line in the corresponding YOLO label `.txt` file.

---

## Column Name Flexibility

The parser recognizes common column name variants (case-insensitive):

| Standard Name | Also Accepted |
|---------------|---------------|
| `filename` | `file`, `image`, `image_name`, `img` |
| `xmin` | `x_min`, `x1`, `left` |
| `ymin` | `y_min`, `y1`, `top` |
| `xmax` | `x_max`, `x2`, `right` |
| `ymax` | `y_max`, `y2`, `bottom` |
| `width` | `w`, `box_width` |
| `height` | `h`, `box_height` |
| `x_center` | `cx`, `center_x` |
| `y_center` | `cy`, `center_y` |
| `class` | `label`, `class_name`, `category`, `cls` |

If column names cannot be matched, the parser falls back to **positional mapping** (first six columns in the expected order for the detected format).

---

## Data Validation

The pipeline automatically performs the following checks:

1. **Missing images**: Rows referencing images not found in the directory are dropped with a warning
2. **NaN / non-numeric values**: Rows with invalid coordinate values are dropped
3. **Out-of-bounds boxes**: Coordinates are clamped to image dimensions
4. **Zero-area boxes**: Boxes with zero or negative width/height are skipped
5. **Filename normalization**: Directory paths in filenames are stripped to basenames
