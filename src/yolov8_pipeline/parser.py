"""CSV annotation parsing and format detection.

Supports three CSV formats:
  - Format A: pixel coordinates (xmin, ymin, xmax, ymax)
  - Format B: pixel coordinates (xmin, ymin, width, height)
  - Format C: normalized YOLO format (x_center, y_center, width, height)
"""

import os

import pandas as pd


def detect_csv_format(df: pd.DataFrame) -> str:
    """
    Auto-detect CSV annotation format from column names and values.

    Args:
        df: DataFrame loaded from the annotation CSV.

    Returns:
        One of 'A', 'B', or 'C'.

    Raises:
        ValueError: If the format cannot be determined.
    """
    cols_lower = [c.lower().strip() for c in df.columns]

    # Explicit normalized flag → Format C
    if "normalized" in cols_lower:
        return "C"

    # xmax / ymax present → Format A
    if "xmax" in cols_lower and "ymax" in cols_lower:
        return "A"

    # width / height present (but no xmax) → Format B
    if "width" in cols_lower and "height" in cols_lower:
        return "B"

    # Positional fallback: check if values look normalized
    if len(df.columns) >= 6:
        numeric_cols = df.iloc[:, 1:5]
        try:
            numeric_vals = numeric_cols.astype(float)
            if numeric_vals.max().max() <= 1.0 and numeric_vals.min().min() >= 0.0:
                return "C"
        except (ValueError, TypeError):
            pass
        return "A"

    raise ValueError(
        "Cannot detect CSV format. Expected columns:\n"
        "  Format A: filename, xmin, ymin, xmax, ymax, class\n"
        "  Format B: filename, xmin, ymin, width, height, class\n"
        "  Format C: filename, x_center, y_center, width, height, class, normalized"
    )


# Accepted column name variants for each standard name
_COLUMN_ALIASES = {
    "A": {
        "filename": ["filename", "file", "image", "image_name", "img"],
        "xmin": ["xmin", "x_min", "x1", "left"],
        "ymin": ["ymin", "y_min", "y1", "top"],
        "xmax": ["xmax", "x_max", "x2", "right"],
        "ymax": ["ymax", "y_max", "y2", "bottom"],
        "class": ["class", "label", "class_name", "category", "cls"],
    },
    "B": {
        "filename": ["filename", "file", "image", "image_name", "img"],
        "xmin": ["xmin", "x_min", "x1", "left", "x"],
        "ymin": ["ymin", "y_min", "y1", "top", "y"],
        "width": ["width", "w", "box_width"],
        "height": ["height", "h", "box_height"],
        "class": ["class", "label", "class_name", "category", "cls"],
    },
    "C": {
        "filename": ["filename", "file", "image", "image_name", "img"],
        "x_center": ["x_center", "cx", "center_x", "x"],
        "y_center": ["y_center", "cy", "center_y", "y"],
        "width": ["width", "w", "box_width"],
        "height": ["height", "h", "box_height"],
        "class": ["class", "label", "class_name", "category", "cls"],
    },
}

# Positional fallback column orders
_POSITIONAL_FALLBACK = {
    "A": ["filename", "xmin", "ymin", "xmax", "ymax", "class"],
    "B": ["filename", "xmin", "ymin", "width", "height", "class"],
    "C": ["filename", "x_center", "y_center", "width", "height", "class"],
}


def normalize_columns(df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """
    Rename CSV columns to standard names based on detected format.

    Also normalizes the ``filename`` column to basenames (strips directory paths).

    Args:
        df: Raw DataFrame.
        fmt: Detected format ('A', 'B', or 'C').

    Returns:
        DataFrame with standardized column names.
    """
    df = df.copy()
    cols = list(df.columns)
    cols_lower = {c.lower().strip(): c for c in cols}

    mapping = {}
    for target, candidates in _COLUMN_ALIASES[fmt].items():
        for c in candidates:
            if c in cols_lower:
                mapping[cols_lower[c]] = target
                break

    # Fall back to positional mapping if not enough columns matched
    if len(mapping) < 6:
        fallback = _POSITIONAL_FALLBACK[fmt]
        mapping = {cols[i]: fallback[i] for i in range(min(len(cols), len(fallback)))}

    df = df.rename(columns=mapping)
    df["filename"] = df["filename"].apply(lambda x: os.path.basename(str(x).strip()))
    return df


def build_class_mapping(classes_series: pd.Series) -> tuple:
    """
    Build a mapping from class label to integer ID.

    If labels are already contiguous integers starting from 0, they are kept as-is.
    Otherwise, string labels are sorted alphabetically and mapped to sequential IDs,
    and non-contiguous integer labels are remapped.

    Args:
        classes_series: Pandas Series of class labels.

    Returns:
        Tuple of ``(mapping_dict, class_names_list)`` where *mapping_dict* maps
        ``str(label) -> int_id`` and *class_names_list* is ordered by ID.
    """
    unique_classes = classes_series.unique()

    # Check if all labels are integers
    all_int = True
    for c in unique_classes:
        try:
            int(c)
        except (ValueError, TypeError):
            all_int = False
            break

    if all_int:
        int_classes = sorted([int(c) for c in unique_classes])
        if int_classes == list(range(len(int_classes))):
            mapping = {str(c): c for c in int_classes}
            names = [str(c) for c in int_classes]
        else:
            mapping = {str(c): i for i, c in enumerate(int_classes)}
            names = [str(c) for c in int_classes]
    else:
        sorted_classes = sorted([str(c) for c in unique_classes])
        mapping = {c: i for i, c in enumerate(sorted_classes)}
        names = sorted_classes

    return mapping, names
