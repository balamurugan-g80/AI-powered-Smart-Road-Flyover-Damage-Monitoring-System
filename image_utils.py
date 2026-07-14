"""
image_utils.py
--------------
Utility functions for handling road/flyover damage IMAGE input:
  - Upload validation & saving
  - Loading / resizing
  - Drawing bounding boxes + confidence scores on detection results
  - Saving annotated output images

Designed to be framework-agnostic: works whether called from a
Streamlit uploader, a CLI script, or a batch job over a dataset folder.
"""

import os
import cv2
import uuid
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Union

logger = logging.getLogger("image_utils")
logging.basicConfig(level=logging.INFO)

# Fallback defaults if config.py is not importable (keeps this file runnable standalone)
try:
    from config import RAW_IMAGES_DIR, PROCESSED_IMAGES_DIR
except ImportError:
    RAW_IMAGES_DIR = Path("data/raw/images")
    PROCESSED_IMAGES_DIR = Path("data/processed/annotated_images")
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Fixed color palette (BGR) per damage class so boxes are visually consistent
DAMAGE_COLORS = {
    "crack": (0, 0, 255),          # red
    "pothole": (0, 140, 255),      # orange
    "spalling": (0, 255, 255),     # yellow
    "joint_failure": (255, 0, 255),# magenta
    "surface_erosion": (255, 255, 0),  # cyan
}
DEFAULT_COLOR = (0, 255, 0)  # green fallback for unknown classes


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------
def validate_image_file(filename: str) -> bool:
    """
    Checks whether a filename has a supported image extension.
    Does NOT check file content/corruption - that happens on load.
    """
    ext = Path(filename).suffix.lower()
    is_valid = ext in VALID_IMAGE_EXTENSIONS
    if not is_valid:
        logger.warning(f"Rejected file '{filename}': unsupported extension '{ext}'")
    return is_valid


# ---------------------------------------------------------------------
# UPLOAD HANDLING
# ---------------------------------------------------------------------
def save_uploaded_image(uploaded_file, destination_dir: Union[str, Path] = RAW_IMAGES_DIR) -> str:
    """
    Persists an uploaded image to disk with a collision-safe unique filename.

    Accepts:
      - A Streamlit `UploadedFile` object (has .name and .getbuffer()/.read())
      - A raw file path (str) to an already-existing image (will be copied)
      - Raw bytes (requires a fallback extension of .jpg)

    Returns:
      Absolute path (str) to the saved file.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]

    # Case 1: Streamlit-style UploadedFile
    if hasattr(uploaded_file, "name") and hasattr(uploaded_file, "getbuffer"):
        if not validate_image_file(uploaded_file.name):
            raise ValueError(f"Unsupported image type: {uploaded_file.name}")
        ext = Path(uploaded_file.name).suffix.lower()
        out_path = destination_dir / f"img_{timestamp}_{unique_id}{ext}"
        with open(out_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Case 2: existing path on disk
    elif isinstance(uploaded_file, (str, Path)):
        src_path = Path(uploaded_file)
        if not src_path.exists():
            raise FileNotFoundError(f"Source image not found: {src_path}")
        if not validate_image_file(src_path.name):
            raise ValueError(f"Unsupported image type: {src_path.name}")
        ext = src_path.suffix.lower()
        out_path = destination_dir / f"img_{timestamp}_{unique_id}{ext}"
        img = cv2.imread(str(src_path))
        if img is None:
            raise ValueError(f"Could not read image (corrupted?): {src_path}")
        cv2.imwrite(str(out_path), img)

    # Case 3: raw bytes
    elif isinstance(uploaded_file, (bytes, bytearray)):
        out_path = destination_dir / f"img_{timestamp}_{unique_id}.jpg"
        with open(out_path, "wb") as f:
            f.write(uploaded_file)

    else:
        raise TypeError(f"Unsupported upload type: {type(uploaded_file)}")

    logger.info(f"Saved uploaded image -> {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------
# LOADING / RESIZING
# ---------------------------------------------------------------------
def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Loads an image as a BGR numpy array (OpenCV convention). Raises if unreadable."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def resize_image(image: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """Downscales an image to max_width while preserving aspect ratio (no upscaling)."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------
# ANNOTATION (bounding boxes + confidence scores)
# ---------------------------------------------------------------------
def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draws bounding boxes, class labels, and confidence scores onto a copy of `image`.

    Each detection dict is expected to have:
        {
          "class_name": str,
          "confidence": float,      # 0.0 - 1.0
          "bbox": (x1, y1, x2, y2)  # pixel coordinates
        }

    Returns:
        Annotated copy of the image (original is left untouched).
    """
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        class_name = det.get("class_name", "damage")
        confidence = det.get("confidence", 0.0)
        color = DAMAGE_COLORS.get(class_name, DEFAULT_COLOR)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        # Label + confidence score background
        label = f"{class_name} {confidence * 100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 4, y1), color, thickness=-1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    return annotated


# ---------------------------------------------------------------------
# SAVING RESULTS
# ---------------------------------------------------------------------
def save_annotated_image(
    image: np.ndarray,
    original_name: str,
    output_dir: Union[str, Path] = PROCESSED_IMAGES_DIR,
) -> str:
    """Saves an annotated image with a name derived from the original, returns saved path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(original_name).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{stem}_annotated_{timestamp}.jpg"

    success = cv2.imwrite(str(out_path), image)
    if not success:
        raise IOError(f"Failed to write annotated image to {out_path}")

    logger.info(f"Saved annotated image -> {out_path}")
    return str(out_path)
