"""
Central configuration for the AI-Powered Smart Road & Flyover
Damage Monitoring and Decision Intelligence System.

This file contains ONLY configuration values (paths, thresholds,
model parameters). No business logic lives here.
"""

import os
from pathlib import Path

# ------------------------------------------------------------------
# BASE PATHS
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw" / "images"
RAW_VIDEOS_DIR = DATA_DIR / "raw" / "videos"
PROCESSED_IMAGES_DIR = DATA_DIR / "processed" / "annotated_images"
PROCESSED_VIDEOS_DIR = DATA_DIR / "processed" / "annotated_videos"
FRAMES_DIR = DATA_DIR / "processed" / "frames"
EXPORTS_DIR = DATA_DIR / "exports"
REPORTS_DIR = EXPORTS_DIR / "reports"
CSV_EXPORTS_DIR = EXPORTS_DIR / "csv_exports"

MODELS_DIR = BASE_DIR / "models"
YOLO_MODEL_PATH = MODELS_DIR / "best.pt"
SEVERITY_MODEL_PATH = MODELS_DIR / "ml" / "severity_classifier.pkl"
PRIORITY_MODEL_PATH = MODELS_DIR / "ml" / "priority_xgb_model.json"
SCALER_PATH = MODELS_DIR / "ml" / "scaler.pkl"

DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "road_monitoring.db"
SCHEMA_PATH = DATABASE_DIR / "schema.sql"

LOGS_DIR = BASE_DIR / "logs"
APP_LOG_PATH = LOGS_DIR / "app.log"
DETECTION_LOG_PATH = LOGS_DIR / "detection.log"
ERROR_LOG_PATH = LOGS_DIR / "error.log"

# Ensure runtime directories exist
for _dir in [
    RAW_IMAGES_DIR, RAW_VIDEOS_DIR, PROCESSED_IMAGES_DIR,
    PROCESSED_VIDEOS_DIR, FRAMES_DIR, REPORTS_DIR,
    CSV_EXPORTS_DIR, LOGS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# YOLOv8 DETECTION SETTINGS
# ------------------------------------------------------------------
YOLO_CONFIDENCE_THRESHOLD = 0.05
YOLO_IOU_THRESHOLD = 0.40
YOLO_IMAGE_SIZE = 1280
YOLO_DEVICE = "cpu"          # set to "cuda" or "0" if GPU available

# If the model is available, use its actual class labels. Otherwise keep
# a compatible fallback for manual entry forms and downstream logic.
DEFAULT_DAMAGE_CLASSES = [
    "pothole",
    "road_crack",
    "bridge_crack",
    "surface_damage",
]

DAMAGE_CLASSES = DEFAULT_DAMAGE_CLASSES.copy()


def get_damage_classes():
    """Return class labels from the YOLO model if available, otherwise fallback."""
    try:
        from ultralytics import YOLO
        if YOLO_MODEL_PATH.exists():
            model = YOLO(str(YOLO_MODEL_PATH))
            names = getattr(model, "names", None)
            if isinstance(names, dict) and names:
                return list(names.values())
    except Exception:
        pass
    return DEFAULT_DAMAGE_CLASSES

# ------------------------------------------------------------------
# VIDEO PROCESSING SETTINGS
# ------------------------------------------------------------------
FRAME_SAMPLE_RATE_FPS = 2          # frames extracted per second of video
MAX_VIDEO_DURATION_SECONDS = 600   # safety cap on processed video length
VIDEO_RESIZE_WIDTH = 1280

# ------------------------------------------------------------------
# SEVERITY SCORING THRESHOLDS
# ------------------------------------------------------------------
SEVERITY_THRESHOLDS = {
    "low": (0.0, 0.25),
    "medium": (0.25, 0.50),
    "high": (0.50, 0.75),
    "critical": (0.75, 1.0),
}

# ------------------------------------------------------------------
# DECISION INTELLIGENCE (XGBoost) SETTINGS
# ------------------------------------------------------------------
PRIORITY_MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "eta": 0.1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

RISK_ALERT_THRESHOLD = 0.75   # priority score above which an alert is raised

# ------------------------------------------------------------------
# SHAP EXPLAINABILITY SETTINGS
# ------------------------------------------------------------------
SHAP_EXPLAINER_TYPE = "tree"          # matches XGBoost tree-based model
SHAP_MAX_DISPLAY_FEATURES = 10

# ------------------------------------------------------------------
# REPORTING SETTINGS
# ------------------------------------------------------------------
REPORT_TITLE = "Road & Flyover Damage Inspection Report"
REPORT_LOGO_PATH = BASE_DIR / "core" / "reporting" / "templates" / "logo.png"
REPORT_PAGE_SIZE = "A4"

# ------------------------------------------------------------------
# STREAMLIT / APP SETTINGS
# ------------------------------------------------------------------
APP_TITLE = "Smart Road & Flyover Damage Monitoring System"
APP_ICON = "🛣️"
DEFAULT_MAP_ZOOM = 12

# ------------------------------------------------------------------
# DATABASE SETTINGS
# ------------------------------------------------------------------
SQLITE_TIMEOUT_SECONDS = 30
ENABLE_FOREIGN_KEYS = True

# ------------------------------------------------------------------
# ENVIRONMENT / SECRETS (loaded via .env in production)
# ------------------------------------------------------------------
ENV = os.getenv("APP_ENV", "development")
DEBUG = ENV == "development"