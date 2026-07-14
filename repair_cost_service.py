"""
repair_cost_service.py
------------------------
Predicts estimated repair cost for detected damage.

Two modes, selected automatically:

  1. ML MODE (preferred): if a trained XGBoost regressor exists at
     `models/ml/repair_cost_xgb_model.json`, it is loaded and used.
     Train it in `training/train_repair_cost_model.py` (not included
     here) once you have historical cost data - features are defined
     in `build_feature_vector()` below so training and inference stay
     in sync.

  2. HEURISTIC MODE (fallback / bootstrap): a transparent unit-cost
     formula, used automatically until a trained model is available so
     the system is usable on day one. Logged clearly so it's never
     silently mistaken for a validated prediction.

IMPORTANT CALIBRATION NOTE:
  Converting bounding-box pixel area to real-world square meters
  requires a `pixels_per_meter` camera calibration constant (depends on
  camera height/angle/lens per inspection rig or vehicle-mounted
  camera). The default in this file is a placeholder - replace it with
  a measured value for your actual camera setup, or costs will be
  directionally right but not precise.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from score_utils import get_severity_weight, round2, pixel_area_to_sqm

logger = logging.getLogger("repair_cost_service")
logging.basicConfig(level=logging.INFO)

try:
    from config import PRIORITY_MODEL_PATH  # reuse models dir convention
    MODELS_ML_DIR = Path(PRIORITY_MODEL_PATH).parent
except ImportError:
    MODELS_ML_DIR = Path("models/ml")

REPAIR_COST_MODEL_PATH = MODELS_ML_DIR / "repair_cost_xgb_model.json"

# ---------------------------------------------------------------------
# HEURISTIC UNIT COSTS (currency units - e.g. INR per square meter)
# Calibrate against your region's actual contractor/PWD rate schedule.
# ---------------------------------------------------------------------
UNIT_COST_PER_SQM: Dict[str, float] = {
    "pothole": 1500.0,
    "crack": 300.0,
    "spalling": 800.0,
    "joint_failure": 2000.0,
    "surface_erosion": 400.0,
}
DEFAULT_UNIT_COST_PER_SQM = 500.0

MOBILIZATION_COST = 5000.0          # fixed cost per repair job (crew, equipment, traffic control)
SEVERITY_COST_MULTIPLIER_RANGE = (1.0, 1.5)  # low-confidence damage costs base rate; high-confidence adds up to +50%

DEFAULT_PIXELS_PER_METER = 100.0    # PLACEHOLDER - calibrate per camera setup


def _severity_cost_multiplier(confidence: float) -> float:
    """Higher-confidence (i.e. more clearly severe) detections scale cost up, within a bounded range."""
    low, high = SEVERITY_COST_MULTIPLIER_RANGE
    return low + (high - low) * confidence


# ---------------------------------------------------------------------
# HEURISTIC MODE
# ---------------------------------------------------------------------
def estimate_repair_cost_heuristic(
    detections: List[Dict],
    pixels_per_meter: float = DEFAULT_PIXELS_PER_METER,
) -> Dict:
    """
    Formula (per detection i):

        area_sqm_i     = area_px_i / (pixels_per_meter ** 2)
        unit_cost_i     = UNIT_COST_PER_SQM[class_i]   (falls back to DEFAULT if unknown class)
        severity_mult_i = 1.0 + 0.5 * confidence_i     (ranges 1.0 - 1.5)

        cost_i = area_sqm_i * unit_cost_i * severity_mult_i

    Total:
        total_cost = MOBILIZATION_COST + Σ cost_i

    Returns:
        {
          "total_cost": float,
          "mobilization_cost": float,
          "line_items": [ {class_name, area_sqm, unit_cost, cost}, ... ]
        }
    """
    line_items = []
    running_total = 0.0

    for det in detections:
        class_name = det.get("class_name", "unknown")
        area_sqm = pixel_area_to_sqm(det.get("area_px", 0.0), pixels_per_meter)
        unit_cost = UNIT_COST_PER_SQM.get(class_name, DEFAULT_UNIT_COST_PER_SQM)
        severity_mult = _severity_cost_multiplier(det.get("confidence", 0.5))
        cost = area_sqm * unit_cost * severity_mult

        line_items.append({
            "class_name": class_name,
            "area_sqm": round2(area_sqm),
            "unit_cost_per_sqm": unit_cost,
            "cost": round2(cost),
        })
        running_total += cost

    total_cost = MOBILIZATION_COST + running_total

    return {
        "total_cost": round2(total_cost),
        "mobilization_cost": MOBILIZATION_COST,
        "line_items": line_items,
        "mode": "heuristic",
    }


# ---------------------------------------------------------------------
# ML MODE
# ---------------------------------------------------------------------
def build_feature_vector(
    detections: List[Dict],
    total_inspected_area_px: float,
    asset_age_years: Optional[float] = None,
    asset_length_m: Optional[float] = None,
) -> Dict[str, float]:
    """
    Builds the feature vector used by BOTH training and inference for the
    XGBoost repair-cost model, so the two never drift out of sync.

    Features:
        total_count, avg_confidence, damage_percentage,
        count_pothole, count_crack, count_spalling,
        count_joint_failure, count_surface_erosion,
        asset_age_years, asset_length_m
    """
    from analytics_service import compute_damage_count, compute_damage_percentage

    count_result = compute_damage_count(detections)
    damage_pct = compute_damage_percentage(detections, total_inspected_area_px)
    avg_conf = (
        sum(d.get("confidence", 0.0) for d in detections) / len(detections)
        if detections else 0.0
    )

    by_class = count_result["count_by_class"]
    features = {
        "total_count": count_result["total_count"],
        "avg_confidence": round2(avg_conf),
        "damage_percentage": damage_pct,
        "count_pothole": by_class.get("pothole", 0),
        "count_crack": by_class.get("crack", 0),
        "count_spalling": by_class.get("spalling", 0),
        "count_joint_failure": by_class.get("joint_failure", 0),
        "count_surface_erosion": by_class.get("surface_erosion", 0),
        "asset_age_years": asset_age_years or 0.0,
        "asset_length_m": asset_length_m or 0.0,
    }
    return features


def _load_ml_model():
    """Attempts to load a trained XGBoost repair-cost model. Returns None if unavailable."""
    if not REPAIR_COST_MODEL_PATH.exists():
        return None
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(REPAIR_COST_MODEL_PATH))
        return model
    except Exception as e:
        logger.warning(f"Failed to load repair cost ML model ({e}); falling back to heuristic.")
        return None


def estimate_repair_cost(
    detections: List[Dict],
    total_inspected_area_px: float,
    pixels_per_meter: float = DEFAULT_PIXELS_PER_METER,
    asset_age_years: Optional[float] = None,
    asset_length_m: Optional[float] = None,
) -> Dict:
    """
    Main entry point: tries the trained XGBoost model first, falls back
    to the heuristic formula if no model is available. Always returns
    the same shape so callers don't need to branch.
    """
    if not detections:
        return {"total_cost": 0.0, "mobilization_cost": 0.0, "line_items": [], "mode": "none"}

    model = _load_ml_model()

    if model is not None:
        import pandas as pd
        features = build_feature_vector(detections, total_inspected_area_px, asset_age_years, asset_length_m)
        feature_df = pd.DataFrame([features])
        predicted_cost = float(model.predict(feature_df)[0])
        logger.info("Repair cost predicted via trained XGBoost model.")
        return {
            "total_cost": round2(max(0.0, predicted_cost)),
            "mobilization_cost": MOBILIZATION_COST,
            "line_items": None,  # ML mode predicts a total directly, not itemized
            "mode": "ml",
            "features_used": features,
        }

    logger.info("No trained repair-cost model found - using heuristic unit-cost formula.")
    return estimate_repair_cost_heuristic(detections, pixels_per_meter)


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_detections = [
        {"class_name": "pothole", "confidence": 0.86, "area_px": 5200},
        {"class_name": "crack", "confidence": 0.55, "area_px": 1200},
        {"class_name": "joint_failure", "confidence": 0.72, "area_px": 3000},
    ]
    result = estimate_repair_cost(sample_detections, total_inspected_area_px=1280 * 720)
    print(f"Mode: {result['mode']}")
    print(f"Total estimated repair cost: {result['total_cost']}")
    if result["line_items"]:
        for item in result["line_items"]:
            print("  ", item)
