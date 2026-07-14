"""
life_prediction_service.py
----------------------------
Predicts Remaining Useful Life (RUL), in years, for a road or flyover
segment: how much longer it can safely serve before major
rehabilitation/replacement is needed.

Two modes, same pattern as repair_cost_service:

  1. ML MODE: trained XGBoost regressor at
     `models/ml/rul_xgb_model.json` (train via
     `training/train_priority_model.py`-style script once historical
     asset-failure/rehabilitation data is available).

  2. HEURISTIC MODE (fallback): an engineering-judgment formula based
     on standard asset design life, current age, and observed
     degradation (health score). Used automatically until a trained
     model exists.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from score_utils import clip, round2, get_asset_lifespan

logger = logging.getLogger("life_prediction_service")
logging.basicConfig(level=logging.INFO)

try:
    from config import PRIORITY_MODEL_PATH
    MODELS_ML_DIR = Path(PRIORITY_MODEL_PATH).parent
except ImportError:
    MODELS_ML_DIR = Path("models/ml")

RUL_MODEL_PATH = MODELS_ML_DIR / "rul_xgb_model.json"

# How strongly degradation (poor health) eats into remaining life.
# 1.0 = a fully-degraded asset (health=0) has 0 years left, regardless of age.
# 0.0 = degradation is ignored entirely (age is the only factor) - not recommended.
DEGRADATION_IMPACT_WEIGHT = 0.85

MIN_RUL_YEARS = 0.0


# ---------------------------------------------------------------------
# HEURISTIC MODE
# ---------------------------------------------------------------------
def estimate_rul_heuristic(
    health_score: float,
    asset_age_years: float,
    asset_type: str = "road",
) -> Dict:
    """
    Formula:

        expected_total_life = ASSET_BASE_LIFESPAN_YEARS[asset_type]
                               (road=15, flyover=50 - design-standard assumptions)

        remaining_base = max(0, expected_total_life - asset_age_years)
                          (life left if degradation were "textbook average" for its age)

        degradation_factor = (100 - health_score) / 100
                              (0 = perfect health, 1 = fully degraded)

        adjusted_rul = remaining_base * (1 - DEGRADATION_IMPACT_WEIGHT * degradation_factor)

        RUL = max(MIN_RUL_YEARS, adjusted_rul)

    A healthy, young asset keeps most of remaining_base. A young asset
    with severe current damage gets sharply discounted, reflecting that
    damage observed today is a leading indicator of accelerated failure,
    not just "normal for its age".

    Returns:
        {"rul_years": float, "expected_total_life": float, "degradation_factor": float}
    """
    expected_total_life = get_asset_lifespan(asset_type)
    remaining_base = max(0.0, expected_total_life - asset_age_years)
    degradation_factor = clip((100.0 - health_score) / 100.0, 0.0, 1.0)

    adjusted_rul = remaining_base * (1 - DEGRADATION_IMPACT_WEIGHT * degradation_factor)
    rul_years = max(MIN_RUL_YEARS, adjusted_rul)

    return {
        "rul_years": round2(rul_years),
        "expected_total_life": expected_total_life,
        "degradation_factor": round2(degradation_factor),
        "mode": "heuristic",
    }


# ---------------------------------------------------------------------
# ML MODE
# ---------------------------------------------------------------------
def build_rul_feature_vector(
    health_score: float,
    severity_score: float,
    asset_age_years: float,
    traffic_level: Optional[str] = None,
    growth_rate_monthly: Optional[float] = None,
) -> Dict[str, float]:
    """Feature vector shared between training and inference for the RUL model."""
    traffic_encoding = {"low": 0, "medium": 1, "high": 2}.get(
        (traffic_level or "low").lower(), 0
    )
    return {
        "health_score": health_score,
        "severity_score": severity_score,
        "asset_age_years": asset_age_years,
        "traffic_level_encoded": traffic_encoding,
        "growth_rate_monthly": growth_rate_monthly or 0.0,
    }


def _load_ml_model():
    if not RUL_MODEL_PATH.exists():
        return None
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(RUL_MODEL_PATH))
        return model
    except Exception as e:
        logger.warning(f"Failed to load RUL ML model ({e}); falling back to heuristic.")
        return None


def estimate_rul(
    health_score: float,
    severity_score: float,
    asset_age_years: float,
    asset_type: str = "road",
    traffic_level: Optional[str] = None,
    growth_rate_monthly: Optional[float] = None,
) -> Dict:
    """
    Main entry point: tries the trained XGBoost model first, falls back
    to the heuristic engineering formula.
    """
    model = _load_ml_model()

    if model is not None:
        import pandas as pd
        features = build_rul_feature_vector(
            health_score, severity_score, asset_age_years, traffic_level, growth_rate_monthly
        )
        feature_df = pd.DataFrame([features])
        predicted_rul = float(model.predict(feature_df)[0])
        logger.info("RUL predicted via trained XGBoost model.")
        return {
            "rul_years": round2(max(MIN_RUL_YEARS, predicted_rul)),
            "mode": "ml",
            "features_used": features,
        }

    logger.info("No trained RUL model found - using heuristic degradation formula.")
    return estimate_rul_heuristic(health_score, asset_age_years, asset_type)


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    result = estimate_rul(
        health_score=42.0,
        severity_score=0.58,
        asset_age_years=18,
        asset_type="flyover",
        traffic_level="high",
    )
    print("RUL Result:", result)
