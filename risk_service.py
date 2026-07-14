"""
risk_service.py
----------------
Computes the Infrastructure Risk Score (0-100) and its categorical
Risk Level (Low / Medium / High / Critical).

Risk is deliberately modeled as the INVERSE of health, then adjusted
for external exposure factors that health alone doesn't capture:
traffic load (more load-cycles accelerate failure consequences) and
a bonus for already-Critical-severity findings.
"""

import logging
from typing import Dict, Optional

from score_utils import clip, round2, score_to_category, get_traffic_multiplier

logger = logging.getLogger("risk_service")
logging.basicConfig(level=logging.INFO)

CRITICAL_SEVERITY_BONUS = 15.0   # flat risk points added if severity_class == "Critical"


def compute_risk_score(
    health_score: float,
    severity_class: str,
    traffic_level: Optional[str] = None,
) -> Dict:
    """
    Formula:

        Base Risk = 100 - health_score

        Traffic Multiplier (from score_utils.TRAFFIC_LOAD_MULTIPLIER):
            low -> 1.00, medium -> 1.15, high -> 1.30 (default 1.00 if unspecified)

        Critical Bonus:
            +15 if severity_class == "Critical", else 0

        Risk Score = clip( Base Risk * traffic_multiplier + critical_bonus, 0, 100 )

        Risk Level = score_to_category(Risk Score)
            0-25 Low | 25-50 Medium | 50-75 High | 75-100 Critical

    Returns:
        {"risk_score": float (0-100), "risk_level": str}
    """
    base_risk = 100.0 - health_score
    traffic_multiplier = get_traffic_multiplier(traffic_level)
    critical_bonus = CRITICAL_SEVERITY_BONUS if severity_class == "Critical" else 0.0

    risk_score = clip(base_risk * traffic_multiplier + critical_bonus, 0.0, 100.0)
    risk_level = score_to_category(risk_score)

    return {
        "risk_score": round2(risk_score),
        "risk_level": risk_level,
    }


def compute_risk_from_detections(
    detections,
    total_inspected_area_px: float,
    asset_age_years: Optional[float] = None,
    traffic_level: Optional[str] = None,
) -> Dict:
    """
    Convenience wrapper chaining analytics -> health -> risk in one call.
    """
    from health_service import compute_health_score_from_detections

    health_result = compute_health_score_from_detections(
        detections, total_inspected_area_px, asset_age_years
    )
    severity_class = health_result["analytics"]["damage_severity"]["severity_class"]

    risk_result = compute_risk_score(
        health_score=health_result["health_score"],
        severity_class=severity_class,
        traffic_level=traffic_level,
    )
    risk_result["health_result"] = health_result
    return risk_result


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_detections = [
        {"class_name": "pothole", "confidence": 0.9, "area_px": 8000},
        {"class_name": "joint_failure", "confidence": 0.85, "area_px": 4000},
    ]
    result = compute_risk_from_detections(
        sample_detections,
        total_inspected_area_px=1280 * 720,
        asset_age_years=20,
        traffic_level="high",
    )
    print("Risk Result:", {k: v for k, v in result.items() if k != "health_result"})
    print("Underlying Health Score:", result["health_result"]["health_score"])
