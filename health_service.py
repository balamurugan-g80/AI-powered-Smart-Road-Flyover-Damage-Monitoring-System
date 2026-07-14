"""
health_service.py
------------------
Computes the Infrastructure Health Score: a single 0-100 number
representing overall structural condition (100 = perfect, 0 = failed).

Depends on analytics_service output (severity_score, damage_percentage,
damage_count) - run analytics_service.analyze() first and pass its
result in, or use compute_health_score_from_detections() to do both
steps in one call.
"""

import logging
from typing import Dict, List, Optional

from score_utils import clip, round2

logger = logging.getLogger("health_service")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# TUNABLE WEIGHTS - Health Degradation Index (HDI) components
# ---------------------------------------------------------------------
W_SEVERITY = 0.50          # weight of damage severity score
W_COVERAGE = 0.30          # weight of damage percentage (surface coverage)
W_DENSITY = 0.20           # weight of raw defect count density

MAX_EXPECTED_DAMAGE_COUNT = 50    # count at/above which density_factor saturates at 1.0

# Age-based penalty (applied AFTER the HDI-based score, as a direct deduction)
AGE_PENALTY_PER_YEAR = 0.3        # health points lost per year of age
MAX_AGE_PENALTY = 15.0            # cap so age alone can't zero out the score


def _damage_count_density_factor(total_count: int) -> float:
    """Normalizes raw defect count to [0, 1] against MAX_EXPECTED_DAMAGE_COUNT."""
    if MAX_EXPECTED_DAMAGE_COUNT <= 0:
        return 0.0
    return clip(total_count / MAX_EXPECTED_DAMAGE_COUNT, 0.0, 1.0)


def compute_health_score(
    severity_score: float,
    damage_percentage: float,
    total_damage_count: int,
    asset_age_years: Optional[float] = None,
) -> Dict:
    """
    Formula:

        Health Degradation Index (HDI), 0-1:
            HDI = clip( W_SEVERITY * severity_score
                        + W_COVERAGE * (damage_percentage / 100)
                        + W_DENSITY  * density_factor ,
                        0, 1 )
            where density_factor = min(1, total_damage_count / MAX_EXPECTED_DAMAGE_COUNT)

        Base Health Score (0-100):
            base_health = 100 * (1 - HDI)

        Age Penalty (only if asset_age_years is supplied):
            age_penalty = min(MAX_AGE_PENALTY, asset_age_years * AGE_PENALTY_PER_YEAR)

        Final Health Score:
            health_score = clip(base_health - age_penalty, 0, 100)

    Returns:
        {"health_score": float (0-100), "hdi": float (0-1), "age_penalty": float}
    """
    density_factor = _damage_count_density_factor(total_damage_count)

    hdi = clip(
        W_SEVERITY * severity_score
        + W_COVERAGE * (damage_percentage / 100)
        + W_DENSITY * density_factor,
        0.0, 1.0,
    )

    base_health = 100 * (1 - hdi)

    age_penalty = 0.0
    if asset_age_years is not None:
        age_penalty = min(MAX_AGE_PENALTY, max(0.0, asset_age_years) * AGE_PENALTY_PER_YEAR)

    health_score = clip(base_health - age_penalty, 0.0, 100.0)

    return {
        "health_score": round2(health_score),
        "hdi": round2(hdi),
        "age_penalty": round2(age_penalty),
    }


def compute_health_score_from_detections(
    detections: List[Dict],
    total_inspected_area_px: float,
    asset_age_years: Optional[float] = None,
) -> Dict:
    """
    Convenience wrapper: runs analytics_service.analyze() then
    compute_health_score() in one call.
    """
    from analytics_service import analyze  # local import avoids a hard circular dependency

    analytics = analyze(detections, total_inspected_area_px)
    result = compute_health_score(
        severity_score=analytics["damage_severity"]["severity_score"],
        damage_percentage=analytics["damage_percentage"],
        total_damage_count=analytics["damage_count"]["total_count"],
        asset_age_years=asset_age_years,
    )
    result["analytics"] = analytics
    return result


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_detections = [
        {"class_name": "pothole", "confidence": 0.86, "area_px": 5200},
        {"class_name": "crack", "confidence": 0.61, "area_px": 1800},
        {"class_name": "joint_failure", "confidence": 0.72, "area_px": 3000},
    ]
    result = compute_health_score_from_detections(
        sample_detections, total_inspected_area_px=1280 * 720, asset_age_years=12
    )
    print("Health Score Result:", result)
