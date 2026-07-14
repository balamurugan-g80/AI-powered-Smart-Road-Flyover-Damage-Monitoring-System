"""
emergency_service.py
---------------------
Computes the Emergency Index (0-100): distinct from Risk Score because
it answers a narrower, more urgent question - "does this need an
emergency response crew NOW", rather than "how risky is this asset
generally". It weighs structural (load-bearing) damage and asset
criticality much more heavily than generic risk does.
"""

import logging
from typing import Dict, List, Optional

from score_utils import (
    clip,
    round2,
    score_to_category,
    is_structural_damage,
    get_asset_criticality,
)

logger = logging.getLogger("emergency_service")
logging.basicConfig(level=logging.INFO)

# Composite weights - MUST sum to 1.0
W_RISK = 0.40
W_STRUCTURAL = 0.30
W_GROWTH = 0.20
W_CRITICALITY = 0.10

IMMEDIATE_ACTION_THRESHOLD = 75.0  # Emergency Index >= this -> flag for emergency dispatch


def _structural_damage_factor(detections: List[Dict]) -> float:
    """
    0-100 score representing the worst observed structural (load-bearing)
    finding. Uses max confidence among structural-class detections
    (joint_failure, spalling) since even ONE high-confidence structural
    defect is dangerous - averaging would dilute that signal.
    """
    structural_confidences = [
        d.get("confidence", 0.0) for d in detections if is_structural_damage(d.get("class_name", ""))
    ]
    if not structural_confidences:
        return 0.0
    return clip(max(structural_confidences) * 100, 0.0, 100.0)


def compute_emergency_index(
    risk_score: float,
    detections: List[Dict],
    asset_type: str = "road",
    growth_factor_0_100: float = 0.0,
) -> Dict:
    """
    Formula:

        structural_factor = max(confidence of structural-class detections) * 100
                             (0 if no structural damage present)

        criticality_factor = ASSET_CRITICALITY_SCORE[asset_type]
                              (flyover=100, road=60)

        growth_factor = caller-supplied 0-100 value representing projected
                        near-term damage growth (see simulation_service) -
                        defaults to 0 if not supplied.

        Emergency Index = clip(
              W_RISK * risk_score
            + W_STRUCTURAL * structural_factor
            + W_GROWTH * growth_factor
            + W_CRITICALITY * criticality_factor,
            0, 100
        )

        Emergency Level = score_to_category(Emergency Index)
        requires_immediate_action = Emergency Index >= IMMEDIATE_ACTION_THRESHOLD (75)

    Returns:
        {
          "emergency_index": float (0-100),
          "emergency_level": str,
          "requires_immediate_action": bool,
          "structural_factor": float
        }
    """
    structural_factor = _structural_damage_factor(detections)
    criticality_factor = get_asset_criticality(asset_type)
    growth_factor = clip(growth_factor_0_100, 0.0, 100.0)

    emergency_index = clip(
        W_RISK * risk_score
        + W_STRUCTURAL * structural_factor
        + W_GROWTH * growth_factor
        + W_CRITICALITY * criticality_factor,
        0.0, 100.0,
    )

    emergency_level = score_to_category(emergency_index)

    return {
        "emergency_index": round2(emergency_index),
        "emergency_level": emergency_level,
        "requires_immediate_action": emergency_index >= IMMEDIATE_ACTION_THRESHOLD,
        "structural_factor": round2(structural_factor),
    }


def compute_emergency_from_detections(
    detections: List[Dict],
    total_inspected_area_px: float,
    asset_type: str = "road",
    asset_age_years: Optional[float] = None,
    traffic_level: Optional[str] = None,
    growth_factor_0_100: float = 0.0,
) -> Dict:
    """Convenience wrapper chaining analytics -> health -> risk -> emergency."""
    from risk_service import compute_risk_from_detections

    risk_result = compute_risk_from_detections(
        detections, total_inspected_area_px, asset_age_years, traffic_level
    )
    emergency_result = compute_emergency_index(
        risk_score=risk_result["risk_score"],
        detections=detections,
        asset_type=asset_type,
        growth_factor_0_100=growth_factor_0_100,
    )
    emergency_result["risk_result"] = risk_result
    return emergency_result


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_detections = [
        {"class_name": "joint_failure", "confidence": 0.91, "area_px": 5000},
        {"class_name": "spalling", "confidence": 0.77, "area_px": 3200},
    ]
    result = compute_emergency_from_detections(
        sample_detections,
        total_inspected_area_px=1280 * 720,
        asset_type="flyover",
        asset_age_years=30,
        traffic_level="high",
        growth_factor_0_100=40,
    )
    print("Emergency Index:", result["emergency_index"])
    print("Emergency Level:", result["emergency_level"])
    print("Immediate action required:", result["requires_immediate_action"])
