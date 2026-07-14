"""
recommendation_service.py
--------------------------
Decision Intelligence Layer: converts detections + the upstream scoring
chain (analytics -> health -> risk -> emergency -> repair_cost ->
life_prediction) into concrete, actionable maintenance recommendations
that can be inserted directly into the `maintenance_recommendations`
table defined in ARCHITECTURE.md.

Design (matches the project's stated principle - "Hybrid decision
logic: not a pure black-box output"):

    ML / composite SCORES  (risk_service, emergency_service)
                    +
    DETERMINISTIC RULES ENGINE  (this file: DAMAGE_ACTION_MAP + escalation)
                    =
    Recommended Action + Priority Level + Due Date + Estimated Cost

Why a rules engine on top of scores, instead of scores alone?
Civic/engineering stakeholders need auditable, explainable reasons for
an action ("why does this say Immediate Repair?"), not just a number.
The rules below encode standard road/bridge-maintenance judgment
(e.g. any crack on a FLYOVER is treated as a structural concern even
though on a ROAD it is purely cosmetic) and are the same rules
`explainability_service.py` narrates back to the user.

Examples this engine is built to reproduce:
    Pothole  (road or flyover, Medium+ severity)  -> "Immediate Repair"
    Bridge/Flyover Crack (any confidence)          -> "Structural Inspection"
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from score_utils import STRUCTURAL_DAMAGE_CLASSES, is_structural_damage

logger = logging.getLogger("recommendation_service")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# 1. RULES ENGINE CONFIGURATION
# ---------------------------------------------------------------------

# Base action per (damage_class, asset_type) - the "default" recommendation
# before any escalation is applied. Calibrate against your agency's SOP.
DAMAGE_ACTION_MAP: Dict[tuple, str] = {
    ("pothole", "road"):           "Pothole Patch Repair",
    ("pothole", "flyover"):        "Immediate Repair",
    ("crack", "road"):             "Monitor / Sealant Repair",
    ("crack", "flyover"):          "Structural Inspection",
    ("spalling", "road"):          "Surface Repair",
    ("spalling", "flyover"):       "Structural Inspection",
    ("joint_failure", "road"):     "Joint Repair",
    ("joint_failure", "flyover"):  "Structural Inspection",
    ("surface_erosion", "road"):   "Resurfacing",
    ("surface_erosion", "flyover"): "Surface Repair",
}
DEFAULT_ACTION = "General Maintenance Inspection"

# On a flyover, these damage classes are ALWAYS treated as a structural
# concern regardless of confidence - a superset of score_utils'
# STRUCTURAL_DAMAGE_CLASSES (which only covers joint_failure/spalling),
# because a crack that is cosmetic on a road surface is a leading
# indicator of structural distress on a load-bearing flyover deck.
FLYOVER_STRUCTURAL_CONCERN_CLASSES = STRUCTURAL_DAMAGE_CLASSES | {"crack"}

# Severity classes (from analytics_service) at/above which a pothole is
# escalated straight to "Immediate Repair" regardless of asset type -
# potholes are a direct vehicle-safety hazard, unlike hairline cracks.
POTHOLE_IMMEDIATE_SEVERITY = {"High", "Critical"}

# Emergency/Risk category -> maintenance_recommendations.priority_level
# (DB schema uses lowercase 'low'/'medium'/'high'/'urgent'; score_utils
# categories are 'Low'/'Medium'/'High'/'Critical' - Critical maps to
# 'urgent' since that is the DB's most severe bucket.)
CATEGORY_TO_PRIORITY_LEVEL = {
    "Low": "low",
    "Medium": "medium",
    "High": "high",
    "Critical": "urgent",
}

# Response-time SLA (calendar days) by priority level - drives due_date.
DUE_DAYS_BY_PRIORITY = {
    "urgent": 3,
    "high": 14,
    "medium": 45,
    "low": 90,
}


# ---------------------------------------------------------------------
# 2. PER-DETECTION ACTION DECISION
# ---------------------------------------------------------------------
def determine_recommended_action(
    class_name: str,
    asset_type: str,
    severity_class: str,
    requires_immediate_action: bool = False,
) -> Dict:
    """
    Applies the rules engine to a single detection.

    Decision order (first match wins - most urgent rule first):
        1. requires_immediate_action (from emergency_service, i.e. the
           whole session's Emergency Index >= IMMEDIATE_ACTION_THRESHOLD)
           AND this detection is structurally significant on a flyover
             -> "Emergency Structural Inspection & Closure Assessment"
        2. requires_immediate_action (any other case)
             -> "Immediate Repair"
        3. Flyover + structurally-significant class (joint_failure,
           spalling, or crack)
             -> "Structural Inspection"
        4. Pothole with High/Critical severity (any asset type)
             -> "Immediate Repair"
        5. Otherwise -> DAMAGE_ACTION_MAP[(class, asset_type)], or
           DEFAULT_ACTION if the combination is unmapped.

    Returns:
        {"action": str, "rule_fired": str}  - rule_fired feeds directly
        into explainability_service's natural-language explanations.
    """
    asset_type = (asset_type or "road").lower()
    is_flyover_structural_concern = (
        asset_type == "flyover" and class_name in FLYOVER_STRUCTURAL_CONCERN_CLASSES
    )

    if requires_immediate_action and is_flyover_structural_concern:
        return {
            "action": "Emergency Structural Inspection & Closure Assessment",
            "rule_fired": "emergency_index_critical_and_structural",
        }
    if requires_immediate_action:
        return {"action": "Immediate Repair", "rule_fired": "emergency_index_critical"}

    if is_flyover_structural_concern:
        return {"action": "Structural Inspection", "rule_fired": "flyover_structural_concern"}

    if class_name == "pothole" and severity_class in POTHOLE_IMMEDIATE_SEVERITY:
        return {"action": "Immediate Repair", "rule_fired": "pothole_high_severity"}

    action = DAMAGE_ACTION_MAP.get((class_name, asset_type), DEFAULT_ACTION)
    return {"action": action, "rule_fired": "default_damage_action_map"}


def recommend_for_detection(detection: Dict, asset_type: str, requires_immediate_action: bool = False) -> Dict:
    """
    Wraps determine_recommended_action() with the per-detection severity
    lookup (a single detection's own confidence/class, not the whole
    session's composite severity) so it can be called detection-by-
    detection for a "per-defect" recommendation list in the dashboard.
    """
    from score_utils import get_severity_weight

    class_name = detection.get("class_name", "unknown")
    confidence = detection.get("confidence", 0.0)
    pseudo_severity_0_1 = confidence * get_severity_weight(class_name)
    severity_class = (
        "Critical" if pseudo_severity_0_1 >= 0.75 else
        "High" if pseudo_severity_0_1 >= 0.50 else
        "Medium" if pseudo_severity_0_1 >= 0.25 else
        "Low"
    )

    decision = determine_recommended_action(
        class_name, asset_type, severity_class, requires_immediate_action
    )
    return {
        "class_name": class_name,
        "confidence": confidence,
        "structural": is_structural_damage(class_name),
        "severity_class": severity_class,
        **decision,
    }


# ---------------------------------------------------------------------
# 3. SESSION-LEVEL RECOMMENDATION (ready for maintenance_recommendations table)
# ---------------------------------------------------------------------
def generate_session_recommendation(
    detections: List[Dict],
    total_inspected_area_px: float,
    asset_type: str = "road",
    road_id: Optional[int] = None,
    prediction_id: Optional[int] = None,
    asset_age_years: Optional[float] = None,
    traffic_level: Optional[str] = None,
    asset_length_m: Optional[float] = None,
    pixels_per_meter: Optional[float] = None,
) -> Dict:
    """
    Full pipeline for one inspection session: chains
    analytics -> health -> risk -> emergency -> repair_cost, then
    applies the rules engine to produce ONE session-level record
    matching `maintenance_recommendations`, plus a `detection_actions`
    list with a recommendation per individual detection.

    Returns:
        {
          "road_id", "prediction_id",
          "priority_level": 'low'|'medium'|'high'|'urgent',
          "recommended_action": str (worst/most urgent action across detections),
          "estimated_cost": float,
          "due_date": 'YYYY-MM-DD',
          "status": "open",
          "emergency_result": {...}, "repair_cost_result": {...},
          "detection_actions": [ per-detection recommend_for_detection() dicts ]
        }
    """
    from emergency_service import compute_emergency_from_detections
    from repair_cost_service import estimate_repair_cost

    emergency_result = compute_emergency_from_detections(
        detections, total_inspected_area_px, asset_type, asset_age_years, traffic_level
    )
    repair_cost_kwargs = {"pixels_per_meter": pixels_per_meter} if pixels_per_meter else {}
    repair_cost_result = estimate_repair_cost(
        detections, total_inspected_area_px,
        asset_age_years=asset_age_years, asset_length_m=asset_length_m,
        **repair_cost_kwargs,
    )

    requires_immediate_action = emergency_result["requires_immediate_action"]
    detection_actions = [
        recommend_for_detection(d, asset_type, requires_immediate_action) for d in detections
    ]

    # The session's headline action is the single most urgent action
    # among all detections (Immediate/Emergency actions outrank Inspect,
    # which outranks routine repair, which outranks Monitor).
    ACTION_URGENCY_RANK = {
        "Emergency Structural Inspection & Closure Assessment": 5,
        "Immediate Repair": 4,
        "Structural Inspection": 3,
        "Pothole Patch Repair": 2, "Joint Repair": 2, "Surface Repair": 2, "Resurfacing": 2,
        "Monitor / Sealant Repair": 1,
        DEFAULT_ACTION: 0,
    }
    headline_action = DEFAULT_ACTION
    if detection_actions:
        headline_action = max(
            detection_actions, key=lambda a: ACTION_URGENCY_RANK.get(a["action"], 0)
        )["action"]

    priority_level = CATEGORY_TO_PRIORITY_LEVEL.get(emergency_result["emergency_level"], "low")
    due_date = (datetime.now(timezone.utc) + timedelta(days=DUE_DAYS_BY_PRIORITY[priority_level])).date().isoformat()

    return {
        "road_id": road_id,
        "prediction_id": prediction_id,
        "priority_level": priority_level,
        "recommended_action": headline_action,
        "estimated_cost": repair_cost_result["total_cost"],
        "due_date": due_date,
        "status": "open",
        "emergency_result": emergency_result,
        "repair_cost_result": repair_cost_result,
        "detection_actions": detection_actions,
    }


def generate_recommendation_id() -> str:
    """Human-readable unique ID, used before the DB assigns a real recommendation_id."""
    return f"REC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_detections = [
        {"class_name": "pothole", "confidence": 0.91, "area_px": 6200},
        {"class_name": "crack", "confidence": 0.68, "area_px": 2100},
    ]

    print("--- Road scenario ---")
    result = generate_session_recommendation(
        sample_detections, total_inspected_area_px=1280 * 720,
        asset_type="road", road_id=1, asset_age_years=15, traffic_level="medium",
    )
    print("Priority:", result["priority_level"], "| Action:", result["recommended_action"])
    print("Due:", result["due_date"], "| Est. cost:", result["estimated_cost"])
    for da in result["detection_actions"]:
        print("  ", da["class_name"], "->", da["action"], f"(rule: {da['rule_fired']})")

    print("\n--- Flyover scenario (bridge crack) ---")
    result2 = generate_session_recommendation(
        [{"class_name": "crack", "confidence": 0.55, "area_px": 1500}],
        total_inspected_area_px=1280 * 720,
        asset_type="flyover", road_id=2, asset_age_years=35, traffic_level="high",
    )
    print("Priority:", result2["priority_level"], "| Action:", result2["recommended_action"])
