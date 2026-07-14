"""
analytics_service.py
---------------------
Turns raw YOLOv8 detections (from detection_service) into the three
foundational metrics every downstream service (health, risk, emergency,
repair cost, RUL, simulation) builds on:

  1. Damage Count       - how many defects, and of what type
  2. Damage Percentage  - what fraction of the inspected surface is damaged
  3. Damage Severity     - a 0-1 index + Low/Medium/High/Critical class

Input contract:
    detections: List[Dict] where each dict has at minimum
        {"class_name": str, "confidence": float (0-1), "area_px": float}
    total_inspected_area_px: total pixel area actually inspected
        - single image  -> image_height * image_width
        - video         -> frame_area * number_of_frames_sampled
          (so "damage percentage" reflects average coverage per sampled
          frame, not double-counting the same defect across frames)
"""

import logging
from collections import Counter
from typing import List, Dict

from score_utils import (
    clip,
    round2,
    get_severity_weight,
    score_to_category,
    SEVERITY_0_1_THRESHOLDS,
)

logger = logging.getLogger("analytics_service")
logging.basicConfig(level=logging.INFO)

# Tunable weights for the composite severity formula - see compute_damage_severity()
SEVERITY_MEAN_WEIGHT = 0.7        # weight on per-instance severity (confidence x class weight)
SEVERITY_COVERAGE_WEIGHT = 0.3    # weight on overall damage percentage contribution
MAX_COUNT_BONUS = 0.5             # cap on the "more instances = worse" bonus
COUNT_BONUS_SCALE = 0.02          # bonus per additional detection instance


# ---------------------------------------------------------------------
# 1. DAMAGE COUNT
# ---------------------------------------------------------------------
def compute_damage_count(detections: List[Dict]) -> Dict:
    """
    Formula: trivial aggregation - no weighting, pure counting.

        total_count   = len(detections)
        count_by_class = Counter(d["class_name"] for d in detections)

    Returns:
        {"total_count": int, "count_by_class": {class_name: int, ...}}
    """
    count_by_class = Counter(d["class_name"] for d in detections)
    return {
        "total_count": len(detections),
        "count_by_class": dict(count_by_class),
    }


# ---------------------------------------------------------------------
# 2. DAMAGE PERCENTAGE
# ---------------------------------------------------------------------
def compute_damage_percentage(detections: List[Dict], total_inspected_area_px: float) -> float:
    """
    Formula:
        Damage % = ( Σ area_px(detection) / total_inspected_area_px ) * 100

    Overlapping bounding boxes are NOT deduplicated (a conservative
    simplification) - if two boxes overlap heavily this will slightly
    overstate coverage. Clipped to 100%.
    """
    if total_inspected_area_px <= 0:
        logger.warning("total_inspected_area_px <= 0; returning 0% damage.")
        return 0.0

    total_damage_area = sum(d.get("area_px", 0.0) for d in detections)
    percentage = (total_damage_area / total_inspected_area_px) * 100
    return round2(clip(percentage, 0.0, 100.0))


# ---------------------------------------------------------------------
# 3. DAMAGE SEVERITY
# ---------------------------------------------------------------------
def compute_damage_severity(detections: List[Dict], total_inspected_area_px: float) -> Dict:
    """
    Composite Damage Severity Index (DSI), 0-1 scale.

    Step 1 - Per-instance severity:
        s_i = confidence_i * severity_weight(class_i)
        (e.g. a 0.9-confidence pothole ~ 0.9*1.0 = 0.9; a 0.6-confidence
         hairline crack ~ 0.6*0.55 = 0.33)

    Step 2 - Mean instance severity across all detections:
        mean_severity = mean(s_i)   [0 if no detections]

    Step 3 - Coverage term (reuses Damage Percentage):
        coverage_term = damage_percentage / 100

    Step 4 - Count bonus (more simultaneous defects = compounding risk,
             capped so a few hundred tiny cracks don't dominate):
        count_bonus = min(MAX_COUNT_BONUS, count * COUNT_BONUS_SCALE)

    Final:
        DSI = clip( SEVERITY_MEAN_WEIGHT * mean_severity
                    + SEVERITY_COVERAGE_WEIGHT * coverage_term
                    + count_bonus ,
                    0, 1 )

        severity_class = Low / Medium / High / Critical via SEVERITY_0_1_THRESHOLDS

    Returns:
        {"severity_score": float (0-1), "severity_class": str}
    """
    if not detections:
        return {"severity_score": 0.0, "severity_class": "Low"}

    per_instance_scores = [
        d.get("confidence", 0.0) * get_severity_weight(d["class_name"])
        for d in detections
    ]
    mean_severity = sum(per_instance_scores) / len(per_instance_scores)

    damage_percentage = compute_damage_percentage(detections, total_inspected_area_px)
    coverage_term = damage_percentage / 100

    count_bonus = min(MAX_COUNT_BONUS, len(detections) * COUNT_BONUS_SCALE)

    dsi = clip(
        SEVERITY_MEAN_WEIGHT * mean_severity
        + SEVERITY_COVERAGE_WEIGHT * coverage_term
        + count_bonus,
        0.0, 1.0,
    )

    # Reuse score_to_category on a 0-100 scale for consistency with other services
    severity_class = score_to_category(dsi * 100, thresholds=[
        (label, upper * 100) for label, upper in SEVERITY_0_1_THRESHOLDS
    ])

    return {"severity_score": round2(dsi), "severity_class": severity_class}


# ---------------------------------------------------------------------
# CONVENIENCE: run all three at once
# ---------------------------------------------------------------------
def analyze(detections: List[Dict], total_inspected_area_px: float) -> Dict:
    """
    Single entry point most callers should use - returns damage count,
    percentage, and severity together, since almost every downstream
    service needs all three.
    """
    return {
        "damage_count": compute_damage_count(detections),
        "damage_percentage": compute_damage_percentage(detections, total_inspected_area_px),
        "damage_severity": compute_damage_severity(detections, total_inspected_area_px),
    }


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_detections = [
        {"class_name": "pothole", "confidence": 0.86, "area_px": 5200},
        {"class_name": "crack", "confidence": 0.61, "area_px": 1800},
        {"class_name": "crack", "confidence": 0.55, "area_px": 900},
        {"class_name": "joint_failure", "confidence": 0.72, "area_px": 3000},
    ]
    frame_area = 1280 * 720

    result = analyze(sample_detections, frame_area)
    print("Damage Count:     ", result["damage_count"])
    print("Damage Percentage:", result["damage_percentage"], "%")
    print("Damage Severity:  ", result["damage_severity"])
