"""
score_utils.py
--------------
Shared constants and pure math helpers used by every analytics/decision
service in this module. Centralizing these means the severity weights,
category thresholds, and asset assumptions are tuned in exactly ONE
place instead of being duplicated (and drifting) across services.

Nothing in this file touches a model, a database, or an image - it is
100% deterministic, side-effect-free math, which makes it trivial to
unit test.
"""

from typing import Dict, List, Tuple, Union

# ---------------------------------------------------------------------
# DAMAGE SEVERITY WEIGHTS
# ---------------------------------------------------------------------
# How structurally serious each damage class is, relative to one another.
# 1.0 = most severe. These are starting assumptions - recalibrate once
# you have historical repair/failure outcomes to validate against.
DAMAGE_SEVERITY_WEIGHTS: Dict[str, float] = {
    "pothole": 1.0,
    "joint_failure": 0.95,
    "spalling": 0.85,
    "crack": 0.55,
    "surface_erosion": 0.45,
}
DEFAULT_SEVERITY_WEIGHT = 0.5  # fallback for any unrecognized class

# Classes considered "structural" (affect load-bearing integrity,
# particularly relevant for flyovers) vs. purely surface-level.
STRUCTURAL_DAMAGE_CLASSES = {"joint_failure", "spalling"}

# ---------------------------------------------------------------------
# ASSET ASSUMPTIONS
# ---------------------------------------------------------------------
ASSET_BASE_LIFESPAN_YEARS: Dict[str, float] = {
    "road": 15.0,
    "flyover": 50.0,
}
DEFAULT_ASSET_LIFESPAN_YEARS = 20.0

ASSET_CRITICALITY_SCORE: Dict[str, float] = {
    "flyover": 100.0,   # failure risk affects many lives / major traffic artery
    "road": 60.0,
}
DEFAULT_ASSET_CRITICALITY = 60.0

TRAFFIC_LOAD_MULTIPLIER: Dict[str, float] = {
    "low": 1.0,
    "medium": 1.15,
    "high": 1.30,
}
DEFAULT_TRAFFIC_MULTIPLIER = 1.0

# ---------------------------------------------------------------------
# CATEGORY THRESHOLDS (apply to any 0-100 score: risk, emergency, etc.)
# List of (label, inclusive_upper_bound), MUST be sorted ascending.
# ---------------------------------------------------------------------
SCORE_CATEGORY_THRESHOLDS: List[Tuple[str, float]] = [
    ("Low", 25.0),
    ("Medium", 50.0),
    ("High", 75.0),
    ("Critical", 100.0),
]

# Same idea but for a 0-1 severity score (used internally before scaling to 100)
SEVERITY_0_1_THRESHOLDS: List[Tuple[str, float]] = [
    ("Low", 0.25),
    ("Medium", 0.50),
    ("High", 0.75),
    ("Critical", 1.0),
]


# ---------------------------------------------------------------------
# GENERIC MATH HELPERS
# ---------------------------------------------------------------------
def clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamps value into [low, high]."""
    return max(low, min(high, value))


def round2(value: float) -> float:
    """Rounds to 2 decimal places - used for all user-facing scores."""
    return round(float(value), 2)


def normalize_0_1(value: float, min_val: float, max_val: float) -> float:
    """
    Min-max normalizes `value` into [0, 1] given an expected [min_val, max_val]
    range. Values outside the range are clipped rather than extrapolated.
    """
    if max_val <= min_val:
        return 0.0
    return clip((value - min_val) / (max_val - min_val), 0.0, 1.0)


def weighted_sum(values: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Computes sum(values[k] * weights[k]) for every key present in both dicts.
    Silently ignores keys missing from either side (keeps callers flexible).
    """
    return sum(values[k] * weights[k] for k in values.keys() & weights.keys())


def score_to_category(score_0_100: float,
                       thresholds: List[Tuple[str, float]] = SCORE_CATEGORY_THRESHOLDS) -> str:
    """
    Maps a 0-100 score to a category label using ascending (label, upper_bound)
    thresholds, e.g. [("Low", 25), ("Medium", 50), ("High", 75), ("Critical", 100)].
    """
    score_0_100 = clip(score_0_100, 0.0, 100.0)
    for label, upper_bound in thresholds:
        if score_0_100 <= upper_bound:
            return label
    return thresholds[-1][0]  # fallback: last (most severe) category


def get_severity_weight(class_name: str) -> float:
    """Looks up the structural-severity weight for a damage class."""
    return DAMAGE_SEVERITY_WEIGHTS.get(class_name, DEFAULT_SEVERITY_WEIGHT)


def pixel_area_to_sqm(area_px: float, pixels_per_meter: float) -> float:
    """
    Converts a bounding-box pixel area to real-world square meters, given a
    camera calibration factor (pixels per linear meter at the road surface).

    IMPORTANT: `pixels_per_meter` must be calibrated per camera/mount
    (height, angle, lens). Without calibration this is a rough estimate -
    treat repair-cost outputs derived from it as directional, not exact,
    until calibration is done.
    """
    if pixels_per_meter <= 0:
        return 0.0
    return area_px / (pixels_per_meter ** 2)


def is_structural_damage(class_name: str) -> bool:
    """True if the damage class affects structural/load-bearing integrity."""
    return class_name in STRUCTURAL_DAMAGE_CLASSES


def get_asset_lifespan(asset_type: str) -> float:
    return ASSET_BASE_LIFESPAN_YEARS.get(asset_type, DEFAULT_ASSET_LIFESPAN_YEARS)


def get_asset_criticality(asset_type: str) -> float:
    return ASSET_CRITICALITY_SCORE.get(asset_type, DEFAULT_ASSET_CRITICALITY)


def get_traffic_multiplier(traffic_level: Union[str, None]) -> float:
    if not traffic_level:
        return DEFAULT_TRAFFIC_MULTIPLIER
    return TRAFFIC_LOAD_MULTIPLIER.get(traffic_level.lower(), DEFAULT_TRAFFIC_MULTIPLIER)
