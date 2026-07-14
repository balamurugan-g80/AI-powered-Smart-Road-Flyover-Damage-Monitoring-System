"""
simulation_service.py
------------------------
Simulates how damage is likely to progress over future months if left
unrepaired, and re-derives Health Score / Risk Level at each future
step so the dashboard can show a trend line, not just a snapshot.

Model: exponential growth, calibrated by a monthly growth rate that
depends on current severity class (worse damage tends to worsen
faster - a hairline crack grows slowly; an active pothole under
traffic load grows fast). This is a simplified deterministic model,
not a physics-based degradation model - treat it as directional
planning input, not a guarantee.
"""

import logging
from typing import Dict, List, Optional

from score_utils import clip, round2

logger = logging.getLogger("simulation_service")
logging.basicConfig(level=logging.INFO)

# Assumed monthly damage-growth rate by current severity class.
# E.g. "Critical" damage (active potholes, spalling) grows ~7%/month
# if left unaddressed; "Low" severity (fine cracks) grows slowly.
MONTHLY_GROWTH_RATE_BY_SEVERITY: Dict[str, float] = {
    "Low": 0.01,
    "Medium": 0.02,
    "High": 0.04,
    "Critical": 0.07,
}
DEFAULT_MONTHLY_GROWTH_RATE = 0.02

# Optimistic/pessimistic bounds as a +/- multiplier on the base rate,
# giving a confidence band rather than a single deterministic line.
OPTIMISTIC_RATE_MULTIPLIER = 0.5
PESSIMISTIC_RATE_MULTIPLIER = 1.5


def get_monthly_growth_rate(severity_class: str) -> float:
    return MONTHLY_GROWTH_RATE_BY_SEVERITY.get(severity_class, DEFAULT_MONTHLY_GROWTH_RATE)


def _project_damage_percentage(current_damage_pct: float, monthly_rate: float, months: int) -> float:
    """
    Formula: exponential growth
        D(t) = D0 * (1 + r) ^ t
    where D0 = current damage percentage, r = monthly growth rate, t = months elapsed.
    Clipped to 100%.
    """
    projected = current_damage_pct * ((1 + monthly_rate) ** months)
    return clip(projected, 0.0, 100.0)


def simulate_damage_growth(
    current_damage_percentage: float,
    current_severity_score: float,
    severity_class: str,
    total_damage_count: int,
    asset_age_years: Optional[float] = None,
    months_ahead: int = 6,
) -> List[Dict]:
    """
    Projects damage percentage, severity score, health score, and risk
    level forward month-by-month, for `months_ahead` months, under
    three growth scenarios: optimistic, expected, pessimistic.

    Formula per month t (for each scenario's rate r):
        damage_pct(t)   = D0 * (1 + r)^t                          [clipped to 100]
        severity(t)     = clip(severity0 * (1 + r)^t, 0, 1)       [same growth applied]
        health(t)       = health_service.compute_health_score(severity(t), damage_pct(t), count)
        risk(t)         = risk_service.compute_risk_score(health(t), severity_class(t))

    Returns:
        List of monthly snapshots:
        [
          {
            "month": int,
            "expected": {"damage_percentage", "severity_score", "health_score", "risk_score", "risk_level"},
            "optimistic": {...},
            "pessimistic": {...},
          },
          ...
        ]
    """
    from health_service import compute_health_score
    from risk_service import compute_risk_score
    from score_utils import score_to_category, SEVERITY_0_1_THRESHOLDS

    base_rate = get_monthly_growth_rate(severity_class)
    scenarios = {
        "optimistic": base_rate * OPTIMISTIC_RATE_MULTIPLIER,
        "expected": base_rate,
        "pessimistic": base_rate * PESSIMISTIC_RATE_MULTIPLIER,
    }

    timeline = []

    for month in range(1, months_ahead + 1):
        month_snapshot = {"month": month}

        for scenario_name, rate in scenarios.items():
            projected_damage_pct = _project_damage_percentage(current_damage_percentage, rate, month)
            projected_severity = clip(current_severity_score * ((1 + rate) ** month), 0.0, 1.0)
            projected_severity_class = score_to_category(projected_severity * 100, thresholds=[
                (label, upper * 100) for label, upper in SEVERITY_0_1_THRESHOLDS
            ])

            health_result = compute_health_score(
                severity_score=projected_severity,
                damage_percentage=projected_damage_pct,
                total_damage_count=total_damage_count,
                asset_age_years=asset_age_years,
            )
            risk_result = compute_risk_score(
                health_score=health_result["health_score"],
                severity_class=projected_severity_class,
            )

            month_snapshot[scenario_name] = {
                "damage_percentage": round2(projected_damage_pct),
                "severity_score": round2(projected_severity),
                "severity_class": projected_severity_class,
                "health_score": health_result["health_score"],
                "risk_score": risk_result["risk_score"],
                "risk_level": risk_result["risk_level"],
            }

        timeline.append(month_snapshot)

    return timeline


def get_growth_factor_for_emergency_index(timeline: List[Dict], scenario: str = "expected",
                                           lookahead_month: int = 3) -> float:
    """
    Extracts a single 0-100 "growth factor" from a simulation timeline for
    use in emergency_service.compute_emergency_index() - specifically,
    how much the risk score is projected to increase by `lookahead_month`
    months out, normalized to 0-100.
    """
    if not timeline:
        return 0.0

    target = next((m for m in timeline if m["month"] == lookahead_month), timeline[-1])
    projected_risk = target[scenario]["risk_score"]
    return clip(projected_risk, 0.0, 100.0)


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    timeline = simulate_damage_growth(
        current_damage_percentage=8.5,
        current_severity_score=0.55,
        severity_class="High",
        total_damage_count=6,
        asset_age_years=15,
        months_ahead=6,
    )
    for snapshot in timeline:
        print(f"Month {snapshot['month']}: expected -> "
              f"health={snapshot['expected']['health_score']}, "
              f"risk={snapshot['expected']['risk_level']} "
              f"({snapshot['expected']['risk_score']})")

    growth_factor = get_growth_factor_for_emergency_index(timeline, lookahead_month=3)
    print("Growth factor (for Emergency Index, 3-month lookahead):", growth_factor)
