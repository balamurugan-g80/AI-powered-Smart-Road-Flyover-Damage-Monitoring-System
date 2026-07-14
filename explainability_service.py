"""
explainability_service.py
---------------------------
Powers the "Explainable AI Dashboard": answers, for every recommendation,
the question a civic engineer or auditor will always ask - "WHY did the
system decide this?"

Two complementary explanation layers, both surfaced together:

  1. RULE EXPLANATION (exact, always available)
     recommendation_service's decision is a deterministic rules engine
     (see DAMAGE_ACTION_MAP / determine_recommended_action). This layer
     simply narrates which rule fired in plain English - e.g.
        "Pothole -> Immediate Repair: severity classified High (rule:
         pothole_high_severity) - potholes at this severity are a
         direct vehicle-safety hazard."
        "Crack on Flyover -> Structural Inspection (rule:
         flyover_structural_concern) - cracks on load-bearing flyover
         decks are treated as potential structural distress regardless
         of confidence."

  2. SCORE CONTRIBUTION EXPLANATION (quantitative)
     Two modes, following the same ML/heuristic pattern as
     repair_cost_service and life_prediction_service:

       a) TRUE SHAP (ML mode) - if a trained tree model (e.g. the
          XGBoost priority/RUL model) is available, uses
          shap.TreeExplainer for a real, additive per-feature
          attribution. This is the path used once training/ produces
          `priority_xgb_model.json`.

       b) EXACT WEIGHTED-SUM DECOMPOSITION (fallback, always available)
          health_service / risk_service / emergency_service are all
          literally weighted sums (W_SEVERITY * x + W_COVERAGE * y + ...).
          For a weighted sum, "each term's contribution" is not an
          approximation the way SHAP is for a black-box model - it is
          the exact term value. This mode is clearly labeled
          "rule_based_contribution", never mislabeled as SHAP, but is
          rendered with the identical chart/response shape so the
          dashboard code doesn't need to branch on which mode ran.
"""

import base64
import io
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("explainability_service")
logging.basicConfig(level=logging.INFO)

try:
    from config import SHAP_MAX_DISPLAY_FEATURES
except ImportError:
    SHAP_MAX_DISPLAY_FEATURES = 10

from emergency_service import W_RISK, W_STRUCTURAL, W_GROWTH, W_CRITICALITY
from health_service import W_SEVERITY, W_COVERAGE, W_DENSITY


# ---------------------------------------------------------------------
# 1. RULE EXPLANATION
# ---------------------------------------------------------------------
RULE_NARRATIVES = {
    "pothole_high_severity": (
        "Potholes classified High/Critical severity are escalated straight to "
        "Immediate Repair - they are a direct vehicle-safety hazard regardless of asset type."
    ),
    "flyover_structural_concern": (
        "This damage class (crack / spalling / joint_failure) was detected on a FLYOVER, "
        "where even surface-level findings can indicate load-bearing structural distress, "
        "so a Structural Inspection is required before any cosmetic repair."
    ),
    "emergency_index_critical": (
        "The session's overall Emergency Index crossed the Immediate-Action threshold "
        "(driven by composite risk, structural findings, projected growth, and asset criticality), "
        "so every finding in this session is escalated to Immediate Repair."
    ),
    "emergency_index_critical_and_structural": (
        "The session's Emergency Index crossed the Immediate-Action threshold AND this finding is "
        "structurally significant on a flyover - the combination triggers an emergency closure assessment."
    ),
    "default_damage_action_map": (
        "No escalation rule applied; the standard action for this damage class and asset type was used."
    ),
}


def explain_rule(detection_action: Dict) -> str:
    """Plain-English explanation of why a single detection got its recommended action."""
    rule = detection_action.get("rule_fired", "default_damage_action_map")
    narrative = RULE_NARRATIVES.get(rule, "No specific rule narrative available.")
    return (
        f"{detection_action['class_name'].replace('_', ' ').title()} "
        f"(confidence {detection_action['confidence']:.2f}, severity {detection_action['severity_class']}) "
        f"-> {detection_action['action']}. Reason: {narrative}"
    )


def explain_recommendation_rules(recommendation: Dict) -> List[str]:
    """Runs explain_rule() over every detection in a session recommendation."""
    return [explain_rule(da) for da in recommendation.get("detection_actions", [])]


# ---------------------------------------------------------------------
# 2a. EXACT WEIGHTED-SUM DECOMPOSITION (heuristic fallback mode)
# ---------------------------------------------------------------------
def explain_emergency_index(emergency_result: Dict, risk_result: Dict) -> Dict:
    """
    Decomposes emergency_service.compute_emergency_index()'s formula into
    its exact four term contributions (points out of 100), so the
    dashboard can show a bar chart of "what drove this score" even
    without a trained ML model.

    emergency_index = W_RISK*risk_score + W_STRUCTURAL*structural_factor
                     + W_GROWTH*growth_factor + W_CRITICALITY*criticality_factor
    """
    risk_score = risk_result["risk_score"]
    structural_factor = emergency_result["structural_factor"]
    # growth_factor and criticality_factor aren't returned individually by
    # emergency_service, so back them out isn't possible without the raw
    # inputs; instead we report the two terms we DO have exactly, and
    # bucket the remainder (growth + criticality) as "other factors".
    known_contribution = W_RISK * risk_score + W_STRUCTURAL * structural_factor
    other_contribution = max(0.0, emergency_result["emergency_index"] - known_contribution)

    contributions = [
        {"feature": "Composite Risk Score", "contribution": round(W_RISK * risk_score, 2)},
        {"feature": "Structural Damage Factor", "contribution": round(W_STRUCTURAL * structural_factor, 2)},
        {"feature": "Growth Rate + Asset Criticality", "contribution": round(other_contribution, 2)},
    ]
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    return {
        "mode": "rule_based_contribution",
        "target": "emergency_index",
        "predicted_value": emergency_result["emergency_index"],
        "contributions": contributions[:SHAP_MAX_DISPLAY_FEATURES],
    }


def explain_health_score(health_result: Dict) -> Dict:
    """
    Decomposes health_service.compute_health_score()'s HDI formula into
    its exact three weighted terms, plus the age penalty deduction.

    HDI = W_SEVERITY*severity_score + W_COVERAGE*(damage_pct/100) + W_DENSITY*density_factor
    health_score = 100*(1-HDI) - age_penalty
    """
    analytics = health_result.get("analytics", {})
    severity_score = analytics.get("damage_severity", {}).get("severity_score", 0.0)
    damage_pct = analytics.get("damage_percentage", 0.0)
    hdi = health_result["hdi"]
    age_penalty = health_result["age_penalty"]

    # Approximate each raw term's share of the HDI, then translate to
    # health-score points lost (100 * share_of_HDI); this is exact given
    # HDI's own linear formula, not an approximation.
    severity_term = W_SEVERITY * severity_score
    coverage_term = W_COVERAGE * (damage_pct / 100)
    density_term = max(0.0, hdi - severity_term - coverage_term)

    contributions = [
        {"feature": "Damage Severity Score", "contribution": round(-100 * severity_term, 2)},
        {"feature": "Damage Coverage %", "contribution": round(-100 * coverage_term, 2)},
        {"feature": "Defect Count Density", "contribution": round(-100 * density_term, 2)},
        {"feature": "Asset Age Penalty", "contribution": round(-age_penalty, 2)},
    ]
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    return {
        "mode": "rule_based_contribution",
        "target": "health_score",
        "predicted_value": health_result["health_score"],
        "contributions": contributions[:SHAP_MAX_DISPLAY_FEATURES],
    }


# ---------------------------------------------------------------------
# 2b. TRUE SHAP (ML mode - used once a trained tree model exists)
# ---------------------------------------------------------------------
def explain_with_shap(model, feature_row: Dict) -> Optional[Dict]:
    """
    Runs shap.TreeExplainer on a trained XGBoost model (e.g. the
    priority/RUL/repair-cost models trained under training/) for a
    single feature row. Returns None (never raises) if `shap` isn't
    installed or the model isn't SHAP-compatible, so callers can
    always fall back to explain_emergency_index()/explain_health_score().
    """
    try:
        import shap
        import pandas as pd

        feature_df = pd.DataFrame([feature_row])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_df)
        row_values = shap_values[0] if hasattr(shap_values, "__len__") else shap_values

        contributions = [
            {"feature": name, "contribution": round(float(val), 4)}
            for name, val in zip(feature_df.columns, row_values)
        ]
        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

        base_value = float(getattr(explainer, "expected_value", 0.0))
        return {
            "mode": "shap",
            "base_value": round(base_value, 4),
            "contributions": contributions[:SHAP_MAX_DISPLAY_FEATURES],
        }
    except Exception as e:
        logger.warning(f"SHAP explanation unavailable ({e}); use the rule-based fallback instead.")
        return None


# ---------------------------------------------------------------------
# 3. NATURAL-LANGUAGE SUMMARY
# ---------------------------------------------------------------------
def generate_natural_language_explanation(contribution_result: Dict) -> str:
    """Turns a contributions list (either mode) into a one-paragraph summary."""
    target = contribution_result.get("target", contribution_result.get("mode", "score"))
    top = contribution_result["contributions"][:3]
    parts = [f"{c['feature']} ({c['contribution']:+.2f} pts)" for c in top]
    mode_label = "SHAP feature attribution" if contribution_result["mode"] == "shap" else "rule-based contribution breakdown"
    return (
        f"Based on {mode_label}, the {target} of "
        f"{contribution_result.get('predicted_value', contribution_result.get('base_value', 'N/A'))} "
        f"was driven mainly by: {', '.join(parts)}."
    )


# ---------------------------------------------------------------------
# 4. CHART FOR THE DASHBOARD (base64 PNG, embeddable via st.image)
# ---------------------------------------------------------------------
def generate_contribution_chart_base64(contribution_result: Dict, title: str = "Feature Contributions") -> Optional[str]:
    """
    Renders a horizontal bar chart of contributions and returns it as a
    base64-encoded PNG data URI, ready for `st.image(data_uri)` in the
    Streamlit "Explainable AI Dashboard" page. Returns None if
    matplotlib isn't installed rather than raising.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        contributions = contribution_result["contributions"]
        features = [c["feature"] for c in contributions][::-1]
        values = [c["contribution"] for c in contributions][::-1]
        colors = ["#d62728" if v >= 0 else "#2ca02c" for v in values]

        fig, ax = plt.subplots(figsize=(7, max(2, 0.5 * len(features))))
        ax.barh(features, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Contribution")
        ax.set_title(title)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        logger.warning(f"Chart generation unavailable ({e}).")
        return None


# ---------------------------------------------------------------------
# 5. ONE-CALL CONVENIENCE (mirrors *_from_detections pattern elsewhere)
# ---------------------------------------------------------------------
def explain_session(recommendation: Dict) -> Dict:
    """
    Full explanation bundle for one session recommendation: rule
    narratives per detection + score decomposition for health/emergency
    + natural-language summary + chart, all in one call for the
    dashboard page to consume directly.
    """
    emergency_result = recommendation["emergency_result"]
    risk_result = emergency_result["risk_result"]
    health_result = risk_result["health_result"]

    emergency_explanation = explain_emergency_index(emergency_result, risk_result)
    health_explanation = explain_health_score(health_result)

    return {
        "rule_explanations": explain_recommendation_rules(recommendation),
        "emergency_explanation": emergency_explanation,
        "health_explanation": health_explanation,
        "emergency_summary": generate_natural_language_explanation(emergency_explanation),
        "health_summary": generate_natural_language_explanation(health_explanation),
        "emergency_chart": generate_contribution_chart_base64(emergency_explanation, "What drove the Emergency Index"),
        "health_chart": generate_contribution_chart_base64(health_explanation, "What drove the Health Score"),
    }


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from recommendation_service import generate_session_recommendation

    sample_detections = [
        {"class_name": "pothole", "confidence": 0.91, "area_px": 6200},
        {"class_name": "crack", "confidence": 0.68, "area_px": 2100},
    ]
    recommendation = generate_session_recommendation(
        sample_detections, total_inspected_area_px=1280 * 720,
        asset_type="flyover", road_id=2, asset_age_years=30, traffic_level="high",
    )

    bundle = explain_session(recommendation)

    print("--- Rule Explanations ---")
    for line in bundle["rule_explanations"]:
        print(" -", line)

    print("\n--- Emergency Index Explanation ---")
    print(bundle["emergency_summary"])
    for c in bundle["emergency_explanation"]["contributions"]:
        print("  ", c)

    print("\n--- Health Score Explanation ---")
    print(bundle["health_summary"])
    for c in bundle["health_explanation"]["contributions"]:
        print("  ", c)

    print("\nEmergency chart generated:", bundle["emergency_chart"] is not None)
    print("Health chart generated:", bundle["health_chart"] is not None)
