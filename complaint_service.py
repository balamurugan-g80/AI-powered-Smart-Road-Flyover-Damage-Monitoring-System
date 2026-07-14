"""
complaint_service.py
---------------------
Municipality Complaint Generator: turns a detection/recommendation
into a formal, submission-ready civic grievance addressed to the
relevant municipal department, referencing the schema fields already
in ARCHITECTURE.md (`roads`, `maintenance_recommendations`, `reports`).

Two outputs:
  1. generate_complaint()      -> structured dict (for the DB / API / UI)
  2. format_complaint_letter() -> a formatted plain-text letter body,
                                   ready to email, print, or hand to
                                   ReportLab (core/reporting) as the
                                   body text of a PDF.

This file intentionally produces GENERIC placeholder salutations
("To the Executive Engineer / Municipal Commissioner") rather than any
real named official, and no real department contact details - fill
those in from your municipality's actual directory/config before
sending.
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("complaint_service")
logging.basicConfig(level=logging.INFO)

try:
    from config import REPORTS_DIR
    COMPLAINTS_DIR = Path(REPORTS_DIR) / "complaints"
except ImportError:
    COMPLAINTS_DIR = Path("data/exports/reports/complaints")
COMPLAINTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# DEPARTMENT ROUTING
# ---------------------------------------------------------------------
# Which civic department a complaint gets routed to, by asset type.
# Structural-concern actions (from recommendation_service) always
# route to the Bridges & Structures desk even on a nominally "road"
# asset, since that action label only fires for genuine structural risk.
DEPARTMENT_BY_ASSET_TYPE = {
    "road": "Public Works Department - Roads Maintenance Division",
    "flyover": "Bridges & Structures Division",
}
STRUCTURAL_ACTIONS = {"Structural Inspection", "Emergency Structural Inspection & Closure Assessment"}
DEFAULT_DEPARTMENT = "Public Works Department - General Complaints Cell"

DAMAGE_DESCRIPTION_TEMPLATES = {
    "pothole": "a pothole",
    "crack": "surface/structural cracking",
    "spalling": "concrete spalling (surface material breaking away, exposing underlying structure)",
    "joint_failure": "expansion joint failure",
    "surface_erosion": "surface erosion / loss of wearing course",
}


def _route_department(asset_type: str, recommended_action: str) -> str:
    if recommended_action in STRUCTURAL_ACTIONS:
        return DEPARTMENT_BY_ASSET_TYPE.get("flyover", DEFAULT_DEPARTMENT)
    return DEPARTMENT_BY_ASSET_TYPE.get((asset_type or "road").lower(), DEFAULT_DEPARTMENT)


def generate_complaint_id(road_id: Optional[int] = None) -> str:
    """e.g. CMP-R2-20260709-A1B2"""
    road_tag = f"R{road_id}" if road_id is not None else "RX"
    return f"CMP-{road_tag}-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"


# ---------------------------------------------------------------------
# STRUCTURED COMPLAINT RECORD
# ---------------------------------------------------------------------
def generate_complaint(
    recommendation: Dict,
    road_name: str,
    asset_type: str = "road",
    location_description: Optional[str] = None,
    gps: Optional[Tuple[float, float]] = None,
    road_id: Optional[int] = None,
    inspector_name: Optional[str] = None,
    evidence_path: Optional[str] = None,
    citizen_contact: Optional[str] = None,
) -> Dict:
    """
    Builds a structured complaint record from a recommendation_service
    output (generate_session_recommendation()) plus location context.

    Returns a dict with every field needed to render a letter, persist
    to the `reports`/`maintenance_recommendations` tables, or return
    from an API endpoint.
    """
    detection_actions = recommendation.get("detection_actions", [])
    damage_classes = sorted({d["class_name"] for d in detection_actions}) or ["unspecified damage"]
    damage_phrases = [DAMAGE_DESCRIPTION_TEMPLATES.get(c, c) for c in damage_classes]

    department = _route_department(asset_type, recommendation.get("recommended_action", ""))

    return {
        "complaint_id": generate_complaint_id(road_id),
        "date_filed": datetime.now(timezone.utc).date().isoformat(),
        "road_id": road_id,
        "road_name": road_name,
        "asset_type": asset_type,
        "location_description": location_description or "Location not specified - see GPS coordinates",
        "gps_coordinates": gps,
        "department": department,
        "damage_types": damage_classes,
        "damage_description": ", ".join(damage_phrases),
        "priority_level": recommendation.get("priority_level", "low"),
        "recommended_action": recommendation.get("recommended_action", "General Maintenance Inspection"),
        "estimated_cost": recommendation.get("estimated_cost"),
        "due_date": recommendation.get("due_date"),
        "inspector_name": inspector_name or "AI Monitoring System (Automated Detection)",
        "evidence_path": evidence_path,
        "citizen_contact": citizen_contact,
        "status": "Submitted",
    }


# ---------------------------------------------------------------------
# FORMATTED LETTER
# ---------------------------------------------------------------------
def format_complaint_letter(complaint: Dict) -> str:
    """Renders `complaint` into a formatted plain-text grievance letter."""
    gps_line = (
        f"GPS Coordinates: {complaint['gps_coordinates'][0]}, {complaint['gps_coordinates'][1]}"
        if complaint.get("gps_coordinates") else "GPS Coordinates: Not available"
    )
    cost = complaint.get("estimated_cost")
    cost_line = f"Estimated Repair Cost: {cost:,.0f}" if cost is not None else "Estimated Repair Cost: Pending assessment"

    letter = f"""
COMPLAINT REFERENCE: {complaint['complaint_id']}
DATE FILED: {complaint['date_filed']}

To,
The Executive Engineer / Municipal Commissioner
{complaint['department']}

Subject: Reported Infrastructure Damage - {complaint['road_name']} - Priority: {complaint['priority_level'].upper()}

Sir/Madam,

This is to bring to your attention that automated infrastructure
monitoring has identified {complaint['damage_description']} on the
following asset:

    Asset Name       : {complaint['road_name']}
    Asset Type       : {complaint['asset_type'].capitalize()}
    Location         : {complaint['location_description']}
    {gps_line}

Based on AI-assisted severity and risk analysis, this finding has been
classified as {complaint['priority_level'].upper()} priority, with the
following recommended corrective action:

    Recommended Action : {complaint['recommended_action']}
    Target Resolution By: {complaint['due_date']}
    {cost_line}

We request that this matter be inspected and addressed within the
timeframe stated above, in the interest of public safety. Photographic/
video evidence supporting this finding is referenced below:

    Evidence Reference: {complaint.get('evidence_path') or 'Attached separately'}

Reported by: {complaint['inspector_name']}
Status: {complaint['status']}

Thank you for your prompt attention to this matter.

Regards,
Smart Road & Flyover Damage Monitoring System
""".strip("\n")
    return letter


def save_complaint(complaint: Dict, letter_text: Optional[str] = None) -> str:
    """
    Writes the complaint letter to COMPLAINTS_DIR as a .txt file (a
    ReportLab-based PDF version can wrap this same text via
    core/reporting/pdf_report_generator.py). Returns the file path.
    """
    letter_text = letter_text or format_complaint_letter(complaint)
    file_path = COMPLAINTS_DIR / f"{complaint['complaint_id']}.txt"
    file_path.write_text(letter_text, encoding="utf-8")
    logger.info(f"Complaint saved to {file_path}")
    return str(file_path)


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_recommendation = {
        "priority_level": "urgent",
        "recommended_action": "Structural Inspection",
        "due_date": "2026-07-16",
        "estimated_cost": 18750.0,
        "detection_actions": [
            {"class_name": "crack", "action": "Structural Inspection", "rule_fired": "flyover_structural_concern"},
        ],
    }
    complaint = generate_complaint(
        sample_recommendation,
        road_name="Anna Salai Flyover",
        asset_type="flyover",
        location_description="Near Teynampet junction, southbound span",
        gps=(13.0432, 80.2456),
        road_id=2,
        evidence_path="data/processed/annotated_images/anna_salai_flyover_004.jpg",
    )
    print("Complaint ID:", complaint["complaint_id"])
    print("Routed to:", complaint["department"])
    print()
    letter = format_complaint_letter(complaint)
    print(letter)
    saved_path = save_complaint(complaint, letter)
    print("\nSaved to:", saved_path)
