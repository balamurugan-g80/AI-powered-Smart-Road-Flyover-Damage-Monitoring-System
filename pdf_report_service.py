"""
pdf_report_service.py
----------------------
Generates the client-facing "PDF Inspection Report" — the final
artifact of the pipeline, combining every upstream score into one
printable document for engineers, auditors, and municipal officials.

Required sections (per spec):
    - Date
    - Uploaded Image (annotated, if available)
    - Detected Damages + Confidence Scores
    - Health Score
    - Risk Score
    - Emergency Index
    - Repair Cost
    - Remaining Useful Life
    - Recommendations
    - Municipality Complaint (letter text)

Uses reportlab (already in requirements.txt) - no new dependency.
Framework-agnostic: takes plain dicts (the same bundle produced by
app.py's run_full_pipeline()) so it can be called from the dashboard,
a CLI script, or a batch job.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable,
)

logger = logging.getLogger("pdf_report_service")
logging.basicConfig(level=logging.INFO)

try:
    from config import REPORTS_DIR, REPORT_TITLE, REPORT_PAGE_SIZE
except ImportError:
    REPORTS_DIR = Path("data/exports/reports")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TITLE = "Road & Flyover Damage Inspection Report"
    REPORT_PAGE_SIZE = "A4"

PAGE_SIZE = A4

# Priority -> accent color for the header band / badges
PRIORITY_COLORS = {
    "low": colors.HexColor("#2ca02c"),
    "medium": colors.HexColor("#e6b800"),
    "high": colors.HexColor("#ff7f0e"),
    "urgent": colors.HexColor("#d62728"),
}


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontSize=20, leading=24, spaceAfter=6,
                               textColor=colors.HexColor("#1a1a2e"), fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="SectionHeading", fontSize=13, leading=16, spaceBefore=14, spaceAfter=6,
                               textColor=colors.HexColor("#16213e"), fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="MetaText", fontSize=9, textColor=colors.grey))
    styles.add(ParagraphStyle(name="BodyText9", fontSize=9.5, leading=13))
    styles.add(ParagraphStyle(name="LetterBody", fontSize=9, leading=13, fontName="Courier"))
    return styles


def _kpi_table(bundle: Dict, styles) -> Table:
    health = bundle["health_result"]["health_score"]
    risk = bundle["risk_result"]["risk_score"]
    risk_level = bundle["risk_result"]["risk_level"]
    emergency = bundle["emergency_result"]["emergency_index"]
    emergency_level = bundle["emergency_result"]["emergency_level"]
    cost = bundle["repair_cost_result"]["total_cost"]
    rul = bundle["rul_result"]["rul_years"]

    data = [
        ["Health Score", "Risk Score", "Emergency Index", "Repair Cost", "Remaining Useful Life"],
        [
            f"{health:.1f} / 100",
            f"{risk:.1f} ({risk_level})",
            f"{emergency:.1f} ({emergency_level})",
            f"₹{cost:,.0f}",
            f"{rul:.1f} yrs",
        ],
    ]
    table = Table(data, colWidths=[3.4 * cm] * 5)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e8eef7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _detections_table(detections: List[Dict], styles) -> Table:
    header = ["#", "Damage Type", "Confidence", "Area (px)"]
    rows = [header]
    for i, d in enumerate(detections, start=1):
        rows.append([
            str(i),
            d.get("class_name", "unknown").replace("_", " ").title(),
            f"{d.get('confidence', 0.0) * 100:.1f}%",
            f"{d.get('area_px', 0.0):,.0f}" if d.get("area_px") is not None else "-",
        ])
    table = Table(rows, colWidths=[1.2 * cm, 6 * cm, 3.5 * cm, 3.5 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _recommendation_block(recommendation: Dict, styles) -> List:
    flow = []
    priority = recommendation.get("priority_level", "low")
    color = PRIORITY_COLORS.get(priority, colors.grey)

    badge_table = Table(
        [[f"PRIORITY: {priority.upper()}"]], colWidths=[5 * cm],
    )
    badge_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    flow.append(badge_table)
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(f"<b>Recommended Action:</b> {recommendation.get('recommended_action', 'N/A')}", styles["BodyText9"]))
    flow.append(Paragraph(f"<b>Target Resolution By:</b> {recommendation.get('due_date', 'N/A')}", styles["BodyText9"]))
    flow.append(Paragraph(f"<b>Estimated Cost:</b> ₹{recommendation.get('estimated_cost', 0):,.0f}", styles["BodyText9"]))

    detection_actions = recommendation.get("detection_actions", [])
    if detection_actions:
        flow.append(Spacer(1, 6))
        rows = [["Damage", "Confidence", "Severity", "Action"]]
        for da in detection_actions:
            rows.append([
                da.get("class_name", "").replace("_", " ").title(),
                f"{da.get('confidence', 0.0) * 100:.1f}%",
                da.get("severity_class", ""),
                da.get("action", ""),
            ])
        t = Table(rows, colWidths=[3.5 * cm, 2.5 * cm, 2.5 * cm, 5.5 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
        ]))
        flow.append(t)
    return flow


def generate_pdf_report(
    bundle: Dict,
    road_name: str,
    asset_type: str,
    annotated_image_path: Optional[str] = None,
    complaint_letter_text: Optional[str] = None,
    complaint_id: Optional[str] = None,
    inspector_name: str = "AI Monitoring System (Automated Detection)",
    output_path: Optional[str] = None,
) -> str:
    """
    Builds the full PDF Inspection Report from a scoring `bundle`
    (the dict returned by app.py's run_full_pipeline()) and returns
    the saved file path.

    bundle keys expected: health_result, risk_result, emergency_result,
    repair_cost_result, rul_result, recommendation, and
    health_result["analytics"] (damage_count / damage_percentage / damage_severity).
    """
    styles = _styles()
    report_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in road_name)[:40]
        output_path = str(Path(REPORTS_DIR) / f"inspection_report_{safe_name}_{stamp}.pdf")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path, pagesize=PAGE_SIZE,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
    )
    story = []

    # ---------------- HEADER ----------------
    story.append(Paragraph(REPORT_TITLE, styles["ReportTitle"]))
    story.append(Paragraph(
        f"Asset: <b>{road_name}</b> ({asset_type.capitalize()}) &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Report Date: <b>{report_date}</b>",
        styles["MetaText"],
    ))
    story.append(Paragraph(f"Inspected/Reported by: {inspector_name}", styles["MetaText"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"), spaceBefore=8, spaceAfter=10))

    # ---------------- UPLOADED / ANNOTATED IMAGE ----------------
    story.append(Paragraph("Uploaded Image (Annotated Detection)", styles["SectionHeading"]))
    if annotated_image_path and Path(annotated_image_path).exists():
        try:
            img = Image(annotated_image_path, width=15 * cm, height=9 * cm, kind="proportional")
            story.append(img)
        except Exception as e:
            logger.warning(f"Could not embed image '{annotated_image_path}': {e}")
            story.append(Paragraph("(Image could not be embedded.)", styles["BodyText9"]))
    else:
        story.append(Paragraph("(No image available for this session — video or manual-entry session.)", styles["BodyText9"]))

    # ---------------- KPI SUMMARY ----------------
    story.append(Paragraph("Decision Intelligence Summary", styles["SectionHeading"]))
    story.append(_kpi_table(bundle, styles))

    # ---------------- DETECTED DAMAGES ----------------
    analytics = bundle["health_result"].get("analytics", {})
    detections = bundle.get("detections") or []
    story.append(Paragraph("Detected Damages & Confidence Scores", styles["SectionHeading"]))
    if detections:
        story.append(_detections_table(detections, styles))
    else:
        story.append(Paragraph("No individual detection records attached to this bundle.", styles["BodyText9"]))

    damage_count = analytics.get("damage_count", {})
    damage_pct = analytics.get("damage_percentage")
    severity = analytics.get("damage_severity", {})
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>Total Detections:</b> {damage_count.get('total_count', 'N/A')} &nbsp; | &nbsp; "
        f"<b>Damage Coverage:</b> {damage_pct}% &nbsp; | &nbsp; "
        f"<b>Severity Score:</b> {severity.get('severity_score', 'N/A')} "
        f"({severity.get('severity_class', 'N/A')})",
        styles["BodyText9"],
    ))

    # ---------------- RECOMMENDATIONS ----------------
    story.append(Paragraph("Recommendations", styles["SectionHeading"]))
    story.extend(_recommendation_block(bundle["recommendation"], styles))

    # ---------------- MUNICIPALITY COMPLAINT ----------------
    story.append(PageBreak())
    story.append(Paragraph("Municipality Complaint", styles["SectionHeading"]))
    if complaint_id:
        story.append(Paragraph(f"<b>Complaint Reference:</b> {complaint_id}", styles["BodyText9"]))
        story.append(Spacer(1, 6))
    if complaint_letter_text:
        for line in complaint_letter_text.split("\n"):
            story.append(Paragraph(line.replace(" ", "&nbsp;") if line.strip() == "" else line, styles["LetterBody"]))
    else:
        story.append(Paragraph(
            "No formal complaint has been filed for this session yet. "
            "Use complaint_service.generate_complaint() to create one, then re-generate this report.",
            styles["BodyText9"],
        ))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Paragraph(
        "Generated automatically by the Smart Road & Flyover Damage Monitoring and Decision "
        "Intelligence System. Scores are AI-assisted engineering estimates - treat repair-cost "
        "and RUL figures as directional pending on-site verification and camera calibration.",
        styles["MetaText"],
    ))

    doc.build(story)
    logger.info(f"PDF inspection report generated -> {output_path}")
    return output_path


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_bundle = {
        "detections": [
            {"class_name": "pothole", "confidence": 0.91, "area_px": 6200},
            {"class_name": "crack", "confidence": 0.68, "area_px": 2100},
        ],
        "health_result": {
            "health_score": 58.4,
            "analytics": {
                "damage_count": {"total_count": 2, "count_by_class": {"pothole": 1, "crack": 1}},
                "damage_percentage": 5.3,
                "damage_severity": {"severity_score": 0.52, "severity_class": "Medium"},
            },
        },
        "risk_result": {"risk_score": 47.2, "risk_level": "Medium"},
        "emergency_result": {"emergency_index": 39.6, "emergency_level": "Medium", "requires_immediate_action": False},
        "repair_cost_result": {"total_cost": 18750.0},
        "rul_result": {"rul_years": 9.2},
        "recommendation": {
            "priority_level": "medium",
            "recommended_action": "Pothole Patch Repair",
            "due_date": "2026-08-23",
            "estimated_cost": 18750.0,
            "detection_actions": [
                {"class_name": "pothole", "confidence": 0.91, "severity_class": "High", "action": "Immediate Repair"},
                {"class_name": "crack", "confidence": 0.68, "severity_class": "Medium", "action": "Monitor / Sealant Repair"},
            ],
        },
    }
    path = generate_pdf_report(
        sample_bundle,
        road_name="MG Road Flyover - Segment 4",
        asset_type="flyover",
        complaint_letter_text="COMPLAINT REFERENCE: CMP-R2-20260709-A1B2\n\nSample complaint letter body...",
        complaint_id="CMP-R2-20260709-A1B2",
    )
    print("Report generated at:", path)
