"""
timeline_service.py
--------------------
Converts the raw, frame-level detection list produced by
`detection_service.process_video()` into a TIME-INDEXED TIMELINE that
the Streamlit dashboard can plot (e.g. Plotly timeline/gantt or a
line chart of damage-severity-over-time), and that the decision
intelligence layer can consume as aggregated features.

Responsibilities:
  - Bucket detections into fixed time intervals (e.g. every 5 seconds)
  - Aggregate count / confidence / damage-type mix per interval
  - Identify "critical moments" (frames worth manual review)
  - Export the timeline to CSV for offline analysis or reporting
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union

logger = logging.getLogger("timeline_service")
logging.basicConfig(level=logging.INFO)

try:
    from config import CSV_EXPORTS_DIR, RISK_ALERT_THRESHOLD
except ImportError:
    CSV_EXPORTS_DIR = Path("data/exports/csv_exports")
    CSV_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RISK_ALERT_THRESHOLD = 0.75


# ---------------------------------------------------------------------
# CONVERSION HELPERS
# ---------------------------------------------------------------------
def detections_to_dataframe(detections: List[Dict]) -> pd.DataFrame:
    """
    Converts the flat list of per-frame detection dicts (as produced by
    DetectionService.process_video) into a pandas DataFrame for analysis.

    Expected input dict shape:
        {"frame_number", "timestamp_sec", "class_name", "confidence",
         "bbox", "area_px"}
    """
    if not detections:
        return pd.DataFrame(
            columns=["frame_number", "timestamp_sec", "class_name", "confidence", "area_px"]
        )

    df = pd.DataFrame(detections)
    # Keep only the columns relevant to timeline analytics
    keep_cols = [c for c in ["frame_number", "timestamp_sec", "class_name", "confidence", "area_px"] if c in df.columns]
    return df[keep_cols].sort_values("timestamp_sec").reset_index(drop=True)


# ---------------------------------------------------------------------
# TIMELINE BUCKETING / AGGREGATION
# ---------------------------------------------------------------------
def build_timeline(
    detections: List[Dict],
    interval_seconds: float = 5.0,
) -> pd.DataFrame:
    """
    Buckets detections into fixed-width time intervals and aggregates
    them per (interval, damage class).

    Returns a DataFrame with columns:
        interval_start, interval_end, class_name,
        detection_count, avg_confidence, max_confidence, total_area_px

    This is the primary structure the Streamlit dashboard's Plotly
    timeline / severity-over-time chart should be built from.
    """
    df = detections_to_dataframe(detections)
    if df.empty:
        logger.info("No detections supplied - returning empty timeline.")
        return pd.DataFrame(
            columns=[
                "interval_start", "interval_end", "class_name",
                "detection_count", "avg_confidence", "max_confidence", "total_area_px",
            ]
        )

    df["interval_index"] = (df["timestamp_sec"] // interval_seconds).astype(int)

    grouped = (
        df.groupby(["interval_index", "class_name"])
        .agg(
            detection_count=("confidence", "count"),
            avg_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
            total_area_px=("area_px", "sum"),
        )
        .reset_index()
    )

    grouped["interval_start"] = grouped["interval_index"] * interval_seconds
    grouped["interval_end"] = grouped["interval_start"] + interval_seconds
    grouped["avg_confidence"] = grouped["avg_confidence"].round(4)
    grouped["max_confidence"] = grouped["max_confidence"].round(4)

    timeline_df = grouped[
        [
            "interval_start", "interval_end", "class_name",
            "detection_count", "avg_confidence", "max_confidence", "total_area_px",
        ]
    ].sort_values(["interval_start", "class_name"]).reset_index(drop=True)

    return timeline_df


# ---------------------------------------------------------------------
# CRITICAL MOMENT IDENTIFICATION
# ---------------------------------------------------------------------
def get_critical_moments(
    detections: List[Dict],
    confidence_threshold: float = RISK_ALERT_THRESHOLD,
) -> pd.DataFrame:
    """
    Filters raw detections down to high-confidence findings that warrant
    manual review or an immediate alert - e.g. for surfacing "jump to
    this timestamp" markers on the video player in the dashboard.

    Returns a DataFrame sorted by timestamp, deduplicated so consecutive
    frames of the same ongoing defect don't spam the review list (keeps
    only the highest-confidence frame per 3-second window per class).
    """
    df = detections_to_dataframe(detections)
    if df.empty:
        return df

    critical = df[df["confidence"] >= confidence_threshold].copy()
    if critical.empty:
        return critical

    critical["dedup_bucket"] = (critical["timestamp_sec"] // 3).astype(int)
    critical = (
        critical.sort_values("confidence", ascending=False)
        .drop_duplicates(subset=["dedup_bucket", "class_name"])
        .drop(columns=["dedup_bucket"])
        .sort_values("timestamp_sec")
        .reset_index(drop=True)
    )
    return critical


# ---------------------------------------------------------------------
# SUMMARY STATISTICS (useful for dashboard KPI cards)
# ---------------------------------------------------------------------
def summarize_video_detections(detections: List[Dict]) -> Dict:
    """
    Produces headline KPIs for a processed video:
        total_detections, unique_damage_types, most_common_type,
        avg_confidence, peak_confidence, critical_count
    """
    df = detections_to_dataframe(detections)
    if df.empty:
        return {
            "total_detections": 0,
            "unique_damage_types": 0,
            "most_common_type": None,
            "avg_confidence": 0.0,
            "peak_confidence": 0.0,
            "critical_count": 0,
        }

    critical_df = get_critical_moments(detections)

    return {
        "total_detections": int(len(df)),
        "unique_damage_types": int(df["class_name"].nunique()),
        "most_common_type": df["class_name"].mode().iat[0] if not df["class_name"].mode().empty else None,
        "avg_confidence": round(float(df["confidence"].mean()), 4),
        "peak_confidence": round(float(df["confidence"].max()), 4),
        "critical_count": int(len(critical_df)),
    }


# ---------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------
def export_timeline_csv(
    timeline_df: pd.DataFrame,
    file_name: str,
    output_dir: Union[str, Path] = CSV_EXPORTS_DIR,
) -> str:
    """Writes a timeline (or any detections DataFrame) to CSV, returns the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / file_name
    timeline_df.to_csv(out_path, index=False)
    logger.info(f"Timeline exported -> {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------
# CLI test entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal smoke test with synthetic data - no model/video required.
    sample_detections = [
        {"frame_number": 0, "timestamp_sec": 0.0, "class_name": "crack", "confidence": 0.62, "bbox": (10, 10, 50, 50), "area_px": 1600},
        {"frame_number": 15, "timestamp_sec": 0.5, "class_name": "pothole", "confidence": 0.81, "bbox": (20, 20, 90, 90), "area_px": 4900},
        {"frame_number": 60, "timestamp_sec": 2.0, "class_name": "crack", "confidence": 0.44, "bbox": (5, 5, 40, 40), "area_px": 1225},
    ]

    tl = build_timeline(sample_detections, interval_seconds=5.0)
    print("Timeline:\n", tl)

    print("\nCritical moments:\n", get_critical_moments(sample_detections, confidence_threshold=0.6))
    print("\nSummary:\n", summarize_video_detections(sample_detections))
