"""
dashboard.py
-------------
Streamlit dashboard for the Smart Road & Flyover Damage Monitoring &
Decision Intelligence System.

Pages:
    Home | Upload Image | Upload Video | Analytics Dashboard |
    Detection History | SHAP Dashboard | Notifications & Complaints

VIDEO PIPELINE (corrected per spec):
    1. Video is read with OpenCV (cv2.VideoCapture) inside
       detection_service.DetectionService.process_video().
    2. Every frame is run through YOLOv8 with conf=0.15, imgsz=640.
    3. Boxes are drawn with `results[0].plot()` (Ultralytics' own
       renderer) - never hand-drawn.
    4. Every annotated frame is written to cv2.VideoWriter at the
       ORIGINAL fps/width/height.
    5. Output is saved to data/processed/annotated_videos/<session>/annotated_video.mp4
    6. Once processing finishes, THIS FILE displays that annotated
       video with st.video(...) - the raw uploaded video is never shown.
    7. While processing, a live progress bar + Current Frame / Frames
       Processed / Remaining Time / Detections Count / FPS panel is shown.
    8. Ultralytics' renderer already overlays damage name + confidence +
       box; detection_service additionally overlays frame number +
       timestamp text on every frame.
    9-14. After processing: detection history, analytics, decision
       intelligence (health/risk/emergency/repair cost/RUL), damage
       growth simulation, SHAP explanation, and an automatic
       notification are all generated and shown on the same page.
    15. Frames containing damage are saved to data/processed/frames/
        (handled inside detection_service.process_video()).
    16. A CSV (Frame Number, Timestamp, Damage Class, Confidence,
        Bounding Box, Severity, Priority) is exported and offered as
        a download.
    17. Every detection is inserted into SQLite via database.py.
    18. "No Damage Detected" is shown when there are zero detections;
        the annotated video is always displayed regardless.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import config
import database
from detection_service import DetectionService
from image_utils import save_uploaded_image, validate_image_file
from video_utils import save_uploaded_video, validate_video_file

logger = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title=config.APP_TITLE, page_icon=config.APP_ICON, layout="wide")
database.init_db()


# =======================================================================
# CACHED RESOURCES
# =======================================================================
@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def get_detection_service() -> Optional[DetectionService]:
    try:
        return DetectionService()
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model: {e}")
        return None


# =======================================================================
# SESSION STATE DEFAULTS
# =======================================================================
_DEFAULTS = {
    "session_id": None,
    "current_detections": [],
    "current_road_name": "",
    "current_asset_type": "road",
    "annotated_video_path": None,
    "annotated_image_path": None,
    "last_pipeline": None,
    "video_result": None,
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)


# =======================================================================
# FULL DECISION-INTELLIGENCE PIPELINE (shared by Image + Video pages)
# =======================================================================
def run_full_pipeline(
    detections: List[Dict],
    total_area_px: float,
    asset_type: str,
    road_name: str,
    asset_age_years: float,
    traffic_level: str,
    session_id: str,
    source_type: str,
    source_filename: str,
    annotated_path: str,
) -> Dict:
    """
    Chains analytics -> health -> risk -> emergency -> repair cost ->
    RUL -> damage growth simulation -> recommendation -> SHAP
    explanation -> automatic notification, then persists everything
    (detections, prediction summary, notification) to SQLite.
    """
    from recommendation_service import generate_session_recommendation
    from life_prediction_service import estimate_rul
    from simulation_service import simulate_damage_growth
    from explainability_service import explain_session
    from notification_service import dispatch_notification

    recommendation = generate_session_recommendation(
        detections, total_area_px, asset_type=asset_type,
        asset_age_years=asset_age_years, traffic_level=traffic_level,
    )
    emergency_result = recommendation["emergency_result"]
    risk_result = emergency_result["risk_result"]
    health_result = risk_result["health_result"]
    analytics = health_result["analytics"]

    rul_result = estimate_rul(
        health_score=health_result["health_score"],
        severity_score=analytics["damage_severity"]["severity_score"],
        asset_age_years=asset_age_years or 0.0,
        asset_type=asset_type,
        traffic_level=traffic_level,
    )

    simulation_timeline = simulate_damage_growth(
        current_damage_percentage=analytics["damage_percentage"],
        current_severity_score=analytics["damage_severity"]["severity_score"],
        severity_class=analytics["damage_severity"]["severity_class"],
        total_damage_count=analytics["damage_count"]["total_count"],
        asset_age_years=asset_age_years,
        months_ahead=6,
    )

    shap_bundle = explain_session(recommendation)

    # --- 14. Notification generated automatically, no user action required ---
    notification_result = dispatch_notification(
        recommendation, road_name or "Unnamed Asset", contacts={}, road_id=None,
    )

    # --- 17. Persist detections + prediction + notification to SQLite ---
    database.insert_detections(
        session_id, detections, road_name=road_name, asset_type=asset_type,
        source_type=source_type, source_filename=source_filename,
        annotated_path=str(annotated_path),
    )
    prediction_row = {
        "damage_count": analytics["damage_count"]["total_count"],
        "damage_percentage": analytics["damage_percentage"],
        "severity_score": analytics["damage_severity"]["severity_score"],
        "severity_class": analytics["damage_severity"]["severity_class"],
        "health_score": health_result["health_score"],
        "risk_score": risk_result["risk_score"],
        "risk_level": risk_result["risk_level"],
        "emergency_index": emergency_result["emergency_index"],
        "emergency_level": emergency_result["emergency_level"],
        "requires_immediate_action": emergency_result["requires_immediate_action"],
        "repair_cost": recommendation["estimated_cost"],
        "rul_years": rul_result["rul_years"],
        "priority_level": recommendation["priority_level"],
        "recommended_action": recommendation["recommended_action"],
        "due_date": recommendation["due_date"],
        "features_json": {},
    }
    database.insert_or_update_prediction(session_id, road_name, asset_type, prediction_row)
    if notification_result.get("alert"):
        database.insert_notifications(
            session_id, road_name, notification_result["alert"], notification_result["delivery_log"]
        )

    return {
        "analytics": analytics,
        "health_result": health_result,
        "risk_result": risk_result,
        "emergency_result": emergency_result,
        "recommendation": recommendation,
        "rul_result": rul_result,
        "simulation_timeline": simulation_timeline,
        "shap_bundle": shap_bundle,
        "notification_result": notification_result,
        "prediction_row": prediction_row,
    }


def flatten_simulation_timeline(timeline: List[Dict]) -> pd.DataFrame:
    rows = []
    for snap in timeline:
        for scenario in ("optimistic", "expected", "pessimistic"):
            rows.append({"month": snap["month"], "scenario": scenario, **snap[scenario]})
    return pd.DataFrame(rows)


def export_detection_csv(session_id: str, detections: List[Dict], recommendation: Dict, asset_type: str) -> str:
    """CSV columns: Frame Number, Timestamp, Damage Class, Confidence, Bounding Box, Severity, Priority."""
    from recommendation_service import recommend_for_detection, CATEGORY_TO_PRIORITY_LEVEL

    requires_immediate = recommendation["emergency_result"]["requires_immediate_action"]
    rows = []
    for det in detections:
        rec = recommend_for_detection(det, asset_type, requires_immediate)
        bbox = det.get("bbox", (None, None, None, None))
        bbox_str = (
            f"({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})" if bbox[0] is not None else ""
        )
        rows.append({
            "Frame Number": det.get("frame_number", ""),
            "Timestamp": det.get("timestamp_sec", ""),
            "Damage Class": det.get("class_name"),
            "Confidence": det.get("confidence"),
            "Bounding Box": bbox_str,
            "Severity": rec["severity_class"],
            "Priority": CATEGORY_TO_PRIORITY_LEVEL.get(rec["severity_class"], "low"),
        })

    df = pd.DataFrame(rows)
    out_dir = Path(config.CSV_EXPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}_detections.csv"
    df.to_csv(out_path, index=False)
    database.insert_report(session_id, "export", str(out_path))
    return str(out_path)


def render_pipeline_results(pipeline: Dict) -> None:
    """Renders analytics, decision intelligence, growth simulation, and SHAP explanation."""
    st.subheader("🧠 Decision Intelligence")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Health Score", f'{pipeline["health_result"]["health_score"]:.1f}')
    c2.metric("Risk Score", f'{pipeline["risk_result"]["risk_score"]:.1f}', pipeline["risk_result"]["risk_level"])
    c3.metric(
        "Emergency Index",
        f'{pipeline["emergency_result"]["emergency_index"]:.1f}',
        pipeline["emergency_result"]["emergency_level"],
    )
    c4.metric("Repair Cost", f'{pipeline["recommendation"]["estimated_cost"]:,.0f}')
    c5.metric("Remaining Useful Life", f'{pipeline["rul_result"]["rul_years"]:.1f} yrs')

    if pipeline["emergency_result"]["requires_immediate_action"]:
        st.error("⚠️ EMERGENCY: Emergency Index has crossed the immediate-action threshold.")

    rec = pipeline["recommendation"]
    st.markdown(
        f"**Recommended Action:** {rec['recommended_action']}  |  "
        f"**Priority:** {rec['priority_level'].upper()}  |  "
        f"**Due by:** {rec['due_date']}"
    )

    with st.expander("Per-Detection Recommendations", expanded=False):
        for da in rec["detection_actions"]:
            st.write(
                f"- **{da['class_name']}** (confidence {da['confidence']:.2f}, "
                f"severity {da['severity_class']}) → {da['action']}"
            )

    st.subheader("📈 Damage Growth Simulation (6-month forecast)")
    sim_df = flatten_simulation_timeline(pipeline["simulation_timeline"])
    fig_growth = px.line(
        sim_df, x="month", y="damage_percentage", color="scenario", markers=True,
        title="Projected Damage % Growth", labels={"month": "Months Ahead", "damage_percentage": "Damage %"},
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    st.subheader("🔍 Explainable AI (SHAP Dashboard)")
    for line in pipeline["shap_bundle"]["rule_explanations"]:
        st.write("- " + line)
    st.write(pipeline["shap_bundle"]["emergency_summary"])
    if pipeline["shap_bundle"]["emergency_chart"]:
        st.image(pipeline["shap_bundle"]["emergency_chart"])
    st.write(pipeline["shap_bundle"]["health_summary"])
    if pipeline["shap_bundle"]["health_chart"]:
        st.image(pipeline["shap_bundle"]["health_chart"])

    st.subheader("📨 Notification Sent Automatically")
    st.json(pipeline["notification_result"]["alert"])


# =======================================================================
# PAGE: HOME
# =======================================================================
def page_home():
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    kpis = database.get_summary_kpis()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Detections", kpis["total_detections"])
    c2.metric("Analysis Sessions", kpis["total_sessions"])
    c3.metric("Urgent Priority", kpis["urgent_count"])
    c4.metric("Open Complaints", kpis["open_complaints"])
    c5.metric("Avg Health Score", kpis["avg_health_score"] if kpis["avg_health_score"] is not None else "N/A")
    st.info(
        "Use **Upload Image** or **Upload Video** in the sidebar to run detection. "
        "Analytics, Decision Intelligence, Growth Simulation and SHAP explanations "
        "are generated automatically once detection completes."
    )


# =======================================================================
# PAGE: UPLOAD IMAGE
# =======================================================================
def page_upload_image(service: Optional[DetectionService]):
    st.header("📤 Upload Image - Damage Detection")

    with st.form("image_meta_form"):
        col1, col2 = st.columns(2)
        with col1:
            road_name = st.text_input("Road / Flyover Name", value=st.session_state["current_road_name"])
            asset_type = st.selectbox("Asset Type", ["road", "flyover"])
        with col2:
            asset_age_years = st.number_input("Asset Age (years)", min_value=0.0, value=10.0, step=1.0)
            traffic_level = st.selectbox("Traffic Level", ["low", "medium", "high"])
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
        submitted = st.form_submit_button("Run Detection on Image")

    if not submitted:
        return
    if uploaded_image is None:
        st.warning("Please upload an image file first.")
        return
    if service is None:
        st.error("YOLO model is not loaded. Check YOLO_MODEL_PATH in config.py and that best.pt exists.")
        return

    session_id = uuid.uuid4().hex[:12]
    saved_path = save_uploaded_image(uploaded_image)

    with st.spinner("Running YOLOv8 inference..."):
        result = service.detect_image(saved_path)

    detections = result["detections"]
    st.session_state.update({
        "session_id": session_id, "current_detections": detections,
        "current_road_name": road_name, "current_asset_type": asset_type,
        "annotated_image_path": result["annotated_path"],
    })

    st.subheader("Detection Results")
    if not detections:
        st.info("No Damage Detected")
    if result["annotated_path"]:
        st.image(result["annotated_path"], caption="Annotated Detection Image", use_container_width=True)

    if not detections:
        return

    st.dataframe(pd.DataFrame(detections), use_container_width=True)

    h, w = result["image_shape"]
    pipeline = run_full_pipeline(
        detections, h * w, asset_type, road_name, asset_age_years, traffic_level,
        session_id, source_type="image", source_filename=Path(saved_path).name,
        annotated_path=result["annotated_path"],
    )
    st.session_state["last_pipeline"] = pipeline
    render_pipeline_results(pipeline)

    csv_path = export_detection_csv(session_id, detections, pipeline["recommendation"], asset_type)
    with open(csv_path, "rb") as f:
        st.download_button("Download Detection CSV", f, file_name=Path(csv_path).name, mime="text/csv")


# =======================================================================
# PAGE: UPLOAD VIDEO  (corrected annotated-video pipeline)
# =======================================================================
def page_upload_video(service: Optional[DetectionService]):
    st.header("🎥 Upload Video - Damage Detection")

    with st.form("video_meta_form"):
        col1, col2 = st.columns(2)
        with col1:
            road_name = st.text_input("Road / Flyover Name", value=st.session_state["current_road_name"])
            asset_type = st.selectbox("Asset Type", ["road", "flyover"])
        with col2:
            asset_age_years = st.number_input("Asset Age (years)", min_value=0.0, value=10.0, step=1.0)
            traffic_level = st.selectbox("Traffic Level", ["low", "medium", "high"])
        uploaded_video = st.file_uploader("Upload an inspection video", type=["mp4", "avi", "mov", "mkv"])
        submitted = st.form_submit_button("Run Detection on Video")

    if not submitted:
        return
    if uploaded_video is None:
        st.warning("Please upload a video file first.")
        return
    if service is None:
        st.error("YOLO model is not loaded. Check YOLO_MODEL_PATH in config.py and that best.pt exists.")
        return

    session_id = uuid.uuid4().hex[:12]
    saved_path = save_uploaded_video(uploaded_video)

    # --- 5. annotated_video.mp4 saved under a per-session subfolder so
    #        concurrent/repeat runs never overwrite each other, while the
    #        filename itself stays exactly "annotated_video.mp4" ---
    output_video_path = Path(config.PROCESSED_VIDEOS_DIR) / session_id / "annotated_video.mp4"

    # --- 20, 21. Live progress bar + stat panel: Current Frame / FPS /
    #     Detection Count / Elapsed Time / Remaining Time ---
    st.subheader("Processing video...")
    progress_bar = st.progress(0)
    stat_cols = st.columns(6)
    current_frame_ph = stat_cols[0].empty()
    frames_processed_ph = stat_cols[1].empty()
    elapsed_time_ph = stat_cols[2].empty()
    remaining_time_ph = stat_cols[3].empty()
    detections_count_ph = stat_cols[4].empty()
    fps_ph = stat_cols[5].empty()

    st.caption("Live Detection History (most recent findings)")
    live_history_ph = st.empty()
    live_history_rows: List[Dict] = []

    def _progress_cb(stats: Dict) -> None:
        total = stats["total_frames"] or 1
        progress_bar.progress(min(1.0, stats["current_frame"] / total))

        # Throttle expensive widget updates to every 3rd frame for smoothness.
        if stats["current_frame"] % 3 == 0 or stats["current_frame"] == stats["total_frames"]:
            current_frame_ph.metric("Current Frame", stats["current_frame"])
            frames_processed_ph.metric(
                "Frames Processed", f'{stats["current_frame"]}/{stats["total_frames"] or "?"}'
            )
            elapsed_time_ph.metric("Elapsed Time", f'{stats["elapsed_seconds"]:.1f}s')
            remaining_time_ph.metric("Remaining Time", f'{stats["eta_seconds"]:.1f}s')
            detections_count_ph.metric("Detections Count", stats["detections_count"])
            fps_ph.metric("Processing FPS", stats["processing_fps"])

        for det in stats["last_frame_detections"]:
            live_history_rows.append({
                "Frame": stats["current_frame"] - 1,
                "Timestamp": stats["timestamp_sec"],
                "Class": det["class_name"],
                "Confidence": det["confidence"],
            })
        if live_history_rows and stats["current_frame"] % 5 == 0:
            live_history_ph.dataframe(pd.DataFrame(live_history_rows[-25:]), use_container_width=True)

    # --- 1, 2, 3, 4: read with OpenCV, infer every frame at conf=0.15/imgsz=640,
    #     draw with results[0].plot(), write via cv2.VideoWriter at source fps/size ---
    with st.spinner("Running YOLOv8 inference frame-by-frame..."):
        result = service.process_video(
            saved_path,
            output_video_path=output_video_path,
            confidence=0.15,
            image_size=640,
            progress_callback=_progress_cb,
        )

    progress_bar.progress(1.0)
    st.success(f"Video processing complete - {result['total_frames_processed']} frames processed.")

    detections = result["all_detections"]
    st.session_state.update({
        "session_id": session_id, "current_detections": detections,
        "current_road_name": road_name, "current_asset_type": asset_type,
        "annotated_video_path": result["output_video_path"], "video_result": result,
    })

    # --- 6, 9, 18, 23: ALWAYS show the annotated video, never the original
    #     upload. If it isn't verified as playable, show the exact reason
    #     plus a fallback preview of the first annotated frame instead of
    #     a blank/broken <video> player. ---
    st.subheader("Annotated Detection Video")
    if not detections:
        st.info("No Damage Detected")

    output_path = result["output_video_path"]
    output_exists = os.path.exists(output_path)
    output_size = os.path.getsize(output_path) if output_exists else 0

    if result.get("playable") and output_exists and output_size > 0:
        st.video(output_path)
    else:
        st.error(
            "⚠️ The annotated video could not be verified as browser-playable. "
            f"Exact reason: {result.get('failure_reason') or 'unknown error'}"
        )
        preview_path = result.get("first_frame_preview_path")
        if preview_path and os.path.exists(preview_path):
            st.image(preview_path, caption="First annotated frame (fallback preview)", use_container_width=True)
        else:
            st.warning("No fallback frame preview could be generated either.")

    # --- 24. Debug diagnostics (video opened / fps / frame count / output
    #     path / output size / codec / VideoWriter status) ---
    with st.expander("🛠️ Video Pipeline Debug Info"):
        st.json({
            "video_opened": result.get("video_opened"),
            "writer_opened": result.get("writer_opened"),
            "video_fps": result.get("video_fps"),
            "total_frames_processed": result.get("total_frames_processed"),
            "resolution": f'{result.get("width")}x{result.get("height")}',
            "output_video_path": output_path,
            "output_size_bytes": result.get("output_size_bytes"),
            "raw_codec_before_conversion": result.get("raw_codec"),
            "conversion_method": result.get("conversion_method"),
            "playable": result.get("playable"),
            "failure_reason": result.get("failure_reason"),
        })

    if not detections:
        return  # nothing further to analyze when the video is clean

    # --- 9. Full detection history for this session ---
    st.subheader("Detection History (this session)")
    st.dataframe(pd.DataFrame(detections), use_container_width=True)
    st.caption(f"{len(result['saved_frame_paths'])} damage frames saved to {config.FRAMES_DIR}")

    # --- 10-14: analytics, decision intelligence, growth simulation, SHAP, notification ---
    total_area_px = result["width"] * result["height"]
    pipeline = run_full_pipeline(
        detections, total_area_px, asset_type, road_name, asset_age_years, traffic_level,
        session_id, source_type="video", source_filename=Path(saved_path).name,
        annotated_path=result["output_video_path"],
    )
    st.session_state["last_pipeline"] = pipeline
    render_pipeline_results(pipeline)

    # --- 16. CSV export ---
    csv_path = export_detection_csv(session_id, detections, pipeline["recommendation"], asset_type)
    with open(csv_path, "rb") as f:
        st.download_button("Download Detection CSV", f, file_name=Path(csv_path).name, mime="text/csv")


# =======================================================================
# PAGE: ANALYTICS DASHBOARD
# =======================================================================
def page_analytics_dashboard():
    st.header("📊 Analytics Dashboard")

    roads = sorted({d["road_name"] for d in database.fetch_all_detections(5000) if d.get("road_name")})
    road_filter = st.selectbox("Filter by Road / Flyover", ["All"] + roads)
    road_name = None if road_filter == "All" else road_filter

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Damage Distribution")
        dist = database.get_damage_class_distribution(road_name)
        if dist:
            fig = px.pie(names=list(dist.keys()), values=list(dist.values()), title="Damage Distribution by Class")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No detections yet.")

    with col2:
        st.subheader("Severity Distribution")
        sev = database.get_severity_class_distribution(road_name)
        if sev:
            fig = px.bar(x=list(sev.keys()), y=list(sev.values()), labels={"x": "Severity", "y": "Sessions"},
                         title="Sessions by Severity Class", color=list(sev.keys()))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analysis sessions yet.")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Risk Trend")
        risk_trend = database.get_risk_trend(road_name)
        if risk_trend:
            df = pd.DataFrame(risk_trend)
            fig = px.line(df, x="created_at", y="risk_score", markers=True, title="Risk Score Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk history yet.")

    with col4:
        st.subheader("Repair Cost")
        cost_trend = database.get_repair_cost_trend(road_name)
        if cost_trend:
            df = pd.DataFrame(cost_trend)
            fig = px.bar(df, x="created_at", y="repair_cost", title="Estimated Repair Cost per Session")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No repair cost history yet.")

    st.subheader("Damage Growth Forecast (most recent session)")
    if st.session_state["last_pipeline"]:
        sim_df = flatten_simulation_timeline(st.session_state["last_pipeline"]["simulation_timeline"])
        fig = px.line(sim_df, x="month", y="damage_percentage", color="scenario", markers=True,
                      title="Projected Damage % Growth - Most Recent Session")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run a detection first to see a growth forecast.")


# =======================================================================
# PAGE: DETECTION HISTORY
# =======================================================================
def page_detection_history():
    st.header("🕘 Detection History")

    detections = database.fetch_all_detections(2000)
    if not detections:
        st.info("No detections recorded yet.")
        return

    df = pd.DataFrame(detections)
    roads = sorted({r for r in df["road_name"].dropna().unique()})
    classes = sorted({c for c in df["class_name"].dropna().unique()})

    col1, col2 = st.columns(2)
    road_filter = col1.selectbox("Road / Flyover", ["All"] + roads)
    class_filter = col2.selectbox("Damage Class", ["All"] + classes)

    filtered = df.copy()
    if road_filter != "All":
        filtered = filtered[filtered["road_name"] == road_filter]
    if class_filter != "All":
        filtered = filtered[filtered["class_name"] == class_filter]

    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download Filtered History (CSV)", filtered.to_csv(index=False).encode("utf-8"),
        file_name="detection_history.csv", mime="text/csv",
    )

    st.subheader("Prediction Sessions")
    st.dataframe(pd.DataFrame(database.fetch_predictions(500)), use_container_width=True)


# =======================================================================
# PAGE: SHAP DASHBOARD
# =======================================================================
def page_shap_dashboard():
    st.header("🔍 Explainable AI (SHAP) Dashboard")

    pipeline = st.session_state["last_pipeline"]
    if not pipeline:
        st.info("Run a detection on the Upload Image or Upload Video page first.")
        return

    st.subheader("Rule-Based Explanations")
    for line in pipeline["shap_bundle"]["rule_explanations"]:
        st.write("- " + line)

    st.subheader("Emergency Index - Feature Contributions")
    st.write(pipeline["shap_bundle"]["emergency_summary"])
    if pipeline["shap_bundle"]["emergency_chart"]:
        st.image(pipeline["shap_bundle"]["emergency_chart"])
    st.dataframe(pd.DataFrame(pipeline["shap_bundle"]["emergency_explanation"]["contributions"]))

    st.subheader("Health Score - Feature Contributions")
    st.write(pipeline["shap_bundle"]["health_summary"])
    if pipeline["shap_bundle"]["health_chart"]:
        st.image(pipeline["shap_bundle"]["health_chart"])
    st.dataframe(pd.DataFrame(pipeline["shap_bundle"]["health_explanation"]["contributions"]))


# =======================================================================
# PAGE: NOTIFICATIONS & COMPLAINTS
# =======================================================================
def page_notifications_complaints():
    st.header("📨 Notifications & Complaints")

    st.subheader("Notifications")
    st.dataframe(pd.DataFrame(database.fetch_notifications(300)), use_container_width=True)

    st.subheader("Complaints")
    st.dataframe(pd.DataFrame(database.fetch_complaints(300)), use_container_width=True)

    st.subheader("Generate a Municipality Complaint")
    pipeline = st.session_state["last_pipeline"]
    if not pipeline:
        st.info("Run a detection first to generate a complaint from its recommendation.")
        return

    if st.button("Generate Complaint from Latest Session"):
        from complaint_service import generate_complaint, format_complaint_letter, save_complaint

        complaint = generate_complaint(
            pipeline["recommendation"],
            road_name=st.session_state["current_road_name"] or "Unnamed Asset",
            asset_type=st.session_state["current_asset_type"],
        )
        letter = format_complaint_letter(complaint)
        file_path = save_complaint(complaint, letter)
        database.insert_complaint(st.session_state["session_id"], complaint, file_path)
        database.insert_report(st.session_state["session_id"], "complaint", file_path)

        st.success(f"Complaint {complaint['complaint_id']} generated and saved.")
        st.text_area("Complaint Letter", letter, height=400)
        with open(file_path, "rb") as f:
            st.download_button("Download Complaint Letter", f, file_name=Path(file_path).name)


# =======================================================================
# MAIN
# =======================================================================
def main():
    service = get_detection_service()
    if service is None:
        st.sidebar.error("⚠️ YOLO model not loaded - detection pages are disabled until best.pt is available.")

    st.sidebar.title(config.APP_TITLE)
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home", "📤 Upload Image", "🎥 Upload Video", "📊 Analytics Dashboard",
         "🕘 Detection History", "🔍 SHAP Dashboard", "📨 Notifications & Complaints"],
    )

    if page == "🏠 Home":
        page_home()
    elif page == "📤 Upload Image":
        page_upload_image(service)
    elif page == "🎥 Upload Video":
        page_upload_video(service)
    elif page == "📊 Analytics Dashboard":
        page_analytics_dashboard()
    elif page == "🕘 Detection History":
        page_detection_history()
    elif page == "🔍 SHAP Dashboard":
        page_shap_dashboard()
    elif page == "📨 Notifications & Complaints":
        page_notifications_complaints()


if __name__ == "__main__":
    main()