"""
app.py
------
FINAL, PRODUCTION-READY entry point for the AI-Powered Smart Road &
Flyover Damage Monitoring and Decision Intelligence System.

Run with:  streamlit run app.py

FULL WORKFLOW (matches the project spec end-to-end):

    Upload Image/Video
          |
    YOLO Detection                 (detection_service.py)
          |
    Analytics Engine                (analytics_service.py)
          |
    Health Score                    (health_service.py)
          |
    Risk Score                      (risk_service.py)
          |
    Emergency Index                 (emergency_service.py)
          |
    Repair Cost Prediction          (repair_cost_service.py)
          |
    Remaining Useful Life           (life_prediction_service.py)
          |
    Damage Growth Simulation        (simulation_service.py)
          |
    Recommendations                 (recommendation_service.py)
          |
    Notifications                   (notification_service.py)
          |
    Database                        (database.py - SQLite)
          |
    Dashboard                       (this file - Streamlit UI)
          |
    PDF Report                      (pdf_report_service.py)

NOTE: This app runs YOLOv8 (ultralytics) if a trained weights file
exists at config.YOLO_MODEL_PATH. If it does not exist, image/video
detection is skipped gracefully and the app falls back to a manual
"Enter Detections" form so every downstream score (health, risk,
emergency, repair cost, RUL, simulation, recommendations, PDF report)
can still be demoed end-to-end without a trained model.
"""

import io
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import database as db
from config import (
    YOLO_MODEL_PATH, DAMAGE_CLASSES, RISK_ALERT_THRESHOLD,
    RAW_IMAGES_DIR, RAW_VIDEOS_DIR, get_damage_classes,
)
from analytics_service import analyze
from health_service import compute_health_score
from risk_service import compute_risk_score
from emergency_service import compute_emergency_index
from repair_cost_service import estimate_repair_cost
from life_prediction_service import estimate_rul
from simulation_service import simulate_damage_growth, get_growth_factor_for_emergency_index
from recommendation_service import generate_session_recommendation
from notification_service import dispatch_notification
from complaint_service import generate_complaint, format_complaint_letter
from explainability_service import explain_session
from pdf_report_service import generate_pdf_report
from image_utils import load_image, resize_image, draw_detections, save_annotated_image, validate_image_file, save_uploaded_image
from video_utils import validate_video_file, save_uploaded_video, get_video_properties
from live_detection import live_detection_page
from email_service import is_smtp_configured, is_valid_email, send_damage_alert, send_test_email, should_send_email

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

st.set_page_config(page_title="Smart Road & Flyover Damage Monitoring", page_icon="🛣️", layout="wide")

db.init_db()


# =======================================================================
# CACHED / SESSION HELPERS
# =======================================================================
@st.cache_resource(show_spinner=False)
def load_detection_model():
    """Loads YOLOv8 once per process. Returns None only on genuine errors."""
    import os
    import traceback
    
    print("\n" + "="*80)
    print("APP: Loading YOLO Detection Model")
    print("="*80)
    print(f"Current working directory: {os.getcwd()}")
    print(f"YOLO_MODEL_PATH (config): {YOLO_MODEL_PATH}")
    print(f"YOLO_MODEL_PATH (resolved): {Path(YOLO_MODEL_PATH).resolve()}")
    print(f"Model file exists: {Path(YOLO_MODEL_PATH).exists()}")
    
    if Path(YOLO_MODEL_PATH).exists():
        size_mb = Path(YOLO_MODEL_PATH).stat().st_size / (1024**2)
        print(f"Model file size: {size_mb:.2f} MB")
    print("="*80 + "\n")
    
    if not Path(YOLO_MODEL_PATH).exists():
        print(f"✗ Model file not found at {YOLO_MODEL_PATH}")
        return None
    
    try:
        print("Attempting to load DetectionService...")
        from detection_service import DetectionService
        model = DetectionService()
        print("✓ DetectionService loaded successfully!")
        return model
    except Exception as e:
        print(f"✗ Failed to load DetectionService: {e}")
        traceback.print_exc()
        logger.exception(f"Could not load YOLOv8 model: {e}")
        return None

def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:10]
    return st.session_state.session_id


def run_full_pipeline(detections, total_inspected_area_px, asset_type, road_id,
                       asset_age_years, traffic_level, asset_length_m=None,
                       pixels_per_meter=None, months_ahead=6):
    """Runs analytics -> health -> risk -> emergency -> repair_cost -> RUL
    -> simulation -> recommendation, all in one call. Returns a dict bundle."""
    analytics = analyze(detections, total_inspected_area_px)

    health_result = compute_health_score(
        severity_score=analytics["damage_severity"]["severity_score"],
        damage_percentage=analytics["damage_percentage"],
        total_damage_count=analytics["damage_count"]["total_count"],
        asset_age_years=asset_age_years,
    )
    health_result["analytics"] = analytics

    risk_result = compute_risk_score(
        health_score=health_result["health_score"],
        severity_class=analytics["damage_severity"]["severity_class"],
        traffic_level=traffic_level,
    )
    risk_result["health_result"] = health_result

    # simulation first (for growth factor input to emergency index)
    simulation_timeline = simulate_damage_growth(
        current_damage_percentage=analytics["damage_percentage"],
        current_severity_score=analytics["damage_severity"]["severity_score"],
        severity_class=analytics["damage_severity"]["severity_class"],
        total_damage_count=analytics["damage_count"]["total_count"],
        asset_age_years=asset_age_years,
        months_ahead=months_ahead,
    )
    growth_factor = get_growth_factor_for_emergency_index(simulation_timeline, lookahead_month=3)

    emergency_result = compute_emergency_index(
        risk_score=risk_result["risk_score"],
        detections=detections,
        asset_type=asset_type,
        growth_factor_0_100=growth_factor,
    )
    emergency_result["risk_result"] = risk_result

    repair_cost_result = estimate_repair_cost(
        detections, total_inspected_area_px,
        asset_age_years=asset_age_years, asset_length_m=asset_length_m,
        **({"pixels_per_meter": pixels_per_meter} if pixels_per_meter else {}),
    )

    rul_result = estimate_rul(
        health_score=health_result["health_score"],
        severity_score=analytics["damage_severity"]["severity_score"],
        asset_age_years=asset_age_years or 0.0,
        asset_type=asset_type,
        traffic_level=traffic_level,
    )

    recommendation = generate_session_recommendation(
        detections, total_inspected_area_px, asset_type=asset_type, road_id=road_id,
        asset_age_years=asset_age_years, traffic_level=traffic_level,
        asset_length_m=asset_length_m, pixels_per_meter=pixels_per_meter,
    )
    # keep our already-computed emergency/repair_cost consistent with the ones
    # generate_session_recommendation computed internally (it recomputes them);
    # we prefer the recommendation's own for internal consistency downstream.
    emergency_result = recommendation["emergency_result"]
    repair_cost_result = recommendation["repair_cost_result"]

    explanation_bundle = explain_session(recommendation)

    return {
        "detections": detections,
        "analytics": analytics,
        "health_result": health_result,
        "risk_result": risk_result,
        "emergency_result": emergency_result,
        "repair_cost_result": repair_cost_result,
        "rul_result": rul_result,
        "simulation_timeline": simulation_timeline,
        "recommendation": recommendation,
        "explanation_bundle": explanation_bundle,
    }


def maybe_send_email_alert(bundle, road_name, road_id, prediction_id, attachment_path=None,
                            frame_number=None, location=None):
    """
    Sends the HTML damage-alert email (with the annotated image/frame
    attached) IF AND ONLY IF:
      1. This session's severity is High or Critical, AND
      2. Auto Email Notifications is enabled in Email Notification
         Settings with a valid receiver address saved.

    Safe to call after every pipeline run regardless of severity/settings
    - it's a no-op otherwise. Every attempt (success or failure) is
    recorded in the email_log table so results are auditable from the
    Notifications & Complaints page.
    """
    severity_class = bundle["health_result"]["analytics"]["damage_severity"]["severity_class"]
    if not should_send_email(severity_class):
        return None

    settings = db.get_email_settings()
    receiver_email = settings.get("receiver_email")
    if not settings.get("auto_send_enabled") or not receiver_email:
        return None

    detections = bundle["detections"]
    top_detection = max(detections, key=lambda d: d.get("confidence", 0.0)) if detections else {}

    detection_data = {
        "damage_type": top_detection.get("class_name", "damage"),
        "severity": severity_class,
        "confidence": top_detection.get("confidence"),
        "risk_score": bundle["risk_result"]["risk_score"],
        "repair_cost": bundle["repair_cost_result"]["total_cost"],
        "road_name": road_name,
        "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": location,
        "frame_number": frame_number if frame_number is not None else top_detection.get("frame_number"),
    }

    result = send_damage_alert(detection_data, receiver_email, attachment_path=attachment_path)
    db.log_email_sent(
        road_id, prediction_id, receiver_email, result["subject"], severity_class,
        "sent" if result["success"] else "failed", error_message=result.get("error"),
    )
    return result


# =======================================================================
# SIDEBAR NAVIGATION
# =======================================================================
st.sidebar.title("🛣️ Smart Road & Flyover Monitoring")
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Analytics Dashboard",
        "📤 Upload & Detect",
        "🔴 Live Detection",
        "🧠 Decision Intelligence",
        "📈 Damage Growth Simulation",
        "🕘 Detection History",
        "🔍 SHAP / Explainability",
        "📨 Notifications & Complaints",
        "📄 PDF Report",
    ],
)

st.sidebar.markdown("---")
roads = db.list_roads()
road_names = ["(New road/flyover)"] + [r["name"] for r in roads]
selected_road_name = st.sidebar.selectbox("Road / Flyover", road_names)

if selected_road_name == "(New road/flyover)":
    new_name = st.sidebar.text_input("New asset name", value="Unnamed Segment")
    new_type = st.sidebar.selectbox("Asset type", ["road", "flyover"])
    if st.sidebar.button("Create asset"):
        rid = db.get_or_create_road(new_name, new_type)
        st.sidebar.success(f"Created '{new_name}' (id={rid})")
        st.rerun()
    current_road_id = None
    current_asset_type = new_type
    current_road_name = new_name
else:
    current_road = next(r for r in roads if r["name"] == selected_road_name)
    current_road_id = current_road["road_id"]
    current_asset_type = current_road["asset_type"]
    current_road_name = current_road["name"]

st.sidebar.markdown("---")
asset_age_years = st.sidebar.number_input("Asset age (years)", min_value=0, max_value=150, value=15)
traffic_level = st.sidebar.selectbox("Traffic load", ["low", "medium", "high"], index=1)


# =======================================================================
# PAGE: UPLOAD & DETECT
# =======================================================================
if page == "📤 Upload & Detect":
    st.title("📤 Upload Image / Video for Damage Detection")
    model = load_detection_model()
    if model is None:
        st.warning(
            f"No trained YOLOv8 weights found at `{YOLO_MODEL_PATH}`. "
            "You can still exercise the full scoring pipeline using the manual "
            "detection entry form below."
        )
    else:
        # Add a collapsible debug panel showing model info
        with st.expander("🔧 Detection Model Debug Panel", expanded=False):
            try:
                info = model.get_model_info()
                st.write("**Model Loaded**: Yes")
                st.write("**Model Path**:", info.get("model_path"))
                st.write("**Model Type**:", info.get("model_type"))
                st.write("**Number of Classes**:", info.get("num_classes"))
                st.write("**Class Names**:")
                st.write(info.get("names"))
            except Exception as e:
                st.write("Could not get model info:", str(e))

    tab_img, tab_vid, tab_manual = st.tabs(["🖼️ Image", "🎞️ Video", "✍️ Manual Detections"])

    # ---------------- IMAGE ----------------
    with tab_img:
        uploaded_image = st.file_uploader("Upload a road/flyover image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_image and st.button("Run Detection on Image", key="run_img"):
            if model is None:
                st.error("No detection model available - use the Manual Detections tab instead.")
            else:
                with st.spinner("Running YOLOv8 inference..."):
                    # Save uploaded image to data/raw/images/ with cross-platform path handling
                    saved_path = save_uploaded_image(uploaded_image)
                    result = model.detect_image(saved_path, save_output=True)

                # Inference debug panel
                with st.expander("🔎 Inference Debug (YOLO)", expanded=True):
                    try:
                        info = model.get_model_info()
                        st.write("**Model Loaded**: Yes")
                        st.write("**Model Path**:", info.get("model_path"))
                        st.write("**Classes (model.names)**:")
                        st.write(info.get("names"))
                        st.write("**Configured DAMAGE_CLASSES (config)**:")
                        st.write(DAMAGE_CLASSES)

                        # Compare class lists
                        model_names = info.get("names") or {}
                        model_set = set(model_names.values()) if isinstance(model_names, dict) else set(model_names)
                        config_set = set(DAMAGE_CLASSES)
                        missing_in_model = config_set - model_set
                        extra_in_model = model_set - config_set
                        st.write("**Classes missing in model compared to config**:", list(missing_in_model))
                        st.write("**Extra classes in model not in config**:", list(extra_in_model))

                        st.write("**Confidence threshold (app)**:", model.confidence_threshold)
                        st.write("**Effective debug threshold applied**: 0.15 (temporary)")
                        st.write("**Inference Time (s)**:", result.get("inference_time"))
                        st.write("**Number of Detections (returned)**:", len(result.get("detections", [])))

                        for i, det in enumerate(result.get("detections", [])):
                            st.write(f"{i+1}. Class: {det.get('class_name')} | Confidence: {det.get('confidence'):.4f} | BBox: {det.get('bbox')}")

                        # Show latest raw YOLO image if present
                        from pathlib import Path
                        raw_dir = Path("data/processed/annotated_images")
                        raw_files = sorted(raw_dir.glob("raw_yolo_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True) if raw_dir.exists() else []
                        if raw_files:
                            st.write("**Latest raw YOLO output**:")
                            st.image(str(raw_files[0]), caption=str(raw_files[0]), use_column_width=True)
                        else:
                            st.write("No raw YOLO output image found yet.")
                    except Exception as e:
                        st.write("Could not show inference debug info:", str(e))

                # Display detection results
                if result["annotated_path"]:
                    st.image(result["annotated_path"], caption="Annotated Detection Result", use_container_width=True)
                
                h, w = result["image_shape"]
                total_area = h * w
                detection_count = len(result["detections"])
                
                # Show detection summary
                if detection_count > 0:
                    st.subheader(f"✓ Found {detection_count} damage(s)")
                    for i, det in enumerate(result["detections"]):
                        det_class = det.get("class_name", "damage")
                        det_conf = det.get("confidence", 0.0)
                        det_area = det.get("area_px", 0.0)
                        bbox = det.get("bbox", (0, 0, 0, 0))
                        st.caption(f"  {i+1}. **{det_class}** - Confidence: {det_conf*100:.1f}% - Area: {det_area:.0f}px - BBox: ({bbox[0]:.0f},{bbox[1]:.0f}) to ({bbox[2]:.0f},{bbox[3]:.0f})")
                else:
                    st.info("✓ No damage detected")

                # Store detections in session state
                session_id = get_session_id()
                db.insert_detections(
                    result["detections"], current_road_id, session_id, "image",
                    source_path=saved_path, annotated_path=result["annotated_path"],
                )
                st.session_state["last_detections"] = result["detections"]
                st.session_state["last_total_area"] = total_area
                st.session_state["last_annotated_image_path"] = result["annotated_path"]
                st.session_state["last_source_type"] = "image"

                # ===== AUTO-RUN FULL SCORING PIPELINE =====
                if detection_count > 0:
                    print("\n" + "="*80)
                    print("[PIPELINE] Auto-running full scoring pipeline after detection...")
                    print("="*80)
                    
                    try:
                        with st.spinner("📊 Analyzing detections and calculating scores..."):
                            bundle = run_full_pipeline(
                                result["detections"], total_area, current_asset_type, current_road_id,
                                asset_age_years, traffic_level,
                            )
                        
                        # Store pipeline results
                        st.session_state["last_bundle"] = bundle
                        
                        # Persist prediction + report
                        prediction_id = db.insert_prediction(
                            current_road_id, session_id, current_asset_type,
                            bundle["health_result"], bundle["risk_result"], bundle["emergency_result"],
                            bundle["repair_cost_result"], bundle["rul_result"], bundle["recommendation"],
                            simulation_timeline=bundle["simulation_timeline"],
                        )
                        st.session_state["last_prediction_id"] = prediction_id
                        
                        db.insert_report(
                            current_road_id, session_id, prediction_id,
                            summary={
                                "health_score": bundle["health_result"]["health_score"],
                                "risk_score": bundle["risk_result"]["risk_score"],
                                "emergency_index": bundle["emergency_result"]["emergency_index"],
                                "repair_cost": bundle["repair_cost_result"]["total_cost"],
                                "rul_years": bundle["rul_result"]["rul_years"],
                            },
                        )
                        
                        print("[PIPELINE] Scoring complete")
                        
                        # Display key metrics automatically
                        st.subheader("📊 Scoring Results")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Health Score", f"{bundle['health_result']['health_score']:.1f}/100")
                        col2.metric("Risk Score", f"{bundle['risk_result']['risk_score']:.1f}", bundle["risk_result"]["risk_level"])
                        col3.metric("Emergency Index", f"{bundle['emergency_result']['emergency_index']:.1f}", bundle["emergency_result"]["emergency_level"])
                        col4.metric("Repair Cost", f"₹{bundle['repair_cost_result']['total_cost']:,.0f}")
                        col5.metric("Remaining Useful Life", f"{bundle['rul_result']['rul_years']:.1f} yrs")
                        
                        # Critical alert
                        if bundle["emergency_result"]["requires_immediate_action"]:
                            st.error("⚠️ IMMEDIATE ACTION REQUIRED — Emergency Index crossed critical threshold")
                        
                        # Show severity
                        severity_class = bundle["health_result"]["analytics"]["damage_severity"]["severity_class"]
                        severity_score = bundle["health_result"]["analytics"]["damage_severity"]["severity_score"]
                        severity_color = "🔴" if severity_class == "Critical" else "🟠" if severity_class == "High" else "🟡" if severity_class == "Medium" else "🟢"
                        st.write(f"{severity_color} **Severity: {severity_class}** ({severity_score:.2f})")
                        
                        # AUTO-GENERATE NOTIFICATIONS/COMPLAINTS IF CRITICAL
                        if severity_class in ("High", "Critical"):
                            print(f"[NOTIFICATIONS] Auto-generating alert for severity={severity_class}")
                            
                            # Auto-send notification
                            rec = bundle["recommendation"]
                            result_notif = dispatch_notification(
                                rec, current_road_name, contacts={"email": "publicworks@city.gov"},
                                road_id=current_road_id, force=True,
                            )
                            if result_notif["alert"]:
                                db.insert_notification(current_road_id, prediction_id, result_notif["alert"], result_notif["delivery_log"])
                                print(f"[NOTIFICATIONS] Alert sent: {result_notif['alert']}")
                                st.info(f"🔔 Auto-alert sent: {result_notif['alert']}")

                            # Auto-send REAL email (Gmail SMTP) if enabled in Email Notification Settings
                            email_result = maybe_send_email_alert(
                                bundle, current_road_name, current_road_id, prediction_id,
                                attachment_path=result["annotated_path"],
                            )
                            if email_result:
                                if email_result["success"]:
                                    st.info(f"📧 Email alert sent to configured recipient.")
                                else:
                                    st.warning(f"📧 Email alert failed: {email_result['error']}")

                            # Auto-generate complaint if Critical
                            if severity_class == "Critical":
                                print(f"[COMPLAINTS] Auto-generating complaint for Critical severity")
                                complaint = generate_complaint(
                                    rec, road_name=current_road_name, asset_type=current_asset_type, road_id=current_road_id,
                                )
                                letter = format_complaint_letter(complaint)
                                db.insert_complaint(current_road_id, prediction_id, complaint, letter)
                                print(f"[COMPLAINTS] Complaint created: {complaint['complaint_id']}")
                                st.warning(f"📄 Auto-complaint created: {complaint['complaint_id']}")
                        
                        print("="*80 + "\n")
                        st.success(f"✓ Pipeline complete: {detection_count} detection(s) scored and stored")
                        
                    except Exception as e:
                        print(f"[ERROR] Pipeline failed: {e}")
                        import traceback
                        traceback.print_exc()
                        st.error(f"Pipeline error: {e}")
                else:
                    st.info("No detections to score - pipeline skipped")

    # ---------------- VIDEO ----------------
    with tab_vid:
        uploaded_video = st.file_uploader("Upload a road/flyover video", type=["mp4", "avi", "mov", "mkv"])

        st.caption("📧 Event-Driven Email Alerts (optional)")
        email_settings = db.get_email_settings()
        col_e1, col_e2 = st.columns(2)
        video_email_alerts_enabled = col_e1.checkbox(
            "Enable live email alerts during processing", value=False, key="video_email_alerts_enabled",
            help="Sends ONE email per unique damage event (tracked by object identity, not per "
                 "frame) to the receiver configured in Email Notification Settings.",
        )
        video_email_cooldown = col_e2.number_input(
            "Cooldown per object (seconds)", min_value=5, max_value=600, value=60, step=5, key="video_email_cooldown",
        )
        if video_email_alerts_enabled and not email_settings.get("receiver_email"):
            st.warning("No receiver email configured yet - set one in Notifications & Complaints → Email Notification Settings.")

        if uploaded_video and st.button("Run Detection on Video", key="run_vid"):
            if model is None:
                st.error("No detection model available - use the Manual Detections tab instead.")
            else:
                saved_path = save_uploaded_video(uploaded_video)
                props = get_video_properties(saved_path)

                st.subheader("🔴 Live Detection Preview")
                live_frame_ph = st.empty()
                progress_bar = st.progress(0.0)
                stat_cols = st.columns(5)
                current_frame_ph = stat_cols[0].empty()
                fps_ph = stat_cols[1].empty()
                detections_ph = stat_cols[2].empty()
                elapsed_ph = stat_cols[3].empty()
                remaining_ph = stat_cols[4].empty()
                email_feed_ph = st.empty()

                def _on_progress(stats: dict) -> None:
                    total = stats.get("total_frames") or 1
                    progress_bar.progress(min(1.0, stats["current_frame"] / total))
                    current_frame_ph.metric("Frame", f'{stats["current_frame"]}/{stats.get("total_frames") or "?"}')
                    fps_ph.metric("Processing FPS", stats.get("processing_fps", 0.0))
                    detections_ph.metric("Detections", stats.get("detections_count", 0))
                    elapsed_ph.metric("Elapsed", f'{stats.get("elapsed_seconds", 0.0):.1f}s')
                    remaining_ph.metric("Remaining", f'{stats.get("eta_seconds", 0.0):.1f}s')

                    # --- Live bounding-box annotation: show the same
                    #     Ultralytics-annotated frame just written to the
                    #     output video, updated in real time as it's processed. ---
                    live_frame = stats.get("live_preview_frame_bgr")
                    if live_frame is not None:
                        live_frame_ph.image(
                            cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB),
                            caption=f'Live detection - frame {stats["current_frame"]} @ {stats.get("timestamp_sec", 0.0):.1f}s',
                            use_container_width=True,
                        )

                    # --- "📧 Email sent for Pothole" live feed ---
                    email_events = stats.get("email_events") or []
                    if email_events:
                        icon = {"queued": "⏳", "sent": "📧", "failed": "⚠️"}
                        lines = [
                            f"{icon.get(ev['status'], '•')} {ev['status'].upper()} - **{ev['class_name']}** "
                            f"(frame {ev['frame_number']}, {ev['confidence']*100:.0f}%)"
                            + (f" - {ev['error']}" if ev.get("error") else "")
                            for ev in reversed(email_events[-5:])
                        ]
                        email_feed_ph.markdown("\n\n".join(lines))

                with st.spinner("Processing video (this may take a while)..."):
                    result = model.process_video(
                        saved_path,
                        progress_callback=_on_progress,
                        live_preview=True,
                        live_preview_every_n_frames=1,
                        email_alerts_enabled=video_email_alerts_enabled,
                        email_receiver=email_settings.get("receiver_email"),
                        email_road_name=current_road_name,
                        email_road_id=current_road_id,
                        email_confidence_threshold=0.5,
                        email_cooldown_seconds=float(video_email_cooldown),
                    )

                live_frame_ph.empty()  # clear the live-preview placeholder once done

                if result.get("playable", True):
                    st.video(result["output_video_path"])
                else:
                    st.error(
                        "⚠️ The annotated video could not be verified as browser-playable. "
                        f"Reason: {result.get('failure_reason') or 'unknown error'}"
                    )
                    preview_path = result.get("first_frame_preview_path")
                    if preview_path and Path(preview_path).exists():
                        st.image(preview_path, caption="First annotated frame (fallback preview)", use_container_width=True)

                detection_count = len(result["all_detections"])
                st.write(f"**Total detections across video: {detection_count}**")
                
                if detection_count > 0:
                    st.subheader("Detections Summary")
                    for i, det in enumerate(result["all_detections"][:10]):  # Show first 10
                        det_class = det.get("class_name", "damage")
                        det_conf = det.get("confidence", 0.0)
                        frame = det.get("frame_number", 0)
                        timestamp = det.get("timestamp_sec", 0.0)
                        st.caption(f"  {i+1}. Frame {frame} @ {timestamp:.1f}s - **{det_class}** {det_conf*100:.1f}%")

                total_area = props["width"] * props["height"]
                session_id = get_session_id()
                db.insert_detections(
                    result["all_detections"], current_road_id, session_id, "video",
                    source_path=saved_path, annotated_path=result["output_video_path"],
                )
                st.session_state["last_detections"] = result["all_detections"]
                st.session_state["last_total_area"] = total_area
                st.session_state["last_annotated_image_path"] = None  # video: no single still frame
                st.session_state["last_source_type"] = "video"

                # ===== AUTO-RUN FULL SCORING PIPELINE FOR VIDEO =====
                if detection_count > 0:
                    print("\n" + "="*80)
                    print(f"[PIPELINE] Auto-running pipeline for video with {detection_count} detections...")
                    print("="*80)
                    
                    try:
                        with st.spinner("📊 Analyzing video detections and calculating scores..."):
                            bundle = run_full_pipeline(
                                result["all_detections"], total_area, current_asset_type, current_road_id,
                                asset_age_years, traffic_level,
                            )
                        
                        st.session_state["last_bundle"] = bundle
                        
                        prediction_id = db.insert_prediction(
                            current_road_id, session_id, current_asset_type,
                            bundle["health_result"], bundle["risk_result"], bundle["emergency_result"],
                            bundle["repair_cost_result"], bundle["rul_result"], bundle["recommendation"],
                            simulation_timeline=bundle["simulation_timeline"],
                        )
                        st.session_state["last_prediction_id"] = prediction_id
                        
                        db.insert_report(
                            current_road_id, session_id, prediction_id,
                            summary={
                                "health_score": bundle["health_result"]["health_score"],
                                "risk_score": bundle["risk_result"]["risk_score"],
                                "emergency_index": bundle["emergency_result"]["emergency_index"],
                                "repair_cost": bundle["repair_cost_result"]["total_cost"],
                                "rul_years": bundle["rul_result"]["rul_years"],
                            },
                        )
                        
                        print("[PIPELINE] Video pipeline complete")
                        
                        # Display results
                        st.subheader("📊 Scoring Results")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Health Score", f"{bundle['health_result']['health_score']:.1f}/100")
                        col2.metric("Risk Score", f"{bundle['risk_result']['risk_score']:.1f}", bundle["risk_result"]["risk_level"])
                        col3.metric("Emergency Index", f"{bundle['emergency_result']['emergency_index']:.1f}", bundle["emergency_result"]["emergency_level"])
                        col4.metric("Repair Cost", f"₹{bundle['repair_cost_result']['total_cost']:,.0f}")
                        col5.metric("Remaining Useful Life", f"{bundle['rul_result']['rul_years']:.1f} yrs")
                        
                        if bundle["emergency_result"]["requires_immediate_action"]:
                            st.error("⚠️ IMMEDIATE ACTION REQUIRED")
                        
                        # AUTO-GENERATE ALERTS FOR VIDEO
                        severity_class = bundle["health_result"]["analytics"]["damage_severity"]["severity_class"]
                        if severity_class in ("High", "Critical"):
                            print(f"[NOTIFICATIONS] Auto-generating alert for video with severity={severity_class}")
                            rec = bundle["recommendation"]
                            result_notif = dispatch_notification(
                                rec, f"{current_road_name} (video)", contacts={"email": "publicworks@city.gov"},
                                road_id=current_road_id, force=True,
                            )
                            if result_notif["alert"]:
                                db.insert_notification(current_road_id, prediction_id, result_notif["alert"], result_notif["delivery_log"])
                                st.info(f"🔔 Auto-alert sent: {result_notif['alert']}")

                            # Auto-send REAL email (Gmail SMTP) if enabled - attach the
                            # first saved detected frame as the representative image
                            # (video has no single "annotated image", only per-frame stills).
                            top_det = (
                                max(result["all_detections"], key=lambda d: d.get("confidence", 0.0))
                                if result["all_detections"] else {}
                            )
                            frame_attachment = result["saved_frame_paths"][0] if result.get("saved_frame_paths") else None
                            email_result = maybe_send_email_alert(
                                bundle, f"{current_road_name} (video)", current_road_id, prediction_id,
                                attachment_path=frame_attachment,
                                frame_number=top_det.get("frame_number"),
                            )
                            if email_result:
                                if email_result["success"]:
                                    st.info("📧 Email alert sent to configured recipient.")
                                else:
                                    st.warning(f"📧 Email alert failed: {email_result['error']}")

                            if severity_class == "Critical":
                                print(f"[COMPLAINTS] Auto-generating complaint for Critical video severity")
                                complaint = generate_complaint(
                                    rec, road_name=f"{current_road_name} (video)", asset_type=current_asset_type, road_id=current_road_id,
                                )
                                letter = format_complaint_letter(complaint)
                                db.insert_complaint(current_road_id, prediction_id, complaint, letter)
                                st.warning(f"📄 Auto-complaint created: {complaint['complaint_id']}")
                        
                        print("="*80 + "\n")
                        st.success(f"✓ Video pipeline complete: {detection_count} detection(s) scored and stored")
                        
                    except Exception as e:
                        print(f"[ERROR] Video pipeline failed: {e}")
                        import traceback
                        traceback.print_exc()
                        st.error(f"Pipeline error: {e}")
                else:
                    st.info("No detections in video - pipeline skipped")

    # ---------------- MANUAL ----------------
    with tab_manual:
        st.caption("Use this to demo the full scoring pipeline without a trained YOLO model.")
        n_rows = st.number_input("Number of detections", min_value=1, max_value=30, value=3)
        manual_detections = []
        damage_classes = get_damage_classes()
        for i in range(int(n_rows)):
            c1, c2, c3 = st.columns(3)
            cls = c1.selectbox(f"Class #{i+1}", damage_classes, key=f"cls_{i}")
            conf = c2.slider(f"Confidence #{i+1}", 0.0, 1.0, 0.7, key=f"conf_{i}")
            area = c3.number_input(f"Area px #{i+1}", min_value=0, value=2000, key=f"area_{i}")
            manual_detections.append({"class_name": cls, "confidence": conf, "area_px": float(area), "bbox": (0, 0, 0, 0)})

        frame_area = st.number_input("Total inspected area (px, e.g. 1280x720=921600)", min_value=1, value=921600)

        if st.button("Save Manual Detections"):
            session_id = get_session_id()
            db.insert_detections(manual_detections, current_road_id, session_id, "image")
            st.session_state["last_detections"] = manual_detections
            st.session_state["last_total_area"] = frame_area
            st.session_state["last_annotated_image_path"] = None
            st.session_state["last_source_type"] = "manual"
            st.success(f"{len(manual_detections)} manual detections saved for this session.")


# =======================================================================
# PAGE: DECISION INTELLIGENCE
# =======================================================================
elif page == "🔴 Live Detection":
    model = load_detection_model()
    live_detection_page(model, road_name=current_road_name, road_id=current_road_id)

elif page == "🧠 Decision Intelligence":
    st.title("🧠 Decision Intelligence — Health, Risk, Emergency, Cost, RUL")

    detections = st.session_state.get("last_detections")
    total_area = st.session_state.get("last_total_area")

    if not detections:
        st.info("No active session detections. Upload an image/video or add manual detections first.")
    else:
        st.caption(f"Scoring {len(detections)} detections for **{current_road_name}** ({current_asset_type}).")

        if st.button("▶️ Run Full Scoring Pipeline", type="primary"):
            with st.spinner("Computing health, risk, emergency, repair cost, RUL, simulation, recommendations..."):
                bundle = run_full_pipeline(
                    detections, total_area, current_asset_type, current_road_id,
                    asset_age_years, traffic_level,
                )
            st.session_state["last_bundle"] = bundle

            # Persist prediction + report
            session_id = get_session_id()
            prediction_id = db.insert_prediction(
                current_road_id, session_id, current_asset_type,
                bundle["health_result"], bundle["risk_result"], bundle["emergency_result"],
                bundle["repair_cost_result"], bundle["rul_result"], bundle["recommendation"],
                simulation_timeline=bundle["simulation_timeline"],
            )
            st.session_state["last_prediction_id"] = prediction_id
            db.insert_report(
                current_road_id, session_id, prediction_id,
                summary={
                    "health_score": bundle["health_result"]["health_score"],
                    "risk_score": bundle["risk_result"]["risk_score"],
                    "emergency_index": bundle["emergency_result"]["emergency_index"],
                    "repair_cost": bundle["repair_cost_result"]["total_cost"],
                    "rul_years": bundle["rul_result"]["rul_years"],
                },
            )
            st.success("Scoring complete and saved to database.")

        bundle = st.session_state.get("last_bundle")
        if bundle:
            health = bundle["health_result"]
            risk = bundle["risk_result"]
            emergency = bundle["emergency_result"]
            cost = bundle["repair_cost_result"]
            rul = bundle["rul_result"]
            rec = bundle["recommendation"]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Health Score", f"{health['health_score']:.1f}/100")
            c2.metric("Risk Score", f"{risk['risk_score']:.1f}", risk["risk_level"])
            c3.metric("Emergency Index", f"{emergency['emergency_index']:.1f}", emergency["emergency_level"])
            c4.metric("Repair Cost", f"₹{cost['total_cost']:,.0f}")
            c5.metric("Remaining Useful Life", f"{rul['rul_years']:.1f} yrs")

            if emergency["requires_immediate_action"]:
                st.error("⚠️ IMMEDIATE ACTION REQUIRED — Emergency Index crossed the critical threshold.")

            st.subheader("📋 Recommendation")
            rc1, rc2, rc3 = st.columns(3)
            rc1.write(f"**Priority:** {rec['priority_level'].upper()}")
            rc2.write(f"**Action:** {rec['recommended_action']}")
            rc3.write(f"**Due date:** {rec['due_date']}")

            with st.expander("Per-detection recommendations"):
                st.dataframe(pd.DataFrame(rec["detection_actions"]))

            st.subheader("📈 Charts")
            colA, colB = st.columns(2)

            with colA:
                dist = health["analytics"]["damage_count"]["count_by_class"]
                if dist:
                    fig_pie = px.pie(
                        names=list(dist.keys()), values=list(dist.values()),
                        title="Damage Type Distribution",
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            with colB:
                sev = health["analytics"]["damage_severity"]
                fig_bar = go.Figure(go.Bar(
                    x=["Severity Score"], y=[sev["severity_score"]],
                    marker_color="crimson" if sev["severity_class"] in ("High", "Critical") else "orange",
                ))
                fig_bar.update_layout(title=f"Severity: {sev['severity_class']}", yaxis_range=[0, 1])
                st.plotly_chart(fig_bar, use_container_width=True)

            fig_cost = px.bar(
                pd.DataFrame(cost.get("line_items") or []),
                x="class_name", y="cost", title="Repair Cost by Damage Type",
            ) if cost.get("line_items") else None
            if fig_cost:
                st.plotly_chart(fig_cost, use_container_width=True)

            # Notifications & Complaint quick actions
            st.subheader("📨 Actions")
            colN, colC = st.columns(2)
            with colN:
                email = st.text_input("Notify email", value="publicworks@city.gov")
                if st.button("Send Notification"):
                    result = dispatch_notification(
                        rec, current_road_name, contacts={"email": email},
                        road_id=current_road_id, force=True,
                    )
                    if result["alert"]:
                        db.insert_notification(current_road_id, st.session_state.get("last_prediction_id"),
                                                result["alert"], result["delivery_log"])
                    st.success("Notification dispatched (simulated) and logged.")

            with colC:
                if st.button("Generate Complaint Letter"):
                    complaint = generate_complaint(
                        rec, road_name=current_road_name, asset_type=current_asset_type, road_id=current_road_id,
                    )
                    letter = format_complaint_letter(complaint)
                    db.insert_complaint(current_road_id, st.session_state.get("last_prediction_id"), complaint, letter)
                    st.session_state["last_complaint_letter"] = letter
                    st.session_state["last_complaint_id"] = complaint["complaint_id"]
                    st.text_area("Complaint Letter", letter, height=300)
                    st.success(f"Complaint {complaint['complaint_id']} saved. You can now generate the PDF Report.")

            st.markdown("---")
            st.info("➡️ Head to the **📄 PDF Report** page in the sidebar to generate and download the final inspection report.")


# =======================================================================
# PAGE: DAMAGE GROWTH SIMULATION
# =======================================================================
elif page == "📈 Damage Growth Simulation":
    st.title("📈 Damage Growth Forecast")
    bundle = st.session_state.get("last_bundle")
    if not bundle:
        st.info("Run the scoring pipeline on the Decision Intelligence page first.")
    else:
        months_ahead = st.slider("Months to project", 1, 12, 6)
        with st.spinner("Simulating..."):
            analytics = bundle["health_result"]["analytics"]
            timeline = simulate_damage_growth(
                current_damage_percentage=analytics["damage_percentage"],
                current_severity_score=analytics["damage_severity"]["severity_score"],
                severity_class=analytics["damage_severity"]["severity_class"],
                total_damage_count=analytics["damage_count"]["total_count"],
                asset_age_years=asset_age_years,
                months_ahead=months_ahead,
            )
        df_rows = []
        for snap in timeline:
            for scenario in ("optimistic", "expected", "pessimistic"):
                df_rows.append({
                    "month": snap["month"], "scenario": scenario,
                    "risk_score": snap[scenario]["risk_score"],
                    "damage_percentage": snap[scenario]["damage_percentage"],
                    "health_score": snap[scenario]["health_score"],
                })
        df = pd.DataFrame(df_rows)

        fig = px.line(
            df, x="month", y="risk_score", color="scenario", markers=True,
            title="Projected Risk Score Trend (Optimistic / Expected / Pessimistic)",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(
            df, x="month", y="damage_percentage", color="scenario", markers=True,
            title="Projected Damage Growth (%)",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df)


# =======================================================================
# PAGE: DETECTION HISTORY
# =======================================================================
elif page == "🕘 Detection History":
    st.title("🕘 Detection History")
    history = db.get_detection_history(current_road_id)
    if not history:
        st.info("No detections recorded yet.")
    else:
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name="detection_history.csv")

        st.subheader("Reports")
        st.dataframe(pd.DataFrame(db.get_reports(current_road_id)), use_container_width=True)

        st.subheader("Predictions")
        st.dataframe(pd.DataFrame(db.get_predictions_history(current_road_id)), use_container_width=True)


# =======================================================================
# PAGE: SHAP / EXPLAINABILITY
# =======================================================================
elif page == "🔍 SHAP / Explainability":
    st.title("🔍 Explainable AI Dashboard")
    bundle = st.session_state.get("last_bundle")
    if not bundle:
        st.info("Run the scoring pipeline on the Decision Intelligence page first.")
    else:
        explanation = bundle["explanation_bundle"]

        st.subheader("Rule-Based Explanations")
        for line in explanation["rule_explanations"]:
            st.write("• " + line)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Emergency Index Drivers")
            st.write(explanation["emergency_summary"])
            df_e = pd.DataFrame(explanation["emergency_explanation"]["contributions"])
            fig_e = px.bar(df_e, x="contribution", y="feature", orientation="h",
                            title="Emergency Index — Contribution Breakdown")
            st.plotly_chart(fig_e, use_container_width=True)

        with col2:
            st.subheader("Health Score Drivers")
            st.write(explanation["health_summary"])
            df_h = pd.DataFrame(explanation["health_explanation"]["contributions"])
            fig_h = px.bar(df_h, x="contribution", y="feature", orientation="h",
                            title="Health Score — Contribution Breakdown")
            st.plotly_chart(fig_h, use_container_width=True)

        st.caption(
            "Mode: rule_based_contribution (exact weighted-sum decomposition). "
            "True SHAP values will be used automatically once a trained XGBoost "
            "priority model is available at config.PRIORITY_MODEL_PATH."
        )


# =======================================================================
# PAGE: NOTIFICATIONS & COMPLAINTS
# =======================================================================
elif page == "📨 Notifications & Complaints":
    st.title("📨 Notifications & Complaints Log")

    # ===================================================================
    # EMAIL NOTIFICATION SETTINGS
    # ===================================================================
    st.subheader("📧 Email Notification Settings")

    if not is_smtp_configured():
        st.warning(
            "Gmail SMTP credentials are not configured on this server, so automatic "
            "emails cannot be sent yet. Set the `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD` "
            "environment variables (see the setup instructions at the top of "
            "`email_service.py`) and restart the app."
        )

    email_settings = db.get_email_settings()
    with st.form("email_settings_form"):
        receiver_email = st.text_input(
            "Receiver Email", value=email_settings.get("receiver_email") or "",
            placeholder="publicworks@city.gov",
        )
        auto_send = st.checkbox(
            "Enable Auto Email Notifications (High/Critical severity)",
            value=bool(email_settings.get("auto_send_enabled", False)),
        )
        col_save, col_test = st.columns(2)
        save_clicked = col_save.form_submit_button("💾 Save Settings", use_container_width=True)
        test_clicked = col_test.form_submit_button("✉️ Send Test Email", use_container_width=True)

    if save_clicked:
        if receiver_email and not is_valid_email(receiver_email):
            st.error(f"'{receiver_email}' does not look like a valid email address.")
        else:
            db.save_email_settings(receiver_email, auto_send)
            st.success("Email settings saved.")
            st.rerun()

    if test_clicked:
        if not receiver_email or not is_valid_email(receiver_email):
            st.error("Enter a valid receiver email above before sending a test.")
        else:
            with st.spinner("Sending test email..."):
                result = send_test_email(receiver_email)
            if result["success"]:
                st.success(f"✅ Test email sent to {receiver_email} - check the inbox (and spam folder).")
            else:
                st.error(f"❌ Test email failed: {result['error']}")

    with st.expander("📜 Email Send Log"):
        email_log = db.fetch_email_log(limit=100)
        if email_log:
            st.dataframe(pd.DataFrame(email_log), use_container_width=True)
        else:
            st.caption("No emails sent yet.")

    st.divider()

    st.subheader("Notifications")
    notifs = db.get_notifications(current_road_id)
    if notifs:
        st.dataframe(pd.DataFrame(notifs), use_container_width=True)
    else:
        st.info("No notifications yet.")

    st.subheader("Complaints")
    complaints = db.get_complaints(current_road_id)
    if complaints:
        df_c = pd.DataFrame(complaints)
        st.dataframe(df_c, use_container_width=True)
        selected = st.selectbox("View letter for complaint:", ["(none)"] + df_c["complaint_id"].tolist())
        if selected != "(none)":
            letter = df_c.loc[df_c["complaint_id"] == selected, "letter_text"].iloc[0]
            st.text_area("Letter", letter, height=300)
    else:
        st.info("No complaints filed yet.")


# =======================================================================
# PAGE: PDF REPORT
# =======================================================================
elif page == "📄 PDF Report":
    st.title("📄 Generate PDF Inspection Report")
    bundle = st.session_state.get("last_bundle")

    if not bundle:
        st.info("Run the scoring pipeline on the Decision Intelligence page first.")
    else:
        st.caption(
            "This report bundles: date, uploaded image, detected damages + confidence "
            "scores, health score, risk score, emergency index, repair cost, remaining "
            "useful life, recommendations, and the municipality complaint (if generated)."
        )

        annotated_path = st.session_state.get("last_annotated_image_path")
        complaint_letter = st.session_state.get("last_complaint_letter")
        complaint_id = st.session_state.get("last_complaint_id")

        c1, c2 = st.columns(2)
        c1.write(f"**Annotated image attached:** {'Yes' if annotated_path else 'No (video/manual session)'}")
        c2.write(f"**Complaint attached:** {complaint_id or 'None generated yet'}")

        inspector_name = st.text_input("Inspector / Reported by", value="AI Monitoring System (Automated Detection)")

        if st.button("🖨️ Generate PDF Report", type="primary"):
            with st.spinner("Building PDF report..."):
                pdf_path = generate_pdf_report(
                    bundle,
                    road_name=current_road_name,
                    asset_type=current_asset_type,
                    annotated_image_path=annotated_path,
                    complaint_letter_text=complaint_letter,
                    complaint_id=complaint_id,
                    inspector_name=inspector_name,
                )
                # Log the report in the database too
                db.insert_report(
                    current_road_id, get_session_id(), st.session_state.get("last_prediction_id"),
                    summary={"generated_via": "pdf_report_service", "path": pdf_path},
                    report_type="inspection", file_path=pdf_path,
                )
            st.success(f"PDF report generated: {pdf_path}")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF Report", data=f.read(),
                    file_name=Path(pdf_path).name, mime="application/pdf",
                )


# =======================================================================
# PAGE: ANALYTICS DASHBOARD (default landing page)
# =======================================================================
else:
    st.title("📊 Analytics Dashboard")
    st.caption("Overview across all recorded inspections and predictions.")

    all_predictions = db.get_predictions_history(current_road_id, limit=1000)
    all_detections = db.get_detection_history(current_road_id, limit=2000)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Detections", len(all_detections))
    kpi2.metric("Inspection Sessions Scored", len(all_predictions))
    if all_predictions:
        latest = all_predictions[-1]
        kpi3.metric("Latest Health Score", f"{latest['health_score']:.1f}" if latest["health_score"] is not None else "N/A")
        kpi4.metric("Latest Risk Level", latest["risk_level"] or "N/A")
    else:
        kpi3.metric("Latest Health Score", "N/A")
        kpi4.metric("Latest Risk Level", "N/A")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ Damage Distribution")
        dist = db.get_damage_distribution()
        if dist:
            df_dist = pd.DataFrame(dist)
            fig = px.pie(df_dist, names="class_name", values="count", title="Damage Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No detections recorded yet.")

    with col2:
        st.subheader("2️⃣ Severity Bar Chart")
        if all_predictions:
            df_pred = pd.DataFrame(all_predictions)
            fig = px.bar(df_pred, x="created_at", y="severity_score", color="severity_class",
                         title="Severity Score Over Sessions")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scored sessions yet.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("3️⃣ Risk Trend")
        if all_predictions:
            df_pred = pd.DataFrame(all_predictions)
            fig = px.line(df_pred, x="created_at", y="risk_score", markers=True, title="Risk Score Trend")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scored sessions yet.")

    with col4:
        st.subheader("4️⃣ Repair Cost Over Time")
        if all_predictions:
            df_pred = pd.DataFrame(all_predictions)
            fig = px.bar(df_pred, x="created_at", y="repair_cost", title="Estimated Repair Cost per Session")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scored sessions yet.")

    st.subheader("5️⃣ Damage Growth Forecast (latest scored session)")
    bundle = st.session_state.get("last_bundle")
    if bundle:
        timeline = bundle["simulation_timeline"]
        df_rows = []
        for snap in timeline:
            for scenario in ("optimistic", "expected", "pessimistic"):
                df_rows.append({
                    "month": snap["month"], "scenario": scenario,
                    "risk_score": snap[scenario]["risk_score"],
                })
        df = pd.DataFrame(df_rows)
        fig = px.line(df, x="month", y="risk_score", color="scenario", markers=True,
                      title="Forecasted Risk Score (6-month projection)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the scoring pipeline (Decision Intelligence page) to see a forecast here.")

    st.markdown("---")
    st.subheader("Recent Detection History")
    if all_detections:
        st.dataframe(pd.DataFrame(all_detections).head(50), use_container_width=True)
    else:
        st.info("No detections yet — go to Upload & Detect to get started.")