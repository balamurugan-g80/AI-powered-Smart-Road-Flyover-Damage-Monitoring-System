"""
live_detection.py
------------------
"Live Detection" page: opens the user's webcam DIRECTLY INSIDE the
Streamlit browser tab (no external OpenCV window) and runs real-time
YOLOv8 inference on every frame, using `streamlit-webrtc` to bridge the
browser's camera feed into a Python video-processing callback.

Architecture
------------
`streamlit-webrtc` runs the actual frame callback (`recv()`) on a
BACKGROUND THREAD separate from Streamlit's main script thread (this is
how it achieves real-time performance without blocking the UI). That
means:

  * `recv()` must never call `st.*` functions directly (Streamlit's
    session state / DOM are not thread-safe from a background thread).
  * Shared stats (FPS, damage counter, last detections) are written to
    a `threading.Lock()`-protected block inside the processor, and the
    main Streamlit thread polls them in a `while ctx.state.playing`
    loop, updating `st.empty()` placeholders - this is the standard
    streamlit-webrtc pattern for live stat overlays.

Both the bounding boxes/labels/confidence AND the severity overlay are
produced by `DetectionService.infer_frame()` + `.annotate_frame()` -
the exact same functions used by the image and video pipelines - so
there is only ONE annotation code path in the whole app to keep in
sync, and this page automatically shares any future fix to that logic.

Event-driven email alerts (one email per unique damage event, not per
frame) reuse `DamageEventTracker` and `fire_damage_event_email_async`
from `detection_service.py` - the SAME object-identity tracking and
background-thread SMTP dispatch used by the batch video pipeline, so
there is only one implementation of that behavior to keep in sync too.
"""

import time
import logging
import threading
from typing import Dict, List, Optional

import streamlit as st

import database as db
from detection_service import DamageEventTracker, fire_damage_event_email_async
from email_service import is_smtp_configured

logger = logging.getLogger("live_detection")
logging.basicConfig(level=logging.INFO)

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# A public STUN server so the browser can negotiate a WebRTC connection
# even when the machine is behind typical home/office NAT. No video data
# ever leaves the machine except directly to this Streamlit session.
_RTC_CONFIGURATION = None
if WEBRTC_AVAILABLE:
    _RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


if WEBRTC_AVAILABLE:
    class YOLOVideoProcessor(VideoProcessorBase):
        """
        Per-connection video processor. streamlit-webrtc instantiates ONE
        of these per active browser stream and calls `recv()` once per
        incoming camera frame on a background thread.
        """

        def __init__(
            self,
            model,
            confidence: Optional[float] = None,
            email_alerts_enabled: bool = False,
            email_receiver: Optional[str] = None,
            email_road_name: Optional[str] = None,
            email_road_id: Optional[int] = None,
            email_confidence_threshold: float = 0.5,
            email_cooldown_seconds: float = 60.0,
        ):
            self.model = model
            self.confidence = confidence
            self.lock = threading.Lock()
            # Stats read by the main Streamlit thread's polling loop below.
            self.frame_count = 0
            self.detection_count = 0
            self.last_fps = 0.0
            self.last_detections: List[Dict] = []
            self.last_error: Optional[str] = None

            # --- Event-driven email alerts (one per unique damage event) ---
            self.email_alerts_enabled = email_alerts_enabled
            self.email_receiver = email_receiver
            self.email_road_name = email_road_name
            self.email_road_id = email_road_id
            self.email_confidence_threshold = email_confidence_threshold
            self.event_tracker = DamageEventTracker(cooldown_seconds=email_cooldown_seconds) if email_alerts_enabled else None
            self.recent_email_events: List[Dict] = []  # bounded log the UI polls

        def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
            img = frame.to_ndarray(format="bgr24")
            t0 = time.time()
            try:
                results, detections = self.model.infer_frame(img, conf=self.confidence)
                annotated = self.model.annotate_frame(
                    img, results, detections,
                    draw_severity=True, draw_frame_info=False,
                )
                elapsed = time.time() - t0
                with self.lock:
                    self.frame_count += 1
                    self.detection_count += len(detections)
                    self.last_fps = round(1.0 / elapsed, 1) if elapsed > 0 else 0.0
                    self.last_detections = detections
                    self.last_error = None
                    frame_number = self.frame_count

                # --- Event-driven email alerts: same one-per-unique-object
                #     logic as the batch video pipeline, reusing the exact
                #     same DamageEventTracker + fire_damage_event_email_async
                #     from detection_service.py so there is only one
                #     implementation of this behavior in the whole app. ---
                if self.email_alerts_enabled and self.email_receiver and self.event_tracker:
                    for det in detections:
                        if det["confidence"] < self.email_confidence_threshold:
                            continue
                        if not det.get("bbox"):
                            continue  # no valid box -> never email, per spec
                        should_send = self.event_tracker.register_detection(det["class_name"], det["bbox"])
                        if not should_send:
                            continue

                        event_entry = {
                            "class_name": det["class_name"], "confidence": det["confidence"],
                            "frame_number": frame_number, "status": "queued", "error": None,
                        }
                        with self.lock:
                            self.recent_email_events.append(event_entry)
                            del self.recent_email_events[:-20]

                        fire_damage_event_email_async(
                            det["class_name"], det["confidence"], annotated.copy(), frame_number,
                            self.email_road_name, self.email_road_id, self.email_receiver,
                            on_email_result=lambda r, _entry=event_entry: _entry.update(
                                {"status": "sent" if r["success"] else "failed", "error": r.get("error")}
                            ),
                        )

                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            except Exception as e:
                # NEVER let an inference error kill the WebRTC stream -
                # log it, surface it to the polling loop, and pass the
                # raw frame through untouched so the video keeps playing.
                logger.exception(f"Live inference failed on a frame: {e}")
                with self.lock:
                    self.last_error = str(e)
                return frame


def live_detection_page(model, road_name: Optional[str] = None, road_id: Optional[int] = None) -> None:
    """
    Renders the "Live Detection" Streamlit page. `model` must be an
    already-loaded DetectionService instance (or None, in which case an
    explanatory message is shown instead of the camera widget).
    `road_name`/`road_id` come from app.py's sidebar Road/Flyover selector
    so event-driven email alerts are tagged consistently with every other
    page.
    """
    st.title("🔴 Live Detection")
    st.caption(
        "Opens your webcam directly in this browser tab and runs real-time "
        "YOLOv8 road/flyover damage detection - no external OpenCV window."
    )

    if not WEBRTC_AVAILABLE:
        st.error(
            "`streamlit-webrtc` (and its `av` dependency) are not installed. "
            "Run `pip install streamlit-webrtc av` and restart the app to "
            "enable this page."
        )
        return

    if model is None:
        st.warning(
            "No detection model available - Live Detection needs a trained "
            "`best.pt` at the path configured in `config.YOLO_MODEL_PATH`. "
            "Use the Manual Detections tab on Upload & Detect instead."
        )
        return

    conf_override = st.slider(
        "Confidence threshold", min_value=0.05, max_value=0.95,
        value=float(model.confidence_threshold), step=0.05,
        help="Lower = more (possibly false-positive) detections. Higher = fewer, more confident detections.",
    )

    st.subheader("📧 Live Event-Driven Email Alerts")
    email_settings = db.get_email_settings()
    col_a, col_b = st.columns(2)
    email_alerts_enabled = col_a.checkbox(
        "Enable Live Email Alerts", value=False,
        help="Sends ONE email per unique damage event (tracked by object identity, "
             "not per frame) using the receiver configured in Email Notification Settings.",
    )
    email_cooldown = col_b.number_input(
        "Cooldown per object (seconds)", min_value=5, max_value=600, value=60, step=5,
    )
    if email_alerts_enabled and not email_settings.get("receiver_email"):
        st.warning(
            "No receiver email is configured yet - set one on the Notifications & "
            "Complaints page (Email Notification Settings) before starting the stream."
        )
    if email_alerts_enabled and not is_smtp_configured():
        st.warning(
            "Gmail SMTP credentials are not configured on this server (GMAIL_ADDRESS / "
            "GMAIL_APP_PASSWORD environment variables) - alerts will fail until that's set up."
        )

    ctx = webrtc_streamer(
        key="live-road-damage-detection",
        video_processor_factory=lambda: YOLOVideoProcessor(
            model, confidence=conf_override,
            email_alerts_enabled=email_alerts_enabled,
            email_receiver=email_settings.get("receiver_email"),
            email_road_name=road_name, email_road_id=road_id,
            email_confidence_threshold=conf_override,
            email_cooldown_seconds=float(email_cooldown),
        ),
        rtc_configuration=_RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.divider()
    st.subheader("📊 Live Stats")
    stat_cols = st.columns(3)
    fps_ph = stat_cols[0].empty()
    count_ph = stat_cols[1].empty()
    frames_ph = stat_cols[2].empty()
    detail_ph = st.empty()
    error_ph = st.empty()
    email_feed_ph = st.empty()

    st.subheader("📜 Notification History")
    history_ph = st.empty()

    if not ctx.state.playing:
        fps_ph.metric("FPS", "—")
        count_ph.metric("Total Damage Detections", "—")
        frames_ph.metric("Frames Processed", "—")
        st.info("Click **START** above and allow camera access to begin live detection.")
        history = db.fetch_email_log(limit=50)
        history_ph.dataframe(history, use_container_width=True) if history else history_ph.caption("No email alerts sent yet.")
        return

    # --- Live polling loop: standard streamlit-webrtc pattern for
    #     surfacing stats computed on the background recv() thread.
    #     Runs only while the stream is active; exits cleanly when the
    #     user clicks STOP or navigates away. ---
    while ctx.state.playing:
        processor = ctx.video_processor
        if processor is None:
            time.sleep(0.2)
            continue

        with processor.lock:
            fps = processor.last_fps
            total_detections = processor.detection_count
            frames_processed = processor.frame_count
            detections = list(processor.last_detections)
            error = processor.last_error
            email_events = list(processor.recent_email_events)

        fps_ph.metric("FPS", fps)
        count_ph.metric("Total Damage Detections", total_detections)
        frames_ph.metric("Frames Processed", frames_processed)

        if error:
            error_ph.error(f"Live inference error (frame skipped): {error}")
        else:
            error_ph.empty()

        if detections:
            detail_ph.write(
                "**Current frame:** "
                + ", ".join(f"{d['class_name']} ({d['confidence']*100:.0f}%)" for d in detections)
            )
        else:
            detail_ph.write("**Current frame:** no damage detected")

        # --- "📧 Email sent for Pothole" live feed ---
        if email_events:
            icon = {"queued": "⏳", "sent": "📧", "failed": "⚠️"}
            lines = [
                f"{icon.get(ev['status'], '•')} {ev['status'].upper()} - **{ev['class_name']}** "
                f"(frame {ev['frame_number']}, {ev['confidence']*100:.0f}%)"
                + (f" - {ev['error']}" if ev.get("error") else "")
                for ev in reversed(email_events[-5:])
            ]
            email_feed_ph.markdown("\n\n".join(lines))
        else:
            email_feed_ph.empty()

        history = db.fetch_email_log(limit=50)
        if history:
            history_ph.dataframe(history, use_container_width=True)
        else:
            history_ph.caption("No email alerts sent yet.")

        time.sleep(0.5)