"""
detection_service.py
---------------------
Core YOLOv8 detection engine for the Smart Road & Flyover Damage
Monitoring system.

  1. Loading the trained YOLOv8 model (best.pt) once.
  2. Running detection on a single image.
  3. Running detection on an uploaded video - CORRECTED PIPELINE:
       - reads the video with OpenCV directly (cv2.VideoCapture)
       - runs YOLOv8 inference on EVERY frame (model.predict(frame, conf=..., imgsz=640))
       - draws boxes/labels/confidence using Ultralytics' OWN renderer
         (`results[0].plot()`) - NOT a hand-rolled cv2.rectangle() pass
       - overlays frame number + timestamp text on top of that rendered frame
       - writes every frame to a cv2.VideoWriter at the source video's
         original fps/width/height
       - saves annotated_video.mp4 to data/processed/annotated_videos/
       - saves every frame that contains a detection to data/processed/frames/
       - reports rich per-frame progress via a callback so a UI (Streamlit)
         can show current frame / frames processed / ETA / detection count / FPS live
  4. Running real-time detection on a live stream (webcam / RTSP / CCTV).
  5. Saving individual frames where damage was detected.

This module has no Streamlit imports - it can be called from a
Streamlit page, a CLI script, or a batch job.
"""

import os
import cv2
import time
import math
import shutil
import logging
import tempfile
import threading
from pathlib import Path
from typing import List, Dict, Generator, Union, Optional, Callable

from ultralytics import YOLO
from score_utils import get_severity_weight

# --- sibling utility modules (this project keeps them as flat top-level
#     modules, e.g. image_utils.py / video_utils.py sit next to app.py -
#     there is no `utils` package here) ---
from image_utils import draw_detections, save_annotated_image, load_image, resize_image
from video_utils import (
    convert_to_h264,
    verify_video_file,
    get_video_codec_fourcc,
    extract_first_frame,
)

logger = logging.getLogger("detection_service")
logging.basicConfig(level=logging.INFO)

try:
    from config import (
        YOLO_MODEL_PATH,
        YOLO_CONFIDENCE_THRESHOLD,
        YOLO_IOU_THRESHOLD,
        YOLO_IMAGE_SIZE,
        YOLO_DEVICE,
        DAMAGE_CLASSES,
        FRAME_SAMPLE_RATE_FPS,
        PROCESSED_VIDEOS_DIR,
        FRAMES_DIR,
        SEVERITY_THRESHOLDS,
    )
except ImportError:
    YOLO_MODEL_PATH = "models/yolo/best.pt"
    YOLO_CONFIDENCE_THRESHOLD = 0.35
    YOLO_IOU_THRESHOLD = 0.45
    YOLO_IMAGE_SIZE = 640
    YOLO_DEVICE = "cpu"
    DAMAGE_CLASSES = ["crack", "pothole", "spalling", "joint_failure", "surface_erosion"]
    FRAME_SAMPLE_RATE_FPS = 2
    PROCESSED_VIDEOS_DIR = Path("data/processed/annotated_videos")
    FRAMES_DIR = Path("data/processed/frames")
    SEVERITY_THRESHOLDS = {"low": (0.0, 0.25), "medium": (0.25, 0.50), "high": (0.50, 0.75), "critical": (0.75, 1.0)}

# Video inference uses a lower, recall-oriented confidence threshold than
# still images by default (0.15) - continuous frames give many chances to
# confirm a defect, so a lower per-frame threshold with visual review is
# preferable to missing damage on a single low-confidence frame.
VIDEO_CONFIDENCE_THRESHOLD = 0.15
VIDEO_IMAGE_SIZE = 640

PROCESSED_VIDEOS_DIR = Path(PROCESSED_VIDEOS_DIR)
FRAMES_DIR = Path(FRAMES_DIR)
PROCESSED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


# =======================================================================
# EVENT-DRIVEN EMAIL ALERTS: object identity tracking
# =======================================================================
class DamageEventTracker:
    """
    Decides whether a given detection is (a) the SAME real-world damage
    object seen in a recent frame, or (b) a genuinely NEW damage event -
    so "one email per unique damage event" can be enforced instead of
    one email per frame.

    Matching is done PER CLASS (a pothole never matches a bridge_crack)
    using EITHER of two independent signals - either is sufficient to
    call it the same object, since a fast-moving camera can make IoU
    drop to 0 between frames for a real match, while a static camera
    with two adjacent potholes can have overlapping IoU for genuinely
    different ones close together, so IoU alone isn't reliable:

      1. IoU (Intersection-over-Union) of the two bounding boxes - good
         for a mostly-static camera where the same object's box barely
         moves frame to frame.
      2. Centroid distance in pixels - good when the camera itself is
         moving (video/dashcam), so the box drifts across the frame but
         is still clearly the same physical object frame-to-frame.

    Once matched to an existing event, `cooldown_seconds` gates repeat
    emails for that SAME object; a detection that matches nothing gets
    treated as a brand new event and always fires immediately (subject
    to the caller's own confidence-threshold check upstream).

    Thread-safe: a lock guards internal state since this can be shared
    across the video-processing thread and a live-stream's recv() thread.
    """

    def __init__(
        self,
        cooldown_seconds: float = 60.0,
        iou_match_threshold: float = 0.3,
        centroid_match_distance_px: float = 80.0,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.iou_match_threshold = iou_match_threshold
        self.centroid_match_distance_px = centroid_match_distance_px
        self._events: List[Dict] = []
        self._lock = threading.Lock()

    @staticmethod
    def _centroid(bbox) -> tuple:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _find_match(self, class_name: str, bbox) -> Optional[Dict]:
        cx, cy = self._centroid(bbox)
        best, best_dist = None, None
        for ev in self._events:
            if ev["class_name"] != class_name:
                continue
            iou = self._iou(ev["bbox"], bbox)
            ex, ey = ev["centroid"]
            dist = math.hypot(cx - ex, cy - ey)
            if iou >= self.iou_match_threshold or dist <= self.centroid_match_distance_px:
                if best_dist is None or dist < best_dist:
                    best, best_dist = ev, dist
        return best

    def register_detection(self, class_name: str, bbox, now: Optional[float] = None) -> bool:
        """
        Call this ONCE per qualifying (confidence-passed, box-visible)
        detection, in frame order. Returns True exactly when an email
        SHOULD be sent for it right now:
          - True the first time this physical object is seen at all.
          - True again only after `cooldown_seconds` has elapsed since
            the LAST email for this same object.
          - False otherwise (same object, still in cooldown).
        A detection of the same class far enough away (or with low
        enough IoU) is treated as a different object and always
        returns True immediately, regardless of any other object's
        cooldown - satisfies "unless a NEW pothole is detected in
        another location".
        """
        now = time.time() if now is None else now
        with self._lock:
            match = self._find_match(class_name, bbox)
            if match is None:
                self._events.append({
                    "class_name": class_name, "bbox": bbox,
                    "centroid": self._centroid(bbox), "last_email_ts": now,
                })
                return True

            match["bbox"] = bbox
            match["centroid"] = self._centroid(bbox)
            if (now - match["last_email_ts"]) >= self.cooldown_seconds:
                match["last_email_ts"] = now
                return True
            return False

    def active_event_count(self) -> int:
        with self._lock:
            return len(self._events)


def fire_damage_event_email_async(
    class_name: str,
    confidence: float,
    annotated_frame,
    frame_number: Optional[int],
    road_name: Optional[str],
    road_id: Optional[int],
    receiver_email: str,
    frames_output_dir: Union[str, Path] = FRAMES_DIR,
    on_email_result: Optional[Callable[[Dict], None]] = None,
) -> None:
    """
    Fires ONE event-driven damage-alert email on a background daemon
    thread, so the video/live detection loop calling this is NEVER
    blocked waiting on an SMTP round-trip ("Continue Detection WITHOUT
    stopping the video").

    IMPORTANT: `annotated_frame` must already have the bounding box,
    label, confidence, and timestamp drawn on it (i.e. the return value
    of `annotate_frame()`) - this function saves EXACTLY the frame it's
    given as the email attachment, never re-annotates or substitutes the
    raw frame.

    Deferred imports (notification_service / email_service / database)
    are deliberate: they keep detection_service.py importable and
    testable on its own (pure CV, no DB/SMTP dependency) when email
    alerts aren't used, while still doing the real work when they are.
    """
    from notification_service import get_action_for_class
    from email_service import send_damage_event_email
    import database as db

    frames_output_dir = Path(frames_output_dir)
    frames_output_dir.mkdir(parents=True, exist_ok=True)
    image_path = frames_output_dir / f"event_{class_name}_{int(time.time() * 1000)}.jpg"
    try:
        cv2.imwrite(str(image_path), annotated_frame)
    except Exception as e:
        logger.error(f"[EMAIL EVENT] Failed to save attachment frame for '{class_name}': {e}")
        image_path = None

    action_info = get_action_for_class(class_name)

    def _worker():
        event = {
            "road_name": road_name or "Unnamed Segment",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "damage_class": class_name,
            "confidence": confidence,
            "severity": action_info["severity"],
            "frame_number": frame_number,
            "recommended_action": action_info["action"],
            "location": None,
            "priority": action_info["priority"],
        }
        result = send_damage_event_email(
            event, receiver_email, attachment_path=str(image_path) if image_path else None,
        )
        db.log_damage_email_event(
            road_id, receiver_email, result["subject"], action_info["severity"],
            "sent" if result["success"] else "failed", class_name, confidence,
            frame_number=frame_number, image_path=str(image_path) if image_path else None,
            error_message=result.get("error"),
        )
        logger.info(
            f"[EMAIL EVENT] class={class_name} confidence={confidence:.2f} "
            f"frame={frame_number} success={result['success']} error={result.get('error')}"
        )
        if on_email_result:
            try:
                on_email_result({
                    "class_name": class_name, "confidence": confidence,
                    "frame_number": frame_number, "success": result["success"],
                    "error": result.get("error"),
                })
            except Exception as cb_err:
                # A broken UI callback must never take down the email worker thread.
                logger.error(f"[EMAIL EVENT] on_email_result callback raised: {cb_err}")

    threading.Thread(target=_worker, daemon=True, name=f"email-event-{class_name}").start()


class DetectionService:
    """
    Wraps a YOLOv8 model instance and exposes high-level detection
    operations for images, uploaded video files, and live streams.

    Instantiate ONCE per process (e.g. cached via st.cache_resource in
    the Streamlit app) - loading the model is the expensive part.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = YOLO_MODEL_PATH,
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold: float = YOLO_IOU_THRESHOLD,
        image_size: int = YOLO_IMAGE_SIZE,
        device: str = YOLO_DEVICE,
    ):
        self.model_path = str(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.model = self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self) -> YOLO:
        """Loads the YOLOv8 model from best.pt. Raises clearly if missing."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YOLOv8 weights not found at '{self.model_path}'. "
                f"Place your trained 'best.pt' there or update config.YOLO_MODEL_PATH."
            )
        logger.info(f"Loading YOLOv8 model from {self.model_path} on device={self.device}")
        model = YOLO(self.model_path)
        return model

    # ------------------------------------------------------------------
    def get_model_info(self) -> Dict:
        """
        Returns basic metadata about the loaded model - used by the
        Streamlit "Detection Model Debug Panel" (model path, class
        names, class count) so the UI can show what's actually loaded
        without touching the model's internals directly.
        """
        names = getattr(self.model, "names", {}) or {}
        return {
            "model_path": self.model_path,
            "model_type": type(self.model).__name__,
            "num_classes": len(names),
            "names": names,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "image_size": self.image_size,
            "device": self.device,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _severity_class_for(class_name: str, confidence: float) -> str:
        """
        Buckets a single detection into low/medium/high/critical severity
        using the same {confidence * class-weight} formula as
        recommendation_service, so the label shown on the video/image
        overlay always agrees with what Decision Intelligence reports.
        """
        pseudo_severity = max(0.0, min(1.0, confidence * get_severity_weight(class_name)))
        for label, (lo, hi) in SEVERITY_THRESHOLDS.items():
            if lo <= pseudo_severity < hi or (label == "critical" and pseudo_severity >= hi):
                return label
        return "low"

    # ------------------------------------------------------------------
    def infer_frame(self, frame, conf: Optional[float] = None, imgsz: Optional[int] = None):
        """
        Runs one YOLOv8 forward pass on a single frame/image and returns
        BOTH the raw Ultralytics `Results` (needed for `.plot()`) and the
        standardized detection dict list (needed for DB/analytics) - this
        is the single inference entry point used by detect_image(),
        process_video(), AND the live webcam page, so all three code
        paths run inference identically.

        Returns: (results, detections)
        """
        results = self.model.predict(
            source=frame,
            conf=conf if conf is not None else self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=imgsz if imgsz is not None else self.image_size,
            device=self.device,
            verbose=False,
        )
        detections = self._results_to_detections(results)
        return results, detections

    # ------------------------------------------------------------------
    def annotate_frame(
        self,
        frame,
        results,
        detections: List[Dict],
        frame_number: Optional[int] = None,
        timestamp_sec: Optional[float] = None,
        draw_severity: bool = True,
        draw_frame_info: bool = True,
    ):
        """
        Turns raw inference output into a fully annotated frame:
        bounding box + class name + confidence (all via Ultralytics'
        own `results[0].plot()` renderer - never hand-drawn, so box
        placement always matches exactly what the model predicted),
        PLUS a severity label per detection and an optional frame
        number/timestamp overlay.

        This is the ONLY place that produces an annotated frame in the
        whole pipeline - detect_image(), process_video(), and the live
        webcam page all call this same function, so a fix here fixes
        every code path at once.

        IMPORTANT (this is what a "boxes not visible" bug usually is):
        the return value of this function is what MUST be written to
        the VideoWriter / shown in Streamlit - never the original
        `frame`. Callers that accidentally write/display `frame`
        instead of this function's return value will get an
        unannotated video even though detections were found correctly.
        """
        result = results[0]
        annotated = result.plot()  # Ultralytics-rendered box + label + confidence

        # --- Self-check: if plot() ever returns something byte-identical
        #     to the raw input frame while detections were found, that's
        #     exactly the "boxes not visible" failure mode - log it LOUDLY
        #     instead of silently shipping a blank-looking frame. ---
        if detections and annotated is frame:
            logger.error(
                "[ANNOTATE BUG] results[0].plot() returned the SAME object as the "
                "input frame despite %d detection(s) - boxes will not be visible. "
                "This usually means an incompatible/downgraded ultralytics version.",
                len(detections),
            )
        elif detections:
            try:
                diff_sum = int(cv2.absdiff(annotated, frame).sum()) if annotated.shape == frame.shape else -1
                if diff_sum == 0:
                    logger.warning(
                        "[ANNOTATE BUG] annotated frame is pixel-identical to the raw "
                        "frame despite %d detection(s) at frame_number=%s - boxes are "
                        "NOT visible. Check ultralytics version / results[0].boxes contents.",
                        len(detections), frame_number,
                    )
            except Exception:
                pass  # diff check is best-effort debug aid only, never fatal

        # --- Severity label per detection, positioned just above each
        #     Ultralytics-drawn box (Ultralytics' own plot() only shows
        #     class+confidence, not severity - this is the one thing we
        #     DO draw manually, deliberately, on top of it). ---
        if draw_severity and detections:
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                severity = self._severity_class_for(det["class_name"], det["confidence"])
                color = {
                    "low": (0, 200, 0), "medium": (0, 200, 255),
                    "high": (0, 100, 255), "critical": (0, 0, 255),
                }.get(severity, (255, 255, 255))
                label = f"severity:{severity}"
                text_y = max(15, y2 + 18)  # just below the box; Ultralytics already labels above it
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, text_y - th - 4), (x1 + tw + 4, text_y + 2), color, -1)
                cv2.putText(annotated, label, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Frame number / timestamp overlay (debug + audit trail) ---
        if draw_frame_info and frame_number is not None:
            overlay_text = f"Frame: {frame_number}"
            if timestamp_sec is not None:
                overlay_text += f"  |  Time: {timestamp_sec:0.2f}s"
            cv2.putText(annotated, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated

    # ------------------------------------------------------------------
    def _run_inference(self, frame, conf: Optional[float] = None, imgsz: Optional[int] = None):
        """
        Runs a single YOLOv8 forward pass on one frame/image.

        Returns the RAW ultralytics `Results` object (not just a dict
        list) so callers that need `results[0].plot()` (the video
        pipeline) have access to it, while `_results_to_detections()`
        below converts it to the standardized dict format used
        everywhere else in the app (image pipeline, DB, analytics).
        """
        results = self.model.predict(
            source=frame,
            conf=conf if conf is not None else self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=imgsz if imgsz is not None else self.image_size,
            device=self.device,
            verbose=False,
        )
        return results

    @staticmethod
    def _results_to_detections(results) -> List[Dict]:
        """Converts one ultralytics Results object into standardized detection dicts."""
        detections = []
        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names.get(class_id, f"class_{class_id}")
            area_px = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": (x1, y1, x2, y2),
                "area_px": round(area_px, 2),
            })
        return detections

    # ------------------------------------------------------------------
    # 1. IMAGE DETECTION  (unchanged - still uses draw_detections for full
    #    control over the still-image annotation style used elsewhere in the app)
    # ------------------------------------------------------------------
    def detect_image(self, image_path: Union[str, Path], save_output: bool = True) -> Dict:
        """
        Runs damage detection on a single image.

        Returns:
            {
              "detections": [ {class_name, confidence, bbox, area_px}, ... ],
              "annotated_path": str | None,
              "image_shape": (h, w),
              "inference_time": float (seconds)
            }
        """
        image = load_image(image_path)
        image = resize_image(image, max_width=1600)

        start = time.time()
        results, detections = self.infer_frame(image)
        elapsed = time.time() - start
        logger.info(f"Image inference on '{image_path}' -> {len(detections)} detections in {elapsed:.3f}s")

        annotated_path = None
        if save_output:
            annotated = draw_detections(image, detections)
            annotated_path = save_annotated_image(annotated, Path(image_path).name)

        return {
            "detections": detections,
            "annotated_path": annotated_path,
            "image_shape": image.shape[:2],
            "inference_time": round(elapsed, 4),
        }

    # ------------------------------------------------------------------
    # 2. VIDEO DETECTION - CORRECTED PIPELINE
    # ------------------------------------------------------------------
    def process_video(
        self,
        video_path: Union[str, Path],
        output_video_path: Optional[Union[str, Path]] = None,
        confidence: float = VIDEO_CONFIDENCE_THRESHOLD,
        image_size: int = VIDEO_IMAGE_SIZE,
        save_detected_frames: bool = True,
        frames_output_dir: Union[str, Path] = FRAMES_DIR,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        live_preview: bool = True,
        live_preview_every_n_frames: int = 1,
        email_alerts_enabled: bool = False,
        email_receiver: Optional[str] = None,
        email_road_name: Optional[str] = None,
        email_road_id: Optional[int] = None,
        email_confidence_threshold: float = 0.5,
        email_cooldown_seconds: float = 60.0,
    ) -> Dict:
        """
        Processes an uploaded video end-to-end using the Ultralytics
        renderer for every frame (no manual cv2.rectangle drawing):

          1. Copies the source video into a private temporary file.
          2. Opens that temp file with cv2.VideoCapture.
          3. Reads every frame.
          4. For every frame: results = model.predict(frame, conf=confidence,
             imgsz=image_size, verbose=False).
          5. annotated_frame = results[0].plot()  <- Ultralytics-rendered
             box + class name + confidence. Nothing is drawn manually,
             except a frame-number/timestamp text overlay on top.
          6. Writes annotated_frame to a cv2.VideoWriter (fourcc='mp4v')
             opened at the SOURCE video's original fps/width/height.
          7. Releases VideoCapture and VideoWriter in a `finally` block.
          8. Verifies the written file exists, is non-empty, and is
             actually decodable before doing anything else with it.
          9. Every frame containing >=1 detection is saved to
             `frames_output_dir`.
          10. Because 'mp4v' is usually NOT decodable by browsers (and
              therefore not by Streamlit's st.video(), which renders an
              HTML5 <video> tag), the raw file is automatically
              transcoded to H.264/yuv420p MP4 - first via a system
              `ffmpeg` subprocess, falling back to `imageio`+
              `imageio-ffmpeg` if ffmpeg isn't installed - and that
              converted file is re-verified before being reported as
              the "playable" output.
          11. Calls `progress_callback(stats)` every frame with:
                {current_frame, total_frames, detections_count,
                 elapsed_seconds, eta_seconds, processing_fps,
                 live_preview_frame_bgr}
              so a Streamlit page can render a live progress bar + metrics
              AND a live bounding-box preview image (the same
              Ultralytics-annotated frame just written to the output
              video, as a raw BGR numpy array - convert with
              cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) before st.image()).
              Pass live_preview=False to omit the frame (stats-only,
              slightly cheaper) or live_preview_every_n_frames=N to only
              include it every Nth frame.

          12. EVENT-DRIVEN EMAIL ALERTS (opt-in via email_alerts_enabled):
              a DamageEventTracker uses IoU + centroid matching to tell
              whether a detection is the SAME physical object seen
              recently (same class, nearby box) or a NEW damage event.
              Every detection that (a) passes email_confidence_threshold,
              (b) has a real bounding box, and (c) is either brand new or
              past its `email_cooldown_seconds` cooldown fires exactly
              ONE email - on a background daemon thread, so the video
              loop is never blocked waiting on SMTP. Each callback's
              stats dict includes an "email_events" snapshot list
              ({class_name, confidence, frame_number, status, error})
              so the UI can show "📧 Email sent for Pothole" live.
          12. If the final file still fails verification, the first
              decodable/annotated frame is saved as a JPEG and the exact
              failure reason is returned so the UI can show it instead
              of a blank/broken video player.

        Returns:
            {
              "output_video_path": str,       # final, browser-safe if playable=True
              "saved_frame_paths": [str, ...],
              "all_detections": [
                  {"frame_number", "timestamp_sec", "class_name",
                   "confidence", "bbox", "area_px"}, ...
              ],
              "total_frames_processed": int,
              "video_fps": float,
              "width": int, "height": int,
              "playable": bool,
              "failure_reason": str | None,
              "first_frame_preview_path": str | None,
              "conversion_method": "ffmpeg" | "imageio" | None,
              "raw_codec": str,
              "output_size_bytes": int,
              "video_opened": bool,
              "writer_opened": bool,
            }
        """
        video_path = Path(video_path)

        # --- 1. Copy the uploaded/source video into a temporary file before
        #     touching it with OpenCV. This isolates VideoCapture from the
        #     original upload path (permissions, in-place overwrite, odd
        #     characters, etc.) and guarantees we always read from a clean,
        #     private, on-disk copy. Cleaned up in the `finally` block. ---
        tmp_suffix = video_path.suffix if video_path.suffix else ".mp4"
        tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=tmp_suffix, prefix="upload_tmp_")
        os.close(tmp_fd)
        tmp_input_path = Path(tmp_path_str)
        shutil.copyfile(video_path, tmp_input_path)

        # --- 2. Open the (temp-file copy of the) video with OpenCV ---
        cap = cv2.VideoCapture(str(tmp_input_path))
        video_opened = cap.isOpened()
        logger.info(f"[VIDEO DEBUG] Video opened: {video_opened} (source={tmp_input_path})")
        if not video_opened:
            tmp_input_path.unlink(missing_ok=True)
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        logger.info(f"[VIDEO DEBUG] FPS: {fps:.2f}")
        logger.info(f"[VIDEO DEBUG] Frame count: {total_frames}")
        logger.info(f"[VIDEO DEBUG] Resolution: {width}x{height}")

        if output_video_path is None:
            output_video_path = PROCESSED_VIDEOS_DIR / "annotated_video.mp4"
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[VIDEO DEBUG] Output path (final, browser-safe): {output_video_path}")

        # --- 6. Write raw annotated frames to an intermediate 'mp4v' file
        #     first. mp4v is what OpenCV's VideoWriter can reliably encode
        #     on virtually any platform, but most browsers (and therefore
        #     Streamlit's st.video(), which renders an HTML5 <video> tag)
        #     cannot decode it. It is converted to real H.264 afterwards
        #     (step 10 below) - never shown to the user directly. ---
        raw_output_path = output_video_path.with_name(output_video_path.stem + "_raw_mp4v.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_output_path), fourcc, fps, (width, height))
        writer_opened = writer.isOpened()
        logger.info(f"[VIDEO DEBUG] VideoWriter status: {'opened' if writer_opened else 'FAILED TO OPEN'}")
        if not writer_opened:
            cap.release()
            tmp_input_path.unlink(missing_ok=True)
            raise IOError(f"Failed to open VideoWriter for: {raw_output_path}")

        frames_output_dir = Path(frames_output_dir)
        frames_output_dir.mkdir(parents=True, exist_ok=True)

        all_detections: List[Dict] = []
        saved_frame_paths: List[str] = []
        frame_number = 0
        start_time = time.time()
        last_annotated_frame = None

        # --- Event-driven email alerts: one tracker per process_video()
        #     call (fresh video = fresh set of tracked objects). Disabled
        #     entirely (zero overhead) unless the caller opts in. ---
        event_tracker = DamageEventTracker(cooldown_seconds=email_cooldown_seconds) if email_alerts_enabled else None
        email_events_log: List[Dict] = []
        email_events_lock = threading.Lock()
        if email_alerts_enabled and not email_receiver:
            logger.warning("[EMAIL EVENT] email_alerts_enabled=True but no email_receiver was given - alerts will be skipped.")

        logger.info(
            f"Processing video '{video_path.name}' -> {raw_output_path} "
            f"(fps={fps:.2f}, {width}x{height}, ~{total_frames} frames)"
        )

        try:
            # --- 3. Read every frame ---
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = frame_number / fps if fps else 0.0

                # --- 4. Inference on EVERY frame (same entry point used by
                #     detect_image() and the live webcam page) ---
                results, frame_detections = self.infer_frame(frame, conf=confidence, imgsz=image_size)

                # --- 5, 22. Ultralytics-rendered box/label/confidence +
                #     severity + frame/timestamp overlay - all via the one
                #     shared annotate_frame() function. THIS return value,
                #     not the raw `frame`, is what gets written below. ---
                annotated_frame = self.annotate_frame(
                    frame, results, frame_detections,
                    frame_number=frame_number, timestamp_sec=timestamp_sec,
                    draw_severity=True, draw_frame_info=True,
                )

                # --- 6. Write annotated frame at ORIGINAL fps/width/height ---
                if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                    annotated_frame = cv2.resize(annotated_frame, (width, height))
                writer.write(annotated_frame)
                last_annotated_frame = annotated_frame

                if frame_number < 3 or frame_number % 100 == 0:
                    logger.info(
                        f"[VIDEO DEBUG] frame={frame_number} detections={len(frame_detections)} "
                        f"classes={[d['class_name'] for d in frame_detections]}"
                    )

                # --- Collect standardized detections for this frame ---
                if frame_detections:
                    for det in frame_detections:
                        all_detections.append({
                            "frame_number": frame_number,
                            "timestamp_sec": round(timestamp_sec, 2),
                            **det,
                        })

                    # --- 12. Save every detected frame ---
                    if save_detected_frames:
                        frame_filename = f"{video_path.stem}_frame{frame_number:06d}.jpg"
                        frame_path = frames_output_dir / frame_filename
                        cv2.imwrite(str(frame_path), annotated_frame)
                        saved_frame_paths.append(str(frame_path))

                    # --- EVENT-DRIVEN EMAIL ALERTS: one email per unique
                    #     damage event (not per frame). Only detections
                    #     that are actually visible with a valid bounding
                    #     box (i.e. already in frame_detections, which only
                    #     contains confidence-passed, drawn detections) are
                    #     eligible - never a class with no box on screen. ---
                    if email_alerts_enabled and email_receiver and event_tracker:
                        for det in frame_detections:
                            if det["confidence"] < email_confidence_threshold:
                                continue
                            if not det.get("bbox"):
                                continue  # no valid box -> never email, per spec

                            should_send = event_tracker.register_detection(det["class_name"], det["bbox"])
                            if not should_send:
                                continue  # same object, still in cooldown

                            event_entry = {
                                "class_name": det["class_name"], "confidence": det["confidence"],
                                "frame_number": frame_number, "status": "queued", "error": None,
                            }
                            with email_events_lock:
                                email_events_log.append(event_entry)
                                del email_events_log[:-20]  # keep the log bounded

                            fire_damage_event_email_async(
                                det["class_name"], det["confidence"], annotated_frame.copy(), frame_number,
                                email_road_name, email_road_id, email_receiver,
                                frames_output_dir=frames_output_dir,
                                on_email_result=lambda r, _entry=event_entry: _entry.update(
                                    {"status": "sent" if r["success"] else "failed", "error": r.get("error")}
                                ),
                            )
                            logger.info(
                                f"[EMAIL EVENT] NEW damage event queued: class={det['class_name']} "
                                f"confidence={det['confidence']:.2f} frame={frame_number}"
                            )

                frame_number += 1

                # --- 20, 21. Rich progress callback for the UI progress bar +
                #     Current Frame / FPS / Detection Count / Elapsed Time /
                #     Remaining Time panel, PLUS a live bounding-box preview
                #     frame so the UI can show real-time annotated detection
                #     while the video is still being processed. ---
                if progress_callback:
                    elapsed = time.time() - start_time
                    processing_fps = frame_number / elapsed if elapsed > 0 else 0.0
                    remaining_frames = max(0, (total_frames - frame_number)) if total_frames else 0
                    eta_seconds = (remaining_frames / processing_fps) if processing_fps > 0 else 0.0

                    include_frame = (
                        live_preview
                        and live_preview_every_n_frames > 0
                        and frame_number % live_preview_every_n_frames == 0
                    )
                    with email_events_lock:
                        email_events_snapshot = list(email_events_log)

                    progress_callback({
                        "current_frame": frame_number,
                        "total_frames": total_frames,
                        "detections_count": len(all_detections),
                        "elapsed_seconds": round(elapsed, 1),
                        "eta_seconds": round(eta_seconds, 1),
                        "processing_fps": round(processing_fps, 2),
                        "last_frame_detections": frame_detections,
                        "timestamp_sec": round(timestamp_sec, 2),
                        # Raw BGR ndarray of the SAME Ultralytics-annotated
                        # frame just written to the output video - i.e. the
                        # live bounding-box/class/confidence overlay, not a
                        # separate re-render. None on throttled-out frames.
                        "live_preview_frame_bgr": annotated_frame if include_frame else None,
                        # Snapshot of every event-driven email fired so far
                        # this video, each: {class_name, confidence,
                        # frame_number, status: queued|sent|failed, error}.
                        # "queued" flips to "sent"/"failed" on a later
                        # callback once the background SMTP thread finishes.
                        "email_events": email_events_snapshot,
                    })

        finally:
            # --- 7. Release VideoCapture and VideoWriter correctly ---
            cap.release()
            writer.release()
            tmp_input_path.unlink(missing_ok=True)

        raw_size = os.path.getsize(raw_output_path) if raw_output_path.exists() else 0
        raw_codec = get_video_codec_fourcc(raw_output_path) if raw_size > 0 else "unknown"
        logger.info(f"[VIDEO DEBUG] Raw output size: {raw_size} bytes")
        logger.info(f"[VIDEO DEBUG] Raw output codec (OpenCV-reported): {raw_codec}")

        # --- 8. Verify the raw file before doing anything else with it ---
        raw_ok, raw_reason = verify_video_file(raw_output_path)
        if not raw_ok:
            logger.error(f"[VIDEO DEBUG] Raw annotated video failed verification: {raw_reason}")

        final_output_path = raw_output_path
        playable = False
        failure_reason = None
        conversion_method = None
        first_frame_preview_path = None

        # --- 10. Convert to a browser-safe H.264 MP4 (ffmpeg, else imageio) ---
        if raw_ok:
            try:
                converted_path, conversion_method = convert_to_h264(
                    raw_output_path, output_video_path, fps=fps,
                )
                ok2, reason2 = verify_video_file(converted_path)
                logger.info(
                    f"[VIDEO DEBUG] H.264 conversion method: {conversion_method}, "
                    f"verified: {ok2}, size: {os.path.getsize(converted_path)} bytes"
                )
                if ok2:
                    playable = True
                    final_output_path = Path(converted_path)
                    # Clean up the intermediate mp4v file - only the final
                    # H.264 file should remain in PROCESSED_VIDEOS_DIR.
                    if raw_output_path != final_output_path:
                        raw_output_path.unlink(missing_ok=True)
                else:
                    failure_reason = reason2
            except Exception as e:
                logger.error(f"[VIDEO DEBUG] H.264 conversion raised: {e}")
                failure_reason = str(e)
        else:
            failure_reason = raw_reason

        # --- 23. If it still cannot be played, save the first annotated
        #     frame so the UI can show *something* plus the exact reason ---
        if not playable:
            preview_frame = extract_first_frame(final_output_path)
            if preview_frame is None:
                preview_frame = last_annotated_frame
            if preview_frame is not None:
                first_frame_preview_path = str(
                    frames_output_dir / f"{video_path.stem}_first_frame_preview.jpg"
                )
                cv2.imwrite(first_frame_preview_path, preview_frame)
            if failure_reason is None:
                failure_reason = "Unknown failure: output video did not pass verification."

        logger.info(
            f"Processed video '{video_path.name}': {len(all_detections)} total detections across "
            f"{frame_number} frames, {len(saved_frame_paths)} damage frames saved -> "
            f"{final_output_path} (playable={playable}, method={conversion_method})"
        )

        return {
            "output_video_path": str(final_output_path),
            "saved_frame_paths": saved_frame_paths,
            "all_detections": all_detections,
            "total_frames_processed": frame_number,
            "video_fps": fps,
            "width": width,
            "height": height,
            # --- playback diagnostics (requirements 8, 10, 23, 24) ---
            "playable": playable,
            "failure_reason": failure_reason,
            "first_frame_preview_path": first_frame_preview_path,
            "conversion_method": conversion_method,
            "raw_codec": raw_codec,
            "output_size_bytes": os.path.getsize(final_output_path) if final_output_path.exists() else 0,
            "video_opened": video_opened,
            "writer_opened": writer_opened,
        }

    # ------------------------------------------------------------------
    # 3. REAL-TIME DETECTION (webcam / RTSP / live CCTV feed)
    # ------------------------------------------------------------------
    def run_realtime_detection(
        self,
        source: Union[int, str] = 0,
        display_window: bool = True,
    ) -> Generator[Dict, None, None]:
        """
        Runs continuous real-time detection on a live source (webcam
        index, RTSP URL, or CCTV stream URL), also using the Ultralytics
        renderer for consistency with the corrected video pipeline.

        Yields:
            {"frame": annotated_bgr_frame, "detections": [...], "timestamp": float}

        Set display_window=False when running inside Streamlit.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open real-time source: {source}")

        logger.info(f"Starting real-time detection on source={source}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Stream ended or frame grab failed.")
                    break

                results, detections = self.infer_frame(frame)
                annotated = self.annotate_frame(frame, results, detections, draw_severity=True, draw_frame_info=False)

                if display_window:
                    cv2.imshow("Real-Time Road Damage Detection (press 'q' to quit)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                yield {
                    "frame": annotated,
                    "detections": detections,
                    "timestamp": time.time(),
                }
        finally:
            cap.release()
            if display_window:
                cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# CLI entry point for quick local testing
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Road damage detection test runner")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--video", type=str, help="Path to a video file")
    parser.add_argument("--webcam", action="store_true", help="Run real-time webcam detection")
    args = parser.parse_args()

    service = DetectionService()

    if args.image:
        result = service.detect_image(args.image)
        print(f"Detections: {result['detections']}")
        print(f"Annotated image saved to: {result['annotated_path']}")

    elif args.video:
        def _print_progress(stats):
            print(
                f"\rFrame {stats['current_frame']}/{stats['total_frames']} | "
                f"Detections: {stats['detections_count']} | "
                f"FPS: {stats['processing_fps']} | ETA: {stats['eta_seconds']}s",
                end="",
            )

        result = service.process_video(args.video, progress_callback=_print_progress)
        print()
        print(f"Annotated video saved to: {result['output_video_path']}")
        print(f"Saved {len(result['saved_frame_paths'])} detected frames.")
        print(f"Total detections across video: {len(result['all_detections'])}")

    elif args.webcam:
        for _ in service.run_realtime_detection(source=0, display_window=True):
            pass

    else:
        parser.print_help()