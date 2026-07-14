"""
video_utils.py
--------------
Utility functions for handling road/flyover damage VIDEO input:
  - Upload validation & saving
  - Reading video properties (fps, resolution, duration)
  - Frame-by-frame generator (with configurable sampling)
  - Video writer creation for saving annotated output video
  - Saving individual detected frames to disk

Kept framework-agnostic (no Streamlit/YOLO imports) so it can be
unit-tested and reused outside the app.
"""

import os
import cv2
import uuid
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Generator, Optional, Tuple, Union

logger = logging.getLogger("video_utils")
logging.basicConfig(level=logging.INFO)

try:
    from config import (
        RAW_VIDEOS_DIR,
        PROCESSED_VIDEOS_DIR,
        FRAMES_DIR,
        FRAME_SAMPLE_RATE_FPS,
        MAX_VIDEO_DURATION_SECONDS,
        VIDEO_RESIZE_WIDTH,
    )
except ImportError:
    RAW_VIDEOS_DIR = Path("data/raw/videos")
    PROCESSED_VIDEOS_DIR = Path("data/processed/annotated_videos")
    FRAMES_DIR = Path("data/processed/frames")
    FRAME_SAMPLE_RATE_FPS = 2
    MAX_VIDEO_DURATION_SECONDS = 600
    VIDEO_RESIZE_WIDTH = 1280
    for d in [RAW_VIDEOS_DIR, PROCESSED_VIDEOS_DIR, FRAMES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------
def validate_video_file(filename: str) -> bool:
    """Checks whether a filename has a supported video extension."""
    ext = Path(filename).suffix.lower()
    is_valid = ext in VALID_VIDEO_EXTENSIONS
    if not is_valid:
        logger.warning(f"Rejected file '{filename}': unsupported extension '{ext}'")
    return is_valid


# ---------------------------------------------------------------------
# UPLOAD HANDLING
# ---------------------------------------------------------------------
def save_uploaded_video(uploaded_file, destination_dir: Union[str, Path] = RAW_VIDEOS_DIR) -> str:
    """
    Persists an uploaded video to disk with a collision-safe unique filename.

    Accepts a Streamlit `UploadedFile` object or an existing file path (str/Path).
    Returns the absolute path (str) to the saved file.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]

    if hasattr(uploaded_file, "name") and hasattr(uploaded_file, "getbuffer"):
        if not validate_video_file(uploaded_file.name):
            raise ValueError(f"Unsupported video type: {uploaded_file.name}")
        ext = Path(uploaded_file.name).suffix.lower()
        out_path = destination_dir / f"vid_{timestamp}_{unique_id}{ext}"
        with open(out_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    elif isinstance(uploaded_file, (str, Path)):
        src_path = Path(uploaded_file)
        if not src_path.exists():
            raise FileNotFoundError(f"Source video not found: {src_path}")
        if not validate_video_file(src_path.name):
            raise ValueError(f"Unsupported video type: {src_path.name}")
        ext = src_path.suffix.lower()
        out_path = destination_dir / f"vid_{timestamp}_{unique_id}{ext}"
        out_path.write_bytes(src_path.read_bytes())

    else:
        raise TypeError(f"Unsupported upload type: {type(uploaded_file)}")

    logger.info(f"Saved uploaded video -> {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------
# VIDEO PROPERTIES
# ---------------------------------------------------------------------
def get_video_properties(video_path: Union[str, Path]) -> dict:
    """
    Returns basic metadata about a video file:
        fps, width, height, frame_count, duration_seconds
    Raises if the file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # some codecs report 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0
    cap.release()

    if duration > MAX_VIDEO_DURATION_SECONDS:
        logger.warning(
            f"Video duration ({duration:.1f}s) exceeds configured cap "
            f"({MAX_VIDEO_DURATION_SECONDS}s). Consider trimming input."
        )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


# ---------------------------------------------------------------------
# FRAME EXTRACTION (generator)
# ---------------------------------------------------------------------
def frame_generator(
    video_path: Union[str, Path],
    resize_width: int = VIDEO_RESIZE_WIDTH,
) -> Generator[Tuple[int, float, "np.ndarray"], None, None]:
    """
    Yields every frame of a video, one at a time, as:
        (frame_number, timestamp_seconds, frame_bgr)

    This is a full-frame generator (used by the video writer so output
    video plays at normal speed). Detection sampling is handled by the
    caller (detection_service) using FRAME_SAMPLE_RATE_FPS, so heavy
    inference isn't run on every single frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if resize_width and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(
                    frame, (resize_width, int(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            timestamp = frame_number / fps
            yield frame_number, timestamp, frame
            frame_number += 1
    finally:
        cap.release()


def should_run_detection(frame_number: int, video_fps: float,
                          sample_rate_fps: float = FRAME_SAMPLE_RATE_FPS) -> bool:
    """
    Decides whether inference should run on this frame number, based on
    the desired detection sampling rate vs. the video's native fps.
    E.g. a 30fps video sampled at 2fps -> detection runs every 15th frame.
    """
    if sample_rate_fps <= 0 or video_fps <= 0:
        return True
    interval = max(1, round(video_fps / sample_rate_fps))
    return frame_number % interval == 0


# ---------------------------------------------------------------------
# VIDEO WRITER
# ---------------------------------------------------------------------
def create_video_writer(
    output_path: Union[str, Path],
    fps: float,
    frame_width: int,
    frame_height: int,
) -> cv2.VideoWriter:
    """
    Creates and returns an OpenCV VideoWriter for saving the annotated
    output video (mp4, H.264-compatible fourcc via 'mp4v').
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise IOError(f"Failed to open VideoWriter for: {output_path}")
    return writer


def get_video_codec_fourcc(video_path: Union[str, Path]) -> str:
    """
    Returns the 4-character codec FOURCC string OpenCV reports for a video
    file (e.g. 'mp4v', 'avc1', 'h264'). Returns 'unknown' if it cannot be
    determined. Used purely for debug logging (requirement #24).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return "unknown"
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    if fourcc_int == 0:
        return "unknown"
    try:
        return "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]).strip()
    except Exception:
        return "unknown"


def verify_video_file(video_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Sanity-checks a written video file before it is ever handed to
    st.video(). Checks, in order:
        1. The file exists on disk.
        2. The file is non-empty (os.path.getsize > 0).
        3. OpenCV can open the container.
        4. OpenCV can decode at least one frame from it.

    Returns (True, "OK") if all checks pass, otherwise
    (False, "<exact human-readable reason>") so the UI can display the
    EXACT reason a video failed (requirement #23).
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return False, f"Output file does not exist: {video_path}"

    size = os.path.getsize(video_path)
    if size <= 0:
        return False, f"Output file is empty (0 bytes): {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False, f"OpenCV could not open the video container (unsupported codec/corrupt file): {video_path}"

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False, f"OpenCV opened the file but could not decode any frame from it: {video_path}"

    return True, "OK"


def ffmpeg_available() -> bool:
    """True if a system `ffmpeg` binary is on PATH."""
    return shutil.which("ffmpeg") is not None


def convert_to_h264(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    fps: Optional[float] = None,
    timeout_seconds: int = 1800,
) -> Tuple[str, str]:
    """
    Converts an arbitrary OpenCV-written video (commonly 'mp4v' fourcc,
    which most browsers - and therefore Streamlit's st.video(), which
    renders an HTML5 <video> tag - cannot decode) into a browser-safe
    H.264 / yuv420p MP4.

    Tries, in order (requirement #10):
        1. System `ffmpeg` via subprocess (fastest, best quality/size).
        2. `imageio` + `imageio-ffmpeg` (pure-Python fallback that ships
           its own static ffmpeg binary - no system install required).

    Returns (output_path, method) where method is "ffmpeg" or "imageio".
    Raises RuntimeError with a clear, exact reason if BOTH attempts fail.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Attempt 1: system ffmpeg ----
    if ffmpeg_available():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "fast",
            "-crf", "23",
            str(output_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
            if proc.returncode == 0 and output_path.exists() and os.path.getsize(output_path) > 0:
                logger.info(f"H.264 conversion via system ffmpeg succeeded -> {output_path}")
                return str(output_path), "ffmpeg"
            logger.warning(
                f"System ffmpeg conversion failed (returncode={proc.returncode}). "
                f"stderr(tail): {proc.stderr[-1000:] if proc.stderr else '<none>'}"
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"System ffmpeg conversion timed out after {timeout_seconds}s.")
        except Exception as e:
            logger.warning(f"System ffmpeg conversion raised an exception: {e}")
    else:
        logger.warning("System 'ffmpeg' binary not found on PATH; will try imageio-ffmpeg fallback.")

    # ---- Attempt 2: imageio + imageio-ffmpeg (pure-Python fallback) ----
    try:
        import imageio.v2 as imageio  # noqa: F401  (imported lazily; optional dependency)

        reader = imageio.get_reader(str(input_path), "ffmpeg")
        meta = reader.get_meta_data()
        out_fps = fps or meta.get("fps") or 25.0

        writer = imageio.get_writer(
            str(output_path),
            fps=out_fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        frame_count = 0
        for frame_rgb in reader:
            writer.append_data(frame_rgb)
            frame_count += 1
        writer.close()
        reader.close()

        if output_path.exists() and os.path.getsize(output_path) > 0 and frame_count > 0:
            logger.info(
                f"H.264 conversion via imageio-ffmpeg succeeded ({frame_count} frames) -> {output_path}"
            )
            return str(output_path), "imageio"
        raise RuntimeError("imageio-ffmpeg produced an empty or zero-frame output file.")

    except Exception as e:
        logger.error(f"imageio-ffmpeg conversion fallback also failed: {e}")
        raise RuntimeError(
            "H.264 conversion failed via both system ffmpeg and imageio-ffmpeg. "
            "Install ffmpeg (`apt-get install -y ffmpeg`) or the Python fallback "
            f"(`pip install imageio[ffmpeg]`). Underlying error: {e}"
        )


def extract_first_frame(video_path: Union[str, Path]):
    """
    Grabs the first decodable frame from a video file (BGR ndarray) for
    UI fallback display when the annotated video itself cannot be played
    (requirement #23). Returns None if no frame could be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def build_output_video_path(original_video_name: str,
                             output_dir: Union[str, Path] = PROCESSED_VIDEOS_DIR) -> str:
    """Builds a unique output path for the annotated version of an uploaded video."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(original_video_name).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(output_dir / f"{stem}_annotated_{timestamp}.mp4")


# ---------------------------------------------------------------------
# SAVE DETECTED FRAMES
# ---------------------------------------------------------------------
def save_detected_frame(
    frame: "np.ndarray",
    frame_number: int,
    source_video_name: str,
    output_dir: Union[str, Path] = FRAMES_DIR,
) -> str:
    """
    Saves a single frame (already annotated) that contained a detection,
    for later review/reporting. Returns the saved file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(source_video_name).stem
    out_path = output_dir / f"{stem}_frame{frame_number:06d}.jpg"
    cv2.imwrite(str(out_path), frame)
    return str(out_path)