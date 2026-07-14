"""
email_service.py
-----------------
Gmail SMTP email alerts for the Smart Road & Flyover Damage Monitoring
System. Sends a professional HTML email (with the annotated detection
image/frame attached) whenever a detection session's severity is
"High" or "Critical".

SETUP (do this once)
---------------------
1. Enable 2-Step Verification on the sending Gmail account:
     https://myaccount.google.com/security
2. Generate an App Password (NOT your normal Gmail password):
     https://myaccount.google.com/apppasswords
   -> choose "Mail" + your device name -> copy the 16-character password.
3. Set two environment variables before launching Streamlit (never hard-
   code credentials in source control):

   Windows (PowerShell):
       setx GMAIL_ADDRESS "youraddress@gmail.com"
       setx GMAIL_APP_PASSWORD "abcd efgh ijkl mnop"
       (close and reopen the terminal so the new vars take effect)

   Linux / macOS (bash/zsh):
       export GMAIL_ADDRESS="youraddress@gmail.com"
       export GMAIL_APP_PASSWORD="abcd efgh ijkl mnop"

   Or place them in a `.env` file and load it at the top of app.py with
   `python-dotenv` (`pip install python-dotenv`; `load_dotenv()`).

WHERE THIS FILE GOES
--------------------
Save as `email_service.py` in the project root, alongside app.py,
detection_service.py, database.py, etc. (flat layout - no `utils/`
package in this project).
"""

import os
import re
import ssl
import smtplib
import logging
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("email_service")
logging.basicConfig(level=logging.INFO)

try:
    from config import APP_TITLE
except ImportError:
    APP_TITLE = "Smart Road & Flyover Damage Monitoring System"

# ---------------------------------------------------------------------
# SMTP CONFIGURATION (Gmail, STARTTLS, App Password auth)
# ---------------------------------------------------------------------
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_TIMEOUT_SECONDS = 20

SENDER_EMAIL = os.getenv("GMAIL_ADDRESS", "balamurugan.g056@gmail.com").strip()
SENDER_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "unrx kgcx qccu elma").strip()

# Only these two severities trigger an automatic email.
SEVERITY_ALERT_LEVELS = {"High", "Critical"}

_EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class EmailConfigError(Exception):
    """Raised for problems detected BEFORE we even try to contact Gmail
    (missing credentials, invalid recipient address, etc.)."""


class EmailSendError(Exception):
    """Raised when the SMTP conversation itself fails (auth rejected,
    no internet, Gmail unreachable, etc.), with a human-readable reason."""


# ---------------------------------------------------------------------
# VALIDATION / STATUS HELPERS
# ---------------------------------------------------------------------
def is_valid_email(address: Optional[str]) -> bool:
    """Basic but reliable email-shape check (not a full RFC 5322 parser -
    that's deliberate; we want to catch obvious typos, not be a mail-format
    pedant)."""
    return bool(address) and bool(_EMAIL_REGEX.match(address.strip()))


def is_smtp_configured() -> bool:
    """True if GMAIL_ADDRESS / GMAIL_APP_PASSWORD environment variables are set."""
    return bool(SENDER_EMAIL) and bool(SENDER_APP_PASSWORD)


def should_send_email(severity_class: Optional[str]) -> bool:
    """Only High/Critical severity should ever trigger an automatic email."""
    return (severity_class or "") in SEVERITY_ALERT_LEVELS


# ---------------------------------------------------------------------
# HTML EMAIL TEMPLATE
# ---------------------------------------------------------------------
def _severity_color(severity_class: str) -> str:
    return {
        "Critical": "#b00020",
        "High": "#e65100",
        "Medium": "#f9a825",
        "Low": "#2e7d32",
    }.get(severity_class, "#555555")


def build_html_email(detection_data: Dict) -> str:
    """
    Builds the HTML body: dark header banner with the app title, a red
    (or orange, for High) warning banner, and a clean summary table.

    `detection_data` may contain any of:
        damage_type, severity, confidence (0-1), risk_score, repair_cost,
        road_name, detected_at, location, frame_number
    All are optional except damage_type/severity - missing fields show "N/A".
    """
    severity = detection_data.get("severity", "Unknown")
    color = _severity_color(severity)

    confidence = detection_data.get("confidence")
    risk_score = detection_data.get("risk_score")
    repair_cost = detection_data.get("repair_cost")

    rows = [
        ("Damage Type", detection_data.get("damage_type", "Unknown")),
        ("Severity", severity),
        ("Confidence Score", f"{confidence * 100:.1f}%" if confidence is not None else "N/A"),
        ("Risk Score", f"{risk_score:.1f}" if risk_score is not None else "N/A"),
        ("Repair Cost", f"₹{repair_cost:,.0f}" if repair_cost is not None else "N/A"),
        ("Road / Flyover", detection_data.get("road_name") or "N/A"),
        ("Date & Time", detection_data.get("detected_at") or "N/A"),
        ("Location", detection_data.get("location") or "Not available"),
    ]
    if detection_data.get("frame_number") is not None:
        rows.append(("Frame Number", str(detection_data["frame_number"])))

    rows_html = "\n".join(
        f'<tr>'
        f'<td style="padding:8px 12px;border-bottom:1px solid #eee;font-weight:600;color:#333;width:40%;">{label}</td>'
        f'<td style="padding:8px 12px;border-bottom:1px solid #eee;color:#333;">{value}</td>'
        f'</tr>'
        for label, value in rows
    )

    return f"""\
<html>
  <body style="margin:0;padding:0;background:#f4f4f7;font-family:Arial,Helvetica,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f7;padding:24px 0;">
      <tr>
        <td align="center">
          <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.15);">
            <tr>
              <td style="background:#0d1b2a;padding:20px 24px;">
                <span style="color:#ffffff;font-size:18px;font-weight:bold;">🛣️ {APP_TITLE}</span>
              </td>
            </tr>
            <tr>
              <td style="background:{color};padding:14px 24px;">
                <span style="color:#ffffff;font-size:16px;font-weight:bold;">
                  ⚠️ {severity.upper()} SEVERITY DAMAGE ALERT
                </span>
              </td>
            </tr>
            <tr>
              <td style="padding:20px 24px;">
                <p style="color:#333;font-size:14px;margin:0 0 16px 0;">
                  An automated inspection has detected road/flyover damage
                  requiring attention. Details are summarized below.
                </p>
                <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:14px;">
                  {rows_html}
                </table>
                <p style="color:#777;font-size:12px;margin-top:20px;">
                  This is an automated alert from the {APP_TITLE}.
                  Please do not reply directly to this email.
                </p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


# ---------------------------------------------------------------------
# CORE SEND FUNCTION
# ---------------------------------------------------------------------
def send_email(
    receiver_email: str,
    subject: str,
    html_body: str,
    attachment_path: Optional[str] = None,
    plain_text_fallback: Optional[str] = None,
) -> None:
    """
    Sends one HTML email via Gmail SMTP (STARTTLS on port 587) using an
    App Password, optionally with a single file attached (the annotated
    detection image, or a saved video frame).

    Raises EmailConfigError for problems detected before we try to send
    (bad recipient address, missing credentials) and EmailSendError for
    anything that goes wrong during the actual SMTP conversation (wrong
    app password, no internet, Gmail unreachable). Callers should catch
    both and show a friendly message - see send_damage_alert() below for
    the recommended pattern.
    """
    if not is_valid_email(receiver_email):
        raise EmailConfigError(f"'{receiver_email}' is not a valid email address.")

    if not is_smtp_configured():
        raise EmailConfigError(
            "Gmail credentials are not configured. Set the GMAIL_ADDRESS and "
            "GMAIL_APP_PASSWORD environment variables (see the setup instructions "
            "at the top of email_service.py) and restart the app."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(plain_text_fallback or "This email requires an HTML-capable email client to view properly.")
    msg.add_alternative(html_body, subtype="html")

    if attachment_path:
        path = Path(attachment_path)
        if path.exists() and path.is_file():
            try:
                data = path.read_bytes()
                subtype = (path.suffix.lstrip(".").lower() or "octet-stream")
                maintype = "image" if subtype in ("jpg", "jpeg", "png", "bmp", "webp") else "application"
                msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=path.name)
                logger.info(f"Attached '{path.name}' to email ({len(data)} bytes)")
            except Exception as e:
                logger.error(f"Failed to read/attach '{attachment_path}': {e}")
        else:
            logger.warning(f"Attachment path does not exist, sending without it: {attachment_path}")

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.send_message(msg)
        logger.info(f"Email sent successfully to {receiver_email} (subject: {subject!r})")

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP authentication failed: {e}")
        raise EmailSendError(
            "Gmail rejected the login credentials. Make sure GMAIL_APP_PASSWORD is "
            "a 16-character App Password (NOT your normal Gmail account password) "
            "and that 2-Step Verification is enabled on the sending account."
        ) from e
    except (smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected, TimeoutError) as e:
        logger.error(f"SMTP connection failed: {e}")
        raise EmailSendError(
            f"Could not connect to {SMTP_HOST}:{SMTP_PORT}. Check the machine's "
            f"internet connection and that port 587 isn't blocked by a firewall."
        ) from e
    except OSError as e:
        # Covers socket.gaierror (DNS/no internet) and similar low-level network errors.
        logger.error(f"Network error while sending email: {e}")
        raise EmailSendError(f"Network error while sending email (check internet connectivity): {e}") from e
    except smtplib.SMTPException as e:
        logger.error(f"SMTP send failed: {e}")
        raise EmailSendError(f"Failed to send email: {e}") from e


# ---------------------------------------------------------------------
# HIGH-LEVEL: DAMAGE ALERT (what the detection pipeline actually calls)
# ---------------------------------------------------------------------
def send_damage_alert(
    detection_data: Dict,
    receiver_email: str,
    attachment_path: Optional[str] = None,
) -> Dict:
    """
    Builds the HTML report and sends it - returns a result dict instead
    of raising, so Streamlit call sites can do:

        result = send_damage_alert(data, receiver, image_path)
        if result["success"]:
            st.success("Email sent")
        else:
            st.error(result["error"])

    without every call site needing its own try/except.

    Returns: {"success": bool, "error": str | None, "subject": str}
    """
    severity = detection_data.get("severity", "Unknown")
    damage_type = detection_data.get("damage_type", "Damage")
    subject = f"🚨 {severity} Severity Road Damage Alert - {damage_type} Detected"
    html_body = build_html_email(detection_data)
    plain_text = (
        f"{severity} severity damage detected.\n"
        f"Type: {damage_type}\n"
        f"Confidence: {detection_data.get('confidence')}\n"
        f"Risk Score: {detection_data.get('risk_score')}\n"
        f"Repair Cost: {detection_data.get('repair_cost')}\n"
        f"Detected At: {detection_data.get('detected_at')}\n"
    )

    try:
        send_email(receiver_email, subject, html_body, attachment_path=attachment_path, plain_text_fallback=plain_text)
        return {"success": True, "error": None, "subject": subject}
    except (EmailConfigError, EmailSendError) as e:
        logger.error(f"send_damage_alert failed: {e}")
        return {"success": False, "error": str(e), "subject": subject}
    except Exception as e:
        # Catch-all so a truly unexpected error (e.g. a corrupt attachment)
        # never crashes the Streamlit page - it's logged and reported back.
        logger.exception(f"Unexpected error in send_damage_alert: {e}")
        return {"success": False, "error": f"Unexpected error: {e}", "subject": subject}


def send_test_email(receiver_email: str) -> Dict:
    """Used by the "Send Test Email" button in the Streamlit settings UI."""
    sample = {
        "damage_type": "pothole",
        "severity": "High",
        "confidence": 0.87,
        "risk_score": 62.5,
        "repair_cost": 18500.0,
        "road_name": "Test Road Segment",
        "detected_at": "This is a TEST email - no real damage was detected.",
        "location": "N/A (test)",
    }
    result = send_damage_alert(sample, receiver_email, attachment_path=None)
    result["subject"] = "✅ Test Email - " + result["subject"]
    return result


# =======================================================================
# EVENT-DRIVEN LIVE/VIDEO DAMAGE ALERTS
# =======================================================================
# Separate, fixed-subject template for the "one email per unique damage
# event" feature (live webcam + batch video processing) - as opposed to
# send_damage_alert() above, which is the full-pipeline High/Critical
# SESSION alert (dynamic subject, used after Decision Intelligence runs).
# Both share send_email()/is_valid_email()/is_smtp_configured() so there
# is still only one place that actually talks to Gmail.

EVENT_ALERT_SUBJECT = "🚨 Smart Road Monitoring Alert"


def build_event_html_email(event: Dict) -> str:
    """
    `event` keys (all optional except damage_class/severity):
        road_name, timestamp, damage_class, confidence, severity,
        frame_number, recommended_action, location, priority
    """
    severity = event.get("severity", "Unknown")
    color = _severity_color(severity)
    confidence = event.get("confidence")

    rows = [
        ("Road / Flyover Name", event.get("road_name") or "N/A"),
        ("Timestamp", event.get("timestamp") or "N/A"),
        ("Damage Type", event.get("damage_class", "Unknown")),
        ("Confidence", f"{confidence * 100:.1f}%" if confidence is not None else "N/A"),
        ("Severity", severity),
        ("Frame Number", str(event["frame_number"]) if event.get("frame_number") is not None else "N/A"),
        ("Recommended Action", event.get("recommended_action") or "N/A"),
        ("Location", event.get("location") or "Not available"),
        ("Inspection Priority", event.get("priority") or "N/A"),
    ]
    rows_html = "\n".join(
        f'<tr>'
        f'<td style="padding:8px 12px;border-bottom:1px solid #eee;font-weight:600;color:#333;width:45%;">{label}</td>'
        f'<td style="padding:8px 12px;border-bottom:1px solid #eee;color:#333;">{value}</td>'
        f'</tr>'
        for label, value in rows
    )

    return f"""\
<html>
  <body style="margin:0;padding:0;background:#f4f4f7;font-family:Arial,Helvetica,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f7;padding:24px 0;">
      <tr>
        <td align="center">
          <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.15);">
            <tr>
              <td style="background:#0d1b2a;padding:20px 24px;">
                <span style="color:#ffffff;font-size:18px;font-weight:bold;">🛣️ {APP_TITLE}</span>
              </td>
            </tr>
            <tr>
              <td style="background:{color};padding:14px 24px;">
                <span style="color:#ffffff;font-size:16px;font-weight:bold;">
                  ⚠️ NEW DAMAGE EVENT DETECTED - {severity.upper()} SEVERITY
                </span>
              </td>
            </tr>
            <tr>
              <td style="padding:20px 24px;">
                <p style="color:#333;font-size:14px;margin:0 0 16px 0;">
                  Live AI-based road/flyover inspection has detected a new,
                  distinct damage event. The annotated frame is attached below.
                </p>
                <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:14px;">
                  {rows_html}
                </table>
                <p style="color:#777;font-size:12px;margin-top:20px;">
                  This is an automated, event-driven alert from the {APP_TITLE}.
                  You will not receive another email for this same object while
                  it remains under active cooldown.
                </p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


def send_damage_event_email(event: Dict, receiver_email: str, attachment_path: Optional[str] = None) -> Dict:
    """
    Sends ONE event-driven alert (fixed subject "🚨 Smart Road Monitoring
    Alert") for a single, newly-tracked damage event. `attachment_path`
    MUST be the ANNOTATED frame (bounding box + label + confidence +
    timestamp already drawn on it) - never the raw frame; callers
    (detection_service.process_video / live_detection.YOLOVideoProcessor)
    are responsible for saving that annotated frame to disk before
    calling this function.

    Returns: {"success": bool, "error": str | None, "subject": str}
    """
    html_body = build_event_html_email(event)
    plain_text = (
        f"New damage event detected.\n"
        f"Road: {event.get('road_name')}\n"
        f"Damage: {event.get('damage_class')}\n"
        f"Confidence: {event.get('confidence')}\n"
        f"Severity: {event.get('severity')}\n"
        f"Frame: {event.get('frame_number')}\n"
        f"Action: {event.get('recommended_action')}\n"
        f"Timestamp: {event.get('timestamp')}\n"
    )
    try:
        send_email(
            receiver_email, EVENT_ALERT_SUBJECT, html_body,
            attachment_path=attachment_path, plain_text_fallback=plain_text,
        )
        return {"success": True, "error": None, "subject": EVENT_ALERT_SUBJECT}
    except (EmailConfigError, EmailSendError) as e:
        logger.error(f"send_damage_event_email failed: {e}")
        return {"success": False, "error": str(e), "subject": EVENT_ALERT_SUBJECT}
    except Exception as e:
        logger.exception(f"Unexpected error in send_damage_event_email: {e}")
        return {"success": False, "error": f"Unexpected error: {e}", "subject": EVENT_ALERT_SUBJECT}