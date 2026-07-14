"""
notification_service.py
------------------------
Turns a recommendation (recommendation_service output) into alerts and
dispatches them through the right channel(s) for its priority level,
producing records shaped for the `alerts` table:

    alerts(alert_id, road_id, detection_id, severity, message, is_resolved, created_at)

CHANNEL ADAPTERS ARE STUBS BY DESIGN.
This module has no real SMTP/SMS/push credentials wired in (none were
provided in this codebase) - `_send_email`, `_send_sms`, and
`_send_push` log what WOULD be sent and return a simulated delivery
receipt. Swap their internals for real providers (e.g. smtplib +
Gmail/SES for email, Twilio for SMS, FCM/APNs for push) without
touching any other function in this file - every public function here
only depends on the {channel, status, timestamp} receipt shape.

This mirrors the project's existing "heuristic fallback, clearly
logged" pattern used in repair_cost_service/life_prediction_service:
never silently pretend a simulated action is real.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("notification_service")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# CHANNEL ROUTING BY PRIORITY LEVEL
# ---------------------------------------------------------------------
# Escalating channel mix - matches DUE_DAYS_BY_PRIORITY in
# recommendation_service: the more urgent the SLA, the more channels
# fire so a low-severity finding doesn't page anyone at 2am, while an
# urgent one reaches every configured contact method.
NOTIFICATION_CHANNELS_BY_PRIORITY: Dict[str, List[str]] = {
    "low":    ["dashboard"],
    "medium": ["dashboard", "email"],
    "high":   ["dashboard", "email", "sms"],
    "urgent": ["dashboard", "email", "sms", "push"],
}

# In-memory de-duplication so the same road+action isn't re-alerted
# every time this process re-evaluates the same session. A production
# deployment should back this with the `alerts` table instead
# (query WHERE road_id=? AND message=? AND is_resolved=0) - this
# in-memory set exists so the module is usable standalone/in tests.
_recent_alert_keys = set()


# ---------------------------------------------------------------------
# PER-CLASS ACTION TABLE (for event-driven live/video email alerts)
# ---------------------------------------------------------------------
# This is a SEPARATE, simpler lookup from NOTIFICATION_CHANNELS_BY_PRIORITY
# above / recommendation_service's rules engine: it maps a single YOLO
# class name DIRECTLY to a fixed action/priority/severity, exactly as
# specified for the live "one email per unique damage event" feature.
# Keys are the model's 4 supported classes (case/spacing-insensitive
# lookup via get_action_for_class() below - "pothole", "Pothole",
# "POTHOLE" all resolve the same way).
DAMAGE_CLASS_ACTION_TABLE: Dict[str, Dict[str, str]] = {
    "pothole": {
        "action": "Immediate patch repair",
        "priority": "High",
        "severity": "High",
    },
    "bridge_crack": {
        "action": "Structural inspection immediately",
        "priority": "Critical",
        "severity": "Critical",
    },
    "longitudinal_crack": {
        "action": "Seal crack and monitor",
        "priority": "Medium",
        "severity": "Medium",
    },
    "surface_damage": {
        "action": "Preventive maintenance",
        "priority": "Low",
        "severity": "Low",
    },
}

_DEFAULT_CLASS_ACTION = {"action": "General Inspection", "priority": "Medium", "severity": "Medium"}


def _normalize_class_name(class_name: str) -> str:
    """'Bridge Crack', 'bridge-crack', 'BRIDGE_CRACK' -> 'bridge_crack'."""
    return (class_name or "").strip().lower().replace(" ", "_").replace("-", "_")


def get_action_for_class(class_name: str) -> Dict[str, str]:
    """
    Returns {"action", "priority", "severity"} for one of the model's 4
    supported classes (pothole / bridge_crack / longitudinal_crack /
    surface_damage). Falls back to a Medium/General-Inspection default
    for any unrecognized class name rather than raising - a
    misclassified/unexpected label should never crash the live pipeline,
    just get a conservative default.
    """
    key = _normalize_class_name(class_name)
    if key not in DAMAGE_CLASS_ACTION_TABLE:
        logger.warning(f"get_action_for_class: '{class_name}' is not one of the 4 supported classes - using default.")
        return dict(_DEFAULT_CLASS_ACTION)
    return dict(DAMAGE_CLASS_ACTION_TABLE[key])


# ---------------------------------------------------------------------
# MESSAGE TEMPLATING
# ---------------------------------------------------------------------
def build_alert_message(recommendation: Dict, road_name: str) -> str:
    """
    Builds the human-readable alert text from a recommendation_service
    output. Kept as a single-line message to fit the `alerts.message`
    TEXT column and SMS length constraints.
    """
    priority = recommendation["priority_level"].upper()
    action = recommendation["recommended_action"]
    due = recommendation["due_date"]
    cost = recommendation["estimated_cost"]
    return (
        f"[{priority}] {road_name}: {action} recommended. "
        f"Due by {due}. Estimated cost: {cost:,.0f}."
    )


def _dedup_key(road_name: str, message: str) -> str:
    return f"{road_name}::{message}"


# ---------------------------------------------------------------------
# CHANNEL ADAPTERS (stubs - replace internals with real providers)
# ---------------------------------------------------------------------
def _send_email(recipient: str, subject: str, body: str) -> Dict:
    logger.info(f"[SIMULATED EMAIL] To: {recipient} | Subject: {subject}")
    return {"channel": "email", "recipient": recipient, "status": "simulated_sent"}


def _send_sms(recipient: str, body: str) -> Dict:
    logger.info(f"[SIMULATED SMS] To: {recipient} | Body: {body[:140]}")
    return {"channel": "sms", "recipient": recipient, "status": "simulated_sent"}


def _send_push(recipient: str, body: str) -> Dict:
    logger.info(f"[SIMULATED PUSH] To: {recipient} | Body: {body}")
    return {"channel": "push", "recipient": recipient, "status": "simulated_sent"}


def _send_dashboard(body: str) -> Dict:
    """Dashboard channel is always 'delivered' - it's just a row the UI reads on next load."""
    logger.info(f"[DASHBOARD ALERT QUEUED] {body}")
    return {"channel": "dashboard", "recipient": "dashboard", "status": "queued"}


_CHANNEL_ADAPTERS = {
    "email": _send_email,
    "sms": _send_sms,
    "push": _send_push,
    "dashboard": _send_dashboard,
}


# ---------------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------------
def dispatch_notification(
    recommendation: Dict,
    road_name: str,
    contacts: Optional[Dict[str, str]] = None,
    road_id: Optional[int] = None,
    detection_id: Optional[int] = None,
    force: bool = False,
) -> Dict:
    """
    Sends the recommendation across every channel appropriate for its
    priority_level, skipping if an identical alert was already sent
    this process lifetime (unless force=True).

    Args:
        recommendation: output of recommendation_service.generate_session_recommendation()
        road_name: display name for the message text
        contacts: {"email": "...", "sms": "+91...", "push": "device_token"}
                  missing channels are skipped with a warning (no crash).
        force: bypass de-duplication (e.g. manual re-notify from the UI)

    Returns:
        {
          "alert": {alert record shaped for the `alerts` table},
          "delivery_log": [ {channel, recipient, status}, ... ]
        }
    """
    contacts = contacts or {}
    priority_level = recommendation.get("priority_level", "low")
    message = build_alert_message(recommendation, road_name)
    dedup_key = _dedup_key(road_name, message)

    if not force and dedup_key in _recent_alert_keys:
        logger.info(f"Skipping duplicate alert for '{road_name}' (already sent this session).")
        return {"alert": None, "delivery_log": [], "skipped_duplicate": True}
    _recent_alert_keys.add(dedup_key)

    channels = NOTIFICATION_CHANNELS_BY_PRIORITY.get(priority_level, ["dashboard"])
    subject = f"Maintenance Alert - {road_name} [{priority_level.upper()}]"

    delivery_log = []
    for channel in channels:
        adapter = _CHANNEL_ADAPTERS[channel]
        try:
            if channel == "email":
                recipient = contacts.get("email")
                if not recipient:
                    logger.warning(f"No email contact configured; skipping email channel for {road_name}.")
                    continue
                receipt = adapter(recipient, subject, message)
            elif channel == "dashboard":
                receipt = adapter(message)
            else:
                recipient = contacts.get(channel)
                if not recipient:
                    logger.warning(f"No {channel} contact configured; skipping {channel} channel for {road_name}.")
                    continue
                receipt = adapter(recipient, message)
            receipt["timestamp"] = datetime.now(timezone.utc).isoformat()
            delivery_log.append(receipt)
        except Exception as e:
            logger.error(f"Notification channel '{channel}' failed: {e}")
            delivery_log.append({"channel": channel, "status": "failed", "error": str(e)})

    alert_record = {
        "road_id": road_id,
        "detection_id": detection_id,
        "severity": priority_level,
        "message": message,
        "is_resolved": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return {"alert": alert_record, "delivery_log": delivery_log, "skipped_duplicate": False}


def notify_from_recommendation(
    recommendation: Dict,
    road_name: str,
    contacts: Optional[Dict[str, str]] = None,
    road_id: Optional[int] = None,
) -> Dict:
    """Convenience alias - the typical entry point called right after
    recommendation_service.generate_session_recommendation()."""
    return dispatch_notification(recommendation, road_name, contacts, road_id)


def resolve_alert(alert_record: Dict) -> Dict:
    """Marks an alert record resolved (e.g. once a repair crew closes the job)."""
    alert_record = dict(alert_record)
    alert_record["is_resolved"] = True
    alert_record["resolved_at"] = datetime.now(timezone.utc).isoformat()
    return alert_record


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_recommendation = {
        "priority_level": "urgent",
        "recommended_action": "Immediate Repair",
        "due_date": "2026-07-12",
        "estimated_cost": 42500.0,
    }
    result = dispatch_notification(
        sample_recommendation,
        road_name="MG Road Flyover - Segment 4",
        contacts={"email": "publicworks@city.gov", "sms": "+911234567890", "push": "device-token-abc"},
        road_id=2,
    )
    print("Alert record:", result["alert"])
    print("Delivery log:")
    for entry in result["delivery_log"]:
        print("  ", entry)

    # Second call with the same inputs is de-duplicated
    repeat = dispatch_notification(
        sample_recommendation, road_name="MG Road Flyover - Segment 4",
        contacts={"email": "publicworks@city.gov"}, road_id=2,
    )
    print("\nRepeat call skipped as duplicate:", repeat["skipped_duplicate"])