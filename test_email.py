"""
test_email.py
-------------
Standalone script to verify your Gmail SMTP setup BEFORE relying on it
inside the Streamlit app. Run from the project root:

    python test_email.py you@example.com

Checks, in order:
  1. GMAIL_ADDRESS / GMAIL_APP_PASSWORD env vars are set.
  2. The recipient address you passed looks valid.
  3. Sends a real test email and reports success/failure with the exact
     reason if it fails (auth error, no internet, etc.).
"""

import sys

from email_service import (
    is_smtp_configured,
    is_valid_email,
    send_test_email,
    SENDER_EMAIL,
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python test_email.py <receiver_email>")
        return 1

    receiver_email = sys.argv[1]

    print("=" * 60)
    print("Smart Road & Flyover Damage Monitoring - Email Test")
    print("=" * 60)

    print(f"[1/3] Checking SMTP credentials...")
    if not is_smtp_configured():
        print("  ❌ FAILED: GMAIL_ADDRESS and/or GMAIL_APP_PASSWORD are not set.")
        print("     See the setup instructions at the top of email_service.py.")
        return 1
    print(f"  ✅ Sender configured: {SENDER_EMAIL}")

    print(f"[2/3] Validating receiver address '{receiver_email}'...")
    if not is_valid_email(receiver_email):
        print(f"  ❌ FAILED: '{receiver_email}' does not look like a valid email address.")
        return 1
    print("  ✅ Address looks valid.")

    print(f"[3/3] Sending test email to {receiver_email}...")
    result = send_test_email(receiver_email)
    if result["success"]:
        print(f"  ✅ SUCCESS: Test email sent (subject: {result['subject']!r}).")
        print("     Check the recipient's inbox (and spam folder).")
        return 0
    else:
        print(f"  ❌ FAILED: {result['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())