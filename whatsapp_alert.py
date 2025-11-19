# whatsapp_alert.py
import requests
import time
import requests.utils

# ---- Configuration ----
# I assumed the number uses +91 country code. If different, update PHONE_NUMBER.
PHONE_NUMBER = "+916394804879"   # <-- replace if you need a different country code (e.g. +63...)
API_KEY = "gLJKSzryX7mt"         # your CallMeBot API key

# Cooldown (to avoid spamming)
LAST_ALERT_TIME = 0
ALERT_COOLDOWN = 60  # seconds between alerts


def send_whatsapp_alert(message: str):
    """Send a WhatsApp alert using CallMeBot API (best-effort, with cooldown)."""
    global LAST_ALERT_TIME

    if time.time() - LAST_ALERT_TIME < ALERT_COOLDOWN:
        # Skip sending if still in cooldown
        return

    try:
        text = requests.utils.quote(message)
        url = f"https://api.callmebot.com/whatsapp.php?phone={PHONE_NUMBER}&text={text}&apikey={API_KEY}"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            print(f"[ALERT] WhatsApp message sent: {message}")
            LAST_ALERT_TIME = time.time()
        else:
            print(f"[ERROR] WhatsApp alert failed (status {resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"[ERROR] WhatsApp alert exception: {e}")
