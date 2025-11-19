
import time
import datetime
import numpy as np
import imutils
import cv2
from math import ceil
from scipy.spatial.distance import euclidean
from tracking import detect_human
from util import rect_distance, progress, kinetic_energy
from colors import RGB_COLORS
from config import (
    SHOW_DETECT, DATA_RECORD, RE_CHECK, RE_START_TIME, RE_END_TIME, SD_CHECK,
    SHOW_VIOLATION_COUNT, SHOW_TRACKING_ID, SOCIAL_DISTANCE,
    SHOW_PROCESSING_OUTPUT, YOLO_CONFIG, VIDEO_CONFIG, DATA_RECORD_RATE,
    ABNORMAL_CHECK, ABNORMAL_ENERGY, ABNORMAL_THRESH, ABNORMAL_MIN_PEOPLE
)
from whatsapp_alert import send_whatsapp_alert
import os
import csv
import traceback

# -------------------------------------------------------------------------
# Configuration / runtime defaults
# -------------------------------------------------------------------------
IS_CAM = VIDEO_CONFIG.get("IS_CAM", True)
HIGH_CAM = VIDEO_CONFIG.get("HIGH_CAM", False)

# alert cooldowns (seconds)
last_violation_alert = 0
last_abnormal_alert = 0
last_restricted_alert = 0
ALERT_COOLDOWN = 60  # seconds

# Debugging: set to True to print per-frame debug logs and write debug CSV.
# You can also add DEBUG_DETECTION = True in config.py and set it there.
DEBUG = getattr(__import__("config"), "DEBUG_DETECTION", True)

# Ensure processed_data exists
os.makedirs("processed_data", exist_ok=True)

# Setup debug_log CSV header if not exists
debug_log_path = "processed_data/debug_log.csv"
if not os.path.exists(debug_log_path):
    with open(debug_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame_count", "humans", "violations", "abnormal_count"])

# -------------------------------------------------------------------------
# Helper record functions
# -------------------------------------------------------------------------
def _record_movement_data(movement_data_writer, movement):
    """Write one track's movement record to CSV."""
    try:
        data = [movement.track_id, movement.entry, movement.exit] + list(np.array(movement.positions).flatten())
        movement_data_writer.writerow(data)
    except Exception as e:
        print("[_record_movement_data] Error:", e)


def _record_crowd_data(record_time, human_count, violate_count, restricted_entry, abnormal_activity, crowd_data_writer):
    """Write one frame's crowd stats to CSV."""
    try:
        data = [record_time, human_count, violate_count, int(restricted_entry), int(abnormal_activity)]
        crowd_data_writer.writerow(data)
    except Exception as e:
        print("[_record_crowd_data] Error:", e)


def _end_video(tracker, frame_count, movement_data_writer):
    """Record exit times for all confirmed tracks when video ends."""
    for t in tracker.tracks:
        if t.is_confirmed():
            try:
                t.exit = frame_count
                _record_movement_data(movement_data_writer, t)
            except Exception as e:
                print("[_end_video] Error:", e)


# -------------------------------------------------------------------------
# Main processing loop
# -------------------------------------------------------------------------
def video_process(cap, frame_size, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer):
    """
    Core per-frame pipeline:
      - YOLOv4-tiny + DeepSORT
      - Records movement/crowd data
      - Detects restricted entry, violations, and abnormal activity
      - Sends WhatsApp alerts with snapshot, with cooldown
      - Saves processed frames for dashboard sync
      - Writes debug log per frame
    """
    global last_violation_alert, last_abnormal_alert, last_restricted_alert

    # Compute FPS / time step handling
    if IS_CAM:
        VID_FPS = None
        DATA_RECORD_FRAME = 1
        TIME_STEP = 1.0
        t0 = time.time()
    else:
        # For video files we must read true fps
        VID_FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
        DATA_RECORD_FRAME = max(1, int(VID_FPS / max(1, DATA_RECORD_RATE)))
        TIME_STEP = DATA_RECORD_FRAME / float(VID_FPS)
        if DEBUG:
            print(f"[INFO] Video mode: VID_FPS={VID_FPS}, DATA_RECORD_FRAME={DATA_RECORD_FRAME}, TIME_STEP={TIME_STEP:.4f}")

    frame_count = 0
    display_frame_count = 0
    RE = False
    ABNORMAL = False

    # loop
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                # end of video or camera failure
                _end_video(tracker, frame_count, movement_data_writer)
                if not VID_FPS and 't0' in locals():
                    VID_FPS = frame_count / max(1, (time.time() - t0))
                break

            frame_count += 1

            # Only process every DATA_RECORD_FRAME frames
            if frame_count % DATA_RECORD_FRAME != 0:
                continue

            display_frame_count += 1

            # Resize to processing size
            frame = imutils.resize(frame, width=frame_size)

            current_datetime = datetime.datetime.now()
            record_time = current_datetime if IS_CAM else frame_count

            # ---------- detection & tracking ----------
            try:
                humans_detected, expired = detect_human(net, ln, frame, encoder, tracker, record_time)
            except Exception as e:
                print("[video_process] detect_human error:", e)
                traceback.print_exc()
                humans_detected, expired = [], []

            # write expired movement tracks
            for movement in expired:
                try:
                    _record_movement_data(movement_data_writer, movement)
                except Exception as e:
                    print("[video_process] error recording expired movement:", e)

            # restricted entry (time window)
            if RE_CHECK:
                RE = any(humans_detected) and (RE_START_TIME < current_datetime.time() < RE_END_TIME)
            else:
                RE = False

            # ---------- compute violations + abnormal individuals ----------
            violate_set = set()
            abnormal_individual = []

            # If many detections, compute pairwise rect distances (always compute for debug)
            for i, track in enumerate(humans_detected):
                try:
                    x1, y1, x2, y2 = map(int, track.to_tlbr().tolist())
                except Exception:
                    continue

                # compute centroid fallback
                if hasattr(track, "positions") and len(track.positions) > 0:
                    cx, cy = map(int, track.positions[-1])
                else:
                    cx, cy = ((x1 + x2) // 2, (y1 + y2) // 2)

                # pairwise distance checks
                for j, t2 in enumerate(humans_detected[i+1:], start=i+1):
                    try:
                        x21, y21, x22, y22 = map(int, t2.to_tlbr().tolist())
                        d = rect_distance((x1, y1, x2, y2), (x21, y21, x22, y22))
                        if d < SOCIAL_DISTANCE:
                            violate_set.update({i, j})
                    except Exception:
                        continue

                # abnormal energy
                if ABNORMAL_CHECK and hasattr(track, "positions") and len(track.positions) >= 2:
                    try:
                        ke = kinetic_energy(track.positions[-1], track.positions[-2], TIME_STEP)
                        if ke > ABNORMAL_ENERGY:
                            abnormal_individual.append(track.track_id)
                    except Exception:
                        pass

            # decide ABNORMAL overall
            if len(humans_detected) > ABNORMAL_MIN_PEOPLE:
                ABNORMAL = (len(abnormal_individual) / max(1.0, len(humans_detected))) > ABNORMAL_THRESH
            else:
                ABNORMAL = False

            # ---------- draw boxes if required ----------
            if SHOW_DETECT:
                for i, track in enumerate(humans_detected):
                    try:
                        x1, y1, x2, y2 = map(int, track.to_tlbr().tolist())
                    except Exception:
                        continue
                    color = RGB_COLORS.get("green", (0, 255, 0))
                    if i in violate_set:
                        color = RGB_COLORS.get("yellow", (0, 255, 255))
                    if track.track_id in abnormal_individual:
                        color = RGB_COLORS.get("blue", (255, 0, 0))
                    if RE:
                        color = RGB_COLORS.get("red", (0, 0, 255))

                    try:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        if SHOW_TRACKING_ID:
                            cv2.putText(frame, str(track.track_id), (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception:
                        pass

            # ---------- record CSV row ----------
            if DATA_RECORD:
                try:
                    _record_crowd_data(record_time, len(humans_detected), len(violate_set), RE, ABNORMAL, crowd_data_writer)
                except Exception as e:
                    print("[video_process] Error writing crowd data:", e)

            # ---------- debug logging to CSV ----------
            try:
                with open(debug_log_path, "a", newline="") as df:
                    writer = csv.writer(df)
                    writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frame_count,
                                     len(humans_detected), len(violate_set), len(abnormal_individual)])
            except Exception:
                pass

            # print debug
            if DEBUG:
                print(f"[DEBUG] Frame {frame_count}: humans={len(humans_detected)}, violations={len(violate_set)}, abnormal_count={len(abnormal_individual)}")

            # ---------- Alerts (with snapshot) ----------
            now = time.time()
            # Violation alert
            if len(violate_set) > 0 and (now - last_violation_alert > ALERT_COOLDOWN):
                try:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = f"processed_data/alert_snapshot_violation_{ts}.jpg"
                    cv2.imwrite(snapshot_path, frame)
                    send_whatsapp_alert(f"🚨 Crowd Alert: {len(violate_set)} Social Distance Violations Detected! Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
                    last_violation_alert = now
                    if DEBUG:
                        print(f"[ALERT] Violation alert sent. Snapshot: {snapshot_path}")
                except Exception as e:
                    print("[ALERT] Violation send error:", e)

            # Abnormal alert
            if ABNORMAL and (now - last_abnormal_alert > ALERT_COOLDOWN):
                try:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = f"processed_data/alert_snapshot_abnormal_{ts}.jpg"
                    cv2.imwrite(snapshot_path, frame)
                    send_whatsapp_alert("⚠️ Abnormal Activity Detected in Crowd Zone!")
                    last_abnormal_alert = now
                    if DEBUG:
                        print(f"[ALERT] Abnormal alert sent. Snapshot: {snapshot_path}")
                except Exception as e:
                    print("[ALERT] Abnormal send error:", e)

            # Restricted entry alert
            if RE and (now - last_restricted_alert > ALERT_COOLDOWN):
                try:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = f"processed_data/alert_snapshot_restricted_{ts}.jpg"
                    cv2.imwrite(snapshot_path, frame)
                    send_whatsapp_alert("🚫 Restricted Entry Detected! A person entered a restricted zone.")
                    last_restricted_alert = now
                    if DEBUG:
                        print(f"[ALERT] Restricted alert sent. Snapshot: {snapshot_path}")
                except Exception as e:
                    print("[ALERT] Restricted send error:", e)

            # ---------- Save latest processed frame for Flask frontend ----------
            try:
                cv2.imwrite("processed_data/latest_frame.jpg", frame)
            except Exception as e:
                print("[video_process] Frame save error:", e)

            # ---------- Show or CLI progress ----------
            if SHOW_PROCESSING_OUTPUT:
                cv2.imshow("Processed Output", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # user requested quit
                    _end_video(tracker, frame_count, movement_data_writer)
                    break
            else:
                progress(display_frame_count)

        except KeyboardInterrupt:
            print("[video_process] Interrupted by user.")
            break
        except Exception as e:
            print("[video_process] Unexpected error:", e)
            traceback.print_exc()
            # continue after short pause
            time.sleep(0.5)
            continue

    # cleanup
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    return VID_FPS
