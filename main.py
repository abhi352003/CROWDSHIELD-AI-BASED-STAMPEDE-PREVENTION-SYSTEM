


"""
main.py — controller for Crowd-Analysis backend

Behavior:
- By default: runs only the video processing (YOLO + DeepSORT) which writes CSV/JSON to processed_data/.
- Does NOT launch movement/crowd/abnormal visualization threads (they open GUI windows).
- If you explicitly run with --enable-analysis, the three analysis scripts will be started in parallel
  (useful for local debugging). Use with care (they may open Matplotlib/OpenCV windows).

Usage:
    python main.py                # safe: backend only, no GUI windows
    python main.py --enable-analysis
"""

import threading
import datetime
import time
import argparse
import numpy as np
import imutils
import cv2
import os
import csv
import json

# Import local modules
from config import YOLO_CONFIG, VIDEO_CONFIG, SHOW_PROCESSING_OUTPUT, DATA_RECORD_RATE, FRAME_SIZE, TRACK_MAX_AGE
from video_process import video_process
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

# ------------------ ARGPARSE ------------------
parser = argparse.ArgumentParser(description="Crowd-Analysis main runner")
parser.add_argument("--enable-analysis", action="store_true",
                    help="Start movement/crowd/abnormal analysis threads (may open GUI windows).")
args = parser.parse_args()
ENABLE_ANALYSIS = args.enable_analysis

# ------------------ BASIC VALIDATION ------------------
if FRAME_SIZE > 1920:
    print("Frame size is too large!")
    quit()
elif FRAME_SIZE < 480:
    print("Frame size is too small! You won't see anything")
    quit()

# ------------------ YOLO + DEEPSORT SETUP ------------------
IS_CAM = VIDEO_CONFIG["IS_CAM"]
cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])

# YOLO paths
WEIGHTS_PATH = YOLO_CONFIG["WEIGHTS_PATH"]
CONFIG_PATH = YOLO_CONFIG["CONFIG_PATH"]

# Load YOLOv3-tiny network
net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Tracker parameters
max_cosine_distance = 0.7
nn_budget = None

# Initialize DeepSort tracker
if IS_CAM:
    max_age = VIDEO_CONFIG["CAM_APPROX_FPS"] * TRACK_MAX_AGE
else:
    max_age = DATA_RECORD_RATE * TRACK_MAX_AGE
    if max_age > 30:
        max_age = 30

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, max_age=max_age)

# Ensure output folder exists
os.makedirs('processed_data', exist_ok=True)

# ------------------ CREATE OUTPUT FILES ------------------
# Open files in append mode now to avoid overwriting existing logs during development.
movement_data_file = open('processed_data/movement_data.csv', 'a', newline='')
crowd_data_file = open('processed_data/crowd_data.csv', 'a', newline='')

movement_data_writer = csv.writer(movement_data_file)
crowd_data_writer = csv.writer(crowd_data_file)

# Write headers if files are new/empty
if os.path.getsize('processed_data/movement_data.csv') == 0:
    movement_data_writer.writerow(['Track ID', 'Entry time', 'Exit Time', 'Movement Tracks'])
    movement_data_file.flush()
if os.path.getsize('processed_data/crowd_data.csv') == 0:
    crowd_data_writer.writerow(['Time', 'Human Count', 'Social Distance violate', 'Restricted Entry', 'Abnormal Activity'])
    crowd_data_file.flush()

# ------------------ FUNCTION DEFINITIONS ------------------
def run_video_process():
    """Main video processing function: runs the detection/tracking pipeline."""
    global VID_FPS, DATA_RECORD_FRAME, END_TIME
    START_TIME = time.time()
    try:
        processing_FPS = video_process(cap, FRAME_SIZE, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer)
    except Exception as e:
        print("[main.run_video_process] Error in video_process:", e)
        processing_FPS = None
    finally:
        # Ensure windows are closed if any left open by video_process
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # flush and close handled in outer scope after join

    END_TIME = time.time()
    PROCESS_TIME = END_TIME - START_TIME
    print(f"\n[INFO] Time elapsed: {round(PROCESS_TIME, 2)} seconds")

    if IS_CAM:
        print(f"[INFO] Processed FPS: {processing_FPS}")
        VID_FPS = processing_FPS
        DATA_RECORD_FRAME = 1
    else:
        try:
            print(f"[INFO] Processed FPS: {round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / PROCESS_TIME, 2)}")
            VID_FPS = cap.get(cv2.CAP_PROP_FPS)
            DATA_RECORD_FRAME = int(VID_FPS / DATA_RECORD_RATE)
            START_TIME_DT = VIDEO_CONFIG["START_TIME"]
            time_elapsed = round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / VID_FPS)
            END_TIME_DT = START_TIME_DT + datetime.timedelta(seconds=time_elapsed)

            video_data = {
                "IS_CAM": IS_CAM,
                "DATA_RECORD_FRAME": DATA_RECORD_FRAME,
                "VID_FPS": VID_FPS,
                "PROCESSED_FRAME_SIZE": FRAME_SIZE,
                "TRACK_MAX_AGE": TRACK_MAX_AGE,
                "START_TIME": START_TIME_DT.strftime("%d/%m/%Y, %H:%M:%S"),
                "END_TIME": END_TIME_DT.strftime("%d/%m/%Y, %H:%M:%S")
            }

            with open('processed_data/video_data.json', 'w') as video_data_file:
                json.dump(video_data, video_data_file)
        except Exception as e:
            print("[main.run_video_process] Error computing video metadata:", e)

    try:
        cap.release()
    except Exception:
        pass

def run_movement_process():
    """Run movement data analysis in parallel (legacy - may open GUI)."""
    try:
        import movement_data_present
        if hasattr(movement_data_present, 'main'):
            movement_data_present.main()
        else:
            exec(open("movement_data_present.py").read(), {})
    except Exception as e:
        print("[main.run_movement_process] Error starting movement_data_present:", e)

def run_crowd_process():
    """Run crowd data analysis in parallel (legacy - may open GUI)."""
    try:
        import crowd_data_present
        if hasattr(crowd_data_present, 'main'):
            crowd_data_present.main()
        else:
            exec(open("crowd_data_present.py").read(), {})
    except Exception as e:
        print("[main.run_crowd_process] Error starting crowd_data_present:", e)

def run_abnormal_process():
    """Run abnormal energy analysis in parallel (legacy - may open GUI)."""
    try:
        import abnormal_data_process
        if hasattr(abnormal_data_process, 'main'):
            abnormal_data_process.main()
        else:
            exec(open("abnormal_data_process.py").read(), {})
    except Exception as e:
        print("[main.run_abnormal_process] Error starting abnormal_data_process:", e)

# ------------------ THREAD SETUP ------------------
video_thread = threading.Thread(target=run_video_process, name="VideoProcessThread", daemon=False)

analysis_threads = []
if ENABLE_ANALYSIS:
    # Only create analysis threads if explicitly requested
    analysis_threads = [
        threading.Thread(target=run_movement_process, name="MovementThread", daemon=True),
        threading.Thread(target=run_crowd_process, name="CrowdThread", daemon=True),
        threading.Thread(target=run_abnormal_process, name="AbnormalThread", daemon=True),
    ]

# ------------------ START ------------------
print("[main] Starting video processing thread...")
video_thread.start()

if ENABLE_ANALYSIS:
    print("[main] --enable-analysis provided: starting analysis threads (movement/crowd/abnormal).")
    for t in analysis_threads:
        t.start()
else:
    print("[main] Analysis threads not started. Backend will only process frames and write processed_data/*")
    print("[main] To enable legacy analysis GUIs (not recommended for production), re-run with --enable-analysis")

# Wait for video processing to complete
video_thread.join()

# If analysis threads were started as daemons, they will exit when main thread exits.
# If you used non-daemon threads, you might want to join them here (not required by default).
for t in analysis_threads:
    if t.is_alive() and not t.daemon:
        t.join(timeout=1)

# Ensure files are flushed and closed cleanly
try:
    movement_data_file.flush()
except Exception:
    pass
try:
    crowd_data_file.flush()
except Exception:
    pass

try:
    movement_data_file.close()
except Exception:
    pass
try:
    crowd_data_file.close()
except Exception:
    pass

print("\n[INFO] Video processing finished. CSVs/JSON are saved to processed_data/.")
print("[INFO] Backend exiting.")
