

import datetime
import os

# ----------------------------
# 🎥 VIDEO CONFIGURATION
# ----------------------------
VIDEO_CONFIG = {
    # Change this to your video path
    "VIDEO_CAP":0,               # or full path: "videos/test.mp4"
    "IS_CAM": True,                       # False = video file mode
    "CAM_APPROX_FPS": 5,                   # used only if IS_CAM=True
    "HIGH_CAM": False,                     # top-down camera = True
    "START_TIME": datetime.datetime.now()  # current start timestamp
}

# ----------------------------
# 🤖 YOLO CONFIGURATION
# ----------------------------
YOLO_CONFIG = {
    "WEIGHTS_PATH": os.path.join("YOLOv4-tiny", "yolov4-tiny.weights"),
    "CONFIG_PATH": os.path.join("YOLOv4-tiny", "yolov4-tiny.cfg")
}

# ----------------------------
# 🧩 GENERAL SETTINGS
# ----------------------------
SHOW_PROCESSING_OUTPUT = True
SHOW_DETECT = True
DATA_RECORD = True
DATA_RECORD_RATE = 5
FRAME_SIZE = 1080
TRACK_MAX_AGE = 3

# ----------------------------
# 🚧 RESTRICTED ENTRY SETTINGS
# ----------------------------
RE_CHECK = False
RE_START_TIME = datetime.time(0, 0, 0)
RE_END_TIME = datetime.time(23, 0, 0)

# ----------------------------
# 📏 SOCIAL DISTANCE SETTINGS
# ----------------------------
SD_CHECK = True
SOCIAL_DISTANCE = 10  # higher for 1080p, lower if resized smaller
SHOW_VIOLATION_COUNT = True
SHOW_TRACKING_ID = True

# ----------------------------
# ⚠️ ABNORMAL CROWD DETECTION
# ----------------------------
ABNORMAL_CHECK = True
ABNORMAL_MIN_PEOPLE = 5
ABNORMAL_ENERGY = 1866
ABNORMAL_THRESH = 0.5   # slightly lowered for realistic detection

# ----------------------------
# 🧠 YOLO + TRACKER PARAMETERS
# ----------------------------
MIN_CONF = 0.3
NMS_THRESH = 0.2

# ----------------------------
# 💾 PATHS & VALIDATIONS
# ----------------------------
os.makedirs("processed_data", exist_ok=True)
os.makedirs("YOLOv4-tiny", exist_ok=True)

# Initialize CSVs if missing
if not os.path.exists("processed_data/crowd_data.csv"):
    with open("processed_data/crowd_data.csv", "w") as f:
        f.write("Time,Human Count,Social Distance violate,Restricted Entry,Abnormal Activity\n")

if not os.path.exists("processed_data/movement_data.csv"):
    with open("processed_data/movement_data.csv", "w") as f:
        f.write("Track ID,Entry time,Exit Time,Movement Tracks\n")

# Verify YOLO files exist
if not (os.path.exists(YOLO_CONFIG["WEIGHTS_PATH"]) and os.path.exists(YOLO_CONFIG["CONFIG_PATH"])):
    print("[⚠️ WARNING] YOLOv4-tiny weights/config not found in 'YOLOv4-tiny' folder!")
    print("Expected:")
    print(f"  {YOLO_CONFIG['WEIGHTS_PATH']}")
    print(f"  {YOLO_CONFIG['CONFIG_PATH']}")
