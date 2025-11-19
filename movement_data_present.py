


import csv
import imutils
import cv2
import json
import math
import numpy as np
import os
import time
from config import VIDEO_CONFIG
from scipy.spatial.distance import euclidean
from colors import RGB_COLORS, gradient_color_RGB

OUTPUT_IMAGE = "processed_data/heatmap.png"


def generate_heatmap():
    """Reads latest movement_data.csv and generates a heatmap PNG (no GUI)."""
    if not os.path.exists('processed_data/movement_data.csv') or not os.path.exists('processed_data/video_data.json'):
        print("[movement_data_present] Waiting for processed data...")
        return None

    try:
        with open('processed_data/video_data.json', 'r') as file:
            data = json.load(file)
            vid_fps = data["VID_FPS"]
            data_record_frame = data["DATA_RECORD_FRAME"]
    except Exception as e:
        print("[movement_data_present] Error reading video_data.json:", e)
        return None

    # Load track data
    tracks = []
    try:
        with open('processed_data/movement_data.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader, None)
            for row in reader:
                if len(row[3:]) > 4:
                    temp = []
                    data_points = row[3:]
                    for i in range(0, len(data_points), 2):
                        try:
                            temp.append([int(data_points[i]), int(data_points[i + 1])])
                        except:
                            continue
                    if len(temp) > 1:
                        tracks.append(temp)
    except Exception as e:
        print("[movement_data_present] Error reading movement_data.csv:", e)
        return None

    if len(tracks) == 0:
        print("[movement_data_present] No movement data yet...")
        return None

    # Initialize camera feed for frame reference
    cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])
    ret, frame = cap.read()
    if not ret:
        print("[movement_data_present] Could not read frame from video source.")
        cap.release()
        return None

    fixed_frame_width = 640
    frame = imutils.resize(frame, width=fixed_frame_width)
    heatmap_frame = np.copy(frame)

    # Parameters
    stationary_threshold_seconds = 2
    stationary_threshold_frame = round(vid_fps * stationary_threshold_seconds / data_record_frame)
    stationary_distance = fixed_frame_width * 0.05
    max_stationary_time = 120
    blob_layer = 50
    max_blob_size = fixed_frame_width * 0.1
    layer_size = max_blob_size / blob_layer
    color_start = 210
    color_end = 0
    color_steps = int((color_start - color_end) / blob_layer)
    scale = 1.5

    stationary_points = []
    movement_points = []

    # Analyze movement & stationary points
    for movement in tracks:
        temp_movement_point = [movement[0]]
        stationary = movement[0]
        stationary_time = 0
        for i in movement[1:]:
            if euclidean(stationary, i) < stationary_distance:
                stationary_time += 1
            else:
                temp_movement_point.append(i)
                if stationary_time > stationary_threshold_frame:
                    stationary_points.append([stationary, stationary_time])
                stationary = i
                stationary_time = 0
        movement_points.append(temp_movement_point)

    # Draw movement tracks
    color1 = (255, 96, 0)
    color2 = (0, 28, 255)
    tracks_frame = np.copy(frame)
    for track in movement_points:
        for i in range(len(track) - 1):
            color = gradient_color_RGB(color1, color2, len(track) - 1, i)
            cv2.line(tracks_frame, tuple(track[i]), tuple(track[i + 1]), color, 2)

    # Function to draw blobs for stationary points
    def draw_blob(frame, coordinates, time_val):
        if time_val >= max_stationary_time:
            layer = blob_layer
        else:
            layer = math.ceil(time_val * scale / layer_size)
        for x in reversed(range(layer)):
            color = color_start - (color_steps * x)
            size = x * layer_size
            cv2.circle(frame, coordinates, int(size), (color, color, color), -1)

    # Generate heatmap
    heatmap = np.zeros((heatmap_frame.shape[0], heatmap_frame.shape[1]), dtype=np.uint8)
    for points in stationary_points:
        draw_heatmap = np.zeros((heatmap_frame.shape[0], heatmap_frame.shape[1]), dtype=np.uint8)
        draw_blob(draw_heatmap, tuple(points[0]), points[1])
        heatmap = cv2.add(heatmap, draw_heatmap)

    lo = np.array([color_start])
    hi = np.array([255])
    mask = cv2.inRange(heatmap, lo, hi)
    heatmap[mask > 0] = color_start

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    lo = np.array([128, 0, 0])
    hi = np.array([136, 0, 0])
    mask = cv2.inRange(heatmap, lo, hi)
    heatmap[mask > 0] = (0, 0, 0)

    # Merge heatmap with background frame
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            if (heatmap[row][col] == np.array([0, 0, 0])).all():
                heatmap[row][col] = heatmap_frame[row][col]
    blended_frame = cv2.addWeighted(heatmap, 0.75, heatmap_frame, 0.25, 1)

    # Save the resulting image instead of showing
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE, blended_frame)
    print(f"[movement_data_present] Heatmap updated -> {OUTPUT_IMAGE}")

    cap.release()
    return True


def main():
    """Continuously refresh and save heatmap."""
    print("[movement_data_present] Headless movement heatmap generator started...")
    while True:
        try:
            success = generate_heatmap()
            if not success:
                time.sleep(3)
            else:
                time.sleep(5)
        except KeyboardInterrupt:
            print("[movement_data_present] Interrupted by user.")
            break
        except Exception as e:
            print("[movement_data_present] Error:", e)
            time.sleep(3)
            continue


if __name__ == "__main__":
    main()
