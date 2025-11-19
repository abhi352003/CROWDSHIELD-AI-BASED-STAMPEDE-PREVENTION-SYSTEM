


"""
detect_human.py
----------------
Handles YOLOv4-tiny + DeepSORT-based human detection and tracking.

✔ Headless (no GUI)
✔ Optimized for real-time inference
✔ Integrated with Crowd-Analysis backend
"""

import numpy as np
import cv2
from config import MIN_CONF, NMS_THRESH
from deep_sort.detection import Detection


def detect_human(net, ln, frame, encoder, tracker, time):
    """
    Detect and track humans using YOLOv4-tiny + DeepSORT.

    Args:
        net (cv2.dnn_Net): YOLO network object.
        ln (list): YOLO output layer names.
        frame (np.ndarray): Current video frame (BGR).
        encoder: DeepSORT feature encoder.
        tracker: DeepSORT tracker instance.
        time: Current frame time (for DeepSORT timestamps).

    Returns:
        tuple:
            tracked_bboxes (list): Active confirmed DeepSORT tracks.
            expired (list): Tracks that expired (exited the frame).
    """

    frame_height, frame_width = frame.shape[:2]

    # ---------------- YOLO Forward Pass ----------------
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    boxes, centroids, confidences = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Class 0 = 'person' in COCO dataset
            if class_id == 0 and confidence > MIN_CONF:
                box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((center_x, center_y))
                confidences.append(float(confidence))

    # ---------------- Non-Max Suppression ----------------
    if len(boxes) == 0:
        tracker.predict()
        expired = tracker.update([], time)
        return [], expired

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    if len(idxs) == 0:
        tracker.predict()
        expired = tracker.update([], time)
        return [], expired

    idxs = idxs.flatten()
    boxes = np.array(boxes)[idxs]
    centroids = np.array(centroids)[idxs]
    confidences = np.array(confidences)[idxs]

    # ---------------- DeepSORT Tracking ----------------
    features = np.array(encoder(frame, boxes))
    detections = [
        Detection(bbox, conf, centroid, feat)
        for bbox, conf, centroid, feat in zip(boxes, confidences, centroids, features)
    ]

    tracker.predict()
    expired = tracker.update(detections, time)

    # ---------------- Collect Active Tracks ----------------
    tracked_bboxes = [
        track for track in tracker.tracks
        if track.is_confirmed() and track.time_since_update <= 5
    ]

    return tracked_bboxes, expired
