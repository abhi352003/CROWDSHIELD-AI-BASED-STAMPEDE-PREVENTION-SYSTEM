from flask import Flask, render_template, Response, jsonify, send_file
import cv2
import json
import time
import os

app = Flask(__name__)

# ------------------ VIDEO STREAM ------------------

def generate_frames():
    """
    Stream the latest processed frame saved by backend (video_process.py)
    instead of directly using the webcam.
    """
    while True:
        try:
            if os.path.exists("processed_data/latest_frame.jpg"):
                frame = cv2.imread("processed_data/latest_frame.jpg")
                if frame is not None:
                    # 🔽 Downscale frame for faster streaming
                    frame = cv2.resize(frame, (640, 360))
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)  # update roughly 10 FPS
        except Exception as e:
            print("[Flask] Video stream error:", e)
            time.sleep(0.5)


# ------------------ ROUTES ------------------

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video stream endpoint (reads processed frames)."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------ IMAGE ENDPOINTS ------------------

@app.route('/energy_plot')
def energy_plot():
    """Serve the latest abnormal energy histogram PNG."""
    path = "processed_data/energy_hist.png"
    return send_file(path, mimetype='image/png') if os.path.exists(path) \
        else send_file("static/placeholder.png", mimetype='image/png')


@app.route('/heatmap')
def heatmap():
    """Serve latest movement heatmap image."""
    path = "processed_data/heatmap.png"
    return send_file(path, mimetype='image/png') if os.path.exists(path) \
        else send_file("static/placeholder.png", mimetype='image/png')


@app.route('/crowd_plot')
def crowd_plot():
    """Serve latest crowd plot image."""
    path = "processed_data/crowd_plot.png"
    return send_file(path, mimetype='image/png') if os.path.exists(path) \
        else send_file("static/placeholder.png", mimetype='image/png')


# ------------------ LIVE DATA STREAM ------------------

@app.route('/data_feed')
def data_feed():
    """Stream combined live data (crowd + movement + abnormal) as Server-Sent Events."""
    def generate():
        while True:
            combined = {
                "crowd": {"time": "-", "human_count": 0, "violation_count": 0,
                          "restricted_entry": False, "abnormal_activity": False},
                "movement": {"track_count": 0},
                "abnormal": {"updated": None}
            }

            # 1️⃣ Crowd data
            try:
                with open('processed_data/crowd_data.csv', 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last = lines[-1].strip().split(',')
                        combined["crowd"] = {
                            "time": last[0],
                            "human_count": int(last[1]),
                            "violation_count": int(last[2]),
                            "restricted_entry": bool(int(last[3])),
                            "abnormal_activity": bool(int(last[4]))
                        }
            except Exception:
                pass

            # 2️⃣ Abnormal (last update time)
            try:
                if os.path.exists("processed_data/energy_hist.png"):
                    combined["abnormal"]["updated"] = os.path.getmtime("processed_data/energy_hist.png")
            except Exception:
                pass

            # 3️⃣ Movement (track count)
            try:
                if os.path.exists("processed_data/movement_data.csv"):
                    with open("processed_data/movement_data.csv", "r") as f:
                        lines = f.readlines()
                        combined["movement"]["track_count"] = max(0, len(lines) - 1)
            except Exception:
                pass

            yield f"data:{json.dumps(combined)}\n\n"
            time.sleep(1)

    return Response(generate(), mimetype='text/event-stream')


# ------------------ MAIN ------------------

if __name__ == '__main__':
    # Create placeholder image if missing
    if not os.path.exists("static/placeholder.png"):
        import numpy as np
        blank = 255 * np.ones((300, 400, 3), np.uint8)
        cv2.putText(blank, "No Data Yet", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite("static/placeholder.png", blank)

    print("[Flask] Crowd Analysis Dashboard running on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
