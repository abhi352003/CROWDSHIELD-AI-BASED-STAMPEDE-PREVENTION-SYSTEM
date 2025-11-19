# 🎥 CrowdShield: AI-Based Stampede Prevention System  
Real-Time Crowd Analytics | YOLOv4-Tiny | DeepSORT | Behaviour Modelling | Flask Dashboard

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ReID-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Server-black.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Overview

**CrowdShield** is an AI-powered real-time video analytics system designed to **detect, track, and analyse human crowd behaviour** to prevent stampedes and dangerous crowd formations.

This system leverages:

- **YOLOv4-Tiny** → Fast, real-time human detection  
- **DeepSORT** → Multi-person identity tracking  
- **Crowd Analytics** → Density estimation, movement heatmaps  
- **Behaviour Analysis** → Speed, acceleration, energy spikes  
- **Alert System** → Automatic WhatsApp alerts  
- **Flask Dashboard** → Live video + heatmaps + graphs  

CrowdShield is built for use in:

✔ Temples  
✔ Railway stations  
✔ Stadiums  
✔ Event gatherings  
✔ Smart city surveillance  

---

## 🚀 Key Features

### 🔹 1. Real-Time Person Detection (YOLOv4-Tiny)
- Detects humans in every frame with high FPS  
- Lightweight model optimized for CPU  

### 🔹 2. Multi-Object Tracking (DeepSORT)
- Maintains **unique identity IDs**  
- Appearance-based Re-ID using 128-D embeddings  
- Minimal ID switching  

### 🔹 3. Behaviour Analysis
- Computes **velocity, acceleration, direction change**  
- Detects **abnormal movement**, running, panic spikes  
- Kinetic energy–based anomaly scoring  

### 🔹 4. Crowd Density & Heatmaps
- Identifies hotspots  
- Highlights stationary points and congestion  
- Uses Gaussian/Blob overlays  

### 🔹 5. Automated Alerts (WhatsApp)
Triggers alert when:
- Density > threshold  
- Abnormal behaviour is detected  
- Restricted area is entered  

### 🔹 6. Interactive Dashboard
- Live processed video feed  
- Heatmap updates  
- Crowd count graph  
- Abnormal activity timeline  
- Full web interface using Flask + JavaScript  

---

## 🧠 System Architecture

```
            ┌──────────────────────┐
            │     IP Camera        │
            └─────────┬────────────┘
                      │  (Live Feed)
                      ▼
        ┌────────────────────────────┐
        │  Frame Preprocessing (CV2) │
        └─────────┬──────────────────┘
                  ▼
        ┌────────────────────────────┐
        │    YOLOv4-Tiny Detector    │
        └─────────┬──────────────────┘
                  ▼
        ┌────────────────────────────┐
        │    DeepSORT Tracking       │
        └─────────┬──────────────────┘
                  ▼
      ┌───────────────────────────────────────┐
      │ Behaviour Analysis                     │
      │ • Speed • Acceleration • Energy       │
      │ • Direction Change • Density          │
      └───────┬───────────────┬──────────────┘
              │               │
     ┌────────▼───────┐   ┌──▼─────────────┐
     │ Heatmap Engine │   │ Graph Generator │
     └────────┬───────┘   └─────┬──────────┘
              │                 │
              ▼                 ▼
       Processed Outputs → Flask Dashboard → Browser
```

---

## 📁 Project Structure

```
CrowdShield/
│── main.py
│── video_process.py
│── tracking.py
│── detection.py
│── tracker.py
│── track.py
│── util.py
│── config.py
│── whatsapp_alert.py
│
├── models/
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights
│   ├── coco.names
│   └── mars-small128.pb
│
├── dashboard/
│   ├── app.py
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── processed_data/
│   ├── movement_data.csv
│   ├── crowd_data.csv
│   ├── video_data.json
│   ├── heatmap.png
│   └── energy_hist.png
│
└── requirements.txt
```

---

## ⚙️ Installation

### **1. Clone the repository**
```bash
git clone https://github.com/abhi352003/Crowd_analysis
cd Crowd_analysis
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Download YOLO weights**
Already included in this project.

### **4. Start Flask Dashboard**
```bash
python dashboard/app.py
```

### **5. Start Real-Time Processing**
```bash
python main.py
```

---

## 📊 Performance Summary

| Component | Value |
|----------|-------|
| YOLO FPS (CPU) | ~3.5 FPS |
| DeepSORT FPS | ~4–5 FPS |
| Full Pipeline FPS | 2–3 FPS |
| Backend → Frontend Delay | 2–3 sec |
| Heatmap Update | 2–3 sec |
| Energy Histogram | 3–4 sec |

---

## 🔥 Future Enhancements

- WebSocket-based frame streaming (0.2s latency)
- ONNX Runtime for faster inference
- GPU acceleration support (70–120 FPS)
- Crowd behaviour prediction using LSTMs
- Multi-camera fusion for large deployments
- Drone-based crowd monitoring

---

## 👥 Team Members

- **Abhishek Kumar (20224003)**  
- **Ayush Pratap Singh (20224042)**  
- **Samit Sonkar (20224131)**  

---

