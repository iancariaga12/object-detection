# YOLO11 Object Detection App

A real-time object detection web app built with YOLO11 and Streamlit.

## Features
- 🖼️ Image upload detection
- 🎬 Video upload with persistent object tracking (ByteTrack)
- 📷 Live webcam feed with real-time detection
- 🎚️ Confidence threshold slider
- 🏷️ Class filter (80 COCO classes)
- 📊 Detection summary with unique object counts

## Tech Stack
- [YOLO11](https://docs.ultralytics.com) — object detection & tracking
- [Streamlit](https://streamlit.io) — web UI
- [OpenCV](https://opencv.org) — image & video processing

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/object-detection.git
cd object-detection
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download YOLO11 model
Download `yolo11m.pt` from [Ultralytics](https://github.com/ultralytics/assets/releases) and place it in the root folder.

### 5. Run the app
```bash
streamlit run app.py
```

App opens at **http://localhost:8501**

## Project Structure
```
object-detection/
├── app.py              ← Streamlit UI
├── requirements.txt    ← Dependencies
├── README.md
└── utils/
    ├── __init__.py
    └── detector.py     ← YOLO11 detection logic
```

## Notes
- YOLO11n (nano) is used for webcam for better performance
- YOLO11m (medium) is used for image and video for better accuracy
- Webcam feature requires a local machine with camera access