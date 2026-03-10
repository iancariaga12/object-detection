import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",   
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


@st.cache_resource
def load_model(model_size: str = "n") -> YOLO:
    model_name = f"yolo11{model_size}.pt"
    model = YOLO(model_name)
    return model

def detect_image(model, image, confidence=0.25, selected_classes=None):

    results = model(image, conf=confidence, verbose=False)[0]

    detections = []
    annotated = image.copy()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = COCO_CLASSES[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if selected_classes and label not in selected_classes:
            continue

        detections.append({
            "label": label,
            "confidence": round(conf, 3),
            "bbox": (x1, y1, x2, y2)
        })

        color = _class_color(cls_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated, detections

def _class_color(cls_id):
    np.random.seed(cls_id * 7)
    return tuple(int(x) for x in np.random.randint(50, 220, size=3))

def detect_video(model, video_path, confidence=0.25, selected_classes=None, progress_callback=None):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = video_path.replace(".mp4", "_detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    seen_ids = {}
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            conf=confidence,
            persist=True,       
            verbose=False
        )[0]

        annotated = frame.copy()

        if results.boxes.id is not None:
            for box, track_id, cls_id in zip(
                results.boxes.xyxy,
                results.boxes.id,
                results.boxes.cls
            ):
                track_id = int(track_id)
                cls_id = int(cls_id)
                label = COCO_CLASSES[cls_id]

                if selected_classes and label not in selected_classes:
                    continue

                if track_id not in seen_ids:
                    seen_ids[track_id] = label

                x1, y1, x2, y2 = map(int, box)
                color = _class_color(cls_id)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                text = f"{label} id:{track_id}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(annotated)
        frame_index += 1

        if progress_callback and total_frames > 0:
            progress_callback(frame_index / total_frames)

    cap.release()
    writer.release()

    return output_path, seen_ids