import cv2
import streamlit as st
from PIL import Image
import numpy as np
import tempfile

from utils.detector import (
    COCO_CLASSES,
    detect_image,
    detect_video,
    load_model,
)

st.set_page_config(
    page_title="YOLO11 Object Detection",
    layout="wide",
)

model = load_model("m")
model_webcam = load_model("n")

st.title("YOLO11 Object Detection")
st.caption("Upload an image or video to detect objects using YOLO11 medium")

with st.sidebar:
    st.header("Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Only show detections above this confidence level"
    )

    st.markdown("---")

    filter_mode = st.radio(
        "Class Filter",
        ["All Classes", "Selected Classes Only"]
    )

    selected_classes = None
    if filter_mode == "Selected Classes Only":
        selected_classes = st.multiselect(
            "Pick classes to detect",
            options=sorted(COCO_CLASSES),
            default=["person", "car", "dog"]
        )
        if not selected_classes:
            st.warning("No classes selected — showing all.")
            selected_classes = None

    st.markdown("---")
    st.caption("YOLO11 medium · 80 COCO classes")

tab_image, tab_video, tab_webcam = st.tabs(["Image", "Video", "Webcam"])


with tab_image:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is None:
        st.info("Upload an image above to get started")

    else:
        pil_image = Image.open(uploaded_file).convert("RGB")
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Original")
            st.image(pil_image, width='stretch')

        with st.spinner("Detecting objects..."):
            annotated, detections = detect_image(model, cv2_image, confidence, selected_classes)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with col_right:
            st.subheader("Detected")
            st.image(annotated_rgb, width='stretch')

        st.success(f"Found {len(detections)} objects!")

        if detections:
            st.markdown("### Detection Summary")
            summary = {}
            for d in detections:
                label = d["label"]
                summary[label] = summary.get(label, 0) + 1

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Class**")
                for label in summary:
                    st.write(label)
            with col2:
                st.markdown("**Count**")
                for count in summary.values():
                    st.write(count)


with tab_video:
    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_video is None:
        st.info("Upload a video above to get started")

    else:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        st.subheader("Original Video")
        st.video(tmp_path)

        if st.button("Run Detection", type="primary"):
            progress_bar = st.progress(0, text="Processing frames...")

            def update_progress(pct):
                progress_bar.progress(pct, text=f"Processing frames... {pct:.0%}")

            with st.spinner(""):
                output_path, seen_ids = detect_video(
                    model, tmp_path, confidence, selected_classes, update_progress
                )

            progress_bar.progress(1.0, text="Done!")

            st.subheader("Detected Video")
            st.video(output_path)

            st.success(f"Tracked {len(seen_ids)} unique objects!")

            if seen_ids:
                st.markdown("### Detection Summary")
                summary = {}
                for obj_label in seen_ids.values():
                    summary[obj_label] = summary.get(obj_label, 0) + 1

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Class**")
                    for class_name in summary:
                        st.write(class_name)
                with col2:
                    st.markdown("**Unique Objects**")
                    for count in summary.values():
                        st.write(count)


with tab_webcam:
    st.subheader("Live Webcam Detection")

    if "run_webcam" not in st.session_state:
        st.session_state["run_webcam"] = False

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start Webcam", type="primary", use_container_width=True):
            st.session_state["run_webcam"] = True
    with col_stop:
        if st.button("Stop Webcam", use_container_width=True):
            st.session_state["run_webcam"] = False

    _, col_center, _ = st.columns([1, 4, 1])
    frame_placeholder = col_center.empty()
    detection_placeholder = st.empty()

    if st.session_state["run_webcam"]:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Cannot access webcam. Make sure no other app is using it.")
            st.session_state["run_webcam"] = False
        else:
            frame_count = 0

            while st.session_state["run_webcam"]:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost webcam feed.")
                    break

                # Shrink frame for faster detection
                small_frame = cv2.resize(frame, (640, 360))

                # Only run detection every 2nd frame
                if frame_count % 2 == 0:
                    annotated, detections = detect_image(
                        model_webcam, small_frame, confidence, selected_classes
                    )
                    last_annotated = annotated
                    last_detections = detections
                else:
                    annotated = last_annotated if frame_count > 0 else small_frame
                    detections = last_detections if frame_count > 0 else []

                frame_count += 1

                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, width='stretch')

                if detections:
                    summary = {}
                    for d in detections:
                        label = d["label"]
                        summary[label] = summary.get(label, 0) + 1

                    summary_text = "  |  ".join(
                        f"{label}: {count}" for label, count in summary.items()
                    )
                    detection_placeholder.info(f"Detected → {summary_text}")
                else:
                    detection_placeholder.info("No objects detected")

            cap.release()
            frame_placeholder.empty()
            detection_placeholder.empty()