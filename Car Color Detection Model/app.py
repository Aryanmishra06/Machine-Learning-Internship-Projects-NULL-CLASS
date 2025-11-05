# app_car_color_final.py
import streamlit as st # pyright: ignore[reportMissingImports]
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO # pyright: ignore[reportMissingImports]
import os

# Automatically downloads yolov8n.pt if not present
yolo_model = YOLO("yolov8n.pt")

st.set_page_config(page_title="Car Color Detection + People Counter", layout="wide")
st.title("🚗 Car Color Detection (Blue vs Other) + People Counter")

# ---------------- Parameters ----------------
IMG_SIZE = 64  # CNN input size
labels = ["blue", "other"]

# ---------------- Load CNN Model ----------------
try:
    cnn_model = tf.keras.models.load_model("car_color_blue_other.h5", compile=False)
    st.success("CNN Model Loaded!")
except Exception as e:
    st.error(f"Error loading CNN model: {e}")
    st.stop()

# ---------------- Load YOLOv8 Model ----------------
yolo_path = "yolov8n.pt"

# Automatically download if not exists
if not os.path.exists(yolo_path):
    st.info("Downloading YOLOv8 model...")
try:
    yolo_model = YOLO(yolo_path)
    st.success("YOLOv8 Model Loaded!")
except Exception as e:
    st.error(f"Error loading YOLOv8 model: {e}")
    st.stop()

# ---------------- Functions ----------------
def predict_car_color(img):
    """Predict car color using CNN"""
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = img_resized.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = cnn_model.predict(img_array, verbose=0)
    class_idx = np.argmax(pred, axis=1)[0]
    confidence = pred[0][class_idx]
    return labels[class_idx], confidence

def process_frame(frame):
    """Detect cars and people, annotate frame"""
    results = yolo_model(frame)[0]
    people_count = 0
    car_count = 0

    if results.boxes is None or len(results.boxes) == 0:
        return frame, people_count, car_count

    for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)

        # Car detection
        if cls_id == 2:  # COCO class 2 = car
            car_count += 1
            car_roi = frame[y1:y2, x1:x2]
            if car_roi.size != 0:
                color_label, conf_score = predict_car_color(car_roi)
                color_box = (0, 0, 255) if color_label == "blue" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(frame, f"{color_label} ({conf_score*100:.1f}%)",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Person detection
        elif cls_id == 0:  # COCO class 0 = person
            people_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.putText(frame, f"People: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame, people_count, car_count

# ---------------- GUI ----------------
st.subheader("Upload Image for Detection")
uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed_frame, people_count, car_count = process_frame(frame)

    st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
             caption=f"People detected: {people_count} | Cars detected: {car_count}",
             use_container_width=True)

    st.write(f"✅ Total People: {people_count}")
    st.write(f"✅ Total Cars: {car_count}")
