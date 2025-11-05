import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dress_color_utils import extract_torso_region, get_dominant_color # pyright: ignore[reportMissingImports]


# === Load Models ===
emotion_model = load_model("models/emotion_model.h5")
age_nat_model = load_model("models/age_nationality_model.keras")
nationality_labels = np.load("models/nationality_labels.npy")  # e.g., ['African', 'Indian', 'Others', 'USA']

# === Emotion Label Mapping (ensure same order as training) ===
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# === Predict Function ===
def predict_all(image_path):
    # Step 1: Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Image not found"}

    resized_img = cv2.resize(img, (128, 128)) / 255.0
    input_img = np.expand_dims(resized_img, axis=0)

    # Step 2: Predict Emotion
    emotion_pred = emotion_model.predict(input_img)
    emotion = emotion_labels[np.argmax(emotion_pred)]

    # Step 3: Predict Age and Nationality
    age_pred, nat_pred = age_nat_model.predict(input_img)
    age = int(age_pred[0][0])
    nationality = nationality_labels[np.argmax(nat_pred[0])]

    # Step 4: Conditional Dress Color Detection
    dress_color = "N/A"
    if nationality in ["Indian", "African"]:
        torso = extract_torso_region(img)
        dress_color = get_dominant_color(torso)

    # Step 5: Package Results
    result = {
        "nationality": nationality,
        "emotion": emotion,
        "age": age if nationality in ["Indian", "USA"] else "N/A",
        "dress_color": dress_color if nationality in ["Indian", "African"] else "N/A"
    }
    return result
