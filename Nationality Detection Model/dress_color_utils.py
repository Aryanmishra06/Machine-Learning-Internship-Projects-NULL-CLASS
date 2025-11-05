import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors # pyright: ignore[reportMissingImports]

# === Step 1: Detect Face ===
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces[0] if len(faces) > 0 else None

# === Step 2: Extract Torso Region ===
def extract_torso_region(image):
    face = detect_face(image)
    if face is None:
        return None

    x, y, w, h = face

    # Define torso region below the face
    torso_y_start = y + h
    torso_y_end = y + int(2.5 * h)
    torso_x_start = max(x - int(0.25 * w), 0)
    torso_x_end = min(x + int(1.25 * w), image.shape[1])

    # Crop the torso region
    torso = image[torso_y_start:torso_y_end, torso_x_start:torso_x_end]
    return torso if torso.size > 0 else None

# === Step 3: Find Closest Color Name ===
def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

# === Step 4: Get Dominant Color ===
def get_dominant_color(image, k=3):
    if image is None:
        return "Unknown"

    image = cv2.resize(image, (100, 100))
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]

    return closest_color(tuple(map(int, dominant_color)))
