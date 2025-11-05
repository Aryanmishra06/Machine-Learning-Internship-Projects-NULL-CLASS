import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout # pyright: ignore[reportMissingImports]

# === Settings ===
dataset_path = "datasets/UTKFace"
img_size = (128, 128)
ethnicity_map = {
    0: 'USA', 1: 'African', 2: 'Indian', 3: 'Others', 4: 'Others', 5: 'Others'
}

# === Load Data ===
images, ages, nationalities = [], [], []

for file in os.listdir(dataset_path):
    try:
        age, gender, ethnicity, _ = file.split("_")
        age = int(age)
        nationality = ethnicity_map.get(int(ethnicity), "Others")

        img = cv2.imread(os.path.join(dataset_path, file))
        img = cv2.resize(img, img_size)
        images.append(img)
        ages.append(age)
        nationalities.append(nationality)
    except:
        continue

images = np.array(images) / 255.0
ages = np.array(ages)

# === Encode Nationality ===
le = LabelEncoder()
nationality_labels = le.fit_transform(nationalities)
num_classes = len(le.classes_)

# === Split Data ===
X_train, X_test, age_train, age_test, nat_train, nat_test = train_test_split(
    images, ages, nationality_labels, test_size=0.2, random_state=42
)

# === Model: Multi-output CNN ===
input_layer = Input(shape=(128, 128, 3))

x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)

# Output 1: Age (regression)
age_output = Dense(1, name='age_output')(x)

# Output 2: Nationality (classification)
nat_output = Dense(num_classes, activation='softmax', name='nationality_output')(x)

# Model
model = Model(inputs=input_layer, outputs=[age_output, nat_output])

model.compile(
    optimizer='adam',
    loss={'age_output': 'mse', 'nationality_output': 'sparse_categorical_crossentropy'},
    metrics={'age_output': 'mae', 'nationality_output': 'accuracy'}
)

# === Train ===
history = model.fit(
    X_train, {'age_output': age_train, 'nationality_output': nat_train},
    epochs=10,
    batch_size=32,
    validation_data=(X_test, {'age_output': age_test, 'nationality_output': nat_test})
)

# === Save Model & Label Encoder ===
os.makedirs("models", exist_ok=True)
model.save("models/age_nationality_model.keras")

np.save("models/nationality_labels.npy", le.classes_)
