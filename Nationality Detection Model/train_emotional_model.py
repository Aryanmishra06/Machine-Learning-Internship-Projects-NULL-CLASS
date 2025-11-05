import os
import tensorflow as tf
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]

# === Paths ===
dataset_path = os.path.join("datasets", "fer2013")
model_save_path = os.path.join("models", "emotion_model.h5")

# === Parameters ===
img_size = (128, 128)
batch_size = 32
num_classes = 7
epochs = 10

# === Data Preparation ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

print("Emotion Labels Mapping:", train_data.class_indices)

# === Model Architecture ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# === Compile ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Train ===
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data
)

# === Save Model ===
os.makedirs("models", exist_ok=True)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
