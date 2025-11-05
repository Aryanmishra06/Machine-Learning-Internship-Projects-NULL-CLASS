from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]

# Settings
image_size = (128, 128)
batch_size = 32

# Data generators for training & validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    directory='datasets/fer2013',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    directory='datasets/fer2013',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Optional: check class index mapping
print("Class Labels:", train_data.class_indices)

