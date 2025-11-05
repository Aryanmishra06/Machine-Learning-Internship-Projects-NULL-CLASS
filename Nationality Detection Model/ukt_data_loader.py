import os
import cv2
import numpy as np

def load_utkface_images('UTKFace', image_size=(128, 128)):
    images, ages, nationalities = [], [], []
    

    for filename in os.listdir(UTKFace): # pyright: ignore[reportUndefinedVariable]
        try:
            age, gender, ethnicity, _ = filename.split('_')
            age = int(age)
            ethnicity = int(ethnicity)
            nationality = ethnicity_map.get(ethnicity, 'Others') # pyright: ignore[reportUndefinedVariable]

            img = cv2.imread(os.path.join(folder_path, filename)) # pyright: ignore[reportUndefinedVariable]
            img = cv2.resize(img, image_size) # pyright: ignore[reportUndefinedVariable]
            images.append(img)
            ages.append(age)
            nationalities.append(nationality)
        except:
            continue

    return np.array(images), np.array(ages), np.array(nationalities)
