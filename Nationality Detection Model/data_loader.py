import pandas as pd # pyright: ignore[reportMissingModuleSource]
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd # pyright: ignore[reportMissingModuleSource]

df = pd.read_csv('utkface_annotations.csv')
df['image_path'] = df['image_path'].apply(lambda x: x.split('\\')[-1])  # Keep only filename
df.to_csv('utkface_annotations.csv', index=False)



def load_data(csv_path, image_dir, target_size=(128, 128)):
    df = pd.read_csv(csv_path)
    df = df[df['gender'].isin(['Male', 'Female'])]  # Clean gender labels

    X = []
    y = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image_path'])

        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, target_size)
        X.append(img)
        y.append(1 if row['gender'] == 'Male' else 0)

    return train_test_split(np.array(X)/255.0, np.array(y), test_size=0.2, random_state=42)
