"""
predict_from_directory.py - this script handles the main pipeline for prediction operations.

It validates the contents of a given directory, then applies the same preprocessing routine applied to our test
dataset. It then allows for the loading of different models that once loaded, iterate over the images in the provided 
directory outputing predictions to a CSV file.

The model predictions csv is stored as `<model_name>_<model_timestamp>_predictions.csv` in `model/model_results`, it is
important to mention that this directory also stores the prediction CSV of our best models.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore

def load_model(
           
):
    """Loads a keras model in .h5 format"""
    load_model()

# --- Config ---
INPUT_DIR = "path/to/predict_dir"
MODEL_PATH = "path/to/model.h5"
OUTPUT_PATH = "predictions.csv"
TARGET_SIZE = (224, 224)  # replace with your model input size

# --- Validate directory ---
if not os.path.isdir(INPUT_DIR) or not os.listdir(INPUT_DIR):
    raise ValueError("Invalid or empty prediction directory.")

# --- Validate images ---
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

image_paths = [
    os.path.join(INPUT_DIR, fname)
    for fname in os.listdir(INPUT_DIR)
    if is_valid_image(os.path.join(INPUT_DIR, fname))
]

# --- Load model ---
model = load_model(MODEL_PATH)

# --- Preprocess and predict ---
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

results = []
for path in image_paths:
    img = preprocess_image(path, TARGET_SIZE)
    pred = model.predict(img)[0][0]  # adjust indexing if needed
    results.append((os.path.basename(path), pred))

# --- Save results ---
df = pd.DataFrame(results, columns=["filename", "prediction"])
df.to_csv(OUTPUT_PATH, index=False)
