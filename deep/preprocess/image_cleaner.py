import cv2
import json
import numpy as np
from datetime import datetime
from deep.constants import IMAGE_DIR, PROCESSED_DIR, IMAGENET_NORM, CLEANER_JSONS
from deep.utils import build_updater

def _preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    result = image.copy()

    if config.get("median_blur", False):
        result = cv2.medianBlur(result, 3)

    if config.get("gaussian_blur", False):
        result = cv2.GaussianBlur(result, (3, 3), 0)

    if config.get("bilateral", False):
        result = cv2.bilateralFilter(result, d=9, sigmaColor=75, sigmaSpace=75)

    if config.get("meanshift", False):
        result = cv2.pyrMeanShiftFiltering(result, sp=5, sr=12)

    if config.get("hist_eq", False):
        ycrcb = cv2.cvtColor(result, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        result = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2RGB)

    if config.get("clahe", False):
        ycrcb = cv2.cvtColor(result, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.00, tileGridSize=(3, 3))
        y_eq = clahe.apply(y)
        result = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2RGB)

    if config.get("gamma_correction", False):
        gamma = config.get("gamma", 1.0)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
        result = cv2.LUT(result, table)

    if config.get("imagenet_norm", False):
        mean = np.array(IMAGENET_NORM["mean"], dtype=np.float32)
        std = np.array(IMAGENET_NORM["std"], dtype=np.float32)
        img_f = result.astype(np.float32) / 255.0
        normed = (img_f - mean) / std
        normed = np.clip(normed, 0.0, 1.0)
        result = (normed * 255.0).clip(0, 255).astype(np.uint8)

    return result

def save_config(
        config: dict
    ) -> None:

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    filename = f"cleaner_config_{timestamp}.json"
    config_path = CLEANER_JSONS / filename

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return config_path

def clean_directory(config: dict) -> None:
    total, failed = 0, 0
    print(f"Starting preprocessing from: {IMAGE_DIR} → {PROCESSED_DIR}")

    for img_path in IMAGE_DIR.glob("*.jpg"):
        total += 1
        print(f"Processing: {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path.name}")
            failed += 1
            continue

        try:
            processed = _preprocess_image(img, config)
            output_path = PROCESSED_DIR / img_path.name
            cv2.imwrite(str(output_path), processed)
        
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            failed += 1

    print(f"\nCompleted. Total: {total}, Failed: {failed}, Success: {total - failed}")
    
    output_path = save_config(config)

    # Update the build
    build_updater.write_to(output_path)