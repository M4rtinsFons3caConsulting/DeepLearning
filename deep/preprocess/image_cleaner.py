"""
image_cleaner.py - A tool for cleaning and preprocessing image datasets.

This script is designed to generate a cleaned directory of image samples by applying various 
image preprocessing techniques. It can be used as a standalone tool or is integrated 
within the `cleaner_routine.ipynb` for more seamless use within a Jupyter notebook environment.

The module works alongside `reset.py` and `regex_delete.py` to provide a comprehensive 
framework for programmatically manipulating and organizing raw image data, allowing for 
easy cleaning of large image directories. 

Although the motivation for these transformations appeared natural, and the suggestions abundant
when simply browsing the internet, support for the techniques presented below was found in several
papers of which we mention:

Citation:
    Ebenezer AS, Kanmani SD, Sivakumar M, Jeba S, Priya. (2022).
    Effect of image transformation on EfficientNet model for COVID-19 CT image classification.
    Materials Today: Proceedings, 51, 2512–2519

Citation: 
    Sadeghi, S., & Ganaie, M. A. (2021). 
    Comparing convolutional neural networks and preprocessing techniques for HEp-2 cell classification 
    in immunofluorescence images. 
    Computers in Biology and Medicine, 138, 104888.
    
It must be said, that these also reference the use of these techniques in augmentation, as per the
course materials.

Key functionalities include:
- Applying transformations such as blurring, normalization, and resizing.
- Generating and saving configuration files for reproducible data preprocessing steps.
- Processing image directories by iterating through and cleaning images in batch.

This tool was found to be particularly useful for preparing datasets for model training, ensuring that 
images are clean and consistent before further processing or model ingestion.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from deep.constants import IMAGE_DIR, PROCESSED_DIR, IMAGENET_NORM, CLEANER_JSONS
from deep.utils import build_updater

def _preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    """
    Preprocesses an image based on the given configuration.

    Applies image transformations like blurring, equalization, and normalization as specified in the config.

    Args:
        image (np.ndarray): The input image to process.
        config (dict): The preprocessing options (e.g., 'median_blur', 'gamma_correction').

    Returns:
        np.ndarray: The processed image.
    """

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

def save_config(config: dict) -> None:
    """
    Saves the preprocessing configuration to a timestamped JSON file.

    Args:
        config (dict): The configuration to save.

    Returns:
        Path: Path to the saved configuration file.
    """

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    filename = f"cleaner_config_{timestamp}.json"
    config_path = CLEANER_JSONS / filename

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    build_updater.write_to(config_path)
    

def clean_directory(config: dict) -> None:
    """
    Processes all images in IMAGE_DIR and saves the results to PROCESSED_DIR.

    Applies the specified preprocessing steps and saves the processed images.

    Args:
        config (dict): The preprocessing configuration to apply.
    """

    total, failed = 0, 0
    print(f"Starting preprocessing from: {IMAGE_DIR} - {PROCESSED_DIR}")

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
    
    save_config(config)


def clean_test_directory(config_path: Path, input_dir: Path, output_dir: Path) -> None:
    """
    Applies preprocessing to all .jpg images in `input_dir` using a config loaded from a JSON file.

    Args:
        config_path (Path): Path to a JSON config inside CLEANER_LOGS.
        input_dir (Path): Directory containing images to process.
        output_dir (Path): Directory where processed images will be saved.
    """
    if config_path.parent != CLEANER_JSONS:
        raise ValueError(f"Config file must be a valid cleaner config sourced from {CLEANER_JSONS}")

    with open(config_path, "r") as f:
        config = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    total, failed = 0, 0
    print(f"Starting test preprocessing from: {input_dir} - Output: {output_dir}")

    for img_path in input_dir.glob("*.jpg"):
        total += 1
        print(f"Processing: {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path.name}")
            failed += 1
            continue

        try:
            processed = _preprocess_image(img, config)
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), processed)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            failed += 1

    print(f"\nCompleted. Total: {total}, Failed: {failed}, Success: {total - failed}")
