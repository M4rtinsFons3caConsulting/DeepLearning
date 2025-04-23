import cv2
import numpy as np
from deep.constants import IMAGE_DIR, IMAGENET_NORM


def _preprocess_image(
        image: np.ndarray
    ) -> np.ndarray:
    """
    Preprocess an input image.

    Args:
        image (np.ndarray): Input image to be preprocessed.

    Returns:
        np.ndarray: Preprocessed image.
    """
    # Denoising (median blur)
    denoised = cv2.medianBlur(image, 3)

    # Mean‑shift filtering (edge‑preserving smoothing, toned down)
    shifted = cv2.pyrMeanShiftFiltering(denoised, sp=5, sr=12)

    # CLAHE on luminance channel
    ycrcb = cv2.cvtColor(shifted, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=1.05, tileGridSize=(3, 3))
    y_eq = clahe.apply(y)
    equalized = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2RGB)

    # Apply ImageNet normalization (mean and std)
    imagenet_mean = np.array(IMAGENET_NORM["mean"], dtype=np.float32)
    imagenet_std = np.array(IMAGENET_NORM["std"], dtype=np.float32)

    img_f = equalized.astype(np.float32) / 255.0
    normed = (img_f - imagenet_mean) / imagenet_std
    normed = np.clip(normed, 0.0, 1.0)

    # Scale back to [0,255] and convert to uint8
    output = (normed * 255.0).clip(0, 255).astype(np.uint8)

    return output


def clean_directory() -> None:
    """
    Processes all images in IMAGE_DIR using `_preprocess_image` and 
    displays each result. Saves it back to the same location upon window close.
    """

    total = 0
    failed = 0

    print(f"Starting preprocessing of images in: {IMAGE_DIR}")
    for img_path in IMAGE_DIR.glob("*.jpg"):
        total += 1
        print(f"Processing: {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path.name}")
            failed += 1
            continue

        processed = _preprocess_image(img)
        cv2.imwrite(str(IMAGE_DIR / img_path.name), processed)
        print(f"Saved: {img_path.name}")

    print(f"\nPreprocessing complete. Total: {total}, Failed: {failed}, Success: {total - failed}")
