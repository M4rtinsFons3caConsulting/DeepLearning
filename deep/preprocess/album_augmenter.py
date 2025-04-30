"""
album_augmenter.py - A tool for the application of image augmentation as an oversampling technique.

This module allows for the augmentation of images using various transformations to balance the class distribution 
in a dataset. It generates an oversampling plan which stores information about the planned transformations to be 
applied to a given image, ensuring reproducibility of augmented images. 

The oversampling process generates transformed versions of minority class samples up to a target ratio or minimum 
amount of samples.

It is primarily used for creating balanced datasets through augmentation before model training, ensuring the dataset 
is ready and properly balanced, and saving both the augmented images and associated metadata for further use.

Key functionalities include:
- Generating an oversample plan based on class distribution.
- Applying transformations like rotation, flipping, and shifting to augment images.
- Saving the augmented dataset along with metadata for further processing.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from deep.constants import PROCESSED_DIR, METADATA_DIR, UPSAMPLE_JSONS
from deep.utils import build_updater

TRANSFORM_GENERATORS: Dict[str, Any] = {
    "rotate_30": lambda x: cv2.warpAffine(
        x,
        cv2.getRotationMatrix2D((x.shape[1] // 2, x.shape[0] // 2), 30, 1),
        (x.shape[1], x.shape[0])
    ),
    "sat_plus": lambda x: cv2.cvtColor(
        cv2.convertScaleAbs(cv2.cvtColor(x, cv2.COLOR_BGR2HSV), alpha=1.0, beta=25),
        cv2.COLOR_HSV2BGR
    ),
    "flip_lr": lambda x: cv2.flip(
        x, 1
    ),
    "bright_plus": lambda x: np.clip(
        x.astype(np.float32) * 3.0, 0, 255
    ).astype(np.uint8),
    "rotate_45": lambda x: cv2.warpAffine(
        x,
        cv2.getRotationMatrix2D((x.shape[1] // 2, x.shape[0] // 2), 45, 1),
        (x.shape[1], x.shape[0])
    ),
    "rotate_15": lambda x: cv2.warpAffine(
        x,
        cv2.getRotationMatrix2D((x.shape[1] // 2, x.shape[0] // 2), 15, 1),
        (x.shape[1], x.shape[0])
    ),
    "rotate_60": lambda x: cv2.warpAffine(
        x,
        cv2.getRotationMatrix2D((x.shape[1] // 2, x.shape[0] // 2), 60, 1),
        (x.shape[1], x.shape[0])
    ),
    "rotate_90": lambda x: cv2.warpAffine(
        x,
        cv2.getRotationMatrix2D((x.shape[1] // 2, x.shape[0] // 2), 90, 1),
        (x.shape[1], x.shape[0])
    ),
    "sat_minus": lambda x: cv2.cvtColor(
        cv2.convertScaleAbs(cv2.cvtColor(x, cv2.COLOR_BGR2HSV), alpha=1.0, beta=-25),
        cv2.COLOR_HSV2BGR
    ),
     "shift": lambda x: cv2.warpAffine(
        x,
        np.float32([[1, 0, x.shape[1] * 0.1], [0, 1, x.shape[0] * 0.1]]),
        (x.shape[1], x.shape[0])
    ),
    "zoom_in": lambda x: cv2.resize(
        x,
        (int(x.shape[1] * 1.2), int(x.shape[0] * 1.2))
    ),
    "zoom_out": lambda x: cv2.resize(
        x,
        (int(x.shape[1] * 1.25), int(x.shape[0] * 1.25))
    )
}

def compute_effective_class_weights(
    df: pd.DataFrame,
    label_column: str,
    beta: float = 0.999,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Computes effective class weights to address class imbalance.

    This function calculates the effective class weights based on the class distribution in the dataset,
    with the option to normalize the weights.

    Citation:
    Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. 
    In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 9268–9277. 
    https://doi.org/10.1109/CVPR.2019.00949

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        label_column (str): Column name containing the labels.
        beta (float, optional): The beta parameter for calculating effective class weights (default is 0.999).
        normalize (bool, optional): Whether to normalize the weights (default is True).

    Returns:
        Dict[str, float]: A dictionary mapping class labels to their effective weights.
    """

    def effective_num(n: int, beta: float) -> float:
        return (1 - beta**n) / (1 - beta)

    label_counts = df[label_column].value_counts()
    effective_nums = label_counts.apply(lambda n: effective_num(n, beta))
    total_effective = effective_nums.sum()
    class_weights = (total_effective / effective_nums).to_dict()

    if normalize:
        mean_weight = np.mean(list(class_weights.values()))
        class_weights = {cls: w / mean_weight for cls, w in class_weights.items()}

    return class_weights

def compute_imbalance_ratio(
    df: pd.DataFrame,
    label_column: str
) -> Dict[str, float]:
    """
    Computes the imbalance ratio for each class in the dataset.

    This function calculates the ratio of the largest class count to each class's count to identify 
    the imbalance ratio for the dataset.

    Citation:
    Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in 
    convolutional neural networks. Neural Networks, 106, 249–259. 
    https://doi.org/10.1016/j.neunet.2018.07.011

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        label_column (str): Column name containing the labels.

    Returns:
        Dict[str, float]: A dictionary mapping class labels to their imbalance ratios.
    """
    counts = df[label_column].value_counts()
    max_count = counts.max()
    return {label: max_count / count for label, count in counts.items()}

def generate_oversample_map(
    df: pd.DataFrame,
    label: str,
    target_ratio: float,
    min_samples: int
) -> List[Dict[str, str]]:
    """
    Generates an oversample plan to balance the class distribution based on a target ratio.

    This function creates an oversample plan that includes transformations to be applied to 
    the underrepresented classes, ensuring that the dataset achieves the desired class distribution.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        label (str): The target label column for oversampling.
        target_ratio (float): The desired class distribution ratio.
        min_samples (int): The minimum number of samples required for oversampling.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing oversample details for each class.
    """

    counts = df[label].value_counts()
    max_count = counts.max()
    majority_label = counts.idxmax()
    plan = []

    for cls_label, count in counts.items():
        if cls_label == majority_label:
            continue

        current_ratio = max_count / count
        if (current_ratio <= target_ratio) and (count > min_samples):
            continue

        target_count = max(int(max_count / target_ratio), (min_samples - count))
        needed = target_count - count
        if needed <= 0:
            continue

        class_df = df[df[label] == cls_label]
        transform_keys = list(TRANSFORM_GENERATORS.keys())
        transform_idx = 0

        while needed > 0 and transform_idx < len(transform_keys):
            transform_key = transform_keys[transform_idx]
            n_samples = min(needed, len(class_df))

            selected = class_df.sample(n_samples)
            selected['transform_key'] = transform_key

            if len(plan) == 0:
                plan = selected
            else:
                plan = pd.concat([plan, selected], ignore_index=True)

            needed -= n_samples
            transform_idx += 1

    config = {
        'label': label,
        'target_ratio': target_ratio,
        'min_samples': min_samples
    }

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")    
    filename = f"{label}_upsample_plan_{timestamp}.json"
    json_path = Path(UPSAMPLE_JSONS) / filename

    plan_with_config = {"config": config, "oversample_plan": plan.to_dict(orient="records")}

    with open(json_path, "w") as f:
        json.dump(plan_with_config, f, indent=4)

    return json_path

def _apply_transformation(
    img: Image.Image,
    transformation_key: str
) -> Image.Image:
    
    """
    Applies a transformation to an image based on the transformation key.

    This function uses a transformation key to apply the corresponding image transformation from 
    the TRANSFORM_GENERATORS dictionary.

    Args:
        img (Image.Image): The input image to be transformed.
        transformation_key (str): The key identifying the transformation to apply.

    Returns:
        Image.Image: The transformed image.
    """

    if transformation_key not in TRANSFORM_GENERATORS:
        raise ValueError(f"Unknown transformation key: {transformation_key}")

    img_array = np.array(img)
    transformed_array = TRANSFORM_GENERATORS[transformation_key](img_array)
    return Image.fromarray(transformed_array)

def oversample_labels(
    label: str,
    output_name: str,
    plan_path: str
) -> None:
    """
    Applies oversampling transformations on the dataset based on a generated plan.

    This function takes the oversample plan generated by `generate_oversample_map`, applies the 
    specified transformations, and saves the augmented images to the dataset.

    Args:
        label (str): The label for which oversampling is being applied.
        output_name (str): The output name for the augmented metadata file.
        plan_path (str): Path to the JSON file containing the oversample plan.
    """

    with open(plan_path, "r") as f:
        plan_data = json.load(f)

    oversample_plan = plan_data.get("oversample_plan", [])
    augmented_rows = []

    for entry in oversample_plan:
        source = PROCESSED_DIR / entry["file_path"]
        filename = f"{source.stem}_{entry[label]}_{entry['transform_key']}{source.suffix}"
        destination = PROCESSED_DIR / filename

        try:
            img = Image.open(source).convert("RGB")
            new_img = _apply_transformation(img, entry["transform_key"])
            new_img.save(destination)

            augmented_rows.append({
                "rare_species_id": entry['rare_species_id'],
                "file_path": str(filename),
                label: entry[label]
            })

        except Exception as e:
            print(f"Failed on {source.name}: {e}")

    df_aug = pd.DataFrame(augmented_rows)
    filepath = METADATA_DIR / f"{output_name}.csv"
    df_aug.to_csv(filepath, mode='a', index=False)

    build_updater.write_to(plan_path)
