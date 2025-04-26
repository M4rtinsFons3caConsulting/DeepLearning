"""
os_tools/oversampler.py

Module for handling class imbalance through oversampling and data augmentation.
Supports CLI generation of augmentation plans (maps), programmatic oversampling, and class analysis tools.
"""

# Built-in and STL
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# 3rd party
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ( # type: ignore
    ImageDataGenerator,
    img_to_array,
    array_to_img,
    load_img,
)

# Root package
from deep.constants import PROCESSED_DIR, METADATA_DIR, UPSAMPLE_JSONS, IMAGE_DIR
from deep.utils import build_updater

# Transformations dict for ImageGenerator
TRANSFORM_GENERATORS: Dict[str, ImageDataGenerator] = {
    "flip_lr": ImageDataGenerator(horizontal_flip=True),
    "bright_plus": ImageDataGenerator(preprocessing_function=lambda x: x * 1.10),
    "bright_minus": ImageDataGenerator(preprocessing_function=lambda x: x * 0.90),
    "sat_plus": ImageDataGenerator(preprocessing_function=lambda x: tf.image.adjust_saturation(x, 1.10)),
    "sat_minus": ImageDataGenerator(preprocessing_function=lambda x: tf.image.adjust_saturation(x, 0.90)),
    "zoom_in": ImageDataGenerator(zoom_range=[1.0, 1.2]),
    "zoom_out": ImageDataGenerator(zoom_range=[0.8, 1.0]),
    "shift": ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1),
    "rotate_15": ImageDataGenerator(rotation_range=15),
    "rotate_30": ImageDataGenerator(rotation_range=30),
    "rotate_45": ImageDataGenerator(rotation_range=45),
    "rotate_60": ImageDataGenerator(rotation_range=60),
    "rotate_90": ImageDataGenerator(rotation_range=90),
}


def compute_effective_class_weights(
    df: pd.DataFrame,
    label_column: str,
    beta: float = 0.999,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Computes effective class weights based on the class distribution.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        label_column (str): The label column for computing class weights.
        beta (float, optional): Beta parameter for effective number calculation. Defaults to 0.999.
        normalize (bool, optional): Whether to normalize the weights. Defaults to True.

    Returns:
        Dict[str, float]: A dictionary containing the class weights.
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
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        label_column (str): The label column to compute the imbalance ratio for.

    Returns:
        Dict[str, float]: A dictionary with the imbalance ratio for each class.
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
    Also writes the plan and configuration to a JSON file with a timestamp.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        label (str): The label column to balance.
        target_ratio (float): The desired target ratio for balancing.
        min_samples (int): The minimum number of samples per class.
        config (dict): Configuration dictionary for transformations.

    Returns:
        List[Dict[str, str]]: A list of oversampling instructions (augmentation plans).
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

    # Write oversample map and config to JSON
    config = {
        'label': label,
        'target_ratio': target_ratio,
        'min_samples': min_samples
    }

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    plan_with_config = {"config": config, "oversample_plan": plan.to_dict(orient="records")}
    json_path = Path(UPSAMPLE_JSONS) / f"oversample_plan_{timestamp}.json"
    
    with open(json_path, "w") as f:
        json.dump(plan_with_config, f, indent=4)
    
    # Update the build
    build_updater.write_to(json_path)

    return plan


def _apply_transformation(
        img: Any,
        transformation_key: str
) -> Any:
    """
    Applies a transformation to an image based on the transformation key.

    Args:
        img (Any): The image to transform.
        transformation_key (str): The key to select the transformation from TRANSFORM_GENERATORS.

    Returns:
        Any: The transformed image.
    """
    if transformation_key not in TRANSFORM_GENERATORS:
        raise ValueError(f"Unknown transformation key: {transformation_key}")

    datagen = TRANSFORM_GENERATORS[transformation_key]
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)
    aug_iter = datagen.flow(img_array, batch_size=1)
    aug_img_array = next(aug_iter)[0].astype(np.uint8)

    return array_to_img(aug_img_array)


def oversample_labels(
        label: str,
        output_name: str,
        plan: str
) -> None:
    """
    Applies oversampling transformations on the dataset based on a generated plan.

    Args:
        label (str): The label column to oversample.
        output_name (str): The output name for the generated oversampled dataset.
        plan (str): The path to the oversampling plan (JSON file).
    """

    with open(plan, "r") as f:
        plan_data = json.load(f)

    oversample_plan = plan_data.get("oversample_plan", [])
    augmented_rows = []

    for entry in oversample_plan:
        source = IMAGE_DIR / entry["file_path"]
        filename = f"{source.stem}_{entry[label]}_{entry['transform_key']}{source.suffix}"
        destination = PROCESSED_DIR / filename

        try:
            img = load_img(source)
            new_img = _apply_transformation(img, entry["transform_key"])
            new_img.save(destination)

            augmented_rows.append({
                "rare_species_id": entry['rare_species_id'],
                "file_path": str(filename)
            })

        except Exception as e:
            print(f"Failed on {source.name}: {e}")

    df_aug = pd.DataFrame(augmented_rows)
    filepath = METADATA_DIR / f"{output_name}.csv"
    df_aug.to_csv(filepath, mode='a', index=False)
