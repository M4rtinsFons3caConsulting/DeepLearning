"""
oversampler.py

This module provides tools to address class imbalance in image classification tasks
through intelligent oversampling and data augmentation strategies.

It supports:
- Calculation of class weights using the "effective number" method for imbalanced learning.
- Analysis of class imbalance ratios.
- Generation of synthetic training examples for minority classes using image augmentations
  such as flipping, rotation, brightness, saturation, and color shifts.
- A CLI interface to customize label selection, imbalance correction thresholds, and reproducibility.

Outputs include augmented images and metadata plans for reproducible experiments.
"""
# Built-in and STL
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# 3rd party
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (  # type:ignore
    ImageDataGenerator,
    img_to_array,
    array_to_img,
    load_img,
) 

# Root package
from deep.constants import IMAGE_DIR, DATA_DIR


# Transformations dict for ImageGenerator
TRANSFORM_GENERATORS: Dict[str, ImageDataGenerator] = {
    "flip_lr": ImageDataGenerator(horizontal_flip=True), # Horizontal flip
    "rotate_20": ImageDataGenerator(rotation_range=20),  # Rotate image by 10 degrees
    "rotate_45": ImageDataGenerator(rotation_range=45),  # Rotate image by 25 degrees
    "bright_plus": ImageDataGenerator(preprocessing_function=lambda x: x * 1.10),   # Increase brightness by 15%
    "bright_minus": ImageDataGenerator(preprocessing_function=lambda x: x * 0.90),  # Decrease brightness by 15%
    "sat_plus": ImageDataGenerator(preprocessing_function=lambda x: tf.image.adjust_saturation(x, 1.15)),   # Increase saturation by 15%
    "sat_minus": ImageDataGenerator(preprocessing_function=lambda x: tf.image.adjust_saturation(x, 0.85)),  # Decrease saturation by 15%
    "ed_plus": ImageDataGenerator(preprocessing_function=lambda x: x + 0.15 * np.array([1, 0, 0])),        # Increase red channel by 12%
    "green_plus": ImageDataGenerator(preprocessing_function=lambda x: x + 0.15 * np.array([0, 1, 0])),      # Increase green channel by 12%
    "blue_plus": ImageDataGenerator(preprocessing_function=lambda x: x + 0.15 * np.array([0, 0, 1])),       # Increase blue channel by 12%
    "red_minus": ImageDataGenerator(preprocessing_function=lambda x: x - 0.15 * np.array([1, 0, 0])),       # Decrease red channel by 12%
    "green_minus": ImageDataGenerator(preprocessing_function=lambda x: x - 0.15 * np.array([0, 1, 0])),     # Decrease green channel by 12%
    "blue_minus": ImageDataGenerator(preprocessing_function=lambda x: x - 0.15 * np.array([0, 0, 1])),      # Decrease blue channel by 12%
}


def compute_effective_class_weights(
    df: pd.DataFrame,
    label_column: str,
    beta: float = 0.999,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute class weights using the "effective number" method to better handle class imbalance.

    Args:
        df (pd.DataFrame): The dataframe containing labeled data.
        label_column (str): Column name that contains the class labels.
        beta (float): Smoothing hyperparameter for effective number calculation.
        normalize (bool): Whether to normalize weights to mean 1.

    Returns:
        Dict[str, float]: Mapping of class labels to their computed weights.
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


def apply_transformation(
    img: Any,  # Runtime: keras_image.Image
    transformation_key: str
) -> Any:     # Runtime: keras_image.Image
    """
    Applies the specified transformation to a PIL image using ImageDataGenerator.

    Args:
        img (keras_image.Image): The input image as a PIL.Image object.
        transformation_key (str): The key corresponding to the transformation.

    Returns:
        keras_image.Image: The augmented image as a PIL.Image object.
    """
    if transformation_key not in TRANSFORM_GENERATORS:
        raise ValueError(f"Unknown transformation key: {transformation_key}")

    datagen = TRANSFORM_GENERATORS[transformation_key]
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)  # Adding batch dimension

    aug_iter = datagen.flow(img_array, batch_size=1)
    aug_img_array = next(aug_iter)[0].astype(np.uint8)

    return array_to_img(aug_img_array)


def compute_imbalance_ratio(
    df: pd.DataFrame,
    label_column: str
) -> Dict[str, float]:
    """
    Calculate the imbalance ratio of each class relative to the majority class.

    Args:
        df (pd.DataFrame): The dataframe containing labeled data.
        label_column (str): Column name that contains the class labels.

    Returns:
        Dict[str, float]: Mapping of class labels to imbalance ratios.
    """
    counts = df[label_column].value_counts()
    max_count = counts.max()
    return {label: max_count / count for label, count in counts.items()}


def _map_oversample(
    df: pd.DataFrame,
    label: str,
    target_ratio: float,
    min_samples: int
) -> List[Dict[str, str]]:
    """
    Create a reproducible augmentation plan to synthetically increase representation
    of underrepresented classes.

    Args:
        df (pd.DataFrame): The metadata dataframe containing file paths and labels.
        label (str): Column name of the label used for balancing.
        target_ratio (float): Desired max imbalance ratio (majority/minority) after augmentation.
        min_samples (int): Minimum acceptable sample count per class.

    Returns:
        List[Dict[str, str]]: A plan detailing which images to augment, with which transformations,
        and for which classes.
    """

    # Obtain value counts, and majority labels
    counts = df[label].value_counts()
    max_count = counts.max()
    majority_label = counts.idxmax()
    plan = []

    # For each class label
    for cls_label, count in counts.items():
        # Ignore if majority
        if cls_label == majority_label:
            continue
        
        # Set the current ratio
        current_ratio = max_count / count
        if (current_ratio <= target_ratio) and (count > min_samples):
            continue
        
        # Obtain the target count of samples to make
        target_count = max(int(max_count / target_ratio), (min_samples - count))
        needed = target_count - count
        if needed <= 0:
            continue
        
        # Initialize the relevant slice of the data, generators and counters
        class_df = df[df[label] == cls_label]
        transform_keys = list(TRANSFORM_GENERATORS.keys())
        generated = 0
        transform_idx = 0

        # Loop while
        while generated < needed and transform_idx < len(transform_keys):
            # Set the transformation to apply
            transform_key = transform_keys[transform_idx]
            
            # Apply for either number needed or lenght of class
            for _ in range(min(needed - generated, len(class_df))):
                # Sample a random row from the class
                row = class_df.sample(1).iloc[0]
                
                # Write transformation to plan
                plan.append({
                    "rare_species_id": row["rare_species_id"],     # Unique ID
                    "file_path": row["file_path"],                 # Path to file
                    "transform_key": transform_key                 # Transformation to apply
                })

                # Halting condition
                generated += 1
                if generated >= needed:
                    break
            
            # Get next ImageGenerator instance
            transform_idx += 1

    return plan


def _oversample(
    output_name: str,
    label: str,
    min_sample: int,
    reproduce: bool,
    target_ratio: float = 3.0
):
    # Normalize output_name (drop any existing extension)
    output_name = Path(output_name).stem

    # Build plan and metadata paths
    plan_path     = DATA_DIR / f"{output_name}.json"
    metadata_path = DATA_DIR / f"{output_name}.csv"

    # Load or create the plan
    if reproduce:
        if not plan_path.exists():
            raise FileNotFoundError(f"No existing plan at {plan_path!r}")
        with plan_path.open("r") as f:
            plan = json.load(f)
    else:
        # read metadata
        meta = pd.read_csv(DATA_DIR / "metadata.csv")
        if label != "is_animal":
            meta = meta[meta["is_animal"] == 1]

        plan = _map_oversample(
            df=meta,
            label=label,
            target_ratio=target_ratio,
            min_samples=min_sample
        )
        # save plan for reproducibility
        plan_path.write_text(json.dumps(plan, indent=2))
        print(f" Created plan with {len(plan)} entries {plan_path}")

    # Prepare metadata CSV
    write_header = not metadata_path.exists()
    mode = "w" if write_header else "a"

    # Iterate, generate images and collect new rows
    augmented_rows = []
    # summary = defaultdict(lambda: defaultdict(int))
    
    for entry in plan:
        rare_id = entry["rare_species_id"]
        img_src_rel = entry["file_path"]
        transform = entry["transform_key"]

        # Resolve source and target paths
        src = IMAGE_DIR / img_src_rel
        out_filename = f"{src.stem}_{label}_{transform}{src.suffix}"
        dst = IMAGE_DIR / out_filename

        try:
            img = load_img(src)
            new_img = apply_transformation(img, transform)
            new_img.save(dst)

            augmented_rows.append({
                "rare_species_id": rare_id,
                "file_path": str(out_filename)
            })

        except Exception as e:
            print(f"\u26A0\uFE0F  Failed on {src.name}: {e}")

    # 4) Flush to CSV
    df_aug = pd.DataFrame(augmented_rows)
    df_aug.to_csv(metadata_path, mode=mode, header=write_header, index=False)
    print(f" Appended {len(df_aug)} rows to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Correct class imbalance for multiple tasks, generating new images and storing them permanently.'
    )

    parser.add_argument(
        '--output_name',
        type=str,
        required=True,
        help="The name of the output file to be generated."
    )

    # CLI argument to specify which label to balance
    parser.add_argument(
        '--label',
        type=str,
        required=True,
        help="The label in the metadata to target."
    )

     # CLI argument to specify the minimum number of samples required for a class
    parser.add_argument(
        '--min_sample',
        type=int,
        default=30,
        help="The minimum number of samples required for a class to be considered adequately represented."
    )

     # CLI argument to specify the minimum number of samples required for a class
    parser.add_argument(
    "--reproduce",
    action="store_true",
    help="Reuse a previously saved augmentation plan"
    )
    
    # CLI argument to specify the maximum imbalance ratio after augmentation
    parser.add_argument(
        '--target_ratio',
        type=float,
        default=3.0,
        help="The target imbalance ratio of the minority sought through augmentation, e.g. if 3:1, if 3 is passed."
    )

    # Pass all arguments through
    _oversample(**vars(parser.parse_args()))