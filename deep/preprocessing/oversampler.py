"""
image_augmentation.py

This module provides tools for addressing class imbalance in image classification tasks
via oversampling. It computes class weights, imbalance ratios,
and generates additional augmented samples for underrepresented classes.

Functionality includes:
- Calculation of effective class weights using the "effective number" formula.
- Measurement of imbalance ratio across classes.
- Augmentation of images via flipping and rotation to synthetically balance dataset.
- CLI interface for specifying label column and target imbalance ratio.
"""

import os
import argparse
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image    # type: ignore
from tensorflow.keras.preprocessing.image import (  # type:ignore
    ImageDataGenerator,
    img_to_array,
    array_to_img,
    load_img,
) 

from deep.constants import IMAGE_DIR, DATA_DIR


TRANSFORM_GENERATORS: Dict[str, ImageDataGenerator] = {
    "_flip_lr": ImageDataGenerator(horizontal_flip=True),  # Horizontal flip
    # "_flip_tb": ImageDataGenerator(vertical_flip=True),  # Vertical flip
    "_rotate_20": ImageDataGenerator(rotation_range=20),  # Rotate image by 10 degrees
    "_rotate_45": ImageDataGenerator(rotation_range=45),  # Rotate image by 25 degrees
    "_bright_plus": ImageDataGenerator(preprocessing_function=lambda x: x * 1.30),  # Increase brightness by 15%
    "_bright_minus": ImageDataGenerator(preprocessing_function=lambda x: x * 0.70),  # Decrease brightness by 15%
    "_sat_plus": ImageDataGenerator(preprocessing_function=lambda x: tf.image.adjust_saturation(x, 1.30)),  # Increase saturation by 15%
    "_sat_minus": ImageDataGenerator(preprocessing_function=lambda x: tf.image.adjust_saturation(x, 0.70)),  # Decrease saturation by 15%
    "_red_plus": ImageDataGenerator(preprocessing_function=lambda x: x + 0.20 * np.array([1, 0, 0])),  # Increase red channel by 12%
    "_green_plus": ImageDataGenerator(preprocessing_function=lambda x: x + 0.20 * np.array([0, 1, 0])),  # Increase green channel by 12%
    "_blue_plus": ImageDataGenerator(preprocessing_function=lambda x: x + 0.20 * np.array([0, 0, 1])),  # Increase blue channel by 12%
    "_red_minus": ImageDataGenerator(preprocessing_function=lambda x: x - 0.20 * np.array([1, 0, 0])),  # Decrease red channel by 12%
    "_green_minus": ImageDataGenerator(preprocessing_function=lambda x: x - 0.20 * np.array([0, 1, 0])),  # Decrease green channel by 12%
    "_blue_minus": ImageDataGenerator(preprocessing_function=lambda x: x - 0.20 * np.array([0, 0, 1])),  # Decrease blue channel by 12%
}



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


def _oversample(
    target_label: str,
    min_num_samples: int,
    target_ratio: float = 3.0
):
    """
    Perform class balancing by augmenting underrepresented classes using image transformations.

    Args:
        target_label (str): Column name of the label to balance.
        min_num_samples (int): Minimum number of samples desired per class.
        target_ratio (float): The desired maximum ratio of majority to minority class.
    """
    # Load the metadata (CSV file) into a dataframe
    metaframe = pd.read_csv(os.path.join('../', DATA_DIR, 'metadata.csv'))
    
    # Get the count of labels in the dataset
    counts = metaframe[target_label].value_counts()
    max_count = counts.max()  # Find the maximum count for any class
    majority_label = counts.idxmax()  # Find the majority class label

    # Variables to store augmented rows and summary of augmentations
    augmented_rows = []
    augmentation_summary = defaultdict(lambda: defaultdict(int))
    try:
        # Iterate through all classes and apply augmentation as needed
        for label, count in counts.items():
            if label == majority_label:
                continue  # Skip the majority class
            
            # Skip classes that are already within the target ratio or have enough few samples
            current_ratio = max_count / count
            if (current_ratio <= target_ratio) and (count > min_num_samples):
                continue
            
            # Calculate the number of samples we need to generate
            target_count = max(int(max_count / target_ratio), (min_num_samples - count))
            needed = target_count - count
            
            if needed <= 0:
                continue  # No need for further augmentation if already sufficient
            
            # Select samples from the underrepresented class
            class_df = metaframe[metaframe[target_label] == label]
            transform_keys: List[str] = list(TRANSFORM_GENERATORS.keys())  # List of transformation keys

            # Initialize generation variables
            generated = 0
            transform_idx = 0
            
            # Image augmentation loop to generate new samples
            while generated < needed and transform_idx < len(transform_keys):
                transform_key = transform_keys[transform_idx]

                for _ in range(min(needed - generated, len(class_df))):
                    # Randomly select an image from the current class
                    row = class_df.sample(1).iloc[0]
                    img_path = row['file_path']
                    base, ext = os.path.splitext(img_path)

                    # Load the image from disk
                    try:
                        img = load_img(os.path.join('..', IMAGE_DIR, img_path))  # Runtime: keras_image.Image
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue

                    # Apply the selected transformation to the image
                    new_img = apply_transformation(img, transform_key)
                    new_filename = f"{base}{transform_key}{ext}"
                    new_img_path = os.path.join('..', IMAGE_DIR, new_filename) 
                    new_img.save(new_img_path)

                    # Update the metadata with the new image path
                    new_row = row.copy()
                    new_row['file_path'] = new_img_path

                    augmented_rows.append(new_row)

                    # Update the augmentation summary
                    augmentation_summary[label][transform_key] += 1
                    generated += 1

                    # Stop if we've generated enough samples
                    if generated >= needed:
                        break

                transform_idx += 1
    finally:
        # Print augmentation summary after processing all classes
        print("\nAugmentation Summary:")
        for label, transform_counts in augmentation_summary.items():
            print(f"Class '{label}':")
            for transform, count in transform_counts.items():
                print(f"  {transform}: {count} images")

        # Save the updated metadata with the augmented samples
        augmented_df = pd.DataFrame(augmented_rows)
        augmented_df.to_csv(os.path.join('../', DATA_DIR, 'metadata.csv'), mode='a', header=False, index=False)
        print("\nOversampling complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Correct class imbalance for multiple tasks, generating new images and storing them permanently.'
    )

    # CLI argument to specify which label to balance
    parser.add_argument(
        '--label',
        type=str,
        required=True,
        help="The label in the metadata to target."
    )

    # CLI argument to specify the maximum imbalance ratio after augmentation
    parser.add_argument(
        '--target_ratio',
        type=float,
        default=3.0,
        help="The maximum imbalance ratio allowed after augmentation, e.g. 3.0 for 3:1."
    )

    # CLI argument to specify the minimum number of samples required for a class
    parser.add_argument(
        '--min_sample',
        type=int,
        default=30,
        help="The minimum number of samples required for a class to be considered adequately represented."
    )

    args = parser.parse_args()
    _oversample(args.label, args.min_sample, args.target_ratio)
