import os
import argparse
from typing import Dict

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image  # type: ignore
import random

from deep.constants import IMAGE_DIR, DATA_DIR


AUGMENTATIONS = {
    "flip": lambda x: x.transpose(image.FLIP_LEFT_RIGHT),  # Example augmentation function
    "rotate": lambda x: x.rotate(30)  # Another example
}


def compute_effective_class_weights(
    df: pd.DataFrame,
    label_column: str,
    beta: float = 0.999,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute class weights based on the Effective Number of Samples method (Cui et al., 2019).
    
    Args:
        df (pd.DataFrame): DataFrame containing image data.
        label_column (str): Name of the column with class labels.
        beta (float): Smoothing parameter, usually close to 1.0.
        normalize (bool): Whether to normalize weights to have mean 1.0.
        
    Returns:
        dict: class label -> weight
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
          df: pd.DataFrame
        , label_column: str
        ) -> Dict[str, float]:
    """
    Computes the imbalance ratio for each class in the label column of the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the label column.
        label_column (str): The name of the column with class labels.

    Returns:
        dict: class label -> imbalance ratio (max count / min count)
    """
    counts = df[label_column].value_counts()
    max_count = counts.max()
    imbalance_ratios = {label: max_count / count for label, count in counts.items()}
    
    return imbalance_ratios


def _oversample(
      target_label: str
    ):  
    # Load the metadata
    metaframe = pd.read_csv(os.path.join('../',DATA_DIR, 'metadata.csv'))

    # Compute imbalance ratio
    imbalance_ratios = compute_imbalance_ratio(metaframe, target_label)

    # List to hold new rows for augmented images
    augmented_rows = []

    # Define target per class based on imbalance ratiO
    target_imbalance_ratio = max(imbalance_ratios.values())  # max ratio for the majority class

    for label, imbalance_ratio in imbalance_ratios.items():
        # Adjust how many images to generate based on imbalance ratio
        num_to_generate = int((imbalance_ratio / target_imbalance_ratio) * 1000)  # example of target oversampling

        if num_to_generate <= 0:
            continue

        class_df = metaframe[metaframe[target_label] == label]
        
        for _ in range(num_to_generate):
            row = class_df.sample(1).iloc[0]
            img_path = row['file_path']
            original_name = img_path[:-4]
            ext = img_path[-4:]

            try:
                img = image.load_img(f'{IMAGE_DIR}/{img_path}')  # .convert('RGB')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

            # Apply random transformation (augmentation)
            transform_name, transform_fn = random.choice(list(augmentations.items()))
            new_img = transform_fn(img)
            new_filename = f"{original_name}{transform_name}{ext}"
            new_img_path = os.path.join(IMAGE_DIR, new_filename)
            new_img.save(new_img_path)

            # Create a new row for the augmented image and append it to augmented_rows
            new_row = row.copy()
            new_row['file_path'] = new_img_path  # Update the file path to the new image
            augmented_rows.append(new_row)

    # Add augmented rows to the original dataframe
    augmented_df = pd.DataFrame(augmented_rows)
    updated_df = pd.concat([metaframe, augmented_df], ignore_index=True)

    # Save the augmented data back
    updated_df.to_csv(os.path.join(DATA_DIR, 'augmented_metadata.csv'), index=False)
    print("Oversamplingcomplete.")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Correct class imbalance for multiple tasks, generating new images and storing them permanently.'
    )
    
    parser.add_argument(
        '--label',
        type=str,
        required=True,
        help="The classification task for which the class imbalance is being corrected."
    )

    args = parser.parse_args()

    _oversample(args.label)

    # Optionally compute class weights for loss weighting in training
    # weights = compute_effective_class_weights(metaframe, args.label, normalize=True)
    # print("Class weights:", weights)
