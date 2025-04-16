"""
Script corrects class imabalance

# 📘 Cui et al., 2019
# Cui, Yin, et al.
# "Class-balanced loss based on effective number of samples."
# CVPR 2019.

# Uses IR to describe long-tail distributions in datasets like iNaturalist, ImageNet-LT, and CIFAR-LT.
# Shows that simple IR (majority/minority) is not enough — proposes "effective number of samples" as a better measure for class balancing.
# Still uses imbalance ratio to define dataset difficulty.

"""
import os
import random
import argparse
from typing import List, Dict
from collections import Counter

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image #type: ignore

from constants import IMAGE_DIR, DATA_DIR


def compute_effective_class_weights(
        df: pd.DataFrame,
        label_column: str,
        beta: float = 0.999,
        normalize: bool = True
    ) -> Dict[str, float]:
    """
    Compute class weights based on the Effective Number of Samples method (Cui et al., 2019).
    
    Parameters:
        df (pd.DataFrame): DataFrame containing image data.
        label_column (str): Name of the column with class labels.
        beta (float): Smoothing parameter, usually close to 1.0.
        normalize (bool): Whether to normalize weights to have mean 1.0.
        
    Returns:
        dict: class label -> weight
    """

    def effective_num(n, beta):
        return (1 - beta**n) / (1 - beta)
    
    label_counts = Counter(df[label_column])
    effective_nums = {cls: effective_num(n, beta) for cls, n in label_counts.items()}
    
    total_effective = sum(effective_nums.values())
    class_weights = {cls: total_effective / e for cls, e in effective_nums.items()}

    if normalize:
        mean_weight = np.mean(list(class_weights.values()))
        class_weights = {cls: w / mean_weight for cls, w in class_weights.items()}

    print(class_weights)

    raise('oops')
    return class_weights


def compute_imbalance_ratio():
    # TODO: 
    pass

def _oversample(
        target_label: str
    ):  

    # Load the metadata
    metaframe = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
    
    # Identify minority classes
    compute_effective_class_weights(metaframe, target_label)
    
    # According 
    # List to hold new rows for augmented images
    augmented_rows = []

    # for label in minority_classes:
    #     class_df = df[df['label'] == label]
    #     num_to_generate = target_per_class

    #     if num_to_generate <= 0:
    #         continue

    #     label_dir = os.path.join(output_dir, str(label))
    #     os.makedirs(label_dir, existok=True)

    #     for _ in range(num_to_generate):
    #         row = class_df.sample(1).iloc[0]
    #         img_path = row['file_path']
    #         original_name = img_path[:-4]
    #         ext = img_path[-4:]

    #         try:
    #             img = image.load(f'{IMAGE_DIR}/{img_path}')  # .convert('RGB')
    #         except Exception as e:
    #             print(f"Error loading {img_path}: {e}")
    #             continue

    #         # Apply random transformation
    #         transform_name, transform_fn = random.choice(list(transformations.items()))
    #         new_img = transform_fn(img)
    #         new_filename = f"{originalname}{transform_name}{ext}"
    #         new_img_path = new_filename
    #         new_img.save(f'{IMAGE_DIR}/{new_filename}')

    #         # Create a new row for the augmented image and append it to augmented_rows
    #         new_row = row.copy()
    #         new_row['file_path'] = new_img_path  # Update the file path to the new image
    #         augmented_rows.append(new_row)

    # # Add augmented rows to the original dataframe
    # augmented_df = pd.DataFrame(augmented_rows)
    # updated_df = pd.concat([df, augmented_df], ignore_index=True)


       #  results.to_csv(IMAGE_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Util corrects class imbalance for multiple tasks, generating new images and storing them permanently.'
        )
    
    parser.add_argument(
        '--label',
        type=str,
        required=True,
        help=f"The classification task for which the class imbalance is being corrected."
    )

    args = parser.parse_args()
    _oversample(args.label)

else:
    pass