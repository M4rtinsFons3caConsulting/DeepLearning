"""
loss_calculator.py - Utility for constructing a weighted categorical focal loss function
for prediction pipeline operations.

This module loads the metadata and corresponding JSON split file to compute effective 
class weights. The weights are then used to instantiate a CategoricalFocalLoss function 
with a fixed gamma value. The function returns this loss object.
"""

import json
from pathlib import Path

import pandas as pd
from deep.modelling.custom_loss import CategoricalFocalLoss
from deep.preprocess.album_augmenter import compute_effective_class_weights
from deep.constants import METADATA_FILE

def get_loss(splits_path: Path):
    """
    Constructs a CategoricalFocalLoss instance using training class frequencies.

    Args:
        splits_path (Path): Path to a JSON file containing 'train_indices' for the training split.

    Returns:
        CategoricalFocalLoss: A focal loss instance with class-specific alpha weights.
    """
    # Load metadata
    data = pd.read_csv(METADATA_FILE)
    data.drop(columns=['phylum', 'is_animal'], inplace=True)

    # Load split indices
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    # Extract training set from indices
    train_indices = splits["train_indices"]
    train_df = data.loc[train_indices].copy()

    # Build label map
    classes = sorted(train_df["family"].unique())
    label_map = {label: idx for idx, label in enumerate(classes)}

    # Compute class weights
    weights_str = compute_effective_class_weights(train_df, 'family')
    weights_idx = {label_map[k]: v for k, v in weights_str.items()}
    alpha = [weights_idx[i] for i in range(len(weights_idx))]

    # Instantiate loss
    loss = CategoricalFocalLoss(gamma=5, alpha=alpha)

    return loss
