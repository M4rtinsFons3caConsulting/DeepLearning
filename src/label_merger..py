"""
_label_merger.py

A utility script for merging manual label CSVs with image metadata.
Designed for use in image classification workflows where manual labels
are collected separately and need to be joined to the metadata by ID.

Supports flexible merge keys and strategies, and can be invoked via
command line or reused as a callable function.
"""

import os
import argparse
import pandas as pd
from typing import Literal
from constants import DATA_DIR


def _merge_labels(
    *,
    csv_file: str,
    left_on: str = 'rare_species_id',
    right_on: str = 'image_id',
    how: Literal['inner', 'left', 'right', 'outer'] = 'inner'
) -> pd.DataFrame:
    """
    Merges a label file with metadata based on provided column keys.

    Args:
        csv_file: Name of the label CSV file (without .csv extension).
        left_on: Column in the metadata to merge on.
        right_on: Column in the label file to merge on.
        how: Type of merge to perform (e.g. 'inner', 'outer').

    Returns:
        A pandas DataFrame containing the merged result.
    """

    if not csv_file:
        raise ValueError("csv_file must be provided as a keyword argument.")

    # Load metadata
    metapath = os.path.join(DATA_DIR, 'metadata.csv')
    metaframe = pd.read_csv(metapath)

    # Load labels
    label_path = os.path.join(DATA_DIR, f'{csv_file}.csv')
    labelframe = pd.read_csv(label_path, header=None, names=['filename', 'label'])

    # Merge
    merged = pd.merge(
        metaframe,
        labelframe,
        left_on=left_on,
        right_on=right_on,
        how=how
    )

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge new labels with metadata.')
    parser.add_argument('--csv_file', type=str, required=True, help='Label file (no .csv extension)')
    parser.add_argument('--left_on', type=str, default='rare_species_id', help='Left key for merge')
    parser.add_argument('--right_on', type=str, default='image_id', help='Right key for merge')
    parser.add_argument('--how', type=str, default='inner', help='Merge strategy (inner, left, right, outer)')

    args = parser.parse_args()
    merged_df = _merge_labels(**vars(args))
    print(merged_df.head())
