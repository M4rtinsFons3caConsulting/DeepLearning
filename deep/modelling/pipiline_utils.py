"""

pipeline_utils.py - provides utility functions for splitting datasets into training, validation, and test sets 
and saving the indices of these splits to a JSON file.

"""

import json
import pandas as pd
from datetime import datetime
from typing import Tuple
from deep.constants import SPLITTER_JSONS 
from deep.utils import build_updater

def save_split_info(
        train_df: pd.DataFrame
        , val_df: pd.DataFrame
        , test_df: pd.DataFrame
    ) -> str:
    """
    Saves the indices of the train, validation, and test splits to a JSON file.

    This function takes the DataFrames of training, validation, and test datasets, 
    extracts their indices, and saves them in a JSON file with a timestamped filename 
    in the directory defined by `SPLITTER_JSONS`.

    Args:
        train_df (pd.DataFrame): The training dataset.
        val_df (pd.DataFrame): The validation dataset.
        test_df (pd.DataFrame): The testing dataset.

    Returns:
        str: The path to the JSON file where split information is saved.
    """
    split_info = {
        "train_indices": train_df.index.tolist(),
        "val_indices": val_df.index.tolist(),
        "test_indices": test_df.index.tolist()
    }

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")    

    json_path = SPLITTER_JSONS / f"split_{timestamp}"

    # Write split info to JSON file
    with open(json_path, "w") as f:
        json.dump(split_info, f, indent=4)

    return json_path

def split_data(
      data: pd.DataFrame
    , label: str
    , seed: int
    , test_size: float = 0.2
    , val_size: float = 0.15
    , save_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training, validation, and test sets.

    This function wraps sklearn's `train_test_split` method to perform the following operations:
    - First, it splits the data into training and test datasets.
    - Then, it further splits the training data into training and validation sets.
    - Optionally, it can save the split indices to a JSON file.

    Args:
        data (pd.DataFrame): The dataset to split.
        label (str): The column name in the DataFrame used as the label for stratified splitting.
        seed (int): The random seed for reproducibility.
        test_size (float): The proportion of data to be used for the test set (default is 0.2).
        val_size (float): The proportion of training data to be used for the validation set (default is 0.15).
        save_split (bool): Whether or not to save the split indices to a JSON file (default is True).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train, validation, and test DataFrames.
    """
    from sklearn.model_selection import train_test_split

    # Split into train and test sets
    train_df, test_df = train_test_split(
        data
        , test_size=test_size
        , stratify=data[label]
        , random_state=seed
    ) 
    
    # Further split train set into train and validation sets
    if val_size > 0:
        train_df, val_df = train_test_split(
            train_df
            , test_size=val_size
            , stratify=train_df[label]
            , random_state=seed
        )  # Create train and validation set

    # Optionally save the split information
    if save_split:
        path = save_split_info(train_df, test_df, val_df)
        build_updater.write_to(path)

    if val_size > 0:
        return train_df, val_df, test_df
    
    else:
        return train_df, test_df
    