import json
import pandas as pd
from datetime import datetime
from typing import Tuple
from deep.constants import SPLITTER_JSONS 
from deep.utils import build_updater

def save_split_info(
        train_df
        , val_df
        , test_df
    ):

    split_info = {
        "train_indices": train_df.index.tolist(),
        "val_indices": val_df.index.tolist(),
        "test_indices": test_df.index.tolist()
    }

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")    

    json_path = SPLITTER_JSONS / f"split_{timestamp}"

    with open(json_path, "w") as f:
        json.dump(split_info, f, indent=4)

    return json_path

def split_data(
      data: pd.DataFrame
    , label: str
    , seed: int
    , test_size=0.2
    , val_size=0.15
    , save_split=True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Wrapper method around sklearn's train test split for ease of use
    """
    from sklearn.model_selection import train_test_split
    # Performing the splits
    train_df, test_df = train_test_split(
        data
        , test_size=test_size
        , stratify=data[label]
        , random_state=seed
    )  # Create test set
    
    train_df, val_df = train_test_split(
        train_df
        , test_size=val_size
        , stratify=train_df[label]
        , random_state=seed
    )  # Create train and validation set

    if save_split:
        path = save_split_info(train_df, test_df, val_df)
        build_updater.write_to(path)

    return train_df, val_df, test_df
