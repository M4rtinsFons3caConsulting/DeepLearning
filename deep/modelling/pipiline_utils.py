from typing import Tuple, Any
import pandas as pd
from deep.constants import MODEL_IMAGE_SIZE, SEEDS

def split_data(
      data: pd.DataFrame
    , label: str
    , seed: int
    , test_size=0.2
    , val_size=0.15
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

    return train_df, val_df, test_df

# Potentially deprecated
#
# def smart_resize_img(
#       image: Any
#     , model: str 
#     ):
#     """Performs image resizing, based on the image size specifications of the model being used."""
#     from tensorflow.keras.preprocessing.image import smart_resize #type: ignore
#     from deep.constants import MODEL_IMAGE_SIZE 

#     try:
#         image_size = MODEL_IMAGE_SIZE[model] 
#     except KeyError:
#         raise("Invalid model specification, please state the correct model name.")
    
#     resized_img = smart_resize(image, image_size[0])  # Resize image
    
#     return resized_img