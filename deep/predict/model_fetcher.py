"""
fetch_model.py - Downloads pretrained model weights from Google Drive.

This script uses the gdown library to download model weight files from 
Google Drive links defined in the WEIGHTS_DICT constant. It stores the 
downloaded file in the current working directory and names it based on 
the selected model. If the download is successful, the function returns 
the full path to the saved file; otherwise, it returns None.
"""

import os
import gdown
from deep.constants import WEIGHTS_DICT

def fetch_model(
        model      
    ) -> str | None:
    """
    Attempts to download a model from the linked drive.

    Returns:
        str | None: Path to the downloaded file if successful, otherwise None.
    """
    try:
        print("Downloading data from Drive...")
        output_path = os.path.join(os.curdir, f"{model}.h5")
        gdown.download(url=WEIGHTS_DICT[model], output=output_path, quiet=False, fuzzy=True)
        print("Download complete.")
        return output_path
    
    except Exception as e:
        print(f"Download failed: {e}")
        return None
