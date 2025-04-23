"""
This script ingests the data from from the google drive where the files have been provided, and validates that both they and 
the additional resources created for this project exist in their expected location in the resources directory.

This script is intented to be ran as part of the setup process, for more details please consult the package level README.md 

"""

import gdown
from deep.constants import ZIP_FILE_INSTRUCTIONS, DRIVE_ZIP_URL

def _fetch_zip() -> bool:
    try:
        print("Downloading data from Drive...")
        output_path = ZIP_FILE_INSTRUCTIONS['drive_data'][0]
        gdown.download(url=DRIVE_ZIP_URL, output=str(output_path), quiet=False, fuzzy=True)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def load_files() -> None:
    """
    Manages the ZIP file by checking its checksum and fetching or extracting if necessary.
    """
    drive_zip, _, _ = ZIP_FILE_INSTRUCTIONS['drive_data']
    additional_zip, _, _ = ZIP_FILE_INSTRUCTIONS['additional_resources']
    
    if not drive_zip.exists():
        if not _fetch_zip():
            raise IOError("Unable to finish downloading the zip file. Aborting")
    else:
        print("File found in directory. Skipping download.")

    if not additional_zip.exists():
        raise FileNotFoundError("Critical additional data is missing. Aborting")
