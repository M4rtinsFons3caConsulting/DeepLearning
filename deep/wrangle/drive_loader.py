"""
drive_loader.py

Handles loading and verification of ZIP data files from both Google Drive and local resources.

This module is used during the setup process to ensure that required datasets and resources are 
available and correctly located in the project’s resources directory.

Main Features:
- Downloads a ZIP file from Google Drive if it is not found locally.
- Validates the presence of essential additional resource files.
- Aborts with clear errors if required files are missing.

Refer to the package-level README.md for usage instructions.
"""


import gdown
from deep.constants import ZIP_FILE_INSTRUCTIONS, DRIVE_ZIP_URL

def _fetch_zip() -> bool:
    """
    Attempts to download the primary ZIP file from Google Drive.

    Returns:
        bool: True if the download was successful, False otherwise.

    Notes:
        - Uses gdown to download the file using a shared Drive URL.
        - Output path is defined in the ZIP_FILE_INSTRUCTIONS constant.
    """
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
    Ensures all required ZIP files are present and valid before proceeding.

    Actions:
        - Downloads the Drive ZIP file if not already present locally.
        - Verifies the existence of the additional resources ZIP.
    
    Raises:
        IOError: If the Drive ZIP download fails.
        FileNotFoundError: If additional resources are missing.
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
