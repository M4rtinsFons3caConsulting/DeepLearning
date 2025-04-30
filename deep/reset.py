"""
reset.py – Utility script to flush and recreate specific image-related directories.

This script helps quickly reset the 'original', 'processed', or 'input' directories to an empty state.
It is useful during development or testing, providing a way to clean up and recreate these directories.
"""

import shutil
import argparse
from pathlib import Path
from deep.constants import PROCESSED_DIR, IMAGE_DIR, INPUT_DIR
from deep.utils import build_updater

def reset_dir(dir: str):
    """Eliminates a specified directory and re-creates it in an empty state.

    Args:
        dir (str): The directory to reset. Can be 'original', 'processed', or 'input'.

    Raises:
        ValueError: If the provided directory is not one of 'original', 'processed', or 'input'.
    """

    if dir == "original":
        dir = IMAGE_DIR
    elif dir == "processed":
        dir = PROCESSED_DIR
        build_updater.clean()  # Cleans the current signature CSV
    elif dir == "input":
        dir = INPUT_DIR
        build_updater.clean()  # Cleans the current signature CSV
    else:
        raise ValueError("Unable to parse path for the requested directory, please try again.")

    dir = Path(dir)
    if dir.exists() and dir.is_dir():
        try:
            shutil.rmtree(dir)
            dir.mkdir(parents=True, exist_ok=True)
            print(f"All content in {dir} has been deleted and the directory has been recreated.")
        except Exception as e:
            print(f"Error deleting {dir}: {e}")
    else:
        print(f"{dir} is not a valid directory or doesn't exist.")

def main():
    """Main function to parse arguments and call the reset_dir function.

    The script expects a directory name as an argument ('original', 'processed', or 'input').
    The corresponding directory will be reset.
    """
    
    parser = argparse.ArgumentParser(description="Delete and recreate a directory.")
    parser.add_argument("dir", type=str, help="Path to the directory to clean.")
    args = parser.parse_args()
    dir = args.dir

    reset_dir(dir)

if __name__ == "__main__":
    main()
