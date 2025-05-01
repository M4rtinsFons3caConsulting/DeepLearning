"""
directory_maker.py - Handles directory creation on setup

Ensures that all required directories for running the package and associated notebooks exist.

This script should be executed during setup to prepare the file system structure needed for 
downstream processing, model training, and resource management.
"""

from deep.constants import MAKE_DIR_LIST

def make_dir():
    """
    Creates the necessary directories for downstream processes.

    Iterates through each path in MAKE_DIR_LIST and creates it if it doesn't already exist.
    """
    # Iterate through each path and create it if it doesn't exist
    for path in MAKE_DIR_LIST:
        path.mkdir(parents=True, exist_ok=True)  # Create directories
        print(f"Created directory: {path}")
