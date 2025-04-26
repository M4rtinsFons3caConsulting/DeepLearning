"""
This tiny loop ensures that all directories to run the package, and its notebooks exist.
"""
from deep.constants import MAKE_DIR_LIST

def make_dir():
    """Creates the necessary directories for downstream processes"""
    # Iterate through each path and create it if it doesn't exist
    for path in MAKE_DIR_LIST:
        path.mkdir(parents=True, exist_ok=True)  # Create directories
        print(f"Created directory: {path}")
