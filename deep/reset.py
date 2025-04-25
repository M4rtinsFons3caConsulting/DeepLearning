from pathlib import Path
import shutil

# Import the directory path
from deep.constants import PROCESSED_DIR

def main():
    # Check if PROCESSED_DIR exists
    if PROCESSED_DIR.exists() and PROCESSED_DIR.is_dir():
        try:
            # Remove the entire contents of the directory by removing the directory itself
            shutil.rmtree(PROCESSED_DIR)
            # Recreate the directory to keep the structure
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            print(f"All content in {PROCESSED_DIR} has been deleted and the directory has been recreated.")
        except Exception as e:
            print(f"Error deleting {PROCESSED_DIR}: {e}")
    else:
        print(f"{PROCESSED_DIR} is not a valid directory or doesn't exist.")

if __name__ == "__main__":
    main()
    
