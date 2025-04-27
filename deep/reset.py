import shutil
import argparse
from pathlib import Path
from deep.constants import PROCESSED_DIR, IMAGE_DIR, INPUT_DIR

def reset_dir(
      dir: str
    ):
    """Eliminates a specified directory, then re-creates it in an empty state."""

    if dir == "original":
        dir = IMAGE_DIR
    elif dir == "processed":
        dir = PROCESSED_DIR
    elif dir == "input":
        dir = INPUT_DIR
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
    """Main argparser"""
    
    parser = argparse.ArgumentParser(description="Delete and recreate a directory.")
    parser.add_argument("dir", type=str, help="Path to the directory to clean.")
    args = parser.parse_args()
    dir = args.dir

    reset_dir(dir)

if __name__ == "__main__":
    main()
