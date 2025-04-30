"""
delete_regex.py - Removes images from a specified directory whose filenames match a given regex pattern.

This utility helps manage image files during offline augmentation by enabling the removal of specific images 
based on regex pattern matching. It assists in the iterative process of quality assessment during augmentation 
and cleaning routines and their applications to images, by ensuring that directories do not get littered with
undesired images after tests.
"""

import re
import os
import argparse
from deep.constants import IMAGE_DIR, PROCESSED_DIR, INPUT_DIR

class InvalidRegexPattern(Exception):
    """
    Custom exception raised when an invalid regex pattern is detected.
    """
    pass

def get_safe(pattern: str) -> re.Pattern:
    """
    Compiles and returns a safe regex pattern.

    This function attempts to compile the provided regex pattern. If the pattern is invalid, it raises an 
    `InvalidRegexPattern` exception.
    """
    try:
        compiled = re.compile(pattern)
        return compiled
    except re.error:
        raise InvalidRegexPattern("Unsafe or invalid regex pattern detected. Aborting.")

def delete_files_by_regex(compiled_pattern: re.Pattern, dir: str) -> None:
    """
    Deletes files from the specified directory whose filenames match the given regex pattern.
    """
    if dir == "original":
        dir = IMAGE_DIR
    elif dir == "processed":
        dir = PROCESSED_DIR
    elif dir == "input":
        dir = INPUT_DIR
    else:
        raise ValueError("Unable to parse path for the requested directory, please try again.")

    for filename in os.listdir(dir):

        file_path = dir / filename
        name_without_ext, _ = os.path.splitext(filename)
        
        if compiled_pattern.search(name_without_ext):
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Delete files in specified directory that match a regex pattern.'
    )
    
    parser.add_argument(
        '--regex_pattern',
        type=str,
        required=True,
        help='A valid regex pattern to match filenames for deletion.'
    )

    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        choices=['original', 'processed', 'input'],
        help='The directory to search for files to delete. Choose from ["original", "processed", "input"].'
    )

    args = parser.parse_args()

    pattern = get_safe(args.regex_pattern)
    delete_files_by_regex(pattern, args.dir)
