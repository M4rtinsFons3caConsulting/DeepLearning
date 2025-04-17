"""
Removes image files whose filenames match a given regex pattern.

This script is intended to assist in dataset curation by removing unwanted or mislabelled 
images from the image directory based on their filenames. It was created to support flexible, 
regex-based pruning of image files after the flattening of the image directory structure. 

Originally, images were organized hierarchically by taxonomy, but with the shift to a 
flat directory structure optimized for multi-schema labelling and use with 
`flow_from_dataframe`, a need emerged for lightweight and pattern-driven data cleaning. 

This utility complements the revised workflow by enabling targeted removal of files 
using safe and user-supplied regular expressions.
"""

import re
import os
import argparse
from constants import IMAGE_DIR

class InvalidRegexPattern(Exception):
    pass

def get_safe(pattern: str) -> re.Pattern:
    try:
        compiled = re.compile(pattern)
        return compiled
    except re.error:
        raise InvalidRegexPattern("Unsafe or invalid regex pattern detected. Aborting.")

def delete_files_by_regex(compiled_pattern: re.Pattern):
    for filename in os.listdir(IMAGE_DIR):
        # Set file_path
        file_path = os.path.join(IMAGE_DIR, filename)

        # Without extension
        name_without_ext, _ = os.path.splitext(filename)

        if compiled_pattern.search(name_without_ext):
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Delete files in IMAGE_DIR that match a regex pattern.'
    )
    
    parser.add_argument(
        '--regex_pattern',
        type=str,
        required=True,
        help='A valid regex pattern to match filenames for deletion.'
    )

    args = parser.parse_args()

    pattern = get_safe(args.regex_pattern)
    delete_files_by_regex(pattern)
