"""
binary_labeller.py

This script handles interactive binary image labeling for machine learning tasks.
Labels are saved incrementally to a CSV file, and the process can resume 
from where it left off.


Usage:
    python binary_labeller.py --output labels
"""
# Imports
import os
import re
import sys
import argparse
from typing import List, Any, Tuple, Optional

# Third-party 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image  # type: ignore

# Local source
from constants import DATA_DIR, IMAGE_DIR


def _display_image(
      img: Any  # Can be image.Image or bool
    , show: bool = True
    ) -> None:
    """
    Displays or closes an image using matplotlib.
    
    Args:
        img: The image to be displayed or closed (resolved dynamically at runtime).
        show: A boolean flag to show (True) or close (False) the image.
    """
    if show:
        plt.figure('Labeller')
        plt.imshow(img)   
        plt.show(block=False)  # Non-blocking display to continue execution
        
        return input(f'1 = "Animal", 0 = No Animal | Press Enter to Quit.\n')
    else:
        plt.close()  # Close the image


def _get_label(
      img_path: str
    ) -> Tuple[Optional[int], Optional[Any]]:  # Runtime: Tuple[Optional[int], Optional[image.Image]]

    """
    Handles the task-specific logic, including binary labeling or cropping.
    
    Args:
        task_name: The name of the task ('binary' or 'from_crop').
        img_path: Path to the image that needs to be labeled.

    Returns:
        A tuple containing the label (either 0, 1, 'SKIP', or None) and the cropped image (if applicable).
    """
    # Store original image from path
    img = image.load_img(img_path)

    user_input = _display_image(img, show=True)

    while True:
        try:
        
            if user_input == '':  # Exit logic
                return None  
            
            # Check if user input is valid
            if user_input in ['1', '0']:
                return int(user_input)
            else:
                print("Invalid input. Try again.")
        
        finally:
            _display_image(None, show=False)  # Close image on exit


def _get_index(
      output_path: str,
      path_list: List[str],
    ) -> List[str]:
    """
    Reads the CSV to find the last labeled file and returns the remaining paths to be labeled.
    
    Args:
        out_df: A dataframe for outputs
        paths_list: List of image paths to be labeled.
    
    Returns:
        A sliced list of paths to continue from the last labeled file.
    """

    labeled_frame = pd.read_csv(output_path)

    # Ensure the 'image_path' column exists in the labeled_frame
    if 'image_path' not in labeled_frame.columns:
        raise KeyError("'image_path' column not found in the labels CSV.")
     
    # Get the last labeled file from the CSV
    last_labeled_file = labeled_frame['image_path'].iloc[-1]

    # Get the index of the last labeled file in the metadata
    last_labeled_index = path_list.index(last_labeled_file) + 1  # next image
    
    print(f"Resuming from {last_labeled_file}.")
    return path_list[last_labeled_index:]  # Slice to the remaining images


def _parse_labels(
      output_name: str
    ) -> None:
    """
    The main function that handles the labeling process. It checks if a labeling CSV exists and resumes from the last labeled item.
    
    Args:
        output_name: The name of the output CSV file to store the labels.
    """
    # Validate output_name will resolve a safe filename 
    if not re.match(r'^[\w\- ]+$', output_name):
        raise ValueError("Invalid output_name")

    # Load the data
    metapath = os.path.join(DATA_DIR, 'metadata.csv')
    metaframe = pd.read_csv(metapath)

    # Get the file keys
    species_id = metaframe['rare_species_id'].tolist()
    metadata_paths = metaframe['file_path'].tolist()

    # Use output name to construct the path to the labels CSV
    output_path = os.path.join(DATA_DIR, f'{output_name}.csv')

    # Check if the output CSV exists, and is non-empty
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        remaining_paths_list = metadata_paths
    else:
        remaining_paths_list = _get_index(output_path, metadata_paths) 
    
    # Fail safe
    if not remaining_paths_list:
        print("Unable to parse list of paths. Now Exiting")
        sys.exit(-1)

    # Initialize 
    results = []

    # Loop until user breaks, or no more paths in path list
    try:
        for path in remaining_paths_list:
            img_path = os.path.join(IMAGE_DIR, path)

            # Run the labelling loop
            label = _get_label(img_path)
            
            # User decided to break
            if label is None:
                break

            # Append to the results
            results.append({
                'rare_species_id': species_id[metadata_paths.index(path)],
                'image_path': path,
                'is_animal': label
            })
            
    except KeyboardInterrupt: # Catch other user interrupt. 
        print("\n[Interrupted by user. Saving progress...]")

    finally: # Ensure results are always stored.
        if results:
            pd.DataFrame(results).to_csv(output_path, mode='a', index=False)
            print(f"[Saved {len(results)} labels to {output_path}]")
        else:
            print("[No labels to save.]")


if __name__ == "__main__":
    # Set argparse for terminal usage
    parser = argparse.ArgumentParser(
        description='A binary labelling assistant'
        )
    
    parser.add_argument(
        '--output_name', 
        type=str, required=True, 
        help='Name of the desired output file without the extension)'
    )

    # Parse and call function
    args = parser.parse_args()
    _parse_labels(args.output_name)
