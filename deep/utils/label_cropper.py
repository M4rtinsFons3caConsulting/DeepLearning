"""
label_cropper.py

This script handles interactive image labeling for machine learning tasks.
It supports binary classification and random cropping-based tasks.
Labels are saved incrementally to a CSV file, and the process can resume 
from where it left off.

Usage:
    python label_cropper.py --task binary --output labels
"""

# Imports
import os
import re
import sys
import random
import argparse
from typing import List, Tuple, Optional, Any

# Third-party 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image  # type: ignore
from keras.preprocessing.image import smart_resize # type: ignore

# Local source
from constants import DATA_DIR, IMAGE_DIR


def _image_display(
          img1: str
        , img2: str = None
        , show: bool = True
        ) -> None:
    if show:
        plt.figure('Labeller')

        # Display first image
        if img2 is None:
            plt.imshow(img1)
            plt.axis('off')
        else:
            plt.subplot(1, 2, 1)
            plt.imshow(img1)
            plt.axis('off')

            # Display second image
            plt.subplot(1, 2, 2)
            plt.imshow(img2)
            plt.axis('off')

        plt.show(block=False)
        plt.show(block=False)
    else:
        plt.close()


def _crop_image(
    img: Any,               # Runtime: image.Image
    crop_mode: int = 9      # 1 = increase, 0 = decrease, 9 = same
) -> Optional[Any]:
    """
    Crops a random patch from the image with dynamic size adjustment and resizes it to 224x224.
    
    Args:
        img: The image to be cropped.
        crop_mode: 1 to increase crop size, 0 to decrease, 9 to keep default size (224).
    
    Returns:
        Resized cropped image or None if crop isn't possible.
    """
    DEFAULT_SIZE = 224
    MIN_SIZE = 144
    STEP = 16
    MAX_STEP = 5

    # Calculate crop size
    if crop_mode == 1:
        crop_size = DEFAULT_SIZE + STEP * MAX_STEP
    elif crop_mode == 0:
        crop_size = DEFAULT_SIZE - STEP * MAX_STEP
    else:
        crop_size = DEFAULT_SIZE

    img_array = image.img_to_array(img)
    height, width, _ = img_array.shape

    if crop_size < MIN_SIZE or height < crop_size or width < crop_size:
        return None

    x = random.randint(0, width - crop_size)
    y = random.randint(0, height - crop_size)

    cropped = img_array[y:y + crop_size, x:x + crop_size]

    # Resize with smart_resize to 224x224
    resized = smart_resize(cropped, (224, 224))

    return image.array_to_img(resized)


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
    # Crop an image
    img = image.load_img(img_path)
    cropped_img = _crop_image(img)
    
    if cropped_img is None:
        print("Image too small to crop. Skipping.")
        return 'SKIP', None

    while True:
        # Build the user prompt
        _image_display(img, cropped_img, show=True)

        # Parse user input
        user_input = input(f'1 - "Bad Crop or Animal", 0 - "Good Crop and No Animal", 9 - "Skip Image" | Press Enter to Quit.\n')

        # Is exit request
        if user_input == '': # Exit logic
            _image_display(None, show=False) # Close image after input
            return None, None    
         
        elif user_input == '0': # Return Result
            _image_display(None, show=False) # Close previous image after valid input
            return int(user_input), cropped_img
        
        elif user_input == '1': # Try again
            # Get new crop
            while True:
                try:
                    crop_action = input('Next crop action | 1 = zoom_out, 0 = zoom_in, 9 = same\n')
                    crop_action = int(crop_action)
                    break
                except ValueError:
                    print("Invalid input. Try again.")

            cropped_img = _crop_image(img, crop_action)

            # Display new crop
            _image_display(None, show=False) # Close previous image before opening the new one
      
        # Is continue request
        elif user_input == '9': # Return None
            _image_display(None, show=False) # Close previous image before opening the new one
            return 'SKIP', None  

        else:  
            # Otherwise user input is invalid
            print("Invalid input. Try again.") 
    

def _get_index(
      output_path: str,
      path_list: List[str],
    ) -> List[str]:
    """
    Reads the CSV to find the last labeled file and returns the remaining paths to be labeled.
    
    Args:
        output_path: The CSV file where previous labels are stored.
        path_list: List of image paths to be labeled.
    
    Returns:
        A sliced list of paths to continue from the last labeled file.
    """
    
    # Read the CSV with the previously labeled data
    labeled_frame = pd.read_csv(output_path)

    # Ensure the 'file_path' column exists in the labeled_frame
    if 'file_path' not in labeled_frame.columns:
        raise KeyError("'file_path' column not found in the labels CSV.")
    
    # Get the last labeled file from the CSV
    last_labeled_file = labeled_frame['file_path'].iloc[-1]

    # Remove _noanimalcrop if present from the filename (to match original filename)
    last_labeled_file_base = last_labeled_file.replace('_noanimalcrop', '')

    # Get the index of the last labeled file (after modification) in the metadata
    try:
        last_labeled_index = path_list.index(last_labeled_file_base) + 1  # next image
    except ValueError:
        print(f"Could not find {last_labeled_file_base} in the provided path list.")
        return []

    print(f"Resuming from {last_labeled_file_base}.")
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
    metadaframe_slice = metaframe.loc[metaframe['is_animal'] == 1]
    metadata_paths = metadaframe_slice['file_path'].tolist()
    
    # Use output name to construct the path to the labels CSV
    output_path = os.path.join(DATA_DIR, f'{output_name}.csv')

    # Check if the output CSV exists, and is non-empty
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        remaining_paths_list = metadata_paths
        headers = True
    else:
        remaining_paths_list = _get_index(output_path, metadata_paths) 
        headers = False
    
    # Fail safe
    if not remaining_paths_list:
        print("Unable to parse list of paths. Now Exiting")
        sys.exit(-1)

    # Initialize 
    results = []
    crops_to_save = []

    try:
        for path in remaining_paths_list:
            img_path = os.path.join(IMAGE_DIR, path)
            label, cropped_img = _get_label(img_path)

            if label is None:
                break

            if label == 0 and cropped_img is not None:
                base, ext = os.path.splitext(path)
                new_filename = f"{base}_noanimalcrop{ext}"
                save_path = os.path.join(IMAGE_DIR, new_filename)

                crops_to_save.append((cropped_img, save_path))

                results.append({
                    'rare_species_id': species_id[metadata_paths.index(path)],
                    'file_path': new_filename
                })

            if label == 'SKIP':
                continue

    except KeyboardInterrupt:
        print("\n[Interrupted by user. Saving progress...]")

    finally:
        for img, save_path in crops_to_save:
            img.save(save_path)
            print(f"[Saved crop to {save_path}]")

        if results:
            pd.DataFrame(results).to_csv(output_path, mode='a', index=False, header=headers)
            print(f"[Saved {len(results)} labels to {output_path}]")
        else:
            print("[No labels to save.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A crop labelling assistant'
    )
    
    parser.add_argument(
        '--output_name', 
        type=str, required=True, 
        help='Name of the desired output file without the extension)'
    )

    args = parser.parse_args()
    _parse_labels(args.output_name)

