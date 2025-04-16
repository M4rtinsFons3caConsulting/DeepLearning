"""
_labbeler.py

This script handles interactive image labeling for machine learning tasks.
It supports binary classification and random cropping-based tasks.
Labels are saved incrementally to a CSV file, and the process can resume 
from where it left off.

Usage:
    python label_parser.py --task binary --output labels
"""

# Imports
import os
import re
import random
import argparse
from typing import List, Tuple, Optional, Any

# Third party
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
    else:
        plt.close()  # Close the image


def _crop_image(
      img: Any  # Runtime: image.Image
    ) -> Optional[Any]:  # Runtime: Optional[image.Image]
    """
    Attempts to crop a random 224x224 section from the image, ensuring the image is large enough.
    
    Args:
        img: The image to be cropped.

    Returns:
        The cropped image if successful, or None if the image is too small.
    """
    img_array = image.img_to_array(img)

    height, width, _ = img_array.shape

    crop_size = 224 # Hardcoded value to work with most pre-trained models

    if height < crop_size or width < crop_size:
        return None  # Return None if the image is too small to crop
    
    # Get a random location on the image
    x = random.randint(0, width - crop_size)
    y = random.randint(0, height - crop_size)

    cropped = img_array[y:y + crop_size, x:x + crop_size]

    return image.array_to_img(cropped)  # Return the cropped image


def _from_label_task(
      task_name: str
    , img_path: str
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
    
    # Set the image to display according to task
    if task_name == 'binary':
        img_to_display = img

    elif task_name == 'from_crop':
        cropped_img = _crop_image(img)
        if cropped_img is None:
            print("Image too small to crop. Skipping.")
            return 'SKIP', None
        img_to_display = cropped_img

    _display_image(img_to_display, show=True)  # Show the image

    while True:
        # Build the user prompt
        prompt = f'Is there an animal in this image? | 1 = Animal, 0 = No Animal'
        if task_name == 'from_crop':
            prompt += ', 9 = Skip (if an animal is present will crop again, unless skipped)'
        prompt += ' | Press Enter to Quit.\n'

        user_input = input(prompt)

        # Parse user input
        if user_input == '': # Exit logic
            _display_image(None, show=False) # Close image after input
            return None, None    
            
        if task_name == 'from_crop':
            if user_input in ['0', '1', '9']:
                _display_image(None, show=False) # Close image after valid input

            if user_input == '0': # Return Result 
                return int(user_input), img_to_display
            
            if user_input == '1': # Try again
                img_to_display = _crop_image(img)
                _display_image(img_to_display, show=True)

            if user_input == '9': # Return None
                return 'SKIP', None  
            
        if task_name == 'binary':    
            if user_input in ['1', '0']:
                return int(user_input), None  
        
        # Otherwise user input is invalid
        print("Invalid input. Try again.") 


def _get_index(
      output_path: str
    , paths_list: List[str]
    ) -> List[str]:
    """
    Reads the CSV to find the last labeled file and returns the remaining paths to be labeled.
    
    Args:
        output_path: Path to the output labels CSV file.
        paths_list: List of image paths to be labeled.
    
    Returns:
        A sliced list of paths to continue from the last labeled file.
    """
    # Read the existing labels file to get the last labeled file
    labeled_frame = pd.read_csv(output_path)

    if not labeled_frame.empty:  # Ensure there are labeled items in the CSV
        last_labeled_file = labeled_frame['filename'].iloc[-1]
        try:
            # Get the index of the last labeled file in the metadata
            last_labeled_index = paths_list.index(last_labeled_file) + 1  # +1 to start from the next image
            print(f"Resuming from {last_labeled_file}.")
            return paths_list[last_labeled_index:]  # Slice to the remaining images
        except ValueError:
            print(f"Warning: Last labeled file '{last_labeled_file}' not found in metadata. Starting from the first image.")
            return paths_list  # Start from the beginning if not found
    else:
        print("The labels CSV is empty. Starting from the first image.")
        return paths_list  # Start from the beginning if the CSV is empty


def _parse_labels(
      output_name: str
    , task: str
    ) -> None:
    """
    The main function that handles the labeling process. It checks if a labeling CSV exists and resumes from the last labeled item.
    
    Args:
        output_name: The name of the output CSV file to store the labels.
        task: The task to be performed ('binary' or 'from_crop').
    """
    # Validate output_name will resolve a safe filename 
    if not re.match(r'^[\w\- ]+$', output_name):
        raise ValueError("Invalid output_name")
   
    # Validate task argument will resolve well
    task = task.lower()
    if task not in ['binary', 'from_crop']:
        raise ValueError('Task not supported.')
    
    metapath = os.path.join(DATA_DIR, 'metadata.csv')
    metaframe = pd.read_csv(metapath)
    paths_list = metaframe['file_path'].tolist()

    # Use the user-defined output name to construct the path to the labels CSV
    output_path = os.path.join(DATA_DIR, f'{output_name}.csv')

    # Check if the labels CSV exists and the task is 'binary'
    if os.path.exists(output_path) and task == 'binary':
        paths_list = _get_index(output_path, paths_list)  # Get the index and updated paths list

    results = []

    try:
        for path in paths_list:
            img_path = os.path.join(IMAGE_DIR, path)

            label, cropped_img = _from_label_task(task, img_path)
            if label is None:
                break
            if label == 'SKIP':
                continue

            # Save non-animal cropped image if the task is 'from_crop' and label is 0
            if task == 'from_crop' and label == 0 and cropped_img is not None:
                base, ext = os.path.splitext(path)
                new_filename = f"{base}_noanimalcrop{ext}"
                save_path = os.path.join(IMAGE_DIR, new_filename)
                cropped_img.save(save_path)
                print(f"[Saved crop to {save_path}]")
                
                # Update filename in results to reflect new name
                filename = new_filename
            else:
                filename = path

            # Append to the results
            results.append({'filename': filename, 'label': label})

    except KeyboardInterrupt:
        print("\n[Interrupted by user. Saving progress...]")
    finally:
        if results:
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"[Saved {len(results)} labels to {output_path}]")
        else:
            print("[No labels to save.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A manual labelling command line interface.')
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output CSV file')
    parser.add_argument('--task', type=str, required=True, help='Labeling task description')

    args = parser.parse_args()
    _parse_labels(args.output_name, args.task)


