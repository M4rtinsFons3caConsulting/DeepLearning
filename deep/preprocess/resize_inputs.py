"""
This script handles the resizing of all images in a specified directory using
TensorFlow's smart_resize function. The smart_resize function ensures that images are resized
without distortion, maintaining the aspect ratio based on the target size defined in the 
MODEL_IMAGE_SIZE configuration for the specified model.
"""

import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import smart_resize  # type: ignore
from tensorflow.keras.preprocessing import image as tf_image  # type: ignore
from deep.constants import MODEL_IMAGE_SIZE, INPUT_DIR

def resize_album(
    model: str,
    path: str
):
    """
    Resizes all images in the specified directory to the target size specified for the given model.

    This function will loop through all the image files in the provided directory, apply the smart_resize 
    function from TensorFlow to resize the images according to the size defined in the MODEL_IMAGE_SIZE 
    dictionary for the given model. The resized images are saved back to their original file paths, 
    effectively overwriting the original images.
     """
    
    size = MODEL_IMAGE_SIZE[model]

    try:
        path = Path(path)  # Convert provided string path to Path object
    except ValueError:
        raise ValueError("Could not convert the provided string to a valid path.")

    # Loop through all images in the provided directory
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        
        try:
            # Ensure that the file is an image (could be extended to check file extension)
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Skipping non-image file: {filename}")
                continue

            # Load and convert the image to an array
            img = tf_image.load_img(file_path)  
            img_array = tf_image.img_to_array(img)

            # Apply smart resize
            resized_img = smart_resize(img_array, size)

            # Construct the path to save the resized image in INPUT_DIR
            output_path = os.path.join(INPUT_DIR, filename)

            # Save the resized image, overwrite the original file
            tf_image.save_img(output_path, resized_img)

        except Exception as e:
            print(f"Error processing {output_path}: {e}")
