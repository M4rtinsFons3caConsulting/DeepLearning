"""Binary classifier utility."""

import os
import sys
import csv
import pandas as pd
import tensorflow as tf

from utils.constants import METADATA_PATH
from utils.display_utils import show_resized_await

def parse_labels(
        output_name: str
    ) -> None:
    
    output_file = os.path.dirname(METADATA_PATH) / output_name

    df = pd.read_csv(METADATA_PATH)
    paths_list = df['file_path'].tolist()

    checkpoint = 0

    # open a file to save the answers
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as file:
            # Look for the last line in the file
            checkpoint = sum(1 for _ in file)

    for index, img in enumerate(paths_list):
        if index < checkpoint - 1:
            continue

        img_id = df[df['file_path'] == img]['rare_species_id'].values[0]
        full_path_image = data_path / img
        img_raw = tf.io.read_file(str(full_path_image))
        img_tensor = tf.image.decode_image(img_raw)

        # Show image in the main thread
        show_resized_await(img_tensor, f'Image {index}')

        # Take user input AFTER the image is closed
        while True:
            user_said = input('1 = Animal, 0 = No Animal:\n to see picture again, type R\n if tired, type Q\n')

            if user_said in ['1', '0', 'Q', 'q']:
                break

            if user_said.upper() == 'R':
                show_resized_await(img_tensor, f'Image {index}')
        
        if user_said.upper() == 'Q':
            sys.exit(0)

        # Write the answer to the file
        with open(output_file, 'a') as file:
            file.write(f"{img_id}, {user_said}\n")        
                

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manually label binary presence of animals.")
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Filename (without extension) to save the labels."
    )
    args = parser.parse_args()

    parse_labels(args.output_name)
