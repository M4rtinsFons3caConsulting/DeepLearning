import os
from PIL import Image
import pandas as pd

def display_image(image_path):
    """Open and display an image."""
    img = Image.open(image_path)
    img.show()

def parse_labels(output_name):
    df = pd.read_csv('metadata.csv')  # Adjust to your metadata file path
    paths_list = df['file_path'].tolist()

    output_file = os.path.join(os.getcwd(), output_name + '.txt')

    for index, img in enumerate(paths_list):
        full_path_image = os.path.join('data_path', img)  # Adjust to your image folder path

        # Display the image
        display_image(full_path_image)

        while True:
            user_said = input('1 = Animal, 0 = No Animal, Q = Quit:\n')

            if user_said in ['1', '0', 'Q', 'q']:
                break

        if user_said.upper() == 'Q':
            break

        # Save the label to the file
        with open(output_file, 'a') as file:
            file.write(f"{img}, {user_said}\n")

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
