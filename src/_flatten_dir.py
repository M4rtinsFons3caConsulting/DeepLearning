""" Moves images to root"""

import os
import shutil
from constants import ROOT_DIR

def move_images_to_root(root_dir):
    for subdir, _, files in os.walk(root_dir):
        if subdir == root_dir:
            continue  # Skip the root directory itself
        for file in files:
            if file.lower().endswith(".jpg"):
                src_path = os.path.join(subdir, file)
                dst_path = os.path.join(root_dir, file)
                
                # Handle potential name collisions
                base, ext = os.path.splitext(file)
                count = 1
                
                while os.path.exists(dst_path):
                    dst_path = os.path.join(root_dir, f"{base}_{count}{ext}")
                    count += 1
                
                shutil.move(src_path, dst_path)
                print(f"Moved: {src_path} -> {dst_path}")

# Example usage
move_images_to_root("/path/to/your/root/folder")
