"""
directory_formatter.py — Flattens and reshapes the image directory and metadata 
to conform with `flow_from_dataframe` requirements used in Keras.

Overview:
    - Flattens nested image folders into a single directory.
    - Cleans and renames image files to ensure uniqueness and consistency.
    - Updates metadata CSV to reflect the new flat structure.
    - Merges binary labels into the metadata for downstream binary classification tasks.

This script is intended to be run once after initial dataset download and before model training.
"""

import os
import shutil
import pandas as pd
from deep.constants import IMAGE_DIR, METADATA_FILE, REGEX_REF, BINLBL_FILE

def _flatten_image_directory() -> None:
    """
    Flattens the nested structure in IMAGE_DIR by moving all .jpg images
    from subdirectories to the root. Assumes filenames are unique.
    """
    for subdir, _, files in os.walk(IMAGE_DIR):
        if subdir == IMAGE_DIR:
            continue
        print(f"Visiting: {subdir}")
        for file in files:
            if file.lower().endswith(".jpg"):
                src_path = os.path.join(subdir, file)
                dst_path = os.path.join(IMAGE_DIR, file)
                shutil.move(src_path, dst_path)
                print(f"Moved: {src_path} -> {dst_path}")

def _delete_empty_subdirs() -> None:
    """
    Removes all empty subdirectories from IMAGE_DIR. 
    If non-empty folders are found, a warning is raised and deletion is aborted.
    """
    for subdir, subdirs, files in os.walk(IMAGE_DIR, topdown=False):
        if os.path.abspath(subdir) == os.path.abspath(IMAGE_DIR):
            continue
        if not subdirs and not files:
            os.rmdir(subdir)
        else:
            raise UserWarning("Images detected, aborting.")

def _rename_images() -> None:
    """
    Renames .jpg images in IMAGE_DIR by retaining only the first two underscore-separated tokens.
    Skips files containing the 'crop' pattern as defined in REGEX_REF.
    """
    for file in os.listdir(IMAGE_DIR):
        if file.lower().endswith(".jpg"):
            if REGEX_REF["crop"].search(file):
                continue
            parts = file.split("_")
            if len(parts) > 2:
                new_name = "_".join(parts[:2]) + ".jpg"
                old_path = os.path.join(IMAGE_DIR, file)
                new_path = os.path.join(IMAGE_DIR, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {file} → {os.path.basename(new_path)}")

def _reshape_metadata() -> None:
    """
    Updates metadata.csv to reflect the new flat directory structure.
    Cleans path columns, removes unused metadata, and drops duplicates.
    """
    metaframe = pd.read_csv(METADATA_FILE)

    metaframe["file_path"] = (
        metaframe["file_path"]
        .apply(lambda x: x.split("/")[-1])
        .apply(lambda x: "_".join(x.split("_")[:2]))
        .apply(lambda x: f"{x}.jpg")
    )
    print("Metadata reshaped")

    metaframe.drop(columns=["eol_content_id", "eol_page_id", "kingdom"], inplace=True)
    metaframe.drop_duplicates(keep="first", inplace=True)

    print("Duplicate values removed")
    
    metaframe.to_csv(METADATA_FILE, index=False)

def _merge_binary_labels() -> None:
    """
    Merges binary classification labels from BINLBL_FILE into the reshaped metadata.
    Ensures the 'is_animal' column is stored as integer type.
    """
    metaframe = pd.read_csv(METADATA_FILE)
    labelframe = pd.read_csv(BINLBL_FILE)
    labelframe["is_animal"] = labelframe["is_animal"].astype("Int64")

    merged = pd.merge(
        metaframe,
        labelframe,
        on="rare_species_id",
        how="inner"
    )

    merged.to_csv(METADATA_FILE, index=False)

def format_structure() -> None:
    """
    Orchestrates the full directory restructuring pipeline:
        1. Flattens the image directory
        2. Deletes leftover subdirectories
        3. Renames image files
        4. Updates metadata file paths
        5. Merges binary labels
    """
    
    print("Starting directory reshaping routine...")

    print("Step 1 Flattening image directory")
    _flatten_image_directory()

    print("Step 2 Deleting empty subdirectories")
    _delete_empty_subdirs()

    print("Step 3 Renaming images")
    _rename_images()

    print("Step 4 Reshaping metadata")
    _reshape_metadata()

    print("Step 5 Merging binary labels")
    _merge_binary_labels()

    print("Directory reshaping complete.")
