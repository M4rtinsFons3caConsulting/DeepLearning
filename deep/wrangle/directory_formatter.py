"""
Moves images to relative root of image directory.

Originally, data was stored in a hierarchy:
    /<image_directory>/<family_fillum>/<eol_content_id>_<eol_page_id>.jpg

It resided in this project’s `/data` directory with other metadata and labelings.
Due to a shift to multiple model architectures and varied labeling schemas,
the structure was flattened to support the `flow_from_dataframe` paradigm.
"""

import os
import shutil
import pandas as pd
from deep.constants import IMAGE_DIR, METADATA_FILE, REGEX_REF, BINLBL_FILE

def _flatten_image_directory() -> None:
    """
    Moves all .jpg images from subdirectories to the root of `IMAGE_DIR`, 
    flattening the folder structure. Assumes all filenames are unique.
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
    Deletes all empty subdirectories inside `IMAGE_DIR`. 
    If any subdirectory contains files or folders, raises a warning.
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
    Renames .jpg images in `IMAGE_DIR` by trimming everything after the 
    second underscore. Skips files matching '_noanimalcrop'.
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
    """Updates `metadata.csv` to support a flat directory structure."""
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
    """Merges binary labels with the reshaped metadata."""
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
