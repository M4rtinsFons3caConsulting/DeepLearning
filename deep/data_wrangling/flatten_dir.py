""" Moves images to relative rood of image directory.

Originally the data was stored using a hierarchy of directories with the following structure: 
    /<image_directory>/<family_fillum>/<eol_content_id>_<eol_eol_page_id>.jpg

;and was kept in this projects /data directory where other data were stored such as metadata 
information and other labelings resulting from our preprocessment. Due to our decision to shift 
towards an architecture that made use of several models however, and depending those models on 
multiple labbelling schemas, we decided to change the image directory to a flat structure that 
allows for image loading using the "flow_from_dataframe" paradigm. 

"""

import os
import shutil
from constants import IMAGE_DIR, DATA_DIR


def _flatten_image_directory():
    """
    Moves all .jpg images from subdirectories to the root of `dir_path`, 
    flattening the folder structure. Assumes all filenames are unique.
    """

    for subdir, _, files in os.walk(DATA_DIR):
        if subdir == DATA_DIR:
            continue  # skip root
        print(f"Visiting: {subdir}")
        for file in files:
            if file.lower().endswith(".jpg"):
                src_path = os.path.join(subdir, file)
                dst_path = os.path.join(DATA_DIR, file)
                shutil.move(src_path, dst_path)
                print(f"Moved: {src_path} -> {dst_path}")


def _delete_empty_subdirs(dir_path):
    """
    Deletes all empty subdirectories inside `dir_path`. 
    If a subdirectory contains files or other folders, raises a warning instead.
    """
    for subdir, subdirs, files in os.walk(DATA_DIR, topdown=False):
        if subdir == DATA_DIR:
            continue  # skip root
        if not subdirs and not files:  # nothing inside
            os.rmdir(subdir)
        else:
            raise UserWarning("Images detected, aborting.")


def _rename_images():
    """
    Renames all .jpg images in `dir_path` by trimming everything after 
    the second underscore in the filename. Assumes all filenames are unique.
    """

    for file in os.listdir(DATA_DIR):
        if file.lower().endswith(".jpg"):
            parts = file.split('_')
            if len(parts) > 2:
                new_name = '_'.join(parts[:2]) + '.jpg'
                old_path = os.path.join(DATA_DIR, file)
                new_path = os.path.join(DATA_DIR, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {file} → {os.path.basename(new_path)}")


def _flatten_metadata():
    """ Updates the provided `metadata.csv` file to support a flat directory structure """
    import pandas as pd

    metapath = os.path.join(DATA_DIR, "metadata.csv")
    metaframe = pd.read_csv(metapath)

    metaframe['file_path'] = (
        metaframe['file_path']
        .apply(lambda x: x.split('/')[-1])            # 1. remove everything before the '/'
        .apply(lambda x: '_'.join(x.split('_')[:2]))  # 2. keep only up to the second underscore
        .apply(lambda x: f"{x}.jpg")                  # 3. append .jpg
)

    metaframe.drop(
        columns=[
              'eol_content_id'
            , 'eol_page_id'
            , 'kingdom'
            ]
        , inplace=True
        )
    
    metaframe.to_csv(metapath, index=False)
    # Done
    
if __name__ == "__main__":
    _flatten_image_directory()
    _delete_empty_subdirs()
    _rename_images()
    _flatten_metadata()

    print("Directory flattened, metadata updated.")