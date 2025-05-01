# Deep_Utils – Labelling Utility Scripts

This folder contains utility scripts used during the development process.  
> **Note:** This folder does **not** include an `__init__.py` file, as these scripts are not intended to be part of a formal module.

## Included Scripts

- **`label_binary.py`**  
  A simple interactive utility for labeling images as either **animal** or **non-animal**.

- **`label_cropper.py`**  
  Used to generate hard negative samples by cropping background regions from suitable source images.

- **`delete_regex.py`**  
  Allows batch deletion of images in a directory by matching filenames against a user-specified regex pattern.

- **`build_updater.py`**
  Utility function that tracks the current treatments applied to a set of images, stored in the 
  `current_signature.csv`.

---

These tools were designed for lightweight, manual labeling and cleanup during early-stage development.  
They are **not meant to be run as part of any routine or automated workflow**.

For more advanced and scalable annotation workflows, consider using:

- [CVAT](https://github.com/opencv/cvat)
- [FiftyOne](https://github.com/voxel51/fiftyone)

While more refined, these tools also require additional setup. For this project's initial implementation, the included scripts were considered sufficient.

## Usage

Each script, apart from `build_updater.py`, can be executed via the command line. To display help or pass runtime arguments, follow this general pattern:


```bash
# For Linux
cd ../deep_utils/
python <script_name>.py --help

# For Windows
cd ..\deep_utils\
python <script_name>.py --help
