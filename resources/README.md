# Deep_Utils - Resources

This directory contains additional resources generated throughout the project's development. For consistency, these are stored in compressed `.zip` files. Manual extraction is possible but **not recommended**, as `setup.py` handles all required unpacking and placement automatically.

Following setup, this directory will also include the original dataset archive used for model training.

## Contents

- **`drive_data.zip`**  
  The original dataset archive provided by the professor. Contains the source images used for training and evaluation.

- **`checksums.csv`**  
  A small file used to validate the success of the ingestion routine, before unpacking and proceeding.

- **`additional_resources.zip`**  
  A bundled archive containing all project-generated assets listed below. Used internally by setup routines listed below:

  - **`binary_labels.csv`**  
    Labels each image as either **animal** or **non-animal**, based on content analysis.

  - **`cropped_images.zip`**  
    Archive of cropped background regions extracted from source images.

  - **`cropped_images.csv`**  
    CSV file containing labels for the cropped images included in the above archive.

  - **`binary_oversampling_map`**  
    Defines an oversampling strategy based on the binary labels (animal vs non-animal).

  - **`family_oversampling_map`**  
    Defines an oversampling strategy based on family-level categorization.


---

These resources were created to support the project's data wrangling and augmentation workflows. They supplement the professor's dataset and are integrated automatically by `setup.py`, particularly during the `/wrangle` and `/preprocess` stages.

