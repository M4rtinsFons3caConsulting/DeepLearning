# Deep_Wrangle – Data Ingestion and Wrangling

This directory hosts scripts intended to be run on a fresh setup of the package. They configure the project directory structure, retrieve the required data, and prepare it for use with the `flow_from_dataframe` method of the Keras library. They are called via the `setup` script.

## Included Scripts

- **`directory_formatter.py`**  
  Converts datasets organized for `flow_from_directory` into a flat structure compatible with `flow_from_dataframe`.

- **`drive_loader.py`**  
  Handles downloading and unpacking of data directly from a Google Drive source, applying necessary wrangling to fit expected structure.

- **`resource_unpacker.py`**  
  A small utility that addresses the need to unpack additional resource for the project.

---

These scripts ensure that:

- All required data is pulled from its original source.
- The dataset layout adheres to `flow_from_dataframe` expectations.
- This project's package is installed into the active conda environment.

