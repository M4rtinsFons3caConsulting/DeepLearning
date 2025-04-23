# Deep_Preprocess – Image Preprocessing and Oversampling

This folder contains utility scripts used during the **image preprocessing phase** of dataset preparation.  
> **Note:** This folder **does include** an `__init__.py` file, allowing optional modular imports if needed.

## Included Scripts

- **`image_cleaner.py`**  
  Applies a standardized image preprocessing pipeline, including denoising, contrast normalization, border trimming, and channel normalization.

- **`oversampler.py`**  
  Handles class imbalance by oversampling underrepresented image categories via controlled duplication, when run as 
  main generates a .json that can be used for repreducibility. When run as part of the setup.py workflow reads all
  .json file from resources and generates the corresponding images.

## Usage

Each script can be executed via the command line. To display help or pass runtime arguments, follow this general pattern:

```bash
# For Linux
cd ../data_utils/
python <script_name>.py --help

# For Windows
cd ..\data_utils\
python <script_name>.py --help
