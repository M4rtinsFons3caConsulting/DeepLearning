import re
from pathlib import Path

# Resolve project Root
ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "image_directory"
METADATA_FILE = DATA_DIR / "metadata.csv"

# Resources Paths
RESOURCES_DIR = ROOT / "resources"
CHECKSUM_FILE = RESOURCES_DIR / "data_checksums.csv"
BINLBL_FILE = RESOURCES_DIR / "additional_resources/binary_labels.csv"

# Unpackaging instructions
ZIP_FILE_INSTRUCTIONS = \
    {
        "drive_data" : (RESOURCES_DIR / "drive_data.zip", IMAGE_DIR, True),
        "additional_resources" : (RESOURCES_DIR / "additional_resources.zip", RESOURCES_DIR, True),
        "cropped_images" : (RESOURCES_DIR / "additional_resources/cropped_images.zip", IMAGE_DIR, False)
    }

# Drive URL
DRIVE_ZIP_URL = "https://drive.google.com/uc?export=download&id=1PyxqW_nsORX4PetkQo6OIL0mUL1pFsTD"

# Upsampling mappings
BINARY_UPSAMPLE = RESOURCES_DIR / "binary_upsample_map.json"
FAMILY_UPSAMPLE = RESOURCES_DIR / "family_upsample_map.json"

# Precompiled regex patterns used for filename recognition or filtering
REGEX_REF = {
    'crop': re.compile(r'crop'),
    'flip_lr': re.compile(r'_flip_lr'),
    'bright_plus': re.compile(r'_bright_plus'),
    'bright_minus': re.compile(r'_bright_minus'),
    'sat_plus': re.compile(r'_sat_plus'),
    'sat_minus': re.compile(r'_sat_minus'),
    'zoom_in': re.compile(r'_zoom_in'),
    'zoom_out': re.compile(r'_zoom_out'),
    'shift': re.compile(r'_shift'),
    'rotate_15': re.compile(r'_rotate_15'),
    'rotate_30': re.compile(r'_rotate_30'),
    'rotate_45': re.compile(r'_rotate_45'),
    'rotate_60': re.compile(r'_rotate_60'),
    'rotate_90': re.compile(r'_rotate_90'),
    'all': re.compile(r'[a-zA-Z]')
}

# ImageNet normalization constants, in CV2 format i.e. BGR
IMAGENET_NORM = {
    "mean": [0.406, 0.456, 0.485],
    "std": [0.225, 0.224, 0.229]
}
