import re
from pathlib import Path

# ------------------ Root ------------------ #
ROOT = Path(__file__).resolve().parent.parent

# ------------------ Data  ------------------ #
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "raw_image_directory"
PROCESSED_DIR = DATA_DIR / "processed_image_directory"

# ------------------ Resources ------------------ #
RESOURCES_DIR = ROOT / "resources"
BINLBL_FILE = RESOURCES_DIR / "additional_resources" / "binary_labels.csv"

# ------------------ Metadata Store ------------------ #
METASTORE_DIR = ROOT / "metadata_store"

CHECKSUM_FILE = METASTORE_DIR / "data_checksums.csv"

METADATA_DIR = METASTORE_DIR / "image_metadata"
METADATA_FILE = METADATA_DIR / "metadata.csv"

CLEANER_JSONS = METASTORE_DIR / "cleaner_logs"
UPSAMPLE_JSONS = METASTORE_DIR / "upsample_logs"
RESULTS_DIR = METASTORE_DIR / "model_results"

SINGATURE_FILE = METADATA_DIR / "current_signature.csv"
SIGNATURE_COLS = ["CLEANER", "UPSAMPLER"]

# ------------------ Zip Instructions ------------------ #
ZIP_FILE_INSTRUCTIONS = {
    "drive_data": (
        RESOURCES_DIR / "drive_data.zip", 
        IMAGE_DIR, 
        True
    ),
    "additional_resources": (
        RESOURCES_DIR / "additional_resources.zip", 
        RESOURCES_DIR, 
        True
    ),
    "cropped_images": (
        RESOURCES_DIR / "additional_resources/cropped_images.zip", 
        IMAGE_DIR, 
        False
    )
}

# ------------------ Move Instructions ------------------ #
MOVE_FILE_INSTRUCTIONS = {
    "metadata": (
        IMAGE_DIR / "metadata.csv",
        METADATA_FILE
    ),
    "cropped_labels": (
        RESOURCES_DIR / "additional_resources/cropped_labels.csv",
        DATA_DIR / "binary_oversample_data.csv"
    )
}

# ------------------ External Resources ------------------ #
DRIVE_ZIP_URL = "https://drive.google.com/uc?export=download&id=1PyxqW_nsORX4PetkQo6OIL0mUL1pFsTD"

# Upsampling mappings
BINARY_UPSAMPLE = RESOURCES_DIR / "binary_upsample_map.json"
FAMILY_UPSAMPLE = RESOURCES_DIR / "family_upsample_map.json"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 20

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

# ------------------ Normalization Constants ------------------ #
# ImageNet normalization in OpenCV BGR order
IMAGENET_NORM = {
    "mean": [0.406, 0.456, 0.485],
    "std": [0.225, 0.224, 0.229]
}
