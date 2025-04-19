from pathlib import Path
import re

# Resolve project Root
ROOT = Path(__file__).resolve().parent.parent

# Resolve critical directories
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "image_directory"

# Precompiled regex patterns used for filename recognition or filtering
REGEX_REF = {
    'crop': re.compile(r'_noanimalcrop'),
    'flip_lr': re.compile(r'_flip_lr'),
    'rotate_10': re.compile(r'_rotate_20'),
    'rotate_45': re.compile(r'_rotate_25'),
    'bright_plus': re.compile(r'_bright_plus'),
    'bright_minus': re.compile(r'_bright_minus'),
    'sat_plus': re.compile(r'_sat_plus'),
    'sat_minus': re.compile(r'_sat_minus'),
    'red_plus': re.compile(r'_red_plus'),
    'green_plus': re.compile(r'_green_plus'),
    'blue_plus': re.compile(r'_blue_plus'),
    'red_minus': re.compile(r'_red_minus'),
    'green_minus': re.compile(r'_green_minus'),
    'blue_minus': re.compile(r'_blue_minus'),
    'all': re.compile(r'[a-zA-Z]')
}
