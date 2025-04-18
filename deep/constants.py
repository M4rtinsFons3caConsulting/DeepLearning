"""A constants file for environmental variables, such as root. Used mostly to import, but can also store values."""

ROOT = "."
DATA_DIR = '../data'
IMAGE_DIR = '../data/image_directory'

REGEX_REF = {
    'crop': r'_noanimalcrop',
    'flip_lr': r'_flip_lr',
    'flip_tb': r'_flip_tb',
    'rotate_10': r'_rotate_10',
    'rotate_25': r'_rotate_25',
    'bright_plus': r'_bright_plus',
    'bright_minus': r'_bright_minus',
    'sat_plus': r'_sat_plus',
    'sat_minus': r'_sat_minus',
    'red_plus': r'_red_plus',
    'green_plus': r'_green_plus',
    'blue_plus': r'_blue_plus',
    'red_minus': r'_red_minus',
    'green_minus': r'_green_minus',
    'blue_minus': r'_blue_minus',
}
