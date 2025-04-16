"""A constants file for environmental variables, such as root. Used mostly to import connstant, but can also store values."""

ROOT = "."
DATA_DIR = '../data'
IMAGE_DIR = '../data/image_directory'


IMAGE_SIZE_STANDARD = {
    'VGG16' :  (224, 224),
    'VGG19' :  [224, 224],
    'Resnet50' : [224, 224]
}