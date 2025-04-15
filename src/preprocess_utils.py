import keras
import tensorflow as tf
from _constants import IMAGE_SIZE_STANDARD

def resize_image( 
    image: tf.Tensor,
    model: str = 'VGG16',
    smart: bool = False
) -> tf.Tensor:
    """
    Resizes the image to the target size.

    Args:
        image (tf.Tensor): The image tensor.
        final_size (List[int]): Target size [height, width] for the resized image.

    Returns:
        tf.Tensor: The resized image.
    """
    if smart:
        return keras.preprocessing.image.smart_resize(image,IMAGE_SIZE_STANDARD[model])
    else:    
        image = tf.image.decode_jpeg(image, channels=3) 
        image = tf.image.resize(image, IMAGE_SIZE_STANDARD[model]) 
        
        return image

def rotate_image(
    image: tf.Tensor,
    rotate: int = 0
) -> tf.Tensor:
    """
    Rotates the image counter-clockwise by 90 degrees a specified number of times.

    Args:
        image (tf.Tensor): The image tensor.
        rotate (int): Number of 90-degree counter-clockwise rotations.

    Returns:
        tf.Tensor: The rotated image.
    """
    for _ in range(rotate):
        image = tf.image.rot90(image)
    return image

def load_image(img_path):
    return keras.utils.load_img(img_path)