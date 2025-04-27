"""
This script defines a customizable EfficientNetB4-based binary classification model 
using TensorFlow Keras. The model includes options for L2 regularization and dropout 
in the fully connected layers after feature extraction.

The EfficientNetB4 model is loaded with pre-trained ImageNet weights and the base 
layers are frozen during initial training. The top layers consist of a combination 
of Global Average Pooling and Global Max Pooling outputs, followed by optional 
regularized and dropout-enabled dense layers, ending in a sigmoid activation for 
binary classification tasks.

Functions:
    - efficient_net(regularizer: bool = False, dropout: bool = False): 
        Constructs and returns the compiled EfficientNetB4 model and its configuration.
"""

from tensorflow.keras import regularizers  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (
    Dense, 
    GlobalAveragePooling2D, 
    Input, 
    BatchNormalization, 
    Dropout, 
    Lambda, 
    GlobalMaxPooling2D, 
    Concatenate
)
from tensorflow.keras.applications.efficientnet import (  # type: ignore
    EfficientNetB0,
    EfficientNetB4, 
    preprocess_input
)
from deep.constants import MODEL_IMAGE_SIZE

def efficient_net(
    regularizer: bool = False
    ,dropout: bool = False
):
    """
    Builds an EfficientNetB4-based binary classification model with optional regularization and dropout.

    Args:
        regularizer (bool): If True, applies L2 regularization to the dense layer.
        dropout (bool): If True, applies 50% dropout after batch normalization.

    Returns:
        Tuple[Model, dict]: A compiled Keras Model instance and a dictionary containing 
        the configuration used (regularizer and dropout flags).
    """
    
    # Save model configuration
    config = {
        'regularizer': regularizer,
        'dropout': dropout
    }

    # Set the input
    input_tensor = Input(shape=(*MODEL_IMAGE_SIZE["efficientnetb4"], 3))
    
    # Apply preprocessing to the input tensor
    x = Lambda(preprocess_input)(input_tensor)

    # Load the pre-trained model
    base_model = EfficientNetB4(include_top=False, weights='imagenet', input_tensor=x)
    
    # Freeze layers
    base_model.trainable = False

    # Add top layers
    gap = GlobalAveragePooling2D()(base_model.output)
    gmp = GlobalMaxPooling2D()(base_model.output)
    x = Concatenate()([gap, gmp])
    
    if regularizer:
        x = Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
    else:
        x = Dense(64, activation='relu')(x)
    
    x = BatchNormalization()(x)
    
    if dropout:
        x = Dropout(0.5)(x)
    
    output = Dense(1, activation='sigmoid')(x)  # Binary classification

    binary_model = Model(inputs=input_tensor, outputs=output)

    return binary_model, config
