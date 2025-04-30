"""
This script defines a customizable EfficientNetB4-based classification model 
using TensorFlow Keras. The model includes options for L2 regularization and dropout 
in the fully connected layers after feature extraction.

The EfficientNetB4 model is loaded with pre-trained ImageNet weights and the base 
layers are frozen during initial training. The top layers consist of a combination 
of Global Average Pooling and Global Max Pooling outputs, followed by optional 
regularized and dropout-enabled dense layers, ending in a sigmoid or softmax activation 
for binary or multiclass classification tasks.

Functions:
    - efficient_net(num_classes: int = 1, regularizer: bool = False, dropout: bool = False, task_type: str = 'binary'): 
        Constructs and returns the compiled EfficientNetB4 model and its configuration. 
        The model can be configured for binary classification or multiclass classification.
        The `task_type` argument determines whether the model uses a sigmoid activation (binary) 
        or a softmax activation (multiclass). Regularization and dropout are optional.

    Arguments for `efficient_net`:
        - num_classes (int): Number of output classes for multiclass classification (1 for binary classification).
        - regularizer (bool): If True, applies L2 regularization to the dense layer.
        - dropout (bool): If True, applies 50% dropout after batch normalization.
        - task_type (str): 'binary' for binary classification (sigmoid activation) or 'multiclass' for multiclass classification (softmax activation).

    Returns:
        Tuple[Model, dict]: A compiled Keras Model instance and a dictionary containing 
        the configuration used (regularizer, dropout, task_type).
"""

from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.layers import (
    Input, Lambda, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, 
    Concatenate, BatchNormalization, Dropout
)
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input # type: ignore
from deep.constants import MODEL_IMAGE_SIZE

def efficient_net(
    num_classes: int = 1,   
    regularizer: bool = False,
    dropout: bool = False,
    task_type: str = 'binary'
):
    """
    Builds an EfficientNetB4-based model for classification (binary or multiclass) with optional regularization and dropout.

    Args:
        num_classes (int): Number of output classes (for multiclass, default is 1 for binary classification).
        regularizer (bool): If True, applies L2 regularization to the dense layer.
        dropout (bool): If True, applies 50% dropout after batch normalization.
        task_type (str): 'binary' for binary classification or 'multiclass' for multiclass classification.

    Returns:
        Tuple[Model, dict]: A compiled Keras Model instance and a dictionary containing 
        the configuration used (regularizer, dropout, task_type).
    """
    
    # Save model configuration
    config = {
        'regularizer': regularizer,
        'dropout': dropout,
        'task_type': task_type
    }

    # Set the input
    input_tensor = Input(shape=(*MODEL_IMAGE_SIZE["efficientnetb4"], 3))
    
    # Apply preprocessing to the input tensor
    x = Lambda(preprocess_input)(input_tensor)

    # Load the pre-trained model
    base_model = EfficientNetB4(include_top=False, weights='imagenet', input_tensor=x)
    
    # Freeze layers
    base_model.trainable = False

    # Add top layers (Global Average + Global Max Pooling)
    gap = GlobalAveragePooling2D()(base_model.output)
    gmp = GlobalMaxPooling2D()(base_model.output)
    x = Concatenate()([gap, gmp])
    
    # Add regularization or simple dense layer
    if regularizer:
        x = Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
    else:
        x = Dense(64, activation='relu')(x)
    
    x = BatchNormalization()(x)
    
    if dropout:
        x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)
    
    # Final output layer (sigmoid for binary, softmax for multiclass)
    if task_type == 'binary':
        output = Dense(1, activation='sigmoid')(x)  # Binary classification
    elif task_type == 'multiclass':
        output = Dense(num_classes, activation='softmax')(x)  # Multiclass classification
    else:
        raise ValueError("Invalid task_type. Choose 'binary' or 'multiclass'.")

    # Create the model
    model = Model(inputs=input_tensor, outputs=output)

    return model, config


def efficient_net_extended(
    num_classes: int = 1,   
    regularizer: bool = False,
    dropout: bool = False,
    task_type: str = 'binary'
):
    """
    Builds an EfficientNetB4-based model for classification (binary or multiclass) with optional regularization and dropout.

    Args:
        num_classes (int): Number of output classes (for multiclass, default is 1 for binary classification).
        regularizer (bool): If True, applies L2 regularization to the dense layer.
        dropout (bool): If True, applies 50% dropout after batch normalization.
        task_type (str): 'binary' for binary classification or 'multiclass' for multiclass classification.

    Returns:
        Tuple[Model, dict]: A compiled Keras Model instance and a dictionary containing 
        the configuration used (regularizer, dropout, task_type).
    """
    
    # Save model configuration
    config = {
        'regularizer': regularizer,
        'dropout': dropout,
        'task_type': task_type
    }

    # Set the input
    input_tensor = Input(shape=(*MODEL_IMAGE_SIZE["efficientnetb4"], 3))
    
    # Apply preprocessing to the input tensor
    x = Lambda(preprocess_input)(input_tensor)

    # Load the pre-trained model
    base_model = EfficientNetB4(include_top=False, weights='imagenet', input_tensor=x)
    
    # Freeze layers
    base_model.trainable = False

    # Add top layers (Global Average + Global Max Pooling)
    gap = GlobalAveragePooling2D()(base_model.output)
    gmp = GlobalMaxPooling2D()(base_model.output)
    x = Concatenate()([gap, gmp])
    
    # Add regularization or simple dense layer
    if regularizer:
        x = Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
    else:
        x = Dense(128, activation='relu')(x)
    
    x = BatchNormalization()(x)
    
    if dropout:
        x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)

    # Final output layer (sigmoid for binary, softmax for multiclass)
    if task_type == 'binary':
        output = Dense(1, activation='sigmoid')(x)  # Binary classification
    elif task_type == 'multiclass':
        output = Dense(num_classes, activation='softmax')(x)  # Multiclass classification
    else:
        raise ValueError("Invalid task_type. Choose 'binary' or 'multiclass'.")

    # Create the model
    model = Model(inputs=input_tensor, outputs=output)

    return model, config


def efficient_net_multilabel(
    num_family_classes: int = 5,   
    num_phylum_classes: int = 10, 
    regularizer: bool = False,
    dropout: bool = False,
    task_type: str = 'multilabel' 
):
    """
    Builds an EfficientNetB4-based model for multiclass multilabel classification with optional regularization and dropout.

    Args:
        num_classes (int): Number of output classes (for multiclass, default is 1 for binary classification).
        num_family_classes (int): Number of classes for family classification (multiclass).
        num_phylum_classes (int): Number of classes for phylum classification (multiclass).
        regularizer (bool): If True, applies L2 regularization to the dense layer.
        dropout (bool): If True, applies 50% dropout after batch normalization.
        task_type (str): 'multilabel' for multiclass multilabel classification.

    Returns:
        Tuple[Model, dict]: A compiled Keras Model instance and a dictionary containing 
        the configuration used (regularizer, dropout, task_type).
    """
    
    # Save model configuration
    config = {
        'regularizer': regularizer,
        'dropout': dropout,
        'task_type': task_type
    }

    # Set the input
    input_tensor = Input(shape=(*MODEL_IMAGE_SIZE["efficientnetb4"], 3))
    
    # Apply preprocessing to the input tensor
    x = Lambda(preprocess_input)(input_tensor)

    # Load the pre-trained model
    base_model = EfficientNetB4(include_top=False, weights='imagenet', input_tensor=x)
    
    # Freeze layers
    base_model.trainable = False

    # Add top layers (Global Average + Global Max Pooling)
    gap = GlobalAveragePooling2D()(base_model.output)
    gmp = GlobalMaxPooling2D()(base_model.output)
    x = Concatenate()([gap, gmp])
    
    # Add regularization
    x = Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)

    # Primary output for 'is_animal' (binary classification)
    is_animal_output = Dense(1, activation='sigmoid', name='is_animal')(x)
    
    # Auxiliary output for 'family' (multi-class classification)
    family_output = Dense(num_family_classes, activation='softmax', name='family')(x)
    
    # Auxiliary output for 'phylum' (multi-class classification)
    phylum_output = Dense(num_phylum_classes, activation='softmax', name='phylum')(x)

    # Define the model with the three outputs
    model = Model(inputs=input_tensor, outputs=[is_animal_output, family_output, phylum_output])

    return model, config
