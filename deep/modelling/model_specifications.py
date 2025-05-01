"""
model_specifications.py - Contains various model specifications explored during the course of the project.

This script defines a customizable classification model based on EfficientNetB(N) using TensorFlow Keras. 
The model is designed with options for L2 regularization and dropout in the fully connected layers after 
feature extraction.

The EfficientNetB(N) model is initialized with pre-trained ImageNet weights, and the base layers are 
frozen during initial training. The top layers consist of a combination of Global Average Pooling and 
Global Max Pooling, followed by optional regularized and dropout-enabled dense layers, culminating in a 
sigmoid or softmax activation for binary or multiclass classification tasks.

Other models were initially explored but were discarded due to poor performance, and are not featured 
in this script.

Additionally, this script includes an experimental model, which was trained on our data as a proof of concept, 
but ultimately performed poorly, as was perhaps predictable.
"""

from tensorflow.keras.models import Model #type:ignore
from tensorflow.keras import regularizers #type: ignore

from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.layers import (
    Input, Lambda, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, 
    Concatenate, BatchNormalization, Dropout, Activation, Flatten, Input, Conv2D, MaxPooling2D
)
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications.efficientnet import EfficientNetB4, EfficientNetB5, preprocess_input # type: ignore
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


def efficient_net_deep(
    num_classes: int = 1,  
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
    
    # Dense Layers
    for units in [256, 128, 64]:
        x = Dense(units, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout:
            x = Dropout(0.1)(x)

    # Output for 'family' (multi-class classification)
    output = Dense(num_classes, activation='softmax', name='family')(x)
    
    # Define the model with the three outputs
    model = Model(inputs=input_tensor, outputs=output)

    return model, config

def efficient_net_b5(
    num_classes: int = 1,   
    regularizer: bool = False,
    dropout: bool = False,
    task_type: str = 'binary'
):
    """
    Builds an EfficientNetB5-based model for classification (binary or multiclass) with optional regularization and dropout.

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
    input_tensor = Input(shape=(*MODEL_IMAGE_SIZE["efficientnetb5"], 3))
    
    # Apply preprocessing to the input tensor
    x = Lambda(preprocess_input)(input_tensor)

    # Load the pre-trained model
    base_model = EfficientNetB5(include_top=False, weights='imagenet', input_tensor=x)
    
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

def small_vgg_model(
    num_classes: int = 1,
    regularizer: bool = False,
    dropout: bool = False,
    task_type: str = 'binary'
):
    """
    Builds a smaller VGG-style CNN for binary or multiclass classification with optional regularization and dropout.

    Args:
        num_classes (int): Number of output classes. Default is 1 (binary).
        regularizer (bool): If True, applies L2 regularization.
        dropout (bool): If True, applies 50% dropout.
        task_type (str): 'binary' or 'multiclass'.

    Returns:
        Tuple[Model, dict]: Compiled Keras model and config dictionary.
    """

    config = {
        'regularizer': regularizer,
        'dropout': dropout,
        'task_type': task_type
    }
    input_tensor = Input(shape=(*MODEL_IMAGE_SIZE["vgg16"], 3))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    if regularizer:
        x = Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)
    else:
        x = Dense(64, activation='relu')(x)

    x = BatchNormalization()(x)

    if dropout:
        x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)

    if task_type == 'binary':
        output = Dense(1, activation='sigmoid')(x)
    elif task_type == 'multiclass':
        output = Dense(num_classes, activation='softmax')(x)
    else:
        raise ValueError("Invalid task_type. Choose 'binary' or 'multiclass'.")

    model = Model(inputs=input_tensor, outputs=output)

    return model, config
