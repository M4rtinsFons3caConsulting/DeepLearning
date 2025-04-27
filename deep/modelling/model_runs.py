"""
This script defines the pipeline to train a binary classification model using Keras. 
The model is built with a customizable configuration, including options for loss function 
(focal loss or binary cross-entropy), and can handle multiple types of data (original, transformed, or upsampled).

The model is trained using augmented image data and includes a callback for early stopping and reducing the learning rate 
if the validation loss plateaus. The pipeline also supports saving and updating baseline performance scores, 
which can be used for tracking model improvements.

Functions:
    - run_binary_model(file_path: str, data: pd.DataFrame, seed: int, model, loss: str, epochs: int, type: str, model_name: str):
        Trains and evaluates a binary classification model, with options for image augmentation, upsampling, 
        and class weighting, while also updating the baseline model performance.
"""

import os
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, smart_resize  # type: ignore

from deep.constants import INPUT_DIR, BATCH_SIZE, METADATA_DIR, MODEL_IMAGE_SIZE
from deep.modelling.metric_utils import get_fitted_model_metrics, plot_confusion_matrix, plot_metrics, show_augmented_images
from deep.modelling.pipiline_utils import split_data
from deep.preprocess.album_augmenter import compute_effective_class_weights
from focal_loss import BinaryFocalLoss

def run_binary_model(
    file_path: str
    ,data: pd.DataFrame
    ,seed: int
    ,model
    ,loss: str
    ,epochs: int
    ,type: str
    ,model_name:str
):
    """
    Runs the training pipeline for a binary classification model, utilizing augmented data and optional upsampling, 
    while also tracking and updating baseline model performance.

    Args:
        file_path (str): Path to the file that stores baseline performance data.
        data (pd.DataFrame): DataFrame containing metadata and file paths for the images.
        seed (int): Random seed for reproducibility.
        model: A Keras model to be trained.
        loss (str): The loss function to use during training ('focal' or 'crossentropy').
        epochs (int): Number of epochs to train the model.
        type (str): Type of dataset to use ('original', 'transformed', or 'upsample').
        model_name (str): The name of the model used (for resizing input images to the correct shape).

    Returns:
        None: This function trains the model, evaluates it, and updates the baseline performance.
    """
    
    import os
    import json

    # Load baseline
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            baseline_data = json.load(f)

        baseline_score = baseline_data['score']

        print(f"Loaded baseline score: {baseline_score}")

    else:
        baseline_score = None
        
        print("No baseline score found, saving current as baseline after run.")
    
    # Split the data
    train_df, val_df, test_df = split_data(data, 'is_animal', seed)

    if type not in ['original', 'transformed', 'upsampled']:
        raise ValueError("Argument type must be one of 'original', 'transformed' or 'upsampled'.")

    if type == 'upsampled':
        
        # Get the upsampled images - cropped and generated
        cropped = pd.read_csv(f'{METADATA_DIR}/cropped_labels.csv')
        upsampled = pd.read_csv(f'{METADATA_DIR}/is_animal_upsample_map.csv')

        # Filter upsample based on the images on the train set
        # Upsampled images from crops
        aux_df1 = upsampled[upsampled['file_path'].str.contains('noanimalcrop', na=False)]

        # Upsampled images from train
        # Exclude the cropped images first
        temp = upsampled[~upsampled['file_path'].isin(aux_df1['file_path'])]

        # Filter to only rare_species_id in train_df
        aux_df2 = temp[temp['rare_species_id'].isin(train_df['rare_species_id'])]

        # Create new upsampled dataframe
        upsampled = pd.concat([
            aux_df1
            ,aux_df2
        ], ignore_index=True
        , axis=0)

        # Add the new metadata to train_df
        train_df = pd.concat([
            train_df
            ,cropped
            ,upsampled
        ], ignore_index=True
        , axis=0
    )

    # Changing target to string
    train_df['is_animal'] = train_df['is_animal'].astype(str)
    val_df['is_animal'] = val_df['is_animal'].astype(str)
    test_df['is_animal'] = test_df['is_animal'].astype(str)

    train_datagen = ImageDataGenerator(
        rotation_range=90,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        channel_shift_range=30.0,
        zoom_range=(0.8, 1.2),
        fill_mode='nearest',
        rescale=1./255
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Train generator
    binary_train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=INPUT_DIR,
        x_col='file_path',
        y_col='is_animal',
        target_size=MODEL_IMAGE_SIZE[model_name],
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=seed,
        shuffle=True,
    )

    # Validation generator
    binary_val_generator = test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=INPUT_DIR,
        x_col='file_path',
        y_col='is_animal',
        target_size=MODEL_IMAGE_SIZE[model_name],
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Test generator
    binary_test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=INPUT_DIR,
        x_col='file_path',
        y_col='is_animal',
        target_size=MODEL_IMAGE_SIZE[model_name],
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Show augmented images - rerun for new batch
    show_augmented_images(binary_train_generator, num_images=8)

    # Compute class weights
    class_weights = compute_effective_class_weights(train_df, 'is_animal')

    # Build the model
    binary_model, config = model

    # Define loss function
    if loss == 'focal':
        loss = BinaryFocalLoss(
            gamma=5
            ,pos_weight=class_weights['1'] / (class_weights['0'] + class_weights['1'])
        )
        
    elif loss == 'crossentropy':
        loss = 'binary_crossentropy'

    else:
        raise('Invalid loss function')

    # Compiling the model
    binary_model.compile(
        optimizer=RMSprop(learning_rate=0.001)
        ,loss=loss
        ,metrics=['accuracy', AUC(), 'precision', 'recall']
    )

    # Fit the model
    fitted_binary = binary_model.fit(
        binary_train_generator
        ,validation_data=binary_val_generator
        ,epochs=epochs
        ,steps_per_epoch=int(np.ceil(len(train_df) / BATCH_SIZE))
        ,validation_steps=int(np.ceil(len(val_df) / BATCH_SIZE))
        ,class_weight=class_weights
        ,callbacks=[
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            ,ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)
        ]
        , verbose=1
    )

    # Get best epoch metrics
    val_precision = get_fitted_model_metrics(fitted_binary)

    # Plot metrics
    plot_metrics(fitted_binary)

    if baseline_score is None or val_precision > baseline_score:
        baseline_data = {
            'score': val_precision,
            'model': config,
            'loss': loss,
            'type': type
        }

        # Save model scores as new baseline
        with open(file_path, "w") as f:
            json.dump(baseline_data, f, indent=2)

        print("Saved current model as new baseline")

    else:
        print("Current model not better than baseline")
        
    # # Make predictions
    # predictions = binary_model.predict(
    #     binary_test_generator
    #     ,steps=len(binary_test_generator)
    #     ,verbose=1
    # )

    # # Get the true labels
    # y_true = binary_test_generator.labels

    # # Convert predictions to class labels
    # y_pred = (predictions > 0.5).astype(int).flatten()

    # # Calculate metrics
    # print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    # print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    # print(f"Recall: {recall_score(y_true, y_pred):.4f}")

    # # Plot confusion matrix
    # plot_confusion_matrix(y_true, y_pred, data['is_animal'].unique())

    # return precision, config
