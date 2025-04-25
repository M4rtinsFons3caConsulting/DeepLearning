import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from focal_loss import BinaryFocalLoss

from deep.constants import IMAGE_DIR, IMG_SIZE, BATCH_SIZE
from deep.preprocess.album_augmenter import compute_effective_class_weights
from deep.models.aux_funcs import split_data, show_augmented_images, smart_resize_img, get_fitted_model_metrics, plot_metrics, plot_confusion_matrix


def run_binary_model(
    data: pd.DataFrame
    ,seed: int
    ,model
    ,loss: str
    ,epochs: int
):
    # Split the data
    train_df, val_df, test_df = split_data(data, 'is_animal', seed)

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
        preprocessing_function=smart_resize_img
    )
    test_datagen = ImageDataGenerator(preprocessing_function=smart_resize_img)

    # Train generator
    binary_train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=f'{IMAGE_DIR}',
        x_col='file_path',
        y_col='is_animal',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=seed,
        shuffle=True  # shuffle for training
    )

    # Validation generator
    binary_val_generator = test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=f'{IMAGE_DIR}',
        x_col='file_path',
        y_col='is_animal',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Test generator
    binary_test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=f'{IMAGE_DIR}',
        x_col='file_path',
        y_col='is_animal',
        target_size=(IMG_SIZE, IMG_SIZE),
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
        ,verbose=1
    )

    # Get best epoch metrics
    precision = get_fitted_model_metrics(fitted_binary)

    # Plot metrics
    plot_metrics(fitted_binary)

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

    return precision, config


def multiple_model_run(
    file_path: str
    ,seeds: list
    ,data: pd.DataFrame
    ,model
    ,loss: str
    ,epochs: int
):
    import os
    import json
    import scipy.stats as stats
    import numpy as np

    # Load baseline
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            baseline_data = json.load(f)

        baseline_scores = baseline_data['scores']

        print(f"Loaded baseline scores: {baseline_scores}")

    else:
        baseline_scores = []
        
        print("No baseline score found, saving current as baseline after run.")

    # Run the model for the first seed
    current_scores = []

    score, config = run_binary_model(
        data=data
        ,seed=seeds[0]
        ,model=model
        ,loss=loss
        ,epochs=epochs
    )

    current_scores.append(score)

    # Checking if baseline exists
    if not baseline_scores:
        print("No baseline available - accepting current model as baseline")
        
        run_all = True

    else:
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(current_scores, baseline_scores, equal_var=False)
        run_all = p_val < 0.05

    # Run remaining seeds if better
    if run_all:
        for seed in seeds[1:]:
            score, _ = run_binary_model(
                data=data
                ,seed=seed
                ,model=model
                ,loss=loss
                ,epochs=epochs
            )

            current_scores.append(score)

        # Final t-test
        t_stat, p_val = stats.ttest_ind(current_scores, baseline_scores, equal_var=False)
        mean = np.mean(current_scores)
        std_err = stats.sem(current_scores)
        ci = stats.t.interval(0.95, len(current_scores) - 1, loc=mean, scale=std_err)

        baseline_data = {
            'scores': current_scores
            ,'model': config
            ,'loss': loss
        }

        # Save model scores as new baseline
        with open(file_path, "w") as f:
            json.dump(baseline_data, f, indent=2)

        print("Saved current model as new baseline")

    else:
        print("Early stop - current model not significantly better than baseline")