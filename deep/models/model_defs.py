from deep.constants import DATA_DIR, IMAGE_DIR, IMG_SIZE, BATCH_SIZE, SEED
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, BatchNormalization, Dropout, Lambda, GlobalMaxPooling2D, Concatenate
from tensorflow.keras import regularizers


def efficient_net(
    regularizer: bool = False
    ,dropout: bool = False
):
    # Save model configuration
    config = {
        'regularizer': regularizer
        ,'dropout': dropout
    }
    # Set the input
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
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