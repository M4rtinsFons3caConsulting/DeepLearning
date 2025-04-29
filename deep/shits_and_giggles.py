from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB4

def create_model_with_predefined_focal_loss():
    # Base model for feature extraction (EfficientNetB4)
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    
    # Primary output for 'is_animal' (binary classification)
    is_animal_output = Dense(1, activation='sigmoid', name='is_animal')(x)
    
    # Auxiliary output for 'family' (multi-class classification)
    family_output = Dense(num_family_classes, activation='softmax', name='family')(x)
    
    # Auxiliary output for 'phylum' (multi-class classification)
    phylum_output = Dense(num_phylum_classes, activation='softmax', name='phylum')(x)

    # Define the model with the three outputs
    model = Model(inputs=base_model.input, outputs=[is_animal_output, family_output, phylum_output])
    
    # Compile the model with Focal Loss for each output (using your pre-defined focal_loss)
    model.compile(
        optimizer='adam',
        loss={
            'is_animal': focal_loss(gamma=2., alpha=0.25, is_binary=True),  # Focal loss for binary classification
            'family': focal_loss(gamma=2., alpha=0.25, is_binary=False),    # Focal loss for multi-class classification
            'phylum': focal_loss(gamma=2., alpha=0.25, is_binary=False),    # Focal loss for multi-class classification
        },
        loss_weights={
            'is_animal': 1.0,  # Primary output (higher weight)
            'family': 0.1,     # Secondary output (smaller weight)
            'phylum': 0.1,     # Secondary output (smaller weight)
        },
        metrics={
            'is_animal': 'accuracy',  # Accuracy for binary classification
            'family': 'accuracy',     # Accuracy for multi-class classification
            'phylum': 'accuracy',     # Accuracy for multi-class classification
        }
    )
    
    return model
