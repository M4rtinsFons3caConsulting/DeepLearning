
def show_augmented_images(generator, num_images=8):
    import matplotlib.pyplot as plt
    import math

    images, labels = next(generator)

    actual_num = min(num_images, len(images))  # Avoid index errors

    max_cols = 4
    rows = math.ceil(actual_num / max_cols)
    plt.figure(figsize=(max_cols * 3, rows * 3))

    for i in range(actual_num):
        plt.subplot(rows, max_cols, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def get_fitted_model_metrics(model):
    import numpy as np

    history = model.history

    # Get best epoch metrics
    best_epoch = np.argmin(history['val_loss'])

    # Loss
    best_train_loss = history['loss'][best_epoch]
    best_val_loss = history['val_loss'][best_epoch]
    print(f"Train Loss: {best_train_loss}\nValidation Loss: {best_val_loss}")
    
    # Precision
    best_train_pre = history['precision'][best_epoch]
    best_val_pre = history['val_precision'][best_epoch]
    print(f"Train Precision:{best_train_pre}\nValidation Precision: {best_val_pre}")

    # Alternatively, to get precision per class
    if 'precision' in history and isinstance(history['precision'], list):
        for i, class_precision in enumerate(history['precision'][best_epoch]):
            print(f"Class {i} Precision: {class_precision}")

    return best_val_pre

def plot_metrics(model, binary=False):
    import matplotlib.pyplot as plt

    history = model.history

    # Defining the variables (multiclass)
    precision = history['precision']
    val_precision = history['val_precision']
    loss = history['loss']
    val_loss = history['val_loss']

    if not binary:
        accuracy = history['accuracy']
        val_accuracy = history['val_accuracy']

    epochs = range(1, len(precision) + 1)

    # Precision Plot
    plt.plot(epochs, precision, 'bo', label='Training Precision')
    plt.plot(epochs, val_precision, 'b', label='Validation Precision')
    plt.title("Training and Validation Precision")
    plt.legend()
    plt.figure()

    if not binary:
        # Accuracy Plot
        plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.figure()

    # Loss Plot
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.legend()

    # Show the plots
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=[*labels],
        yticklabels=[*labels]
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    