
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

    return best_val_pre


def plot_metrics(model):
    import matplotlib.pyplot as plt

    history = model.history

    # Defining the variables
    pre = history['precision']
    val_pre = history['val_precision']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(pre) + 1)

    # Recall plot
    plt.plot(
        epochs
        ,pre
        ,'bo'
        ,label='Training Precision'
    )
    plt.plot(
        epochs
        ,val_pre
        ,'b'
        ,label='Validation Precision'
    )
    plt.title("Training and Validation Precision")
    plt.legend()
    plt.figure()

    # Loss plot
    plt.plot(
        epochs
        ,loss
        ,'bo'
        ,label='Training Loss'
    )
    plt.plot(
        epochs
        ,val_loss
        ,'b'
        ,label='Validation Loss'
    )
    plt.title("Training and Validation Loss")
    plt.legend()
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