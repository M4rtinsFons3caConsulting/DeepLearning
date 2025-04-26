from deep.constants import IMG_SIZE, SEED


def split_data(data, label):
    from sklearn.model_selection import train_test_split

    # Performing the splits
    train_df, test_df = train_test_split(
        data, 
        test_size=0.2, 
        stratify=data[label], 
        random_state=SEED
        )  # Create test set
    
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.15, 
        stratify=train_df[label], 
        random_state=SEED)
      # Create train and validation set

    return train_df, val_df, test_df


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


def smart_resize_img(image, target_size=(IMG_SIZE, IMG_SIZE)):
    from tensorflow.keras.preprocessing.image import smart_resize

    resized_img = smart_resize(image, target_size)  # Resize image
    resized_img /= 255.0  # Normalize the image
    
    return resized_img


def get_fitted_model_metrics(model):
    import numpy as np

    history = model.history

    # Get best epoch metrics
    best_epoch = np.argmin(history['val_loss'])

    # Loss
    best_train_loss = history['loss'][best_epoch]
    best_val_loss = history['val_loss'][best_epoch]
    print(f"Train Loss: {best_train_loss}\nValidation Loss: {best_val_loss}")
    # Accuracy
    best_train_acc = history['accuracy'][best_epoch]
    best_val_acc = history['val_accuracy'][best_epoch]
    print(f"Train Accuracy:{best_train_acc}\nValidation Accuracy: {best_val_acc}")


def plot_metrics(model):
    import matplotlib.pyplot as plt

    history = model.history

    # Defining the variables
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Accuracy plot
    plt.plot(
        epochs
        ,acc
        ,'bo'
        ,label='Training Accuracy'
    )
    plt.plot(
        epochs
        ,val_acc
        ,'b'
        ,label='Validation Accuracy'
    )
    plt.title("Training and Validation Accuracy")
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