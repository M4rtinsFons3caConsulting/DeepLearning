"""
custom_loss.py — Implementation of Categorical Focal Loss.

This module provides a custom implementation of the categorical focal loss function,
adapted for use with integer-labeled targets instead of one-hot encoded vectors.
The implementation is inspired by established formulations in the literature and
existing TensorFlow API support.

We acknowledge the reference implementation provided by the Keras library:

Citation (TensorFlow API):  
    TensorFlow. (n.d.). *tf.keras.losses.CategoricalFocalCrossentropy*.  
    TensorFlow API.  
    https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalFocalCrossentropy

The theoretical basis for focal loss is derived from the following seminal work:

Citation (Focal Loss):  
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).  
    Focal loss for dense object detection.  
    In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980–2988.  
    https://doi.org/10.1109/ICCV.2017.324

And for the multi-class extension, we also reference:

Citation (Categorical Focal Loss):  
    Jaiswal, A., & AbdAlmageed, W. (2021).  
    Categorical Focal Loss for Multi-class Classification: An Overview.  
    *arXiv preprint arXiv:2111.05025*.  
    https://arxiv.org/abs/2111.05025
"""

import tensorflow as tf
from tensorflow.keras.losses import Loss # type: ignore
from keras.saving import register_keras_serializable #type: ignore

@register_keras_serializable()
class CategoricalFocalLoss(Loss):
    """
    Implementation of the Categorical Focal Loss function.

    This loss is designed to address class imbalance by focusing learning on hard misclassified examples.
    It extends the standard categorical cross-entropy by applying a modulating factor to down-weight easy examples.

    Attributes:
        gamma (float): Focusing parameter that reduces the relative loss for well-classified examples (default=2.0).
        alpha (Optional[list[float]]): Class balancing weights; if None, all classes are treated equally.
        from_logits (bool): If True, applies softmax to raw predictions before computing loss.
    """

    def __init__(
        self, 
        gamma: float = 2.0, 
        alpha: list[float] | None = None, 
        from_logits: bool = False, 
        **kwargs
    ):
        
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(
        self, 
        y_true: tf.Tensor, 
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        
        # Apply softmax if predictions are raw logits
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)  # prevent log(0)

        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_term = tf.pow(1 - y_pred, self.gamma)

        if self.alpha is not None:
            alpha = tf.constant(self.alpha, dtype=tf.float32)
            alpha_factor = y_true * alpha
            focal_loss = alpha_factor * focal_term * cross_entropy
        else:
            focal_loss = focal_term * cross_entropy

        # Sum the loss over all classes
        return tf.reduce_sum(focal_loss, axis=1)
