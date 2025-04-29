import tensorflow as tf
from tensorflow.keras.losses import Loss # type: ignore

class CategoricalFocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=None, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_term = tf.pow(1 - y_pred, self.gamma)

        if self.alpha is not None:
            alpha = tf.constant(self.alpha, dtype=tf.float32)
            alpha_factor = y_true * alpha
            focal_loss = alpha_factor * focal_term * cross_entropy
        else:
            focal_loss = focal_term * cross_entropy

        return tf.reduce_sum(focal_loss, axis=1)