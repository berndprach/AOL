
import tensorflow as tf


class MakeLabelOneHot:
    """
    Makes label one-hot of type float32.
    """
    def __init__(self, label_dim):
        self.label_dim = label_dim

    def on_train_set(self, *args):
        return self.on_val_set(*args)

    def on_val_set(self, img, label_id, *args):
        label_oh = tf.one_hot(label_id, self.label_dim, dtype="float32")
        return (img, label_oh, *args)
