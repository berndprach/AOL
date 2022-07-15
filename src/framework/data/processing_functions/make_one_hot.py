
import tensorflow as tf


class MakeOneHot:
    """
    Makes label one-hot and a float32.
    """
    def __init__(self, label_dim, is_training=False):
        self.label_dim = label_dim
        self.is_training = is_training

    def __call__(self, img, label_id, *args):
        label_oh = tf.one_hot(label_id, self.label_dim, dtype="float32")
        return (img, label_oh, *args)
