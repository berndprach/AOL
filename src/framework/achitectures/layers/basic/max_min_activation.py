
import tensorflow as tf

from framework.achitectures.layers.layer import Layer


class MaxMinActivation(Layer):
    print_name = "MaxMinActivation"

    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def call(x, *args):
        # Requires the final dimension to be even!!
        old_shape = [tf.shape(x)[0]] + [d for d in x.shape[1:-1]] + [x.shape[-1]]
        new_shape = [tf.shape(x)[0]] + [d for d in x.shape[1:-1]] + [x.shape[-1]//2, 2]

        x = tf.reshape(x, shape=new_shape)
        x_max = tf.reduce_max(x, axis=-1, keepdims=True)
        x_min = tf.reduce_min(x, axis=-1, keepdims=True)
        x_maxmin = tf.concat([x_max, x_min], axis=-1)
        return tf.reshape(x_maxmin, shape=old_shape)


