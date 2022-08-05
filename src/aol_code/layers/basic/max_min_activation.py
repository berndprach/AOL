"""
MaxMin activation function, introduced by
Norm-preserving Orthogonal Permutation Linear Unit Activation Functions (OPLU)
(https://arxiv.org/abs/1604.02313), and
Sorting Out Lipschitz Function Approximation
(https://arxiv.org/abs/1811.05381).
"""

import tensorflow as tf

from aol_code.layers.layer import Layer


class MaxMinActivation(Layer):
    @staticmethod
    def call(x, *args):
        # Requires the final dimension to be even!!
        input_shape = [
            tf.shape(x)[0],
            *(d for d in x.shape[1:-1]),
            x.shape[-1],
        ]
        intermediate_shape = [
            tf.shape(x)[0],
            *(d for d in x.shape[1:-1]),
            x.shape[-1] // 2,
            2,
        ]

        x = tf.reshape(x, shape=intermediate_shape)
        x_max = tf.reduce_max(x, axis=-1, keepdims=True)
        x_min = tf.reduce_min(x, axis=-1, keepdims=True)
        x_maxmin = tf.concat([x_max, x_min], axis=-1)
        return tf.reshape(x_maxmin, shape=input_shape)
