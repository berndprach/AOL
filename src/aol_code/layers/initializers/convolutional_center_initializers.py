"""
Will initialize a kernel matrix (e.g. for a 3x3xcxc kernel)
such that the (spatial) center is e.g. an orthogonal matrix,
and the other values are 0.
"""
import tensorflow as tf


class IdentityCenterInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        # Expected shape: [ks1, ks2, nrof_input_channels, nrof_channels].
        ks1, ks2, n_in, n_out = shape

        if n_in != n_out:
            raise ValueError("Number of channels of input and output"
                             "should be the same for "
                             "IdentityCenterInitializer!!"
                             f"(Got {n_in} and {n_out}.)")

        center_initializer = tf.keras.initializers.Identity()
        central_values = center_initializer(shape=(n_in, n_out))
        return fill_with_zeros(shape, central_values)


class OrthogonalCenterInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        # Expected shape: [ks1, ks2, nrof_input_channels, nrof_channels].
        ks1, ks2, n_in, n_out = shape

        center_initializer = tf.keras.initializers.Orthogonal()
        central_values = center_initializer(shape=(n_in, n_out))
        return fill_with_zeros(shape, central_values)


def fill_with_zeros(shape, central_values):
    ks1, ks2, n_in, n_out = shape
    p1 = (ks1 - 1) // 2
    p2 = (ks2 - 1) // 2

    kernel = tf.zeros(shape=shape)
    kernel = tf.Variable(kernel)
    kernel[p1, p2, :, :].assign(central_values)
    return kernel
