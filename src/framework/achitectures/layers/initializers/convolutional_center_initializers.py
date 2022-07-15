"""
Will initialize a kernel matrix (e.g. for a 3x3xcxc kernel)
such that the (spatial) center is e.g. an orthogonal matrix,
and the other values are 0.
"""
import tensorflow as tf


class CenterInitializer(tf.keras.initializers.Initializer):
    def __init__(self, inner_initializer, enforce_same_size=True):
        super().__init__()
        self.inner_initializer = inner_initializer
        self.enforce_same_size = enforce_same_size  # asserts c_in == c_out

    def __call__(self, shape, dtype=None):  # expected shape: [ks1, ks2, nrof_input_channels, nrof_channels].

        ks1, ks2, n_in, n_out = shape
        p1 = (ks1-1)//2
        p2 = (ks2-1)//2

        if self.enforce_same_size and n_in != n_out:
            raise ValueError(f"The input and output dimension given to the CenterInitializer will be different! "
                             f"(shape: {shape}) (Set enforce_same_size->False to allow different sizes.)")

        parameters = tf.zeros(shape=shape)
        parameters = tf.Variable(parameters)

        central_parameters = self.inner_initializer((n_in, n_out))
        parameters[p1, p2, :, :].assign(central_parameters)

        return parameters


class IdentityCenterInitializer(CenterInitializer):
    def __init__(self):
        super().__init__(tf.keras.initializers.Identity())


class OrthogonalCenterInitializer(CenterInitializer):
    def __init__(self):
        super().__init__(tf.initializers.orthogonal())
