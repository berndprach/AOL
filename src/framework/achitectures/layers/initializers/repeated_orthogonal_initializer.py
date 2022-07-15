"""
Initializer that ensure each pair of columns (of the jacobian)
is either identical or orthogonal
(as long as the kernel size equals the stride).

If nrof_rows <= nrof_cols, the initializer will ensure J^T J is the identity.

If nrof_cols < nrof_rows, some rows will have to be repeated,
and the initializer will sample columns from an orthonormal basis,
and rescaled according to how often they were sampled.

Note that for a 1 x 1 x c_in x c_out kernel parameter,
the jacobian will consist of blocks of size c_out x c_in,
so the kernel parameter will have orthogonal rows instead of columns.
"""
import tensorflow as tf


class RepeatedOrthogonalInitializer:
    def __init__(self):
        self.orthogonal_initializer = tf.initializers.orthogonal()
        pass

    def __call__(self, shape, dtype=None):  # expected shape: [ks1, ks2, #input_channels, #output_channels]

        nrof_inputs = 1
        for dim in shape[:-1]:
            nrof_inputs *= dim
        nrof_outputs = shape[-1]

        if nrof_inputs <= nrof_outputs:
            # initializer.orthogonal already gives a matrix with orthogonal rows.
            return self.orthogonal_initializer(shape)

        # Sample columns (of the jacobian) from an orthonormal basis with replacement:
        orthonormal_basis = self.orthogonal_initializer((nrof_outputs, nrof_outputs))  # orthogonal rows

        sampled_columns = tf.random.uniform(shape=[nrof_inputs], minval=0, maxval=nrof_outputs, dtype=tf.int64)
        sampled_columns_oh = tf.one_hot(sampled_columns, depth=nrof_outputs)  # shape [nrof_inputs, nrof_outputs]

        counts = tf.reduce_sum(sampled_columns_oh, axis=0, keepdims=True)
        sampled_columns_oh_rescaled = sampled_columns_oh / tf.maximum(1, counts) ** (1 / 2)

        weight = tf.matmul(a=sampled_columns_oh_rescaled, b=orthonormal_basis)  # shape [nrof_inputs, nrof_outputs]
        return tf.reshape(weight, shape=shape)
