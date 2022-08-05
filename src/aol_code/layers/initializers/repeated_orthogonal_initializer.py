"""
Initializer that ensure each pair of rows is either identical or orthogonal.

If nrof_rows <= nrof_cols, the initializer will ensure J^T J is the identity.

If nrof_cols < nrof_rows, some rows will have to be repeated,
and the initializer will sample columns from an orthonormal basis,
and rescaled according to how often they were sampled.

Just like e.g. in tf.initializers.Orthogonal, if the shape is more than
two-dimensional, all but the last dimension will be reshaped into a single
one before initialization, and the matrix with be reshaped back into the
desired shape after initialiation.
"""
import tensorflow as tf


class RepeatedOrthogonalInitializer:
    def __call__(self, shape, dtype=None):
        # Expected shape: [ks1, ks2, #input_channels, #output_channels]

        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]

        orthogonal_initializer = tf.initializers.Orthogonal()
        if num_rows <= num_cols:
            # Orthogonal initializer gives a matrix with orthogonal rows.
            return orthogonal_initializer(shape)

        # Sample rows from an orthonormal basis with replacement:
        orthonormal_basis = orthogonal_initializer((num_cols, num_cols))

        sampled_column_indexes = tf.random.uniform(shape=[num_rows],
                                                   minval=0,
                                                   maxval=num_cols,
                                                   dtype=tf.int64)
        sampled_columns_oh = tf.one_hot(sampled_column_indexes, depth=num_cols)
        # shape [num_rows, num_cols]

        counts = tf.reduce_sum(sampled_columns_oh, axis=0, keepdims=True)
        resclings = tf.maximum(1, counts) ** (1 / 2)
        sampled_columns_oh_rescaled = sampled_columns_oh / resclings
        # shape [num_rows, num_cols]

        # Fill the weight matrix with the sampled rescaled rows:
        weights = tf.matmul(a=sampled_columns_oh_rescaled,
                            b=orthonormal_basis)  # shape [num_rows, num_cols]
        return tf.reshape(weights, shape=shape)
