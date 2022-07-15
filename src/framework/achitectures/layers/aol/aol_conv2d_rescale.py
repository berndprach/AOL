
import tensorflow as tf


def aol_conv2d_rescale(kernel_parameters):
    """
    Takes a convolutional parameter matrix as an input,
    and returns the rescaled version of it
    that guarantes the convolutions to be 1-Lipschitz
    (with respect to the L2 norm).
    """
    channel_rescaling_values = get_aol_conv2d_rescale(kernel_parameters)
    rescaled_kernel_weights = kernel_parameters * channel_rescaling_values[None, None, :, None]
    return rescaled_kernel_weights


def get_aol_conv2d_rescale(kernel_parameters, epsilon=1e-6):
    w = kernel_parameters  # shape: [ks1, ks2, nrof_in, nrof_out]

    w_transposed = tf.transpose(w, [2, 0, 1, 3])  # abuse the channel dimension for iterating over input dimensions.
    w_kernel = tf.transpose(w, [0, 1, 3, 2])  # want to multply together corresponding of output dimensions!

    # Padding needed to guarantee to pick up any positions with kernel overlap.
    p1 = w.shape[0] - 1  # kernel_size1 - 1
    p2 = w.shape[1] - 1  # kernel_size2 - 1

    v = tf.nn.conv2d(
        input=w_transposed,
        filters=w_kernel,
        strides=[1, 1, 1, 1],
        padding=[[0, 0], [p1, p1], [p2, p2], [0, 0]]
    )  # shape [nrof_in, 2*ks1-1, 2*ks2-1, nrof_in]
    lipschitz_bounds_squared = tf.reduce_sum(tf.abs(v), axis=(1, 2, 3))  # shape [nrof_in]
    rescaling_factors = (lipschitz_bounds_squared + epsilon) ** (-1 / 2)
    return rescaling_factors
