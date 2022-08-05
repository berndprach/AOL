import tensorflow as tf


def aol_conv2d_rescale(kernel_parameters):
    """
    Takes a convolutional parameter kernel as an input,
    and returns the rescaled version of it
    that guarantes the convolutions to be 1-Lipschitz
    (with respect to the L2 norm).
    """
    channel_rescaling_values = get_aol_conv2d_rescale(kernel_parameters)
    rescaled_kernel_weights \
        = kernel_parameters * channel_rescaling_values[None, None, :, None]
    return rescaled_kernel_weights


def get_aol_conv2d_rescale(kernel_parameters, epsilon=1e-6):
    w = kernel_parameters  # shape: [ks1, ks2, nrof_in, nrof_out]

    # For each i and j, we want to convolve kernel[:, :, i, :]
    # with kernel[:, :, j, :] in order to calculate the bound.
    # We can do this for all i and j in parallel by using the
    # batch dimension and the output dimension of a standard
    # implementation of a convolution. (tf.nn.conv2d)

    w_input_dimension_as_batch = tf.transpose(w, [2, 0, 1, 3])
    w_input_dimension_as_output = tf.transpose(w, [0, 1, 3, 2])

    # Padding needed to guarantee to pick up any positions with kernel overlap:
    p1 = w.shape[0] - 1  # kernel_size1 - 1
    p2 = w.shape[1] - 1  # kernel_size2 - 1

    v = tf.nn.conv2d(
        input=w_input_dimension_as_batch,
        filters=w_input_dimension_as_output,
        strides=[1, 1, 1, 1],
        padding=[[0, 0], [p1, p1], [p2, p2], [0, 0]]
    )  # shape [nrof_in, 2*ks1-1, 2*ks2-1, nrof_in]

    # Sum the absolute value of v over one of the input
    # channel dimension (axis 3),
    # as well as over the spatial dimensions (axis 1 and 2):
    lipschitz_bounds_squared = tf.reduce_sum(tf.abs(v),
                                             axis=(1, 2, 3))  # shape [nrof_in]
    rescaling_factors = (lipschitz_bounds_squared + epsilon) ** (-1 / 2)
    return rescaling_factors
