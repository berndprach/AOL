
import tensorflow as tf


def aol_rescale(parameter_matrix):
    """
    Takes a parameter matrix (size: [nrof_inputs, nrof_outputs]) as an input,
    and returns the rescaled version of it
    that is guaranteed to be 1-Lipschitz
    (with respect to the L2 norm).
    """
    rescaling_factors = get_rescaling_factors(parameter_matrix)
    return parameter_matrix * rescaling_factors[:, None]


def get_rescaling_factors(parameter_matrix, epsilon=1e-6):
    mmT = tf.linalg.matmul(a=parameter_matrix,
                           b=parameter_matrix,
                           transpose_b=True)
    mmT_abs = tf.abs(mmT)
    lipschitz_bounds_squared = tf.reduce_sum(mmT_abs, axis=1)

    rescaling_factors = (lipschitz_bounds_squared + epsilon) ** (-1 / 2)
    return rescaling_factors
