
import tensorflow as tf


def get_kernel_regularizer(kernel_regularizer):
    if kernel_regularizer is None:
        return None
    elif isinstance(kernel_regularizer, float):
        return tf.keras.regularizers.l2(kernel_regularizer)
    elif callable(kernel_regularizer):
        return kernel_regularizer
    else:
        raise ValueError(f"Not sure what to do with kernel regularizer "
                         f"{kernel_regularizer}! (It is neither None, nor"
                         f"a float, nor a callable.)")
