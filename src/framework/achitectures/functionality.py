
import tensorflow as tf


def partial_reshape(x, newshape, indices=None):
    newshape = list(newshape)  # to make tuples work
    if indices is None:
        indices = [i for i in range(len(newshape))]
    xshape = x.shape
    for i in range(len(newshape)):
        if newshape[i] is None:
            newshape[i] = xshape[indices[i]]
    xnew = tf.reshape(x, shape=newshape)
    return xnew
