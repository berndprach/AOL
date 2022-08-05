
import numpy as np


def split_by_label(xs, ys, valsize, trainsize, nrof_classes=10):
    """
    Takes x and y values, and splits them into two datasets
    of given sizes with equal numbers of examples per label.
    """
    if valsize < nrof_classes: print("The validation set will be empty!")
    if trainsize < nrof_classes: print("The training set will be empty!")

    counts = {}
    x_val, y_val = [], []
    x_train, y_train = [], []

    for x, y in zip(xs, ys):
        counts[y] = counts.get(y, 0) + 1
        if counts[y] <= valsize//nrof_classes:
            x_val.append(x)
            y_val.append(y)
        elif counts[y] <= valsize//nrof_classes + trainsize//nrof_classes:
            x_train.append(x)
            y_train.append(y)

    return (x_val, y_val), (x_train, y_train)


def shuffle_multiple(*dss, seed=None):
    """ Shuffles all datasets in the same way. """
    rng = np.random.default_rng(seed)
    shuffle_indices = rng.permutation(len(dss[0]))
    return [[ds[i] for i in shuffle_indices] for ds in dss]
