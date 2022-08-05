import numpy as np

import aol_code.data.load_data_functions.util as data_util

from typing import Protocol, Optional, List, Callable


class Metadata(Protocol):
    name: str
    train_size: int
    val_size: int
    label_dim: int

    label_strings: Optional[List[str]]
    load_data_function = Callable


def load_data_from_metadata(md: Metadata):
    (x_train_val, y_train_val), (x_test, y_test) = md.load_data_function()

    x_train_val = x_train_val / 255
    x_test = x_test / 255

    # Change ndim for the labels to 1 (from 2):
    y_train_val = np.reshape(y_train_val, newshape=[-1])
    y_test = np.reshape(y_test, newshape=[-1])

    # Shuffle in case data is ordered
    x_train_val, y_train_val = data_util.shuffle_multiple(x_train_val,
                                                          y_train_val,
                                                          seed=1111)

    # Split into training and validation data:
    split_data = data_util.split_by_label(xs=x_train_val,
                                          ys=y_train_val,
                                          trainsize=md.train_size,
                                          valsize=md.val_size,
                                          nrof_classes=md.label_dim)
    (x_val, y_val), (x_train, y_train) = split_data

    # Shuffle again because selection method introduces correlation
    # between the first few training labels:
    x_train, y_train = data_util.shuffle_multiple(x_train, y_train, seed=1111)
    x_val, y_val = data_util.shuffle_multiple(x_val, y_val, seed=1111)

    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }
