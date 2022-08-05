"""
Applies preprocessing to a tensorflow dataset.
"""

import tensorflow as tf

from aol_code.location_manager import location_manager as loc
from typing import Protocol, Tuple, Any, List, Dict


class Preprocessor(Protocol):
    def on_train_set(self, *args) -> Tuple:
        ...

    def on_val_set(self, *args) -> Tuple:
        ...


tf_ds_type = Any


class SequentialPreprocessor:
    def __init__(self, preprocessors: List[Preprocessor]):
        self.preprocessors = preprocessors

    def __call__(self, dataset: Dict[str, tf_ds_type], batch_size):
        preprocessed_tf_dataset = {}

        for partition_name, tf_ds in dataset.items():
            is_training = (partition_name == "train")

            if is_training:
                tf_ds = tf_ds.shuffle(buffer_size=10 * batch_size)

            for preprocessor in self.preprocessors:
                if is_training:
                    preprocessing_function = preprocessor.on_train_set
                else:
                    preprocessing_function = preprocessor.on_val_set
                tf_ds = tf_ds.map(preprocessing_function,
                                  num_parallel_calls=tf.data.AUTOTUNE)

            tf_ds = tf_ds.batch(batch_size)
            tf_ds = tf_ds.prefetch(2)
            preprocessed_tf_dataset[partition_name] = tf_ds

        print_result(preprocessed_tf_dataset)
        return preprocessed_tf_dataset


def print_result(preprocessed_tf_dataset):
    for x, y in preprocessed_tf_dataset["train"]:
        x_shape_str = str(x.numpy().shape) + ","
        y_shape_str = str(y.numpy().shape) + ","
        loc.print_pro("Created tensorflow dataset with preprocessing. "
                      "Train data batch has:")
        loc.print_pro(f"x shape: {x_shape_str:20} x dtype: {x.dtype}")
        loc.print_pro(f"y shape: {y_shape_str:20} y dtype: {y.dtype}")
        break
