"""
Applies preprocessing to a tensorflow dataset.
"""

import tensorflow as tf

from framework.data.processing_functions.processing_functions_factory import processing_functions_factory

from framework.location_manager import location_manager as loc
from typing import List, Tuple, Dict, Any

tf_ds = Any


class SequentialPreprocessor:
    def __init__(self, preprocessor_configurations: List[Tuple[str, Dict]]):
        self.preprocessor_configurations = preprocessor_configurations

        loc.add_settings("Preprocessor Settings",
                         **{str(i+1): config for i, config in enumerate(self.preprocessor_configurations)})

    def get_preprocessed_dataset(self, dataset: Dict[str, tf_ds], batch_size):
        preprocessed_tf_dataset = {}

        for partition_name, partition in dataset.items():
            is_training = (partition_name == "train")

            if is_training:
                partition = partition.shuffle(buffer_size=10 * batch_size)

            for preprocessor_configuration in self.preprocessor_configurations:
                preprocessor_name, preprocessor_kwargs = preprocessor_configuration
                preprocessing_function = processing_functions_factory.create(preprocessor_name,
                                                                             is_training=is_training,
                                                                             **preprocessor_kwargs)
                partition = partition.map(preprocessing_function,
                                          num_parallel_calls=tf.data.AUTOTUNE)

            preprocessed_tf_dataset[partition_name] = partition.batch(batch_size).prefetch(2)

        print_result(preprocessed_tf_dataset)

        return preprocessed_tf_dataset


def print_result(preprocessed_tf_dataset):
    for x, y in preprocessed_tf_dataset["train"]:
        x_shape_str = str(x.numpy().shape) + ","
        y_shape_str = str(y.numpy().shape) + ","
        loc.print_pro("Created datasets from data. Train data batch has:")
        loc.print_pro(f"x shape: {x_shape_str:20} x dtype: {x.dtype}")
        loc.print_pro(f"y shape: {y_shape_str:20} y dtype: {y.dtype}")
        break
