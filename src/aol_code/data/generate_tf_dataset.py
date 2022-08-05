"""
Generates a tensorflow dataset from data.
"""

import tensorflow as tf

from aol_code.location_manager import location_manager as loc
from typing import Dict, Any, Tuple

tf_ds_type = Any


def generate_tf_datasets(data: Dict[str, Tuple]):
    """ Takes the data and creates a tensorflow dataset with it. """

    tf_dataset: Dict[str, tf_ds_type] = {}

    for partition_name, partition_data_tuple in data.items():

        if len(partition_data_tuple[0]) == 0:
            loc.print_pro(f"Dataset partition \"{partition_name}\" "
                          f"has no data, so there will not be corresponding "
                          f"tensorflow dataset!")
            continue

        tf_ds = tf.data.Dataset.from_tensor_slices(partition_data_tuple)
        tf_dataset[partition_name] = tf_ds
    return tf_dataset
