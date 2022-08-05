
import tensorflow.keras.datasets.cifar10 as cifar10

from aol_code.data.load_data_functions.load_data_from_metadata import (
    load_data_from_metadata
)

from aol_code.location_manager import location_manager as loc
from dataclasses import dataclass, asdict


@dataclass
class CIFAR10Metadata:
    name: str = "cifar10"
    train_size: int = 40_000
    val_size: int = 10_000
    label_dim: int = 10

    label_strings = ["airplane", "automobile", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]

    @staticmethod
    def load_data_function():
        return cifar10.load_data()


def load_cifar10(**kwargs):
    md = CIFAR10Metadata(**kwargs)

    # Log the data settiings:
    loc.add_settings(
        name="Data Settings",
        **{(key if key != "name" else "dataset_name"): val
           for key, val in asdict(md).items()}
    )

    return load_data_from_metadata(md)
