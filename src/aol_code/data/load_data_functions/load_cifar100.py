
import tensorflow.keras.datasets.cifar100 as cifar100

# from aol_code.data.load_data_functions import load_data_from_metadata
from aol_code.data.load_data_functions.load_data_from_metadata import (
    load_data_from_metadata
)

from aol_code.location_manager import location_manager as loc
from dataclasses import dataclass, asdict


@dataclass
class CIFAR100Metadata:
    name: str = "cifar100"
    train_size: int = 40_000
    val_size: int = 10_000
    label_dim: int = 100

    label_strings = ["class " + str(i+1) for i in range(100)]

    @staticmethod
    def load_data_function():
        return cifar100.load_data()


def load_cifar100(**kwargs):
    md = CIFAR100Metadata(**kwargs)

    # Log the data settiings:
    loc.add_settings(
        name="Data Settings",
        **{(key if key != "name" else "dataset_name"): val
           for key, val in asdict(md).items()}
    )

    return load_data_from_metadata(md)
