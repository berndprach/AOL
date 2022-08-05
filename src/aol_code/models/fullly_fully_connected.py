import tensorflow as tf

import aol_code.layers as layers
# import aol_code.layers.initializers as initializers

from dataclasses import dataclass, asdict
from aol_code.location_manager import location_manager as loc
from typing import Tuple, Optional, Union, Callable


@dataclass
class FullyFullyConnectedHyperparameters:
    nrof_layers: int = 8
    input_size: Tuple[int, int] = (32, 32)
    width: Optional[int] = None
    nrof_classes: int = 10
    kernel_regularizer: Union[Callable, float] = 0.
    nrof_input_channels: int = 3
    kernel_initializer: str = "identity"

    def __post_init__(self):
        if self.width is None:
            self.width = 4 * self.input_size[0] * self.input_size[1]


def get_fully_fully_connected_model(**kwargs):
    hp = FullyFullyConnectedHyperparameters(**kwargs)

    loc.add_settings(name="Model Setting",
                     model_name="ffc",
                     **asdict(hp))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(hp.input_size[0],
                                                   hp.input_size[1],
                                                   hp.nrof_input_channels)))

    model.add(layers.aol.AOLDense(units=hp.width,
                                  kernel_regularizer=hp.kernel_regularizer,
                                  kernel_initializer="orthogonal"
                                  ))

    for _ in range(hp.nrof_layers - 1):
        model.add(layers.basic.MaxMinActivation())
        model.add(layers.aol.AOLDense(units=hp.width,
                                      kernel_regularizer=hp.kernel_regularizer,
                                      kernel_initializer=hp.kernel_initializer,
                                      ))

    model.add(layers.basic.FirstChannels(nrof_channels=hp.nrof_classes,
                                         ndim=2))
    return model
