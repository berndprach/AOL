
import tensorflow as tf

from framework.achitectures.layers.layer_factory import layer_factory
from framework.achitectures.layers.aol.aol_dense import AOLDense
from framework.achitectures.layers.basic.first_channels_layers import FirstChannelsLayer
from framework.achitectures.layers.basic.max_min_activation import MaxMinActivation

from dataclasses import dataclass, asdict
from framework.location_manager import location_manager as loc
from typing import Optional, Union, Callable


@dataclass
class FullyFullyConnectedHyperparameters:
    nrof_layers: int = 8
    input_sidelength: int = 32
    width: Optional[int] = None
    nrof_classes: int = 10
    kernel_regularizer: Union[Callable, float] = 0.
    nrof_input_channels: int = 3
    kernel_initializer: str = "identity"  # "orthogonal" also possible.


def __post_init__(self):
        if self.width is None:
            self.width = 4 * self.input_sidelength * self.input_sidelength


def get_fully_fully_connected_model(**kwargs):
    hp = FullyFullyConnectedHyperparameters(**kwargs)

    loc.add_settings(name="Model Setting",
                     model_name="ffc",
                     **asdict(hp))

    sl = hp.input_sidelength
    reg = hp.kernel_regularizer

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(sl, sl, hp.nrof_input_channels)))

    model.add(AOLDense(units=hp.width,
                       kernel_regularizer=reg,
                       kernel_initializer="orthogonal"
                       ))

    for _ in range(hp.nrof_layers-1):
        model.add(MaxMinActivation())
        model.add(AOLDense(units=hp.width,
                           kernel_regularizer=reg,
                           kernel_initializer=hp.kernel_initializer,
                           ))

    model.add(FirstChannelsLayer(nrof_channels=hp.nrof_classes,
                                 ndim=2))
    # model.add(tf.keras.layers.Flatten())
    return model


def get_fully_fully_connected_model_old(**kwargs):
    hp = FullyFullyConnectedHyperparameters(**kwargs)

    loc.add_settings(name="Model Setting",
                     model_name="ffc",
                     **asdict(hp))

    sl = hp.input_sidelength
    reg = hp.kernel_regularizer

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape(target_shape=[1, 1, sl * sl * hp.nrof_input_channels],
                                      input_shape=(sl, sl, hp.nrof_input_channels)))

    for layer_idx in range(hp.nrof_layers):
        # model.add(AOLConv2D(filters=28 * 28, kernel_size=(1, 1)))
        # model.add(MaxMinActivation())
        model.add(layer_factory.create("AOLConv2D",
                                       filters=hp.width,
                                       kernel_size=(1, 1),
                                       kernel_regularizer=reg,
                                       # kernel_initializer=kernel_initializer,
                                       kernel_initializer="repeated_orthogonal",
                                       ))

        if layer_idx < hp.nrof_layers-1:
            model.add(layer_factory.create("MaxMinActivation"))

    # model.add(AOLConv2D(filters=10, kernel_size=(1, 1), kernel_initializer="repeated_orthogonal"))
    model.add(layer_factory.create("FirstChannels",
                                   nrof_channels=hp.nrof_classes))
    model.add(tf.keras.layers.Flatten())
    return model

