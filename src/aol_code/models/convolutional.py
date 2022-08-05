import tensorflow as tf

import aol_code.layers as layers
import aol_code.layers.initializers as initializers

from aol_code.location_manager import location_manager as loc
from dataclasses import dataclass, asdict
from typing import Union, Callable, Tuple, Any


@dataclass
class ConvolutionalHyperparameters:
    base_width: int = 16
    nrof_layers_per_block: int = 5
    kernel_regularizer: Union[Callable, float] = 0.
    nrof_classes: int = 10
    nrof_input_channels: int = 3


def get_convolutional_model(**kwargs):
    hp = ConvolutionalHyperparameters(**kwargs)

    loc.add_settings(name="Model Setting",
                     model_name="convolutional",
                     **asdict(hp))

    w = hp.base_width
    reg = hp.kernel_regularizer

    ki1x1 = initializers.IdentityCenterInitializer
    # ki3x3 = initializers.IdentityCenterInitializer
    ki_width_change = initializers.RepeatedOrthogonalInitializer

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(
        input_shape=(32, 32, hp.nrof_input_channels)))

    # Down to shape 16 x 16 x 16:
    add_block(model,
              width=hp.base_width,
              number_of_layers=hp.nrof_layers_per_block,
              kernel_regularizer=hp.kernel_regularizer)

    # Down to 8 x 8 x 64:
    add_block(model,
              width=4 * hp.base_width,
              number_of_layers=hp.nrof_layers_per_block,
              kernel_regularizer=hp.kernel_regularizer)

    # Down to 4 x 4 x 256:
    add_block(model,
              width=16 * hp.base_width,
              number_of_layers=hp.nrof_layers_per_block,
              kernel_regularizer=hp.kernel_regularizer)

    # Down to 2 x 2 x 1024:
    add_block(model,
              width=64 * hp.base_width,
              number_of_layers=hp.nrof_layers_per_block,
              kernel_size=(1, 1),
              kernel_initializer=ki1x1,
              kernel_regularizer=hp.kernel_regularizer)

    # Down to 1 x 1 x 1024:
    model.add(layers.aol.AOLConv2D(filters=64 * w,
                                   kernel_size=(1, 1),
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki_width_change,
                                   ))
    model.add(layers.basic.FirstChannels(nrof_channels=16 * w))
    add_block(model,
              width=64 * hp.base_width,
              number_of_layers=hp.nrof_layers_per_block,
              kernel_size=(1, 1),
              kernel_initializer=ki1x1,
              kernel_regularizer=hp.kernel_regularizer)

    # Down to 1x1x10:
    model.add(layers.aol.AOLConv2D(filters=64 * w,
                                   kernel_size=(1, 1),
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki1x1,
                                   ))
    model.add(layers.basic.FirstChannels(nrof_channels=hp.nrof_classes))
    model.add(tf.keras.layers.Flatten())
    return model


@dataclass
class ConvolutionalBlockHyperparameters:
    number_of_layers: int = 5
    kernel_size: Tuple[int] = (3, 3)
    kernel_regularizer: float = 0.
    first_kernel_initializer: Any = initializers.RepeatedOrthogonalInitializer
    kernel_initializer: Any = initializers.IdentityCenterInitializer


def add_block(model, width, **kwargs):
    hp = ConvolutionalBlockHyperparameters(**kwargs)
    model.add(layers.basic.ConcatenationPooling(pool_size=(2, 2)))
    model.add(
        layers.aol.AOLConv2D(filters=width,
                             kernel_size=(1, 1),
                             padding="same",
                             kernel_regularizer=hp.kernel_regularizer,
                             kernel_initializer=hp.first_kernel_initializer,
                             ))
    model.add(layers.basic.MaxMinActivation())

    for _ in range(hp.number_of_layers - 1):
        model.add(
            layers.aol.AOLConv2D(filters=width,
                                 kernel_size=hp.kernel_size,
                                 padding="same",
                                 kernel_regularizer=hp.kernel_regularizer,
                                 kernel_initializer=hp.kernel_initializer,
                                 ))
        model.add(layers.basic.MaxMinActivation())
