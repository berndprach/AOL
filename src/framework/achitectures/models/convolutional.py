import tensorflow as tf

from framework.achitectures.layers.layer_factory import layer_factory

from framework.location_manager import location_manager as loc
from dataclasses import dataclass, asdict
from typing import Union, Callable


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
    nl = hp.nrof_layers_per_block
    reg = hp.kernel_regularizer

    ki3x3 = "identity_center"
    ki1x1 = "identity_center"
    ki_width_change = "repeated_orthogonal"

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, hp.nrof_input_channels)))

    # Down to shape 16 x 16 x 16:
    add_block(model, width=w, nl=nl, ks=(3, 3), reg=reg, ki_first=ki1x1, ki=ki3x3)

    # Down to 8 x 8 x 64:
    add_block(model, width=4*w, nl=nl, ks=(3, 3), reg=reg, ki_first=ki1x1, ki=ki3x3)

    # Down to 4 x 4 x 256:
    add_block(model, width=16*w, nl=nl, ks=(3, 3), reg=reg, ki_first=ki1x1, ki=ki3x3)

    # Down to 2 x 2 x 1024:
    add_block(model, width=64*w, nl=nl, ks=(1, 1), reg=reg, ki_first=ki1x1, ki=ki1x1)

    # Down to 1 x 1 x 1024:
    model.add(layer_factory.create("AOLConv2D",
                                   filters=64*w,
                                   kernel_size=(1, 1),
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki_width_change,
                                   ))
    model.add(layer_factory.create("FirstChannels",
                                   nrof_channels=16*w))
    add_block(model, width=64*w, nl=nl, ks=(1, 1), reg=reg, ki_first=ki1x1, ki=ki1x1)

    # Down to 1x1x10:
    model.add(layer_factory.create("AOLConv2D",
                                   filters=64*w,
                                   kernel_size=(1, 1),
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki1x1,
                                   ))
    model.add(layer_factory.create("FirstChannels",
                                   nrof_channels=hp.nrof_classes))
    model.add(tf.keras.layers.Flatten())
    return model


def add_block(model, width, nl, ks, reg, ki_first, ki):
    model.add(layer_factory.create("ConcatenationPooling",
                                   pool_size=(2, 2)))
    model.add(layer_factory.create("AOLConv2D",
                                   filters=width,
                                   kernel_size=(1, 1),
                                   padding="same",
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki_first,
                                   ))
    model.add(layer_factory.create("MaxMinActivation"))
    for _ in range(nl-1):
        model.add(layer_factory.create("AOLConv2D",
                                       filters=width,
                                       kernel_size=ks,
                                       padding="same",
                                       kernel_regularizer=reg,
                                       kernel_initializer=ki,
                                       ))
        model.add(layer_factory.create("MaxMinActivation"))
