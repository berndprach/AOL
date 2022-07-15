import tensorflow as tf

from framework.achitectures.layers.layer_factory import layer_factory

from framework.location_manager import location_manager as loc

from dataclasses import dataclass, asdict
from typing import Union, Callable


@dataclass
class PatchwiseHyperparameters:
    patch_size: int = 4
    input_sidelength: int = 32
    filters_after_downsize: int = 8
    patch_encoding_factor: int = 12
    nrof_encoding_layers: int = 12
    nrof_aggregation_layers: int = 14
    kernel_regularizer: Union[Callable, float] = 0.
    nrof_classes: int = 10
    nrof_input_channels: int = 3


def get_patchwise_model(**kwargs):
    hp = PatchwiseHyperparameters(**kwargs)
    ps = hp.patch_size
    np = hp.input_sidelength // ps
    fd = hp.filters_after_downsize
    s = hp.patch_encoding_factor
    reg = hp.kernel_regularizer

    loc.add_settings(name="Model Settings",
                     number_of_patches=np * np,
                     **asdict(hp))

    ki3x3 = "identity_center"
    ki1x1 = "identity_center"
    ki_width_change = "repeated_orthogonal"

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, hp.nrof_input_channels)))

    # Extract Patches:
    model.add(layer_factory.create("ConcatenationPooling",
                                   pool_size=(ps, ps)))
    model.add(layer_factory.create("AOLConv2D",
                                   filters=ps * ps * s,
                                   kernel_size=(1, 1),
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki_width_change,
                                   ))
    model.add(layer_factory.create("MaxMinActivation"))

    for _ in range(hp.nrof_encoding_layers):
        model.add(layer_factory.create("AOLConv2D",
                                       filters=s * ps * ps,
                                       kernel_size=(3, 3),
                                       padding="same",
                                       kernel_regularizer=reg,
                                       kernel_initializer=ki3x3,
                                       ))
        model.add(layer_factory.create("MaxMinActivation"))

    # Reduce number of channels:
    model.add(layer_factory.create("AOLConv2D",
                                   filters=s * ps * ps,
                                   kernel_size=(1, 1),
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki1x1,
                                   ))
    model.add(layer_factory.create("FirstChannels",
                                   nrof_channels=fd))

    # Reshape to 1x1 spatial size:
    model.add(tf.keras.layers.Flatten())

    for layer_idx in range(hp.nrof_aggregation_layers):
        model.add(layer_factory.create("AOLDense",
                                       units=fd * np * np,
                                       kernel_regularizer=reg,
                                       ))
        if layer_idx < hp.nrof_aggregation_layers - 1:
            model.add(layer_factory.create("MaxMinActivation"))

    model.add(layer_factory.create("FirstChannels",
                                   nrof_channels=hp.nrof_classes,
                                   ndim=2))

    """
    # Reshape to 1x1 spatial size:
    model.add(layer_factory.create("ConcatenationPooling",
                                   pool_size=(np, np)))
    model.add(layer_factory.create("AOLConv2D",
                                   filters=fd * np * np,
                                   kernel_size=(1, 1),
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki1x1,
                                   ))

    model.add(layer_factory.create("MaxMinActivation"))

    for layer_idx in range(hp.nrof_aggregation_layers):
        model.add(layer_factory.create("AOLConv2D",
                                       filters=fd * np * np,
                                       kernel_size=(1, 1),
                                       kernel_regularizer=reg,
                                       kernel_initializer=ki1x1,
                                       ))
        if layer_idx < hp.nrof_aggregation_layers - 1:
            model.add(layer_factory.create("MaxMinActivation"))

    model.add(layer_factory.create("FirstChannels",
                                   nrof_channels=hp.nrof_classes))
    model.add(tf.keras.layers.Flatten())
    """
    return model
