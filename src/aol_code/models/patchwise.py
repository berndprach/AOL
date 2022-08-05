import tensorflow as tf

import aol_code.layers as layers
import aol_code.layers.initializers as initializers

from aol_code.location_manager import location_manager as loc
from dataclasses import dataclass, asdict
from typing import Tuple, Union, Callable


@dataclass
class PatchwiseHyperparameters:
    patch_size: Tuple[int, int] = (4, 4)
    input_size: Tuple[int, int] = (32, 32)
    filters_after_downsize: int = 8
    patch_encoding_sizes: int = 4*4*12
    nrof_encoding_layers: int = 12
    nrof_aggregation_layers: int = 14
    kernel_regularizer: Union[Callable, float] = 0.
    nrof_classes: int = 10
    nrof_input_channels: int = 3

    @property
    def nrof_patches(self):
        fits0 = self.input_size[0] // self.patch_size[0]
        fits1 = self.input_size[1] // self.patch_size[1]
        return fits0 * fits1

    @property
    def dense_units(self):
        return self.nrof_patches * self.filters_after_downsize


def get_patchwise_model(**kwargs):

    hp = PatchwiseHyperparameters(**kwargs)
    reg = hp.kernel_regularizer

    loc.add_settings(name="Model Settings",
                     nrof_patches=hp.nrof_patches,
                     dense_units=hp.dense_units,
                     **asdict(hp))

    ki3x3 = initializers.IdentityCenterInitializer
    ki1x1 = initializers.IdentityCenterInitializer
    ki_width_change = initializers.RepeatedOrthogonalInitializer

    patch_extraction_layers = [
        tf.keras.layers.InputLayer(
            input_shape=(32, 32, hp.nrof_input_channels)),
        # Extract Patches:
        layers.basic.ConcatenationPooling(pool_size=hp.patch_size),
        layers.aol.AOLConv2D(filters=hp.patch_encoding_sizes,
                             kernel_size=(1, 1),
                             kernel_regularizer=reg,
                             kernel_initializer=ki_width_change,
                             ),
        layers.basic.MaxMinActivation(),
    ]

    patch_processing_layers = []
    for _ in range(hp.nrof_encoding_layers):
        patch_processing_layers.append(
            layers.aol.AOLConv2D(filters=hp.patch_encoding_sizes,
                                 kernel_size=(3, 3),
                                 padding="same",
                                 kernel_regularizer=reg,
                                 kernel_initializer=ki3x3,
                                 )
        )
        patch_processing_layers.append(layers.basic.MaxMinActivation())

    # Reduce number of channels:
    channel_reduction_layers = [
        layers.aol.AOLConv2D(filters=hp.patch_encoding_sizes,
                             kernel_size=(1, 1),
                             kernel_regularizer=reg,
                             kernel_initializer=ki1x1,
                             ),
        layers.basic.FirstChannels(nrof_channels=hp.filters_after_downsize),
    ]

    # Reshape to 1x1 spatial size:
    flatten_layer = tf.keras.layers.Flatten()

    dense_layers = []
    for layer_idx in range(hp.nrof_aggregation_layers):
        dense_layers.append(
            layers.aol.AOLDense(units=hp.dense_units,
                                kernel_initializer="identity",
                                kernel_regularizer=reg,
                                ))
        if layer_idx < hp.nrof_aggregation_layers - 1:
            dense_layers.append(layers.basic.MaxMinActivation())

    final_first_channel_layer = layers.basic.FirstChannels(
        nrof_channels=hp.nrof_classes, ndim=2)

    model = tf.keras.Sequential([
        *patch_extraction_layers,
        *patch_processing_layers,
        *channel_reduction_layers,
        flatten_layer,
        *dense_layers,
        final_first_channel_layer,
    ])

    return model
