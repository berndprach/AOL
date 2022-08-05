import tensorflow as tf

import aol_code.layers as layers
import aol_code.layers.initializers as initializers

from aol_code.location_manager import location_manager as loc


def get_standard_convolutional_model(kernel_regularizer=0.,
                                     nrof_classes=10,
                                     nrof_layers_per_block=5):
    reg = kernel_regularizer

    loc.add_settings(name="Model Settings",
                     kernel_regularizer=kernel_regularizer,
                     nrof_classes=nrof_classes)

    ki1x1 = initializers.IdentityCenterInitializer
    ki3x3 = initializers.IdentityCenterInitializer
    ki_width_change = initializers.RepeatedOrthogonalInitializer

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, 3)))

    # To 32x32x32:
    model.add(layers.aol.AOLConv2D(filters=32,
                                   kernel_size=(1, 1),
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki_width_change,
                                   ))
    model.add(layers.basic.MaxMinActivation())

    for block_width in [32, 64, 128, 256, 512]:
        for _ in range(nrof_layers_per_block-1):
            model.add(layers.aol.AOLConv2D(filters=block_width,
                                           kernel_size=(3, 3),
                                           padding="same",
                                           kernel_regularizer=reg,
                                           kernel_initializer=ki3x3,
                                           ))
            model.add(layers.basic.MaxMinActivation())
        model.add(layers.basic.ConcatenationPooling(pool_size=(2, 2)))
        model.add(layers.aol.AOLConv2D(filters=2 * block_width,
                                       kernel_size=(1, 1),
                                       kernel_regularizer=reg,
                                       kernel_initializer=ki_width_change,
                                       ))

    # model.add(layer_factory.create("MaxMinActivation"))
    model.add(layers.aol.AOLConv2D(filters=1024,
                                   kernel_size=(1, 1),
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki1x1,
                                   ))
    model.add(layers.basic.FirstChannels(nrof_channels=nrof_classes))
    model.add(tf.keras.layers.Flatten())
    return model
