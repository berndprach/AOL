import tensorflow as tf

from framework.achitectures.layers.layer_factory import layer_factory

from framework.location_manager import location_manager as loc


def get_standard_convolutional_model(kernel_regularizer=0., nrof_classes=10):
    reg = kernel_regularizer

    loc.add_settings(name="Model Settings",
                     kernel_regularizer=kernel_regularizer,
                     nrof_classes=nrof_classes)

    ki3x3 = "identity_center"
    ki1x1 = "identity_center"
    ki_width_change = "repeated_orthogonal"

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, 3)))

    # To 32x32x32:
    model.add(layer_factory.create("AOLConv2D",
                                   filters=32,
                                   kernel_size=(1, 1),
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki_width_change,
                                   ))
    model.add(layer_factory.create("MaxMinActivation"))

    for w in [32, 64, 128, 256, 512]:
        for _ in range(4):
            model.add(layer_factory.create("AOLConv2D",
                                           filters=w,
                                           kernel_size=(3, 3),
                                           padding="same",
                                           kernel_regularizer=reg,
                                           kernel_initializer=ki3x3,
                                           ))
            model.add(layer_factory.create("MaxMinActivation"))
        model.add(layer_factory.create("ConcatenationPooling",
                                       pool_size=(2, 2)))
        model.add(layer_factory.create("AOLConv2D",
                                       filters=2 * w,
                                       kernel_size=(1, 1),
                                       kernel_regularizer=reg,
                                       kernel_initializer=ki_width_change,
                                       ))

    # model.add(layer_factory.create("MaxMinActivation"))
    model.add(layer_factory.create("AOLConv2D",
                                   filters=1024,
                                   kernel_size=(1, 1),
                                   kernel_regularizer=reg,
                                   kernel_initializer=ki1x1,
                                   ))
    model.add(layer_factory.create("FirstChannels",
                                   nrof_channels=nrof_classes))
    model.add(tf.keras.layers.Flatten())
    return model
