from framework.general_code.factory import Factory

from framework.achitectures.layers.basic.concatenation_pooling import ConcatenationPooling
from framework.achitectures.layers.basic.first_channels_layers import FirstChannelsLayer
from framework.achitectures.layers.basic.max_min_activation import MaxMinActivation

basic_layer_dict = {
    "ConcatenationPooling": ConcatenationPooling,
    "FirstChannels": FirstChannelsLayer,
    "MaxMinActivation": MaxMinActivation,
}

basic_layer_factory = Factory(name="Basic Layer Factory",
                              products=basic_layer_dict)

