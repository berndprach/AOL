from framework.general_code.factory import Factory

from framework.achitectures.layers.aol.aol_dense import AOLDense
from framework.achitectures.layers.aol.aol_conv2d import AOLConv2D

aol_layer_dict = {
    "AOLDense": AOLDense,
    "AOLConv2D": AOLConv2D,
}

aol_layer_factory = Factory(name="AOL Layer Factory",
                            products=aol_layer_dict)
