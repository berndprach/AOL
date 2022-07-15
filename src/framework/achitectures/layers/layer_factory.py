"""
Factory that produces layers.
"""

from framework.general_code.factory import Factory

from framework.achitectures.layers.aol.aol_layer_factory import aol_layer_factory
from framework.achitectures.layers.basic.basic_layer_factory import basic_layer_factory
from framework.achitectures.layers.basic.keras_layer_factory import keras_layer_factory

layer_factory = Factory(name="Layers", products={})
layer_factory.add_subfactory(aol_layer_factory, safe_add=True)
layer_factory.add_subfactory(basic_layer_factory, safe_add=True)
layer_factory.add_subfactory(keras_layer_factory, safe_add=True)
