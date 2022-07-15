"""
Basic functionality of a custom layer.
"""

import tensorflow.keras.layers as layers

from framework.general_code.functionality import add_article

from abc import ABC, abstractmethod


class Layer(layers.Layer, ABC):
    nrof_layers = 0  # Keeps count of the number of layers initialized.
    print_name = "Layer"

    def __init__(self):
        print(f"Initialising {self.print_name}!.")

        Layer.nrof_layers = Layer.nrof_layers + 1
        name = self.print_name.lower().replace(" ", "_") + str(Layer.nrof_layers)

        super(Layer, self).__init__(name=name)

    def __str__(self):
        return f"I am {add_article(self.print_name)}!"

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def build(self, input_shape):  # input_shape includes batch_size.
        print(f"Building {self.print_name} with input shape {input_shape}.")

    @abstractmethod
    def call(self, x, *args):
        pass
