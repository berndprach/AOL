"""
Layer class that makes sure all layers have different names,
by adding the layer index to the name.
"""

import tensorflow.keras.layers as layers


class Layer(layers.Layer):
    nrof_layers = 0  # Keeps count of the number of layers initialized.

    def __init__(self):
        Layer.nrof_layers = Layer.nrof_layers + 1
        name = self.__class__.__name__ + "Layer" + str(Layer.nrof_layers)
        # print(f"Initialising {name}.")

        super(Layer, self).__init__(name=name)

    def __str__(self):
        return f"I am {add_article(self.name)}!"

    def build(self, input_shape):  # input_shape includes batch_size.
        print(f"Building {self.name} "
              f"with input shape {input_shape}.")


def add_article(word: str) -> str:
    if word[0].lower() in ["a", "e", "i", "o", "u"]:
        return f"an {word}"
    else:
        return f"a {word}"
