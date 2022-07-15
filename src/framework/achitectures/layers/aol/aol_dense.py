import tensorflow as tf

from framework.achitectures.layers.aol.aol_rescale import aol_rescale

from framework.achitectures.layers.layer import Layer
from framework.achitectures.layers.initializers.initializer_factory import initializer_factory

from dataclasses import dataclass
from typing import Tuple, Optional, Union, Callable


@dataclass
class AOLDenseHyperparameters:
    activation: Optional[str] = None
    use_bias: bool = True
    kernel_initializer: str = "identity"
    kernel_regularizer: Optional[Union[Callable, float]] = None

    nrof_inputs: Optional[int] = None
    nrof_outputs: Optional[int] = None


class AOLDense(Layer):
    print_name = "AOL Dense"

    def __init__(self, units, **kwargs):
        super(AOLDense, self).__init__()

        self.hp = AOLDenseHyperparameters(nrof_outputs=units,
                                          **kwargs)

        if isinstance(self.hp.kernel_regularizer, float):  # Default regularizer: L2
            self.kernel_regularizer = tf.keras.regularizers.l2(self.hp.kernel_regularizer)
        else:
            self.kernel_regularizer = self.hp.kernel_regularizer

        self.activation_fn = tf.keras.activations.get(self.hp.activation)

        self.kernel_parameters = None
        self.bias_parameters = None

    def __str__(self):
        return f"I am a {self.print_name}, {self.hp.nrof_inputs} -> {self.hp.nrof_outputs}."

    @property
    def kernel_weights(self):
        return aol_rescale(self.kernel_parameters)

    def build(self, input_shape):
        print(f"Building {self.print_name} with input shape {input_shape}.")
        self.hp.nrof_inputs = input_shape[-1]
        self.initialize_weights()

    def call(self, x, *args):
        if self.kernel_regularizer is not None:
            regularization_loss = self.kernel_regularizer(self.kernel_parameters)  # Loss on parameters!
            self.add_loss(regularization_loss)

        x_new = tf.matmul(a=x, b=self.kernel_weights)
        x_new = x_new + self.bias_parameters[None, :]
        x_new = self.activation_fn(x_new)
        return x_new

    def initialize_weights(self):
        self.kernel_parameters = self.add_weight(
            name=f"kernel_parameters",  # Needed name to be able to save weights.
            shape=[self.hp.nrof_inputs, self.hp.nrof_outputs],
            initializer=initializer_factory[self.hp.kernel_initializer],
            trainable=True,
        )

        self.bias_parameters = self.add_weight(
            name=f"bias_parameters",  # Needed name to be able to save weights.
            shape=[self.hp.nrof_outputs],
            initializer=tf.keras.initializers.Constant(value=0.),
            trainable=self.hp.use_bias,
        )
