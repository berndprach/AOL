
import tensorflow as tf

from aol_code.layers.aol.aol_rescale import aol_rescale

from aol_code.layers.layer import Layer
from aol_code.layers.get_kernel_regularizer import get_kernel_regularizer
import aol_code.layers.initializers as initializers

from dataclasses import dataclass
from typing import Optional, Union, Callable


@dataclass
class AOLDenseHyperparameters:
    activation: Optional[str] = None
    use_bias: bool = True
    kernel_initializer: str = "repeated_orthogonal"  # "identity"
    kernel_regularizer: Optional[Union[Callable, float]] = None

    nrof_inputs: Optional[int] = None
    nrof_outputs: Optional[int] = None


class AOLDense(Layer):
    def __init__(self, units, **kwargs):
        super(AOLDense, self).__init__()

        self.hp = AOLDenseHyperparameters(nrof_outputs=units, **kwargs)

        self.activation_fn = tf.keras.activations.get(self.hp.activation)
        self.kernel_regularizer = get_kernel_regularizer(
            self.hp.kernel_regularizer)

        self.kernel_parameters = None
        self.bias_parameters = None

    @property
    def kernel_weights(self):
        return aol_rescale(self.kernel_parameters)

    def build(self, input_shape):
        print(f"Building {self.name} with input shape {input_shape}.")
        self.hp.nrof_inputs = input_shape[-1]
        self.initialize_weights()

    def call(self, x, *args):
        if self.kernel_regularizer is not None:
            regularization_loss = self.kernel_regularizer(
                self.kernel_parameters)  # Loss on parameters!
            self.add_loss(regularization_loss)

        x_new = tf.matmul(a=x, b=self.kernel_weights)
        x_new = x_new + self.bias_parameters[None, :]
        x_new = self.activation_fn(x_new)
        return x_new

    def initialize_weights(self):
        self.kernel_parameters = self.add_weight(
            name=f"kernel_parameters",
            shape=[self.hp.nrof_inputs, self.hp.nrof_outputs],
            initializer=initializers.get(self.hp.kernel_initializer),
            trainable=True,
        )

        self.bias_parameters = self.add_weight(
            name=f"bias_parameters",
            shape=[self.hp.nrof_outputs],
            initializer=tf.keras.initializers.Constant(value=0.),
            trainable=self.hp.use_bias,
        )
