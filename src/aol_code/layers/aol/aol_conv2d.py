import tensorflow as tf

from aol_code.layers.aol.aol_conv2d_rescale import aol_conv2d_rescale

from aol_code.layers.layer import Layer
from aol_code.layers.get_kernel_regularizer import get_kernel_regularizer
import aol_code.layers.initializers as initializers

from dataclasses import dataclass
from typing import Tuple, Optional, Union, Callable


@dataclass
class AOLConv2DHyperparameters:
    filters: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int] = (1, 1)
    padding: str = "valid"
    dilation_rate: Tuple[int, int] = (1, 1)
    activation: Optional[str] = None
    use_bias: bool = True
    kernel_initializer: str = "orthogonal_center"
    kernel_regularizer: Optional[Union[Callable, float]] = None

    nrof_input_channels: Optional[int] = None


class AOLConv2D(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(AOLConv2D, self).__init__()
        self.hp = AOLConv2DHyperparameters(filters, kernel_size, **kwargs)

        self.activation_fn = tf.keras.activations.get(self.hp.activation)
        self.kernel_regularizer = get_kernel_regularizer(
            self.hp.kernel_regularizer)

        if self.hp.strides[0] * self.hp.strides[1] > 1:
            raise NotImplementedError(
                "Use a ConcatenationPooling layer to apply a stride instead,"
                "in order to have a tighter upper bound!")

        self.kernel_parameters = None
        self.bias_parameters = None

    @property
    def kernel_weights(self):
        return aol_conv2d_rescale(self.kernel_parameters)

    def build(self, input_shape):
        print(f"Building {self.name} with input shape {input_shape}.")
        self.hp.nrof_input_channels = input_shape[-1]
        self.initialize_weights()

    def call(self, x, *args):
        if self.kernel_regularizer is not None:
            # Apply Loss (to the kernel parameters).
            regularization_loss = self.kernel_regularizer(
                self.kernel_parameters)
            self.add_loss(regularization_loss)

        strides = [1, self.hp.strides[0], self.hp.strides[1], 1]
        x_new = tf.nn.conv2d(
            x,
            self.kernel_weights,
            strides=strides,
            padding=self.hp.padding.upper(),
            dilations=self.hp.dilation_rate
        )

        x_new = x_new + self.bias_parameters[None, None, None, :]
        x_new = self.activation_fn(x_new)
        return x_new

    def initialize_weights(self):
        self.kernel_parameters = self.add_weight(
            name=f"kernel_parameters",
            shape=[self.hp.kernel_size[0],
                   self.hp.kernel_size[1],
                   self.hp.nrof_input_channels,
                   self.hp.filters],
            initializer=initializers.get(self.hp.kernel_initializer),
            trainable=True,
        )

        self.bias_parameters = self.add_weight(
            name=f"bias_parameters",
            shape=[self.hp.filters],
            initializer=tf.keras.initializers.Constant(value=0.),
            trainable=self.hp.use_bias,
        )

