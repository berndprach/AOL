
import tensorflow as tf

from framework.achitectures.layers.aol.aol_conv2d_rescale import aol_conv2d_rescale

from framework.achitectures.layers.layer import Layer
from framework.achitectures.layers.initializers.initializer_factory import initializer_factory

from dataclasses import dataclass
from typing import Tuple, Optional, Union, Callable


@dataclass
class AOLConv2DHyperparameters:
    strides: Tuple[int, int] = (1, 1)
    padding: str = "valid"
    dilation_rate: Tuple[int, int] = (1, 1)
    activation: Optional[str] = None
    use_bias: bool = True
    kernel_initializer: str = "identity_center"
    kernel_regularizer: Optional[Union[Callable, float]] = None

    ks1: Optional[int] = None
    ks2: Optional[int] = None
    s1: Optional[int] = None
    s2: Optional[int] = None

    nrof_input_channels: Optional[int] = None
    nrof_output_channels: Optional[int] = None


class AOLConv2D(Layer):
    print_name = "AOL Conv2D"

    def __init__(self, filters, kernel_size, **kwargs):
        super(AOLConv2D, self).__init__()

        self.hp = AOLConv2DHyperparameters(**kwargs)

        if isinstance(self.hp.kernel_regularizer, float):  # Default regularizer: L2
            self.kernel_regularizer = tf.keras.regularizers.l2(self.hp.kernel_regularizer)
        else:
            self.kernel_regularizer = self.hp.kernel_regularizer

        self.activation_fn = tf.keras.activations.get(self.hp.activation)
        self.hp.nrof_output_channels = filters

        assert type(kernel_size) is tuple, "I don't wanna deal with this, just give me a tuple for kernel_size!"
        assert type(self.hp.strides) is tuple, "I don't wanna deal with this, just give me a tuple for strides!"
        self.hp.ks1, self.hp.ks2 = kernel_size
        self.hp.s1, self.hp.s2 = self.hp.strides

        if self.hp.s1 * self.hp.s2 > 1:
            raise NotImplementedError("With a stride, channel-wise rescaling is not optimal!\n"
                                      + "Instead, rescaling depending on the spatial location "
                                      + "within the kernel tends to perform better.\n"
                                      + "Use a ConcatenationPooling layer to apply a stride instead!")

        self.kernel_parameters = None
        self.bias_parameters = None

    def __str__(self):
        outstr = f"I am a {self.print_name}, " \
                 + f"[{self.hp.ks1}, {self.hp.ks2}, {self.hp.nrof_input_channels}] -> {self.hp.nrof_output_channels}" \
                 + "."
        return outstr

    @property
    def kernel_weights(self):
        return aol_conv2d_rescale(self.kernel_parameters)

    def build(self, input_shape):
        print(f"Building {self.print_name} with input shape {input_shape}.")
        self.hp.nrof_input_channels = input_shape[-1]
        self.initialize_weights()

    def call(self, x, *args):
        if self.kernel_regularizer is not None:
            regularization_loss = self.kernel_regularizer(self.kernel_parameters)  # Loss on parameters!
            self.add_loss(regularization_loss)

        strides = [1, self.hp.s1, self.hp.s2, 1]
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
            name=f"kernel_parameters",  # Needed name to be able to save weights.
            shape=[self.hp.ks1, self.hp.ks2, self.hp.nrof_input_channels, self.hp.nrof_output_channels],
            initializer=initializer_factory[self.hp.kernel_initializer],
            trainable=True,
        )

        self.bias_parameters = self.add_weight(
            name=f"bias_parameters",  # Needed name to be able to save weights.
            shape=[self.hp.nrof_output_channels],
            initializer=tf.keras.initializers.Constant(value=0.),
            trainable=self.hp.use_bias,
        )
