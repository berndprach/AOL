
import tensorflow as tf

from framework.achitectures.layers.layer import Layer

from typing import Optional, Tuple


class ConcatenationPooling(Layer):
    print_name = "Concatenation Pooling Layer"

    def __init__(self,
                 pool_size: Optional[Tuple] = None,
                 strides: Optional[Tuple] = None,
                 padding: str = "valid"
                 ) -> None:

        super().__init__()

        if pool_size is None: pool_size = (2, 2)
        self.ks1 = pool_size[0]
        self.ks2 = pool_size[1]

        if strides is None: strides = (self.ks1, self.ks2)
        self.s1 = strides[0]
        self.s2 = strides[1]

        self.padding = padding

    def call(self, x, training=None):
        patches = tf.image.extract_patches(
            x,
            sizes=[1, self.ks1, self.ks2, 1],
            strides=[1, self.s1, self.s2, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding.upper(),
        )  # shape: [batch, height, width, ks1*ks2*c_in]
        return patches

