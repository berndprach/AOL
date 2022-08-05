
import tensorflow as tf

from aol_code.layers.layer import Layer

from typing import Optional, Tuple


class ConcatenationPooling(Layer):
    def __init__(self,
                 pool_size: Optional[Tuple] = None,
                 strides: Optional[Tuple] = None,
                 padding: str = "valid"
                 ) -> None:

        super().__init__()

        if pool_size is None: pool_size = (2, 2)
        self.pool_size = pool_size

        if strides is None: strides = pool_size
        self.strides = strides

        self.padding = padding

    def call(self, x, training=None):
        patches = tf.image.extract_patches(
            x,
            sizes=[1, self.pool_size[0], self.pool_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding.upper(),
        )  # shape: [bs, new_height, new_width, ks1*ks2*c_in]
        return patches

