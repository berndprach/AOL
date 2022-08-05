"""
Default values taken from
https://github.com/moritzhambach/Image-Augmentation-in-Keras-CIFAR-10-
"""
import tensorflow as tf

from dataclasses import dataclass


@dataclass
class SpatialAugmentationParameters:
    rotation_range: int = 5
    horizontal_flip: bool = True
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1


class SpatialAugmentation:
    def __init__(self, **kwargs):
        self.parameters = SpatialAugmentationParameters(**kwargs)

        self.rotation_layer = None
        self.shift_layer = None

        self.define_layers()

    def define_layers(self):
        # I guess it would be more efficient to put those layers in the model directly!

        self.rotation_layer = tf.keras.layers.RandomRotation(factor=self.parameters.rotation_range / 360)

        self.shift_layer = tf.keras.layers.RandomTranslation(
            height_factor=self.parameters.height_shift_range,
            width_factor=self.parameters.width_shift_range,
        )

    def on_train_set(self, img, *args):
        img = self.rotation_layer(img)
        img = self.shift_layer(img)
        img = tf.image.random_flip_left_right(img)

        return (img, *args)

    # @staticmethod
    def on_val_set(self, img, *args):
        return (img, *args)
