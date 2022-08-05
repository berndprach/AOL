
import tensorflow as tf

from dataclasses import dataclass


@dataclass
class ColorAugmentationParameters:
    hue: float = 0.02
    saturation1: float = .3
    saturation2: float = 2.
    brightness: float = 0.1
    contrast1: float = .5
    contrast2: float = 2.


class ColorAugmentation:
    def __init__(self, **kwargs):
        self.parameters = ColorAugmentationParameters(**kwargs)

    def on_train_set(self, img, *args):
        img = tf.image.random_hue(img, self.parameters.hue)
        img = tf.image.random_saturation(img, self.parameters.saturation1, self.parameters.saturation2)
        img = tf.image.random_brightness(img, self.parameters.brightness)
        img = tf.image.random_contrast(img, self.parameters.contrast1, self.parameters.contrast2)

        img = tf.clip_by_value(img, 0., 1.)

        return (img, *args)

    @staticmethod
    def on_val_set(img, *args):
        return (img, *args)

