
import unittest
import tensorflow as tf

from aol_code.layers.basic import FirstChannels


class TestFirstChannels(unittest.TestCase):

    def test_shape(self):
        conv_first_channels_layer = FirstChannels(nrof_channels=10)
        input_tensor = tf.zeros(shape=[100, 32, 32, 32])
        output_tensor = conv_first_channels_layer(input_tensor)
        output_shape = list(output_tensor.shape)
        goal_shape = [100, 32, 32, 10]
        self.assertListEqual(goal_shape, output_shape)

        dense_first_channels_layer = FirstChannels(nrof_channels=256,
                                                   ndim=2)
        input_tensor = tf.zeros(shape=[100, 1024])
        output_tensor = dense_first_channels_layer(input_tensor)
        output_shape = list(output_tensor.shape)
        goal_shape = [100, 256]
        self.assertListEqual(goal_shape, output_shape)


if __name__ == "__main__":
    unittest.main()
