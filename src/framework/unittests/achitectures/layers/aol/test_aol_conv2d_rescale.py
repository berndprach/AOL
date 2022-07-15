
import unittest
import tensorflow as tf

from framework.achitectures.layers.aol.aol_conv2d_rescale import get_aol_conv2d_rescale


class TestGetAOLConv2DRescale(unittest.TestCase):

    def test_output_shape(self):
        input_tensor = tf.zeros([3, 3, 32, 64])
        output_tensor = get_aol_conv2d_rescale(input_tensor)
        output_shape = list(output_tensor.shape)
        goal_shape = [32]
        self.assertListEqual(goal_shape, output_shape)

    def test_on_orthogonal_input(self):
        input_tensor = tf.initializers.orthogonal()(shape=(1, 1, 7, 7))
        r = get_aol_conv2d_rescale(input_tensor)
        difference = tf.reduce_sum((r-1.)**2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_on_zero_input(self):
        input_tensor = tf.zeros(shape=(1, 1, 7, 7))
        r = get_aol_conv2d_rescale(input_tensor, epsilon=1e-6)
        difference = tf.reduce_sum((r-1e3)**2).numpy()
        self.assertAlmostEqual(difference, 0.)


if __name__ == "__main__":
    print("test")
    unittest.main()
