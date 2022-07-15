
import unittest
import tensorflow as tf

from framework.achitectures.layers.aol.aol_rescale import get_rescaling_factors, aol_rescale


class TestGetAOLConv2DRescale(unittest.TestCase):

    def test_output_shape(self):
        input_tensor = tf.zeros([27, 64])
        output_tensor = get_rescaling_factors(input_tensor)
        output_shape = list(output_tensor.shape)
        goal_shape = [27]
        self.assertListEqual(goal_shape, output_shape)

    def test_on_orthogonal_input(self):
        input_tensor = tf.initializers.orthogonal()(shape=(13, 13))
        output_tensor = aol_rescale(input_tensor)
        difference = tf.reduce_sum((input_tensor-output_tensor)**2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_on_zero_input(self):
        input_tensor = tf.zeros(shape=(13, 13))
        r = get_rescaling_factors(input_tensor, epsilon=1e-6)
        difference = tf.reduce_sum((r-1e3)**2).numpy()
        self.assertAlmostEqual(difference, 0.)


if __name__ == "__main__":
    print("test")
    unittest.main()
