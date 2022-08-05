
import unittest
import tensorflow as tf

from aol_code.layers.basic import ConcatenationPooling


class TestConcatenationPooling(unittest.TestCase):

    def test_shape(self):
        concatenation_pooling_layer = ConcatenationPooling()
        input_tensor = tf.zeros(shape=[100, 32, 32, 3])
        output_tensor = concatenation_pooling_layer(input_tensor)
        output_shape = list(output_tensor.shape)
        # print(output_shape)
        goal_shape = [100, 16, 16, 12]
        self.assertListEqual(goal_shape, output_shape)

    def test_values(self):
        concatenation_pooling_layer = ConcatenationPooling()
        input_values = tf.convert_to_tensor([[1., 2.], [3., 4.]])
        input_tensor = input_values[None, :, :, None]
        output_tensor = concatenation_pooling_layer(input_tensor)
        goal_output_values = tf.convert_to_tensor([1., 2., 3., 4.])
        goal_output = goal_output_values[None, None, None, :]
        difference = tf.reduce_sum((goal_output-output_tensor)**2).numpy()
        self.assertAlmostEqual(difference, 0.)


if __name__ == "__main__":
    unittest.main()
