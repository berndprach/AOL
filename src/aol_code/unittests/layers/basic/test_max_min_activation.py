
import unittest
import tensorflow as tf

from aol_code.layers.basic import MaxMinActivation


class TestMaxMinActivation(unittest.TestCase):

    def test_shape(self):
        print("\n\n*** Shape Test: ***")
        max_min_acitvation = MaxMinActivation()
        input_tensor = tf.zeros([1, 10, 10, 6])
        output_tensor = max_min_acitvation(input_tensor)
        output_shape = list(output_tensor.shape)
        goal_shape = [1, 10, 10, 6]
        self.assertListEqual(goal_shape, output_shape)

    def test_values1D(self):
        print("\n\n*** Values Test with (batched) 1D input: ***")
        max_min_acitvation = MaxMinActivation()
        input_tensor = tf.convert_to_tensor([-1., 1., 2., -2.])[None, :]
        output_tensor = max_min_acitvation(input_tensor)
        print(output_tensor)
        goal_output = tf.convert_to_tensor([1., -1., 2., -2.])[None, :]
        difference = tf.reduce_sum((output_tensor-goal_output)**2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_values2D(self):
        print("\n\n*** Values Test with (batched) 2D input: ***")
        max_min_acitvation = MaxMinActivation()
        input_tensor = tf.convert_to_tensor([-1., 1., 2., -2.])[None, None, :]
        output_tensor = max_min_acitvation(input_tensor)
        print(output_tensor)
        goal_output = tf.convert_to_tensor([1., -1., 2., -2.])[None, None, :]
        difference = tf.reduce_sum((output_tensor-goal_output)**2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_values3D(self):
        print("\n\n*** Values Test with (batched) 3D input: ***")
        max_min_acitvation = MaxMinActivation()
        input_tensor = tf.convert_to_tensor([1., 0., 2., 3.])
        input_tensor = input_tensor[None, None, None, :]
        output_tensor = max_min_acitvation(input_tensor)
        print(output_tensor)
        goal_output = tf.convert_to_tensor([1., 0., 3., 2.])
        goal_output = goal_output[None, None, None, :]
        difference = tf.reduce_sum((output_tensor-goal_output)**2).numpy()
        self.assertAlmostEqual(difference, 0.)


if __name__ == "__main__":
    unittest.main()
