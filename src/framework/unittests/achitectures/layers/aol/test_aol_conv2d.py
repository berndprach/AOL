import unittest
import numpy as np
import tensorflow as tf

from framework.achitectures.layers.aol.aol_conv2d import AOLConv2D


class TestAOLConv2D(unittest.TestCase):

    def test_output_shape(self):
        aol_conv = AOLConv2D(filters=5, kernel_size=(1, 1))
        input_tensor = tf.zeros([1, 10, 10, 5])
        output_tensor = aol_conv(input_tensor)
        output_shape = list(output_tensor.shape)
        goal_shape = [1, 10, 10, 5]
        self.assertListEqual(goal_shape, output_shape)

    def test_layer_is_identity_map_at_initialization(self):
        aol_conv = AOLConv2D(filters=5, kernel_size=(1, 1))
        input_tensor = tf.convert_to_tensor([1., 2., 3., 4., 5.])[None, None, None, :]
        output_tensor = aol_conv(input_tensor)
        difference = tf.reduce_sum((input_tensor - output_tensor) ** 2).numpy()
        self.assertAlmostEqual(difference, 0.)

        aol_conv = AOLConv2D(filters=5, kernel_size=(3, 3), padding="same")
        input_tensor = tf.reshape(tf.range(45, dtype=tf.float32), [1, 3, 3, 5])
        output_tensor = aol_conv(input_tensor)
        difference = tf.reduce_sum((input_tensor - output_tensor) ** 2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_lipschitzness(self):
        aol_conv = AOLConv2D(filters=3, kernel_size=(1, 1))

        # Build layer and then change kernel parameters.
        # The rescaling should make sure that the layer is still Lipschitz!
        input_tensor = tf.convert_to_tensor([1., 2., 3.])[None, None, None, :]
        aol_conv(input_tensor)  # build layer
        aol_conv.kernel_parameters = tf.reshape(tf.range(9, dtype=tf.float32), [1, 1, 3, 3]) / 20
        # print(f"Kernel weights:\n{aol_conv.kernel_weights}")

        input_tensor1 = tf.convert_to_tensor([1., 2., 3.])[None, None, None, :]
        output_tensor1 = aol_conv(input_tensor1)

        input_tensor2 = tf.convert_to_tensor([2., 2., 2.])[None, None, None, :]
        output_tensor2 = aol_conv(input_tensor2)

        input_difference = tf.reduce_sum((input_tensor1 - input_tensor2) ** 2).numpy()
        output_difference = tf.reduce_sum((output_tensor1 - output_tensor2) ** 2).numpy()
        print(f"Input and Output differences: {input_difference}, {output_difference}")
        self.assertLess(output_difference, input_difference)

        # Do multiple pseudo-random tests:
        np.random.seed(1111)
        for test_idx in range(10):
            input_tensor1 = tf.convert_to_tensor(np.random.normal(size=3))[None, None, None, :]
            output_tensor1 = aol_conv(input_tensor1)

            input_tensor2 = tf.convert_to_tensor(np.random.normal(size=3))[None, None, None, :]
            output_tensor2 = aol_conv(input_tensor2)

            input_difference = tf.reduce_sum((input_tensor1 - input_tensor2) ** 2).numpy()
            output_difference = tf.reduce_sum((output_tensor1 - output_tensor2) ** 2).numpy()
            print(f"Test {test_idx}: Input and Output differences: {input_difference}, {output_difference}")

            self.assertLess(output_difference, input_difference)

    def test_orthogonality(self):
        aol_conv = AOLConv2D(filters=16,
                             kernel_size=(1, 1),
                             kernel_initializer="repeated_orthogonal")

        input_tensor1 = tf.convert_to_tensor([1., 2., 3.])[None, None, None, :]
        output_tensor1 = aol_conv(input_tensor1)

        input_tensor2 = tf.convert_to_tensor([2., 2., 2.])[None, None, None, :]
        output_tensor2 = aol_conv(input_tensor2)

        input_difference = tf.reduce_sum((input_tensor1 - input_tensor2) ** 2).numpy()
        output_difference = tf.reduce_sum((output_tensor1 - output_tensor2) ** 2).numpy()
        print(f"Input and Output differences: {input_difference}, {output_difference}")

        self.assertLess(output_difference, input_difference)
        self.assertAlmostEqual(output_difference, input_difference, places=3)

        # Do multiple pseudo-random tests:
        np.random.seed(1111)
        for test_idx in range(10):
            input_tensor1 = tf.convert_to_tensor(np.random.normal(size=3))[None, None, None, :]
            output_tensor1 = aol_conv(input_tensor1)

            input_tensor2 = tf.convert_to_tensor(np.random.normal(size=3))[None, None, None, :]
            output_tensor2 = aol_conv(input_tensor2)

            input_difference = tf.reduce_sum((input_tensor1 - input_tensor2) ** 2).numpy()
            output_difference = tf.reduce_sum((output_tensor1 - output_tensor2) ** 2).numpy()
            print(f"Test {test_idx}: Input and Output differences: {input_difference}, {output_difference}")

            self.assertLess(output_difference, input_difference)
            self.assertAlmostEqual(output_difference, input_difference, places=3)


if __name__ == "__main__":
    print("test")
    unittest.main()
