import unittest
import numpy as np
import tensorflow as tf

from framework.achitectures.layers.aol.aol_dense import AOLDense


class TestAOLConv2D(unittest.TestCase):

    def test_output_shape(self):
        aol_dense = AOLDense(units=20)
        input_tensor = tf.zeros([16, 10])
        output_tensor = aol_dense(input_tensor)
        output_shape = list(output_tensor.shape)
        goal_shape = [16, 20]
        self.assertListEqual(goal_shape, output_shape)

    def test_lipschitzness(self):
        aol_dense = AOLDense(units=20)

        # Build layer and then change kernel parameters.
        # The rescaling should make sure that the layer is still Lipschitz!
        input_tensor = tf.range(10, dtype=tf.float32)[None, :]
        aol_dense(input_tensor)  # build layer
        aol_dense.kernel_parameters = tf.reshape(tf.range(7, 207, dtype=tf.float32), [10, 20]) / 20
        # print(f"Kernel weights:\n{aol_conv.kernel_weights}")

        input_tensor1 = tf.range(10, dtype=tf.float32)[None, :]
        output_tensor1 = aol_dense(input_tensor1)

        input_tensor2 = tf.convert_to_tensor([5.]*10)[None, :]
        output_tensor2 = aol_dense(input_tensor2)

        input_difference = tf.reduce_sum((input_tensor1 - input_tensor2) ** 2).numpy()
        output_difference = tf.reduce_sum((output_tensor1 - output_tensor2) ** 2).numpy()
        print(f"Input and Output differences: {input_difference}, {output_difference}")
        self.assertLess(output_difference, input_difference)

        # Do multiple pseudo-random tests:
        np.random.seed(1111)
        for test_idx in range(10):
            input_tensor1 = tf.convert_to_tensor(np.random.normal(size=10))[None, :]
            output_tensor1 = aol_dense(input_tensor1)

            input_tensor2 = tf.convert_to_tensor(np.random.normal(size=10))[None, :]
            output_tensor2 = aol_dense(input_tensor2)

            input_difference = tf.reduce_sum((input_tensor1 - input_tensor2) ** 2).numpy()
            output_difference = tf.reduce_sum((output_tensor1 - output_tensor2) ** 2).numpy()
            print(f"Test {test_idx}: Input and Output differences: {input_difference}, {output_difference}")

            self.assertLess(output_difference, input_difference)

    def test_orthogonality(self):
        aol_dense = AOLDense(units=16,
                             kernel_initializer="orthogonal")

        input_tensor1 = tf.range(16, dtype=tf.float32)[None, :]
        output_tensor1 = aol_dense(input_tensor1)

        input_tensor2 = tf.convert_to_tensor([5.]*16)[None, :]
        output_tensor2 = aol_dense(input_tensor2)

        input_difference = tf.reduce_sum((input_tensor1 - input_tensor2) ** 2).numpy()
        output_difference = tf.reduce_sum((output_tensor1 - output_tensor2) ** 2).numpy()
        print(f"Input and Output differences: {input_difference}, {output_difference}")

        self.assertLess(output_difference, input_difference)
        self.assertAlmostEqual(output_difference, input_difference, places=2)

        # Do multiple pseudo-random tests:
        np.random.seed(1111)
        for test_idx in range(10):
            input_tensor1 = tf.convert_to_tensor(np.random.normal(size=16))[None, :]
            output_tensor1 = aol_dense(input_tensor1)

            input_tensor2 = tf.convert_to_tensor(np.random.normal(size=16))[None, :]
            output_tensor2 = aol_dense(input_tensor2)

            input_difference = tf.reduce_sum((input_tensor1 - input_tensor2) ** 2).numpy()
            output_difference = tf.reduce_sum((output_tensor1 - output_tensor2) ** 2).numpy()
            print(f"Test {test_idx}: Input and Output differences: {input_difference}, {output_difference}")

            self.assertLess(output_difference, input_difference)
            self.assertAlmostEqual(output_difference, input_difference, places=2)


if __name__ == "__main__":
    print("test")
    unittest.main()
