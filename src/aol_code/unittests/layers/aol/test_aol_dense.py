
import unittest
import numpy as np
import tensorflow as tf

from aol_code.layers.aol.aol_dense import AOLDense


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
        entries = tf.range(7, 207, dtype=tf.float32) / 20
        aol_dense.kernel_parameters = tf.reshape(entries, [10, 20])

        # Do multiple tests with pseudo-random inputs:
        np.random.seed(1111)
        for test_idx in range(10):
            input_batch1 = np.random.normal(size=[1, 10])
            input_tensor1 = tf.convert_to_tensor(input_batch1)
            output_tensor1 = aol_dense(input_tensor1)

            input_batch2 = np.random.normal(size=[1, 10])
            input_tensor2 = tf.convert_to_tensor(input_batch2)
            output_tensor2 = aol_dense(input_tensor2)

            input_difference_vector = input_tensor1 - input_tensor2
            input_difference_sq = tf.reduce_sum(input_difference_vector ** 2)
            output_difference_vector = output_tensor1 - output_tensor2
            output_difference_sq = tf.reduce_sum(output_difference_vector ** 2)
            print(f"Test {test_idx}: Input and Output differences: "
                  f"{input_difference_sq:.3f}, {output_difference_sq:.3f}")

            self.assertLess(output_difference_sq.numpy(),
                            input_difference_sq.numpy())

    def test_orthogonality(self):
        aol_dense = AOLDense(units=16,
                             kernel_initializer="orthogonal")

        # Do multiple tests with pseudo-random inputs:
        np.random.seed(1111)
        for test_idx in range(10):
            input_batch1 = np.random.normal(size=[1, 10])
            input_tensor1 = tf.convert_to_tensor(input_batch1)
            output_tensor1 = aol_dense(input_tensor1)

            input_batch2 = np.random.normal(size=[1, 10])
            input_tensor2 = tf.convert_to_tensor(input_batch2)
            output_tensor2 = aol_dense(input_tensor2)

            input_difference_vector = input_tensor1 - input_tensor2
            input_difference_sq = tf.reduce_sum(input_difference_vector ** 2)
            input_difference_sq = input_difference_sq.numpy()
            output_difference_vector = output_tensor1 - output_tensor2
            output_difference_sq = tf.reduce_sum(output_difference_vector ** 2)
            output_difference_sq = output_difference_sq.numpy()
            print(f"Test {test_idx}: Input and Output differences: "
                  f"{input_difference_sq:.3f}, {output_difference_sq:.3f}")

            self.assertLess(output_difference_sq, input_difference_sq)
            self.assertAlmostEqual(output_difference_sq,
                                   input_difference_sq,
                                   places=2)


if __name__ == "__main__":
    unittest.main()
