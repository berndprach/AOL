import unittest
import numpy as np
import tensorflow as tf

from aol_code.layers.aol import AOLConv2D


class TestAOLConv2D(unittest.TestCase):

    def test_output_shape(self):
        print("\n\n*** Output Shape Test: ***")
        aol_conv = AOLConv2D(filters=7,
                             kernel_size=(3, 3),
                             padding="valid")
        input_tensor = tf.zeros([1, 10, 10, 5])
        output_tensor = aol_conv(input_tensor)
        output_shape = list(output_tensor.shape)
        goal_shape = [1, 8, 8, 7]
        self.assertListEqual(goal_shape, output_shape)

    def test_layer_is_identity_map_at_initialization(self):
        print("\n\n*** Identity Initialization Test: ***")
        aol_conv = AOLConv2D(filters=5,
                             kernel_size=(3, 3),
                             padding="same",
                             kernel_initializer="identity_center")
        input_tensor = tf.reshape(tf.range(45, dtype=tf.float32), [1, 3, 3, 5])
        output_tensor = aol_conv(input_tensor)
        difference = tf.reduce_sum((input_tensor - output_tensor) ** 2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_lipschitzness(self):
        print("\n\n*** Lipschitzness Test: ***")
        input_width, output_width = 8, 12
        aol_conv = AOLConv2D(filters=output_width,
                             kernel_size=(3, 3))

        # Build layer and then change kernel parameters.
        # The rescaling should make sure that the layer is still Lipschitz!
        # aol_conv.build(input_shape=(1, 10, 10, 8))
        aol_conv(tf.zeros(shape=[1, 10, 10, input_width], dtype=tf.float32))
        entries = tf.range(10, 10 + 3 * 3 * input_width * output_width,
                           dtype=tf.float32) / 1000
        kernel_parameters = tf.reshape(entries,
                                       [3, 3, input_width, output_width])
        aol_conv.kernel_parameters = kernel_parameters
        # print(aol_conv.kernel_parameters[:2, :2, :2, :2])

        # Do multiple tests with pseudo-random inputs:
        np.random.seed(1111)
        for test_idx in range(10):
            input_batch1 = np.random.normal(size=[1, 10, 10, input_width])
            input_tensor1 = tf.convert_to_tensor(input_batch1)
            output_tensor1 = aol_conv(input_tensor1)

            input_batch2 = np.random.normal(size=[1, 10, 10, input_width])
            input_tensor2 = tf.convert_to_tensor(input_batch2)
            output_tensor2 = aol_conv(input_tensor2)

            input_difference_vector = input_tensor1 - input_tensor2
            input_difference_sq = tf.reduce_sum(input_difference_vector ** 2)
            input_difference_sq = input_difference_sq.numpy()
            output_difference_vector = output_tensor1 - output_tensor2
            output_difference_sq = tf.reduce_sum(output_difference_vector ** 2)
            output_difference_sq = output_difference_sq.numpy()
            print(f"Test {test_idx}: Input and Output differences: "
                  f"{input_difference_sq:.3f}, {output_difference_sq:.3f}")

            self.assertLess(output_difference_sq, input_difference_sq)

        # Check that the layer did not re-build() again,
        # and the kernel parameters are as initialized:
        # print(aol_conv.kernel_parameters[:2, :2, :2, :2])
        self.assertAlmostEqual(aol_conv.kernel_parameters[0, 0, 0, 0].numpy(),
                               10 / 1000)

    def test_orthogonality(self):
        print("\n\n*** Othogonality Test: ***")
        aol_conv = AOLConv2D(filters=16,
                             kernel_size=(3, 3),
                             kernel_initializer="orthogonal_center",
                             padding="same")

        # Do multiple tests with pseudo-random inputs:
        np.random.seed(1111)
        for test_idx in range(10):
            input_batch1 = np.random.normal(size=[1, 5, 5, 16])
            input_tensor1 = tf.convert_to_tensor(input_batch1)
            output_tensor1 = aol_conv(input_tensor1)

            input_batch2 = np.random.normal(size=[1, 5, 5, 16])
            input_tensor2 = tf.convert_to_tensor(input_batch2)
            output_tensor2 = aol_conv(input_tensor2)

            input_difference_vector = input_tensor1 - input_tensor2
            input_difference_sq = tf.reduce_sum(input_difference_vector ** 2)
            input_difference_sq = input_difference_sq.numpy()
            output_difference_vector = output_tensor1 - output_tensor2
            output_difference_sq = tf.reduce_sum(output_difference_vector ** 2)
            output_difference_sq = output_difference_sq.numpy()
            print(f"Test {test_idx}: Input and Output differences: "
                  f"{input_difference_sq:.3f}, {output_difference_sq:.3f}")

            self.assertLess(output_difference_sq, input_difference_sq)
            self.assertAlmostEqual(output_difference_sq/input_difference_sq,
                                   1., places=3)

