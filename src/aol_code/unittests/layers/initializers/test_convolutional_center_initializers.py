import unittest
import tensorflow as tf

from aol_code.layers.initializers.convolutional_center_initializers import (
    IdentityCenterInitializer,
    OrthogonalCenterInitializer,
)


class TestConvolutionalCenterInitializers(unittest.TestCase):

    def test_identity_center_initializer(self):
        identity_center_initializer = IdentityCenterInitializer()
        parameters = identity_center_initializer(shape=[3, 3, 2, 2])
        zero_matrix = [[0, 0], [0, 0]]
        bunch_of_zeros = [zero_matrix, zero_matrix, zero_matrix]
        goal_parameters = [
            bunch_of_zeros,
            [zero_matrix, [[1, 0], [0, 1]], zero_matrix],
            bunch_of_zeros
        ]
        goal_parameters = tf.convert_to_tensor(goal_parameters,
                                               dtype=tf.float32)
        difference = tf.reduce_sum((parameters - goal_parameters) ** 2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_different_dimensions_rise_error(self):
        identity_center_initializer = IdentityCenterInitializer()
        with self.assertRaises(ValueError):
            identity_center_initializer(shape=[1, 1, 3, 4])

    def test_orthogonal_center_initializer(self):
        orthogonal_center_initializer = OrthogonalCenterInitializer()
        parameters = orthogonal_center_initializer(shape=[3, 3, 5, 5])

        # Assert 0s outside the center:
        for i in range(3):
            for j in range(3):
                if i == j: continue
                vector_norm = tf.reduce_sum(parameters[i, j, :, :] ** 2)
                vector_norm = vector_norm.numpy()
                self.assertAlmostEqual(vector_norm, 0.)

        # Assert orthgonal center:
        for i in range(5):
            vector_i = parameters[1, 1, i, :]
            for j in range(5):
                vector_j = parameters[1, 1, j, :]
                pointwise_product = vector_i * vector_j
                inner_product = tf.reduce_sum(pointwise_product).numpy()
                goal_value = 1. if i == j else 0.
                self.assertAlmostEqual(inner_product, goal_value, places=5)

