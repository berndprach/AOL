import unittest
import tensorflow as tf

from framework.achitectures.layers.initializers.convolutional_center_initializers import (
    IdentityCenterInitializer,
    OrthogonalCenterInitializer,
)


class TestConvolutionalCenterInitializers(unittest.TestCase):

    def test_identity_center_initializer(self):
        identity_center_initializer = IdentityCenterInitializer()
        parameters = identity_center_initializer(shape=[3, 3, 2, 2])
        bunch_of_zeros = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        goal_parameters = [
            bunch_of_zeros,
            [[[0, 0], [0, 0]], [[1, 0], [0, 1]], [[0, 0], [0, 0]]],
            bunch_of_zeros
        ]
        goal_parameters = tf.convert_to_tensor(goal_parameters, dtype=tf.float32)
        difference = tf.reduce_sum((parameters - goal_parameters) ** 2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_orthogonal_center_initializer(self):
        orthogonal_center_initializer = OrthogonalCenterInitializer()
        parameters = orthogonal_center_initializer(shape=[3, 3, 5, 5])

        # Assert 0s outside the center:
        for i in range(3):
            for j in range(3):
                if i == j: continue
                vector_norm = tf.reduce_sum(parameters[i, j, :, :] ** 2).numpy()
                self.assertAlmostEqual(vector_norm, 0.)

        # Assert orthgonal center:
        for i in range(5):
            for j in range(5):
                pointwise_product = parameters[1, 1, i, :] * parameters[1, 1, j, :]
                inner_product = tf.reduce_sum(pointwise_product).numpy()
                goal_value = 1. if i == j else 0.
                self.assertAlmostEqual(inner_product, goal_value, places=5)

    def test_different_dimensions_rise_error(self):
        identity_center_initializer = IdentityCenterInitializer()
        with self.assertRaises(ValueError):
            identity_center_initializer(shape=[1, 1, 3, 4])


if __name__ == "__main__":
    unittest.main()
