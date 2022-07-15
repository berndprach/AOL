
import unittest
import tensorflow as tf

from framework.achitectures.layers.initializers.repeated_orthogonal_initializer import RepeatedOrthogonalInitializer


class TestMaxMinActivation(unittest.TestCase):

    def test_pointwise_square_shape(self):
        repeated_orthogonal_initializer = RepeatedOrthogonalInitializer()
        parameters = repeated_orthogonal_initializer(shape=[1, 1, 6, 6])
        parameters = parameters[0, 0, :, :]
        jacobian = tf.matmul(a=parameters, b=parameters, transpose_b=True)
        # print(f"\n\nJacobian:\n{jacobian}")
        goal = tf.eye(6, dtype=tf.float32)
        difference = tf.reduce_sum((jacobian-goal)**2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_pointwise_higher_output_dimensions(self):
        repeated_orthogonal_initializer = RepeatedOrthogonalInitializer()
        parameters = repeated_orthogonal_initializer(shape=[1, 1, 4, 5])
        parameters = parameters[0, 0, :, :]
        jacobian = tf.matmul(a=parameters, b=parameters, transpose_b=True)
        print(f"\n\nJacobian:\n{jacobian}")
        goal = tf.eye(4, dtype=tf.float32)
        difference = tf.reduce_sum((jacobian-goal)**2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_pointwise_higher_input_dimension(self):
        repeated_orthogonal_initializer = RepeatedOrthogonalInitializer()
        parameters = repeated_orthogonal_initializer(shape=[1, 1, 5, 4])
        parameters = parameters[0, 0, :, :]
        jacobian = tf.matmul(a=parameters, b=parameters, transpose_b=True)
        # print(f"\n\nJacobian:\n{jacobian}")

        # Test the rows of the jacobian sum to 1:
        row_sums = tf.reduce_sum(jacobian, axis=1)
        difference = tf.reduce_sum((row_sums-1.)**2).numpy()
        self.assertAlmostEqual(difference, 0.)

        # Check every entry of the jacobian is either equal to the row maximum or zero:
        diff_0_sq = jacobian**2
        row_max = tf.reduce_max(jacobian, axis=1, keepdims=True)
        diff_max_sq = (jacobian-row_max)**2
        min_diff_sq = tf.minimum(diff_0_sq, diff_max_sq)
        difference = tf.reduce_sum(min_diff_sq).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_spatial_2x2_shape(self):
        repeated_orthogonal_initializer = RepeatedOrthogonalInitializer()
        parameters = repeated_orthogonal_initializer(shape=[2, 2, 3, 12])
        parameters = tf.reshape(parameters, [12, 12])
        jacobian = tf.matmul(a=parameters, b=parameters, transpose_b=True)
        # print(f"\n\nJacobian:\n{jacobian}")
        goal = tf.eye(12, dtype=tf.float32)
        difference = tf.reduce_sum((jacobian-goal)**2).numpy()
        self.assertAlmostEqual(difference, 0.)


if __name__ == "__main__":
    unittest.main()
