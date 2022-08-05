
import unittest
import tensorflow as tf

from aol_code.layers.initializers import RepeatedOrthogonalInitializer


class TestRepeatedOrthogonalInitializer(unittest.TestCase):

    def test_square_matrix_intialization(self):
        repeated_orthogonal_initializer = RepeatedOrthogonalInitializer()
        parameters = repeated_orthogonal_initializer(shape=[6, 6])
        jacobian = tf.matmul(a=parameters, b=parameters, transpose_b=True)
        print(f"\n\nJacobian:\n{tf.round(100*jacobian)/100}")
        goal = tf.eye(6, dtype=tf.float32)
        difference = tf.reduce_sum((jacobian-goal)**2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_intialization_of_matrix_with_more_rows_than_columns(self):
        repeated_orthogonal_initializer = RepeatedOrthogonalInitializer()
        parameters = repeated_orthogonal_initializer(shape=[4, 5])
        jacobian = tf.matmul(a=parameters, b=parameters, transpose_b=True)
        print(f"\n\nJacobian:\n{tf.round(100*jacobian)/100}")
        goal = tf.eye(4, dtype=tf.float32)
        difference = tf.reduce_sum((jacobian-goal)**2).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_intialization_of_matrix_with_more_columns_than_rows(self):
        repeated_orthogonal_initializer = RepeatedOrthogonalInitializer()
        parameters = repeated_orthogonal_initializer(shape=[5, 4])
        jacobian = tf.matmul(a=parameters, b=parameters, transpose_b=True)
        print(f"\n\nJacobian:\n{tf.round(100*jacobian)/100}")

        # Test the rows of the jacobian sum to 1:
        row_sums = tf.reduce_sum(jacobian, axis=1)
        difference = tf.reduce_sum((row_sums-1.)**2).numpy()
        self.assertAlmostEqual(difference, 0.)

        # Check every entry of the jacobian is either equal to the row maximum
        # or zero:
        diff_0_sq = jacobian**2
        row_max = tf.reduce_max(jacobian, axis=1, keepdims=True)
        diff_max_sq = (jacobian-row_max)**2
        min_diff_sq = tf.minimum(diff_0_sq, diff_max_sq)
        difference = tf.reduce_sum(min_diff_sq).numpy()
        self.assertAlmostEqual(difference, 0.)

    def test_intialization_of_2x2_kernel_tensor(self):
        repeated_orthogonal_initializer = RepeatedOrthogonalInitializer()
        parameters = repeated_orthogonal_initializer(shape=[2, 2, 3, 12])
        parameters = tf.reshape(parameters, [12, 12])
        jacobian = tf.matmul(a=parameters, b=parameters, transpose_b=True)
        goal = tf.eye(12, dtype=tf.float32)
        difference = tf.reduce_sum((jacobian-goal)**2).numpy()
        self.assertAlmostEqual(difference, 0.)


if __name__ == "__main__":
    unittest.main()
