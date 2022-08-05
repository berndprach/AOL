
import math
import unittest
import tensorflow as tf

from aol_code.metrics import CertifiedRobustAccuracyFromScore


class TestCertifiedRobustAccuracyFromScore(unittest.TestCase):

    def test_L2_cra(self):
        cra_from_scores = CertifiedRobustAccuracyFromScore(
            maximal_perturbation=1., norm="L2")

        e1_vector = tf.convert_to_tensor([(1. if i == 0 else 0.)
                                          for i in range(10)])

        label1 = e1_vector
        pred1 = e1_vector  # score margin < sqrt(2)

        label2 = e1_vector
        pred2 = 2*e1_vector  # score margin > sqrt(2)

        label_batch = tf.stack([label1, label2], axis=0)
        pred_batch = tf.stack([pred1, pred2], axis=0)

        cra = cra_from_scores(label_batch, pred_batch)
        print(f"\n\nCRA: {cra}")

        # self.assertAlmostEqual(cra.numpy(), 1/2)
        self.assertAlmostEqual(cra[0], 0.)
        self.assertAlmostEqual(cra[1], 1.)


if __name__ == "__main__":
    unittest.main()
