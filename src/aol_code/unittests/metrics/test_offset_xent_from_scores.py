
import math
import unittest
import tensorflow as tf

from aol_code.metrics import OffsetXentFromScores


class TestXent(unittest.TestCase):

    def test_uniform_offset(self):
        offset_xent_from_scores = OffsetXentFromScores(offset=1.)
        nrof_classes = 10
        pred_tensor = tf.convert_to_tensor([0. for _ in range(10)])
        label_tensor = tf.convert_to_tensor([(1. if i == 0 else 0.)
                                             for i in range(10)])
        loss = offset_xent_from_scores(label_tensor, pred_tensor)
        self.assertAlmostEqual(loss,
                               1 + math.log(math.exp(-1.) + nrof_classes-1))


if __name__ == "__main__":
    unittest.main()
