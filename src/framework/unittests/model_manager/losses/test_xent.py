
import math
import unittest
import tensorflow as tf

from framework.model_related.loss.loss_function_factory import loss_function_directory


class TestXent(unittest.TestCase):

    def test_uniform(self):
        Xent = loss_function_directory.get("xent")
        xent = Xent()
        nrof_classes = 10
        pred_tensor = tf.convert_to_tensor([1/nrof_classes for _ in range(10)])
        label_tensor = tf.convert_to_tensor([(1. if i == 0 else 0.) for i in range(10)])
        loss = xent(label_tensor, pred_tensor)
        self.assertAlmostEqual(loss, math.log(nrof_classes))

    def test_uniform_from_scores(self):
        Xent = loss_function_directory.get("xent", "from_scores")
        xent = Xent()
        nrof_classes = 10
        pred_tensor = tf.convert_to_tensor([0. for _ in range(10)])
        label_tensor = tf.convert_to_tensor([(1. if i == 0 else 0.) for i in range(10)])
        loss = xent(label_tensor, pred_tensor)
        self.assertAlmostEqual(loss, math.log(nrof_classes))

    def test_perfect_prediction(self):
        Xent = loss_function_directory.get("xent")
        xent = Xent()
        nrof_classes = 10
        pred_tensor = tf.convert_to_tensor([(1. if i == 0 else 0.) for i in range(nrof_classes)])
        label_tensor = tf.convert_to_tensor([(1. if i == 0 else 0.) for i in range(nrof_classes)])
        loss = xent(label_tensor, pred_tensor).numpy()
        # print(loss)
        self.assertAlmostEqual(loss, 0., places=6)

    def test_uniform_offset(self):
        Xent = loss_function_directory.get("offset_xent", "from_scores")
        xent = Xent(offset=1.)
        nrof_classes = 10
        pred_tensor = tf.convert_to_tensor([0. for _ in range(10)])
        label_tensor = tf.convert_to_tensor([(1. if i == 0 else 0.) for i in range(10)])
        loss = xent(label_tensor, pred_tensor)
        self.assertAlmostEqual(loss, 1 + math.log(math.exp(-1.) + nrof_classes-1))


if __name__ == "__main__":
    unittest.main()
