
import tensorflow as tf

from framework.general_code.directory import Directory

from framework.model_related.metrics.metric import Metric
from framework.model_related.metrics.metric_options import MetricOption


class MeanCertainty(Metric):
    name = "cert"

    def __call__(self, y_true_batch, y_pred_batch):
        certs = y_true_batch * y_pred_batch
        certs = tf.reduce_sum(certs, axis=-1)
        return certs


class MeanCertaintyFromScores(Metric):
    name = "cert"

    def __call__(self, y_true_batch, s_pred_batch):
        y_pred_batch = tf.math.softmax(s_pred_batch, axis=-1)
        certs = y_true_batch * y_pred_batch
        certs = tf.reduce_sum(certs, axis=-1)
        return certs


def register_certainty_metrics(metric_directory: Directory):
    metric_directory.register(
        MeanCertainty,
        "certainty",
    )

    metric_directory.register(
        MeanCertaintyFromScores,
        "certainty",
        MetricOption.FROM_SCORES.value,
    )
