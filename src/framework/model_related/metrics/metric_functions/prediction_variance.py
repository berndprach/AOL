"""
Considers a batch,
centers it (subtracts the mean),
and caluculates the squared L2 norms.
"""
import tensorflow as tf

from framework.general_code.directory import Directory

from framework.model_related.metrics.metric import Metric
from framework.model_related.metrics.metric_options import MetricOption


class Variance(Metric):
    name = "var"
    percentable = False

    def __call__(self, y_true_batch, s_pred_batch):
        s_pred_batch = s_pred_batch - tf.reduce_mean(s_pred_batch, axis=0, keepdims=True)
        return tf.reduce_sum(s_pred_batch**2, axis=-1)


def register_variance_metrics(metric_directory: Directory):
    metric_directory.register(
        Variance,
        "variance",
    )

    metric_directory.register(
        Variance,
        "variance",
        MetricOption.FROM_SCORES.value,
    )
