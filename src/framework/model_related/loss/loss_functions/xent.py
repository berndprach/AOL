
import tensorflow as tf

from framework.model_related.loss.loss_functions.loss_function import LossFunction
from framework.model_related.metrics.metric_options import MetricOption

from framework.general_code.directory import Directory


class Xent(LossFunction):
    name = "Xent"

    def __init__(self, from_logits=False):
        super().__init__()
        self.from_logits = from_logits

    def __call__(self, label_batch, prediction_batch):
        return tf.keras.metrics.categorical_crossentropy(
            label_batch,
            prediction_batch,
            from_logits=self.from_logits,
        )


class XentFromScores(Xent):
    def __init__(self):
        super().__init__(from_logits=True)


class OffsetXentFromScores(LossFunction):
    def __init__(self, offset=0., temperature=1.):
        super().__init__()
        self.offset = offset
        self.temperature = temperature
        self.name = f"{self.offset:.3g}-OffsetXent"

    def __call__(self, label_batch, prediction_batch):
        offset_prediction_batch = prediction_batch - self.offset * label_batch
        offset_prediction_batch /= self.temperature
        return tf.keras.metrics.categorical_crossentropy(
            label_batch,
            offset_prediction_batch,
            from_logits=True,
        ) * self.temperature


def register_xent(loss_function_directory: Directory):
    loss_function_directory.register(
        Xent,
        "xent",
    )

    loss_function_directory.register(
        XentFromScores,
        "xent",
        MetricOption.FROM_SCORES.value,
    )

    loss_function_directory.register(
        OffsetXentFromScores,
        "offset_xent",
        MetricOption.FROM_SCORES.value,
    )
