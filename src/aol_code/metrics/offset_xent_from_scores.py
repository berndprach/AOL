
import tensorflow as tf


class OffsetXentFromScores:
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
