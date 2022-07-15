
import tensorflow as tf

from typing import Dict, List


class LogModelLossCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history_dict: Dict[str: List] = None

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            return  # I think losses are still symbolic at this point?

        model_loss = 0.
        for layer in self.model.layers:
            for loss in layer.losses:
                model_loss += loss.numpy()

        logs["model_loss"] = model_loss
        # logs["total_loss"] = logs.get("loss", 0.) + model_loss
