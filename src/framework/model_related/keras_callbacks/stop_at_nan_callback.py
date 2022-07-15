
import math
import tensorflow as tf

from framework.location_manager import location_manager as loc


class StopAtNaNCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if math.isnan(logs["loss"]):
            loc.print_pro(f"Loss is NaN! Stopping training.")
            self.model.stop_training = True


