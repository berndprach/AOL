import tensorflow as tf
import time


class LogTimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.val_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs["time"] = time.time() - self.epoch_start_time
        logs["val_time"] = time.time() - self.val_start_time

    def on_test_begin(self, logs=None):
        self.val_start_time = time.time()


